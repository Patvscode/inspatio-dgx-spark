#!/usr/bin/env python3
"""Streaming DiT inference server — runs INSIDE Docker container.

Loads model once, then continuously generates frames block-by-block.
Reads camera pose from interactive_io/pose.json
Writes frames to interactive_io/frames/frame_NNNN.jpg

Usage (from host):
    docker exec inspatio-world bash -c "cd /workspace/inspatio-world && \
        TORCH_CUDA_ARCH_LIST=12.1a \
        TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
        TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache \
        python3 dit_stream.py"
"""

import gc
import json
import os
import sys
import time

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file

# Add project to path
sys.path.insert(0, "/workspace/inspatio-world")

from demo_utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb, gpu
from pipeline import CausalInferencePipeline
from pipeline.causal_inference import denoise_block
from utils.misc import set_seed
from utils.render_warper import convert_mask_video

# Live camera control
try:
    from live_camera import LiveCamera
    LIVE_CAMERA_AVAILABLE = True
except ImportError:
    LIVE_CAMERA_AVAILABLE = False
    print("[WARN] live_camera module not found, using pre-baked trajectory only", flush=True)

IO_DIR = "/workspace/inspatio-world/interactive_io"
POSE_FILE = os.path.join(IO_DIR, "pose.json")
FRAMES_DIR = os.path.join(IO_DIR, "frames")
STATUS_FILE = os.path.join(IO_DIR, "status.json")
QUALITY_FILE = os.path.join(IO_DIR, "quality.json")

# Initialize for single-GPU (no distributed)
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29517"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["LOCAL_RANK"] = "0"
torch.distributed.init_process_group(backend='nccl', world_size=1, rank=0)
torch.cuda.set_device(0)

device = torch.device("cuda")
set_seed(42)

def write_status(status, **kwargs):
    data = {"status": status, "timestamp": time.time(), **kwargs}
    with open(STATUS_FILE, 'w') as f:
        json.dump(data, f)
    print(f"[STATUS] {status}", flush=True)

def read_pose():
    try:
        with open(POSE_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {"yaw": 0, "pitch": 0, "zoom": 1.0, "paused": False, "stop": False}

def save_frame(frame_tensor, frame_idx):
    """Save a single frame as JPEG. frame_tensor is [H, W, 3] float [0,1]."""
    img = (frame_tensor.clamp(0, 1) * 255).byte().cpu().numpy()
    Image.fromarray(img).save(
        os.path.join(FRAMES_DIR, f"frame_{frame_idx:06d}.jpg"),
        quality=85
    )

def main():
    os.makedirs(FRAMES_DIR, exist_ok=True)
    
    # Clean old frames
    for f in os.listdir(FRAMES_DIR):
        os.remove(os.path.join(FRAMES_DIR, f))
    
    write_status("loading_model")
    
    # ── Load config ──
    config = OmegaConf.load("configs/inference_1.3b.yaml")
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    
    # Read quality settings from quality.json
    init_h, init_w = 240, 416
    init_steps = [1000, 250]
    try:
        with open(QUALITY_FILE, 'r') as f:
            qcfg = json.load(f)
        res_map = {"scout": (240, 416), "draft": (360, 624), "full": (480, 832)}
        init_h, init_w = res_map.get(qcfg.get("quality", "scout"), (240, 416))
        step_map = {2: [1000, 250], 3: [1000, 500, 250], 4: [1000, 750, 500, 250]}
        init_steps = step_map.get(qcfg.get("steps", 2), [1000, 250])
        print(f"[QUALITY] Startup: {init_h}x{init_w}, steps={init_steps}", flush=True)
    except Exception:
        print("[QUALITY] No quality.json, using defaults (240x416, 2 steps)", flush=True)
    
    config.dataset.video_size = [init_h, init_w]
    config.denoising_step_list = init_steps
    
    num_frame_per_block = getattr(config, "num_frame_per_block", 3)
    
    # ── Initialize pipeline ──
    write_status("loading_pipeline")
    pipeline = CausalInferencePipeline(config, device=device)
    
    checkpoint_path = "./checkpoints/InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors"
    print(f"Loading checkpoint from {checkpoint_path}", flush=True)
    state_dict = load_file(checkpoint_path)
    pipeline.generator.load_state_dict(state_dict, strict=False)
    
    pipeline = pipeline.to(dtype=torch.bfloat16)
    
    free_vram = get_cuda_free_memory_gb(gpu)
    low_memory = free_vram < 40
    print(f"Free VRAM: {free_vram:.1f} GB, low_memory={low_memory}", flush=True)
    
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    else:
        pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    
    # ── Load TAE ──
    write_status("loading_tae")
    from utils.taehv import TAEHV
    tae_path = "./checkpoints/taehv/taew2_1.pth"
    tae_model = TAEHV(checkpoint_path=tae_path).to(device, torch.float16)
    tae_model.eval()
    
    # TAE warmup
    print("Warming up TAE...", flush=True)
    with torch.no_grad():
        dummy = torch.randn(1, 3, 3, 240, 416, device=device, dtype=torch.float16)
        _ = tae_model.decode_video(
            torch.randn(1, 1, tae_model.latent_channels, 30, 52, device=device, dtype=torch.float16),
            show_progress_bar=False
        )
        del dummy
    torch.cuda.synchronize()
    print("TAE warmup done.", flush=True)
    
    # ── KV cache ──
    pipeline._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=device)
    
    def reset_kv_cache():
        for bc in pipeline.kv_cache1:
            bc['k'].detach_().zero_()
            bc['v'].detach_().zero_()
    
    # ── Optional DiT compile ──
    lat_h, lat_w = init_h // 8, init_w // 8
    use_torch_compile = os.environ.get("INSPATIO_USE_TORCH_COMPILE", "0").lower() in ("1", "true", "yes")
    if use_torch_compile:
        write_status("compiling_dit")
        import torch._inductor.config as ic
        ic.fx_graph_cache = True
        torch._dynamo.config.cache_size_limit = 32
        cache_dir = "/dev/shm/torchinductor_stream"
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir

        pipeline.generator.model = torch.compile(
            pipeline.generator.model,
            mode="reduce-overhead",
            fullgraph=False,
            dynamic=False,
            backend="inductor",
        )
        print("DiT compiled.", flush=True)

        # ── DiT warmup ──
        write_status("warming_up")
        F_warmup = num_frame_per_block
        print(f"Latent dims: {lat_h}x{lat_w} (from {init_h}x{init_w})", flush=True)

        dummy_noise = torch.randn(1, F_warmup, 16, lat_h, lat_w, device=device, dtype=torch.bfloat16)
        dummy_render = torch.randn(1, F_warmup, 20, lat_h, lat_w, device=device, dtype=torch.bfloat16)
        dummy_cond = {"prompt_embeds": torch.randn(1, 512, 4096, device=device, dtype=torch.bfloat16)}

        warmup_sizes = [3]
        for wi, n_ctx in enumerate(warmup_sizes):
            kv_size = n_ctx * (lat_h * lat_w)
            dummy_ctx = torch.randn(1, n_ctx, 36, lat_h, lat_w, device=device, dtype=torch.bfloat16)
            print(f"  Compile warmup {wi+1}/{len(warmup_sizes)} (kv_size={kv_size})...", flush=True)
            reset_kv_cache()
            denoise_block(
                pipeline.generator, pipeline.scheduler, dummy_noise, dummy_cond,
                pipeline.kv_cache1,
                context_frames=dummy_ctx, context_no_grad=True, context_freqs_offset=0,
                render_block=dummy_render, denoising_kv_size=kv_size,
                denoising_steps=pipeline.denoising_step_list,
            )
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()

        del dummy_noise, dummy_render, dummy_cond
        torch.cuda.empty_cache()
        gc.collect()
        reset_kv_cache()
    else:
        write_status("warming_up", message="Skipping Torch compile for faster startup")
        print(f"Latent dims: {lat_h}x{lat_w} (from {init_h}x{init_w})", flush=True)
        print("[STARTUP] INSPATIO_USE_TORCH_COMPILE=0, skipping torch.compile warmup for faster first frame", flush=True)
    
    write_status("ready", message="Model loaded. Waiting for scene data.")
    print("="*60, flush=True)
    print("STREAMING SERVER READY", flush=True)
    print("="*60, flush=True)
    
    # ── Load dataset ──
    from datasets.video_dataset import VideoDataset
    
    dataset_config = OmegaConf.to_container(config.dataset, resolve=True)
    json_path = "./user_input/new.json"
    if not os.path.exists(json_path):
        write_status("error", message="No scene data. Run batch pipeline first.")
        return
    
    dataset_config['json_path'] = json_path
    dataset_config['video_size'] = [init_h, init_w]
    dataset = VideoDataset(**dataset_config)
    
    if len(dataset) == 0:
        write_status("error", message="No videos in dataset")
        return
    
    write_status("loading_scene")
    batch = dataset[0]
    
    # Prepare tensors
    render_videos_ori = batch["render_video"].unsqueeze(0).to(device, dtype=torch.bfloat16)
    render_videos_ori = rearrange(render_videos_ori, 'b t c h w -> b c t h w')
    mask_videos_ori = batch["mask_video"].unsqueeze(0).to(device, dtype=torch.bfloat16)
    mask_videos_ori = rearrange(mask_videos_ori, 'b t c h w -> b c t h w')
    
    text_prompts = [batch["text"]] if isinstance(batch["text"], str) else batch["text"]
    
    source_video = batch["source_video"].unsqueeze(0).to(device, dtype=torch.bfloat16)
    source_video = rearrange(source_video, 'b t c h w -> b c t h w')
    
    # ── Live Camera Setup ──
    live_cam = None
    if LIVE_CAMERA_AVAILABLE:
        try:
            with open(json_path, 'r') as f:
                scene_data = json.load(f)
            if scene_data:
                da3_path = scene_data[0].get("vggt_depth_path", "")
                if da3_path.startswith("./"):
                    da3_path = os.path.join("/workspace/inspatio-world", da3_path[2:])
                live_cam = LiveCamera(da3_path, device, init_w, init_h)
                if live_cam.load():
                    write_status("camera_ready", message="Live camera control active")
                    print("[CAMERA] Live camera control ACTIVE", flush=True)
                else:
                    live_cam = None
                    print("[CAMERA] Failed to load, using pre-baked trajectory", flush=True)
        except Exception as e:
            print(f"[CAMERA] Init error: {e}, using pre-baked trajectory", flush=True)
            live_cam = None
    
    # TAE encode
    def tae_encode(video_bcthw):
        video = video_bcthw.permute(0, 2, 1, 3, 4)
        video = ((video * 0.5 + 0.5).clamp(0, 1)).to(torch.float16)
        latent = tae_model.encode_video(video, show_progress_bar=False)
        return latent.to(torch.bfloat16)
    
    def tae_decode(latent):
        video = tae_model.decode_video(latent.to(torch.float16), show_progress_bar=False)
        return video.float()
    
    write_status("encoding")
    render_latent = tae_encode(render_videos_ori)
    mask_latent = convert_mask_video(mask_videos_ori)
    ref_latent = tae_encode(source_video)
    
    latent_length = render_latent.shape[1]
    if latent_length % num_frame_per_block != 0:
        num_output_frames = latent_length - latent_length % num_frame_per_block
    else:
        num_output_frames = latent_length
    
    render_latent = render_latent[:, :num_output_frames].to(device, dtype=torch.bfloat16)
    mask_latent = mask_latent[:, :num_output_frames].to(device, dtype=torch.bfloat16)
    ref_latent = ref_latent[:, :num_output_frames].to(device, dtype=torch.bfloat16)
    
    # Text encoding
    conditional_dict = pipeline.text_encoder(text_prompts=text_prompts)
    
    B = 1
    C = 16
    num_blocks = num_output_frames // num_frame_per_block
    
    write_status("streaming", total_blocks=num_blocks, total_frames=num_output_frames)
    print(f"Starting streaming: {num_blocks} blocks, {num_output_frames} frames", flush=True)
    
    # ── STREAMING LOOP ──
    frame_counter = 0
    block_idx = 0
    last_pred = None
    last_reset_token = None
    reset_kv_cache()
    
    noise = torch.randn(
        [1, num_output_frames, C, lat_h, lat_w], device=device, dtype=torch.bfloat16
    )
    
    seq_len = lat_h * lat_w  # 30*52 = 1560
    
    while True:
        start_index = block_idx * num_frame_per_block
        if start_index >= num_output_frames:
            # Loop back to start for continuous playback
            write_status("looping", message="Restarting from beginning")
            block_idx = 0
            start_index = 0
            last_pred = None
            reset_kv_cache()
            # Re-generate noise for variety
            noise = torch.randn_like(noise)
            continue
        
        t_block_start = time.time()
        
        noisy_input = noise[:, start_index:start_index + num_frame_per_block]
        ref_block = ref_latent[:, start_index:start_index + num_frame_per_block]
        
        # ── Camera control: live render or pre-baked ──
        pose = read_pose()
        reset_token = pose.get("resetToken")
        if live_cam is not None and reset_token != last_reset_token:
            live_cam.reset()
            last_reset_token = reset_token
            print("[CAMERA] Reset live camera accumulators", flush=True)

        # Left joystick: moveX/moveY (translation), Right joystick: lookX/lookY (rotation)
        joystick_yaw = pose.get("lookX", pose.get("yaw", 0))    # right stick X = yaw
        joystick_pitch = pose.get("lookY", pose.get("pitch", 0))  # right stick Y = pitch
        joystick_zoom = pose.get("zoom", 1.0)
        joystick_move_x = pose.get("moveX", 0)  # left stick = strafe
        joystick_move_y = pose.get("moveY", 0)  # left stick = dolly
        joystick_active = (abs(joystick_yaw) > 0.05 or abs(joystick_pitch) > 0.05 or
                           abs(joystick_move_x) > 0.05 or abs(joystick_move_y) > 0.05 or
                           abs(joystick_zoom - 1.0) > 0.03)
        
        if live_cam is not None and joystick_active:
            # Live rendering from point cloud with joystick control.
            # TAEHV compresses roughly 3 RGB frames -> 1 latent frame, so to drive
            # a 3-latent streaming block we must render 9 RGB frames here.
            try:
                live_rgb_frames = num_frame_per_block * 3
                render_video_block, mask_video_block = live_cam.render_block(
                    start_index * 3, live_rgb_frames,
                    joystick_yaw, joystick_pitch, joystick_zoom,
                    joystick_move_x, joystick_move_y,
                )
                if render_video_block is not None:
                    # TAE encode the live-rendered frames
                    with torch.no_grad():
                        rv = render_video_block.permute(0, 2, 1, 3, 4)  # b c t h w -> b t c h w
                        rv = ((rv * 0.5 + 0.5).clamp(0, 1)).to(torch.float16)
                        render_block = tae_model.encode_video(rv, show_progress_bar=False).to(torch.bfloat16)

                        # Match the streaming block length expected by the generator
                        if render_block.shape[1] != num_frame_per_block:
                            print(f"[CAMERA] Encoded live render produced {render_block.shape[1]} latent frames, expected {num_frame_per_block}", flush=True)
                            render_block = render_latent[:, start_index:start_index + num_frame_per_block]

                        # Mask: downsample to latent size
                        mv = mask_video_block  # [1, 1, T, H, W]
                        mv = rearrange(mv, 'b c t h w -> (b t) c h w')
                        mv = torch.nn.functional.interpolate(mv, size=(lat_h, lat_w), mode='bilinear', align_corners=False)
                        mv = rearrange(mv, '(b t) c h w -> b t c h w', b=1)
                        # Pad: replicate first frame 4x, then concat rest (matching convert_mask_video)
                        mv_padded = torch.cat([mv[:, 0:1].repeat(1, 4, 1, 1, 1), mv[:, 1:]], dim=1)
                        trim = mv_padded.shape[1] - mv_padded.shape[1] % 4
                        mv_padded = mv_padded[:, :trim]
                        mask_block = mv_padded.view(1, mv_padded.shape[1] // 4, 4, lat_h, lat_w)
                        if mask_block.shape[1] != num_frame_per_block:
                            print(f"[CAMERA] Encoded live mask produced {mask_block.shape[1]} latent frames, expected {num_frame_per_block}", flush=True)
                            mask_block = mask_latent[:, start_index:start_index + num_frame_per_block]
                else:
                    render_block = render_latent[:, start_index:start_index + num_frame_per_block]
                    mask_block = mask_latent[:, start_index:start_index + num_frame_per_block]
            except Exception as e:
                print(f"[CAMERA] Live control fallback: {e}", flush=True)
                render_block = render_latent[:, start_index:start_index + num_frame_per_block]
                mask_block = mask_latent[:, start_index:start_index + num_frame_per_block]
        else:
            # Pre-baked trajectory (no joystick input)
            render_block = render_latent[:, start_index:start_index + num_frame_per_block]
            mask_block = mask_latent[:, start_index:start_index + num_frame_per_block]
            if live_cam is not None:
                # Decay offsets back toward the base path when controls are idle
                live_cam.yaw_accum *= 0.95
                live_cam.pitch_accum *= 0.95
                live_cam.strafe_accum *= 0.90
                live_cam.forward_accum *= 0.90
                live_cam.zoom_accum += (1.0 - live_cam.zoom_accum) * 0.15
        
        render_input = torch.cat([mask_block, render_block], dim=2)
        
        kv_size = seq_len * 3
        
        # Build context
        zero_latents = torch.zeros_like(ref_block)
        ref_padded = torch.cat([ref_block, zero_latents[:, :, :4], zero_latents], dim=2)
        
        if start_index == 0:
            context_frames = ref_padded
        else:
            zero_lp = torch.zeros_like(last_pred)
            last_pred_padded = torch.cat([last_pred, zero_lp[:, :, :4], zero_lp], dim=2)
            context_frames = torch.cat([ref_padded, last_pred_padded], dim=1)
            kv_size = kv_size + seq_len * 3
        
        # Denoise
        with torch.no_grad():
            denoised_pred, _ = denoise_block(
                pipeline.generator, pipeline.scheduler, noisy_input, conditional_dict,
                pipeline.kv_cache1,
                context_frames=context_frames,
                context_no_grad=True,
                context_freqs_offset=0,
                render_block=render_input,
                denoising_kv_size=kv_size,
                denoising_steps=pipeline.denoising_step_list,
            )
        
        last_pred = denoised_pred.clone().detach()
        
        # Decode this block immediately
        with torch.no_grad():
            decoded = tae_decode(denoised_pred)  # [B, T, C, H, W]
        
        torch.cuda.synchronize()
        t_block_end = time.time()
        block_time = t_block_end - t_block_start
        
        # Save frames
        for fi in range(decoded.shape[1]):
            frame = rearrange(decoded[0, fi], 'c h w -> h w c')
            save_frame(frame, frame_counter)
            frame_counter += 1
        
        fps = num_frame_per_block / block_time
        print(f"Block {block_idx}: {block_time:.2f}s ({fps:.1f} FPS) | Frame {frame_counter}", flush=True)
        
        write_status("streaming", 
                     block=block_idx, 
                     frame=frame_counter, 
                     block_time=round(block_time, 2),
                     fps=round(fps, 2))
        
        block_idx += 1
        
        # Hot-reload quality.json (denoising steps)
        try:
            with open(QUALITY_FILE, 'r') as f:
                qcfg = json.load(f)
            new_steps = qcfg.get("steps", 2)
            step_map = {
                2: [1000, 250],
                3: [1000, 500, 250],
                4: [1000, 750, 500, 250],
            }
            new_step_list = step_map.get(new_steps, [1000, 250])
            if new_step_list != pipeline.denoising_step_list:
                pipeline.denoising_step_list = new_step_list
                print(f"[QUALITY] Denoising steps changed to {new_steps}: {new_step_list}", flush=True)
        except Exception:
            pass
        
        # Check for pause/stop (pose already read above for camera control)
        if pose.get("stop", False):
            write_status("stopped")
            print("Stop signal received. Exiting.", flush=True)
            break
        
        # Pause loop — wait until unpaused
        while pose.get("paused", False):
            write_status("paused", block=block_idx, frame=frame_counter)
            time.sleep(0.5)
            pose = read_pose()
            if pose.get("stop", False):
                write_status("stopped")
                print("Stop signal received during pause. Exiting.", flush=True)
                return


if __name__ == "__main__":
    main()
