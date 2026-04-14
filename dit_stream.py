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

IO_DIR = "/workspace/inspatio-world/interactive_io"
POSE_FILE = os.path.join(IO_DIR, "pose.json")
FRAMES_DIR = os.path.join(IO_DIR, "frames")
STATUS_FILE = os.path.join(IO_DIR, "status.json")

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
    
    # Force scout resolution for speed
    config.dataset.video_size = [240, 416]
    config.denoising_step_list = [1000, 250]  # 2 steps for speed
    
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
    
    # ── Compile DiT ──
    write_status("compiling_dit")
    import torch._inductor.config as ic
    ic.fx_graph_cache = True
    torch._dynamo.config.cache_size_limit = 32
    cache_dir = "/dev/shm/torchinductor_stream"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    
    pipeline.generator.model = torch.compile(
        pipeline.generator.model,
        mode="max-autotune",
        fullgraph=False,
        dynamic=False,
        backend="inductor",
    )
    print("DiT compiled.", flush=True)
    
    # ── DiT warmup ──
    write_status("warming_up")
    F_warmup = num_frame_per_block
    # Latent dims for 240x416: h=30, w=52
    lat_h, lat_w = 30, 52
    
    dummy_noise = torch.randn(1, F_warmup, 16, lat_h, lat_w, device=device, dtype=torch.bfloat16)
    dummy_render = torch.randn(1, F_warmup, 20, lat_h, lat_w, device=device, dtype=torch.bfloat16)
    dummy_cond = {"prompt_embeds": torch.randn(1, 512, 4096, device=device, dtype=torch.bfloat16)}
    
    warmup_sizes = [3, 6]
    for wi, n_ctx in enumerate(warmup_sizes):
        kv_size = n_ctx * (lat_h * lat_w)
        dummy_ctx = torch.randn(1, n_ctx, 36, lat_h, lat_w, device=device, dtype=torch.bfloat16)
        print(f"  Compile warmup {wi+1}/{len(warmup_sizes)} (kv_size={kv_size})...", flush=True)
        for _ in range(3):
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
    dataset_config['video_size'] = [240, 416]
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
        render_block = render_latent[:, start_index:start_index + num_frame_per_block]
        mask_block = mask_latent[:, start_index:start_index + num_frame_per_block]
        ref_block = ref_latent[:, start_index:start_index + num_frame_per_block]
        
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
        
        # Check for pause/stop
        pose = read_pose()
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
