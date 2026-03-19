import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import CausalInferencePipeline
from utils.misc import set_seed

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller
from safetensors.torch import load_file

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint file")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--json_path", type=str, help="Path to the json file")
parser.add_argument("--version", type=str, default="version_0", help="Output version subfolder name")
args = parser.parse_args()

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    rank = 0
    set_seed(args.seed)

print(f'[Rank {rank}] Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

# Initialize pipeline
pipeline = CausalInferencePipeline(config, device=device)

checkpoint_name = "None"
method_name = "default"
if args.checkpoint_path:
    print(f"[Rank {rank}] Loading checkpoint from {args.checkpoint_path}")
    state_dict = load_file(args.checkpoint_path)
    mismatch, missing = pipeline.generator.load_state_dict(state_dict, strict=False)
    print(f"[Rank {rank}] Mismatch: {mismatch}, Missing: {missing}")
    checkpoint_name = args.checkpoint_path.split("/")[-2]
    method_name = args.checkpoint_path.split("/")[-3]

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
else:
    pipeline.text_encoder.to(device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)

from datasets.video_dataset import VideoDataset
dataset_config = OmegaConf.to_container(config.dataset, resolve=True)
if args.json_path:
    dataset_config['json_path'] = args.json_path
dataset = VideoDataset(**dataset_config)
print(f"[Rank {rank}] Number of videos: {len(dataset)}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

output_dir = os.path.join(args.output_folder, method_name, checkpoint_name)
os.makedirs(output_dir, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

from utils.render_warper import convert_mask_video

for i, batch_data in tqdm(enumerate(dataloader), total=len(dataloader), disable=(rank != 0), desc=f"Rank {rank}"):
    if dist.is_initialized():
        global_idx = i * world_size + rank
    else:
        global_idx = i

    batch = batch_data if isinstance(batch_data, dict) else batch_data[0]

    # Load pre-rendered render/mask videos from batch (produced by offline point cloud rendering)
    render_videos_ori = batch["render_video"].to(device, dtype=torch.bfloat16)
    render_videos_ori = rearrange(render_videos_ori, 'b t c h w -> b c t h w')
    mask_videos_ori = batch["mask_video"].to(device, dtype=torch.bfloat16)
    mask_videos_ori = rearrange(mask_videos_ori, 'b t c h w -> b c t h w')

    render_latent = pipeline.vae.encode_to_latent(render_videos_ori).to(device, dtype=torch.bfloat16)
    mask_latent = convert_mask_video(mask_videos_ori)

    text_prompts = batch["text"]
    if "target_video" in batch:
        target_video = batch["target_video"].to(device=device, dtype=torch.bfloat16)
    else:
        target_video = batch["source_video"].to(device=device, dtype=torch.bfloat16)
    target_video = rearrange(target_video, 'b t c h w -> b c t h w')
    latent = pipeline.vae.encode_to_latent(target_video).to(device=device, dtype=torch.bfloat16)

    ref_video = batch["source_video"].to(device=device, dtype=torch.bfloat16)
    ref_video = rearrange(ref_video, 'b t c h w -> b c t h w')
    ref_latent = pipeline.vae.encode_to_latent(ref_video).to(device=device, dtype=torch.bfloat16)

    latent_length = latent.shape[1]
    if latent_length % config.num_frame_per_block != 0:
        num_output_frames = latent_length - latent_length % config.num_frame_per_block
    else:
        num_output_frames = latent_length
    sampled_noise = torch.randn(
        [args.num_samples, num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
    )

    render_latent = render_latent[:, :num_output_frames, ...].to(device=device, dtype=torch.bfloat16)
    mask_latent = mask_latent[:, :num_output_frames, ...].to(device=device, dtype=torch.bfloat16)
    latent = latent[:, :num_output_frames, ...].to(device=device, dtype=torch.bfloat16)

    video = pipeline.inference(
        noise=sampled_noise,
        text_prompts=text_prompts,
        ref_latent=ref_latent,
        render_latent=render_latent,
        mask_latent=mask_latent,
    )
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()

    source_video = rearrange(target_video, 'b c t h w -> b t h w c').cpu()
    source_video = (source_video * 0.5 + 0.5).clamp(0, 1)

    render_video = rearrange(render_videos_ori, 'b c t h w -> b t h w c').cpu()
    render_video = (render_video * 0.5 + 0.5).clamp(0, 1)

    pred_video = 255.0 * current_video
    source_video_out = 255.0 * source_video
    render_video_out = 255.0 * render_video
    pipeline.vae.model.clear_cache()

    output_dir = os.path.join(args.output_folder, method_name, checkpoint_name, args.version)
    os.makedirs(output_dir, exist_ok=True)

    for seed_idx in range(args.num_samples):
        write_video(os.path.join(output_dir, f'{global_idx}-pred_video_rank{rank}.mp4'), pred_video[seed_idx], fps=24)
        write_video(os.path.join(output_dir, f'{global_idx}-source_video_rank{rank}.mp4'), source_video_out[seed_idx], fps=24)
        write_video(os.path.join(output_dir, f'{global_idx}-render_video_rank{rank}.mp4'), render_video_out[seed_idx], fps=24)

        if 'target_extrinsics' in batch:
            target_extrinsics = batch["target_extrinsics"].float().to(device=device)
            torch.save(target_extrinsics, os.path.join(output_dir, f'extrinsics_{global_idx}.pt'))

if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()

print(f"[Rank {rank}] Inference completed!")
