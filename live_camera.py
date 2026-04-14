#!/usr/bin/env python3
"""Live camera control module for dit_stream.py.

Loads point cloud data and renders new views on-the-fly based on
joystick input from pose.json. Replaces the pre-baked trajectory
render_latent with live-rendered frames.
"""

import glob
import json
import math
import os
import time

import cv2
import numpy as np
import torch
from plyfile import PlyData


# ── Point cloud loading ──

def load_ply_data(ply_path, device):
    """Load point cloud from PLY file. Returns (points, colors) tensors."""
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    if len(vertex) == 0:
        return None, None
    pts = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1).astype(np.float32)
    if 'red' in vertex:
        r, g, b = vertex['red'], vertex['green'], vertex['blue']
        colors = np.stack([r, g, b], axis=-1).astype(np.float32)
        if colors.max() > 1.0:
            colors /= 255.0
    else:
        colors = np.ones_like(pts) * 0.5
    return torch.from_numpy(pts).to(device), torch.from_numpy(colors).to(device)


def load_ply_sequence(pcd_dir, device, max_frames=None, stride=1):
    """Load PLY sequence. Returns lists of (points, colors).
    
    stride > 1 loads every Nth frame to save memory.
    """
    ply_files = sorted(glob.glob(os.path.join(pcd_dir, "*.ply")))
    if not ply_files:
        return [], []
    if stride > 1:
        ply_files = ply_files[::stride]
    if max_frames:
        ply_files = ply_files[:max_frames]
    
    print(f"[CAMERA] Loading {len(ply_files)} point clouds from {pcd_dir}", flush=True)
    points_list, colors_list = [], []
    for pf in ply_files:
        pts, cols = load_ply_data(pf, device)
        points_list.append(pts)
        colors_list.append(cols)
    return points_list, colors_list


def load_intrinsic(da3_dir, device):
    """Load intrinsic matrix (3x3)."""
    path = os.path.join(da3_dir, "intrinsic.txt")
    data = np.loadtxt(path)
    K = data[0:3, :3].astype(np.float32)
    return torch.tensor(K, device=device, dtype=torch.float32)


def load_extrinsics(da3_dir, device):
    """Load extrinsics. Returns (initial_c2w, list_of_c2ws)."""
    path = os.path.join(da3_dir, "extrinsic.txt")
    data = np.loadtxt(path)
    num_frames = data.shape[0] // 3
    c2ws = []
    for i in range(num_frames):
        w2c_34 = data[i * 3:(i + 1) * 3, :4].astype(np.float32)
        w2c = np.vstack([w2c_34, np.array([[0, 0, 0, 1]], dtype=np.float32)])
        w2c_t = torch.tensor(w2c, dtype=torch.float32, device=device)
        c2w = torch.linalg.inv(w2c_t)
        c2ws.append(c2w)
    return c2ws[0], c2ws


def scale_intrinsic(K, target_w, target_h):
    """Scale intrinsic to target resolution."""
    orig_cx = K[0, 2].item()
    orig_cy = K[1, 2].item()
    sx = target_w / (orig_cx * 2)
    sy = target_h / (orig_cy * 2)
    K_s = K.clone()
    K_s[0, 0] *= sx
    K_s[1, 1] *= sy
    K_s[0, 2] = target_w / 2.0
    K_s[1, 2] = target_h / 2.0
    return K_s


# ── Rendering ──

MIN_DEPTH = 0.1
DEPTH_EPS = 1e-4
_splat_cache = {}


def render_frame(points, colors, c2w, K, width, height, point_size=2, ss_ratio=2.0):
    """Render point cloud from camera pose. Returns (rgb_tensor, mask_tensor).
    
    rgb_tensor: [3, H, W] float32 in [-1, 1]
    mask_tensor: [1, H, W] float32 in [0, 1]
    """
    if points is None or colors is None:
        device = c2w.device
        return (torch.zeros(3, height, width, device=device),
                torch.zeros(1, height, width, device=device))
    
    device = points.device
    H_hi = int(height * ss_ratio)
    W_hi = int(width * ss_ratio)
    
    K_hi = K.clone()
    K_hi[0, :] *= ss_ratio
    K_hi[1, :] *= ss_ratio
    
    ps = int(point_size * ss_ratio)
    if ps % 2 == 0:
        ps += 1
    
    # World → camera
    w2c = torch.linalg.inv(c2w)
    N = points.shape[0]
    pts_h = torch.cat([points, torch.ones(N, 1, device=device)], dim=1)
    cam = (pts_h @ w2c.T)[:, :3]
    z = cam[:, 2]
    
    valid = z > MIN_DEPTH
    if valid.sum() == 0:
        return (torch.zeros(3, height, width, device=device),
                torch.zeros(1, height, width, device=device))
    
    xyz = cam[valid]
    rgb = colors[valid]
    z = z[valid]
    
    u = (K_hi[0, 0] * xyz[:, 0] / z) + K_hi[0, 2]
    v = (K_hi[1, 1] * xyz[:, 1] / z) + K_hi[1, 2]
    
    # Splatting
    if ps > 1:
        key = (ps, device)
        if key not in _splat_cache:
            r = ps // 2
            rng = torch.arange(-r, r + 1, device=device)
            dy, dx = torch.meshgrid(rng, rng, indexing='ij')
            _splat_cache[key] = (dx.flatten(), dy.flatten())
        dx, dy = _splat_cache[key]
        
        u_f = torch.round(u.unsqueeze(1) + dx.unsqueeze(0)).long().view(-1)
        v_f = torch.round(v.unsqueeze(1) + dy.unsqueeze(0)).long().view(-1)
        z_f = z.unsqueeze(1).expand(-1, dx.shape[0]).reshape(-1)
        rgb_f = rgb.unsqueeze(1).expand(-1, dx.shape[0], 3).reshape(-1, 3)
    else:
        u_f = torch.round(u).long()
        v_f = torch.round(v).long()
        z_f = z
        rgb_f = rgb
    
    ok = (u_f >= 0) & (u_f < W_hi) & (v_f >= 0) & (v_f < H_hi)
    u_f, v_f, z_f, rgb_f = u_f[ok], v_f[ok], z_f[ok], rgb_f[ok]
    
    # Z-buffer
    idx = v_f * W_hi + u_f
    dbuf = torch.full((H_hi * W_hi,), float('inf'), device=device)
    dbuf.scatter_reduce_(0, idx, z_f, reduce='min', include_self=True)
    closest = z_f <= dbuf[idx] + DEPTH_EPS
    
    fu, fv, frgb = u_f[closest], v_f[closest], rgb_f[closest]
    
    canvas = torch.zeros(3, H_hi, W_hi, device=device)
    canvas[:, fv, fu] = frgb.permute(1, 0)
    
    mask = torch.zeros(1, H_hi, W_hi, device=device)
    mask[0, fv, fu] = 1.0
    
    # Downsample via avg pool
    canvas_4d = canvas.unsqueeze(0)  # [1, 3, H_hi, W_hi]
    mask_4d = mask.unsqueeze(0)      # [1, 1, H_hi, W_hi]
    
    rgb_out = torch.nn.functional.interpolate(
        canvas_4d, size=(height, width), mode='bilinear', align_corners=False
    ).squeeze(0)  # [3, H, W]
    
    mask_out = torch.nn.functional.interpolate(
        mask_4d, size=(height, width), mode='bilinear', align_corners=False
    ).squeeze(0)  # [1, H, W]
    
    # Convert RGB to [-1, 1] range expected by TAE
    rgb_out = rgb_out * 2.0 - 1.0
    
    return rgb_out, mask_out


# ── Camera math ──

def rotation_matrix(axis, angle_rad, device):
    """Create a 3x3 rotation matrix around axis ('x', 'y', or 'z')."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    if axis == 'x':
        return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], device=device, dtype=torch.float32)
    elif axis == 'y':
        return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], device=device, dtype=torch.float32)
    elif axis == 'z':
        return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], device=device, dtype=torch.float32)


def apply_joystick_to_c2w(base_c2w, yaw_rad, pitch_rad, zoom_factor, device):
    """Apply yaw/pitch/zoom offsets to a base camera-to-world matrix.
    
    Yaw rotates around world Y axis.
    Pitch rotates around camera's local X axis.
    Zoom moves along camera's Z (forward) axis.
    """
    c2w = base_c2w.clone()
    
    # Extract rotation and translation
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    
    # Apply yaw (world Y rotation)
    if abs(yaw_rad) > 1e-6:
        Ry = rotation_matrix('y', yaw_rad, device)
        R = Ry @ R
        # Rotate translation around world origin  
        t = Ry @ t
    
    # Apply pitch (local X rotation)
    if abs(pitch_rad) > 1e-6:
        Rx = rotation_matrix('x', pitch_rad, device)
        R = R @ Rx  # apply in local frame
    
    # Apply zoom (move along camera forward direction)
    if abs(zoom_factor - 1.0) > 1e-3:
        forward = R[:, 2]  # camera Z axis in world coords
        t = t + forward * (zoom_factor - 1.0) * 0.5
    
    c2w[:3, :3] = R
    c2w[:3, 3] = t
    return c2w


def apply_translation_to_c2w(c2w, strafe_offset, forward_offset, vertical_offset, device):
    """Translate camera using local strafing/forward plus world-up lift.

    strafe_offset moves along camera right vector.
    forward_offset moves along camera forward vector.
    vertical_offset moves along world up.
    """
    moved = c2w.clone()
    R = moved[:3, :3]
    t = moved[:3, 3]

    right = R[:, 0]
    forward = R[:, 2]
    world_up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=t.dtype)

    t = t + right * strafe_offset + forward * forward_offset + world_up * vertical_offset
    moved[:3, 3] = t
    return moved


class LiveCamera:
    """Manages live camera state and point cloud rendering for dit_stream.
    
    Usage:
        cam = LiveCamera(da3_dir, device, width, height)
        cam.load()
        
        # Each block:
        rgb_frames, mask_frames = cam.render_block(
            block_idx, num_frames_per_block, joystick_yaw, joystick_pitch, joystick_zoom
        )
    """
    
    def __init__(self, da3_dir, device, width, height, point_size=2):
        self.da3_dir = da3_dir
        self.device = device
        self.width = width
        self.height = height
        self.point_size = point_size
        
        self.points_list = []
        self.colors_list = []
        self.K = None
        self.initial_c2w = None
        self.source_c2ws = []
        
        # Accumulated joystick state
        self.yaw_accum = 0.0
        self.pitch_accum = 0.0
        self.zoom_accum = 1.0
        self.strafe_accum = 0.0
        self.forward_accum = 0.0
        self.vertical_accum = 0.0
        
        self.loaded = False
    
    def load(self):
        """Load point clouds and camera data."""
        t0 = time.time()
        
        # Find the _da3_tmp directory with PLY files
        pcd_dir = os.path.join(self.da3_dir, "frames_pcd")
        if not os.path.exists(pcd_dir):
            # Try parent's _da3_tmp
            base = os.path.basename(self.da3_dir)
            parent = os.path.dirname(self.da3_dir)
            tmp_dir = os.path.join(parent, base + "_da3_tmp")
            pcd_dir = os.path.join(tmp_dir, "frames_pcd")
            if os.path.exists(pcd_dir):
                # Use tmp dir for intrinsics/extrinsics too
                self.da3_dir = tmp_dir
        
        if not os.path.exists(pcd_dir):
            print(f"[CAMERA] No PLY files found at {pcd_dir}", flush=True)
            return False
        
        # Load every 4th point cloud to save GPU memory (~120 instead of 478)
        self.points_list, self.colors_list = load_ply_sequence(
            pcd_dir, self.device, stride=4
        )
        
        if not self.points_list:
            print("[CAMERA] No point clouds loaded", flush=True)
            return False
        
        K_orig = load_intrinsic(self.da3_dir, self.device)
        self.K = scale_intrinsic(K_orig, self.width, self.height)
        self.initial_c2w, self.source_c2ws = load_extrinsics(self.da3_dir, self.device)
        
        self.loaded = True
        print(f"[CAMERA] Loaded {len(self.points_list)} point clouds, "
              f"{len(self.source_c2ws)} poses in {time.time()-t0:.1f}s", flush=True)
        return True
    
    def reset(self):
        """Reset joystick accumulation."""
        self.yaw_accum = 0.0
        self.pitch_accum = 0.0
        self.zoom_accum = 1.0
        self.strafe_accum = 0.0
        self.forward_accum = 0.0
        self.vertical_accum = 0.0
    
    def render_block(self, block_start_frame, num_frames, yaw_input, pitch_input, zoom_input,
                     move_x_input=0.0, move_y_input=0.0, move_z_input=0.0,
                     sensitivity=1.0, dt=0.18):
        """Render a block of frames from point clouds using joystick input.
        
        Args:
            block_start_frame: starting frame index in the trajectory
            num_frames: frames per block (typically 3)
            yaw_input: raw joystick yaw [-1, 1]
            pitch_input: raw joystick pitch [-1, 1]
            zoom_input: raw joystick zoom factor
            sensitivity: multiplier for joystick input
            dt: time step for integration (matches block generation time)
        
        Returns:
            render_video: [1, C, T, H, W] tensor in [-1, 1] bfloat16
            mask_video: [1, C, T, H, W] tensor in [0, 1] bfloat16
        """
        if not self.loaded:
            return None, None
        
        # Accumulate joystick input
        yaw_rate = yaw_input * sensitivity * 0.05  # radians per update
        pitch_rate = pitch_input * sensitivity * 0.03
        self.yaw_accum += yaw_rate * dt * 10  # scale by dt
        self.pitch_accum += pitch_rate * dt * 10
        self.zoom_accum = max(0.3, min(3.0, zoom_input if zoom_input > 0 else 1.0))

        # Left-stick translation in local camera frame
        strafe_rate = move_x_input * sensitivity * 0.04
        forward_rate = (-move_y_input) * sensitivity * 0.06
        vertical_rate = move_z_input * sensitivity * 0.05
        self.strafe_accum += strafe_rate * dt * 10
        self.forward_accum += forward_rate * dt * 10
        self.vertical_accum += vertical_rate * dt * 10
        self.strafe_accum = max(-2.5, min(2.5, self.strafe_accum))
        self.forward_accum = max(-3.0, min(3.0, self.forward_accum))
        self.vertical_accum = max(-2.5, min(2.5, self.vertical_accum))
        
        # Clamp pitch to avoid gimbal lock
        self.pitch_accum = max(-math.pi / 3, min(math.pi / 3, self.pitch_accum))
        
        rgb_frames = []
        mask_frames = []
        
        num_pcds = len(self.points_list)
        num_source_poses = len(self.source_c2ws)
        
        for fi in range(num_frames):
            frame_idx = block_start_frame + fi
            
            # Get base trajectory pose (loops)
            pose_idx = frame_idx % num_source_poses
            base_c2w = self.source_c2ws[pose_idx]
            
            # Apply joystick offsets
            c2w = apply_joystick_to_c2w(
                base_c2w, self.yaw_accum, self.pitch_accum, 
                self.zoom_accum, self.device
            )
            c2w = apply_translation_to_c2w(
                c2w,
                self.strafe_accum,
                self.forward_accum,
                self.vertical_accum,
                self.device,
            )
            
            # Get point cloud (loops with stride)
            pcd_idx = (frame_idx // 4) % num_pcds  # match stride=4 loading
            pts = self.points_list[pcd_idx]
            cols = self.colors_list[pcd_idx]
            
            # Render
            rgb, mask = render_frame(
                pts, cols, c2w, self.K, self.width, self.height,
                point_size=self.point_size
            )
            rgb_frames.append(rgb)      # [3, H, W]
            mask_frames.append(mask)    # [1, H, W]
        
        # Stack: [T, C, H, W] → [1, C, T, H, W]
        render_video = torch.stack(rgb_frames, dim=0).unsqueeze(0)  # [1, T, 3, H, W]
        render_video = render_video.permute(0, 2, 1, 3, 4).to(torch.bfloat16)  # [1, 3, T, H, W]
        
        mask_video = torch.stack(mask_frames, dim=0).unsqueeze(0)  # [1, T, 1, H, W]
        mask_video = mask_video.permute(0, 2, 1, 3, 4).to(torch.bfloat16)  # [1, 1, T, H, W]
        
        return render_video, mask_video
