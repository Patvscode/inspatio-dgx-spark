#!/usr/bin/env python3
"""InSpatio-World Stream Viewer v3 — full-screen game-style HUD.

Video fills the screen. Controls float on top as transparent overlays.
Joysticks at bottom corners. Compact action bar. Pull-up settings drawer.
"""

import asyncio
import base64
import glob
import json
import os
import queue as queue_mod
import shutil
import subprocess
import threading
import time

from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import uvicorn

HOST_DIR = os.path.expanduser("~/Desktop/AI-apps-workspace/inspatio-world")
IO_DIR = os.path.join(HOST_DIR, "interactive_io")
FRAMES_DIR = os.path.join(IO_DIR, "frames")
STATUS_FILE = os.path.join(IO_DIR, "status.json")
POSE_FILE = os.path.join(IO_DIR, "pose.json")
RECORD_DIR = os.path.join(IO_DIR, "recording")
THUMBS_DIR = os.path.join(IO_DIR, "thumbnails")
USER_INPUT = os.path.join(HOST_DIR, "user_input")
PORT = 7861

app = FastAPI()

# ── State ──
session_state = {
    "paused": False,
    "running": True,
    "timer_end": None,
    "timer_minutes": 60,
    "servers_stopped": True,
    "recording": False,
    "active_scene": None,  # set at startup from new.json
    "processing_scene": None,
}


def write_pose(**kwargs):
    try:
        with open(POSE_FILE, 'r') as f:
            pose = json.load(f)
    except Exception:
        pose = {"yaw": 0, "pitch": 0, "zoom": 1.0}
    pose.update(kwargs)
    with open(POSE_FILE, 'w') as f:
        json.dump(pose, f)


def read_status():
    try:
        with open(STATUS_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {"status": "unknown"}


def restore_llama_servers():
    try:
        subprocess.run(["systemctl", "--user", "start", "llama-main.service"],
                       timeout=15, capture_output=True)
        session_state["servers_stopped"] = False
    except Exception:
        pass


def stop_llama_servers():
    try:
        subprocess.run(["systemctl", "--user", "stop", "llama-main.service"],
                       timeout=10, capture_output=True)
        subprocess.run(["pkill", "-f", "llama-server"], timeout=5, capture_output=True)
        session_state["servers_stopped"] = True
    except Exception:
        pass


def stop_dit_stream():
    write_pose(stop=True)
    session_state["running"] = False
    time.sleep(2)
    # Kill inside Docker container (host-side pkill can't reach it)
    try:
        subprocess.run(["docker", "exec", "inspatio-world", "bash", "-c", "pkill -9 -f dit_stream"],
                       timeout=10, capture_output=True)
    except Exception:
        pass
    try:
        subprocess.run(["pkill", "-f", "dit_stream.py"], timeout=5, capture_output=True)
    except Exception:
        pass


def get_available_scenes():
    scenes = []
    seen_names = set()
    for f in sorted(os.listdir(USER_INPUT)):
        if not f.endswith(('.mp4', '.mov', '.avi', '.MOV')):
            continue
        name = f.rsplit('.', 1)[0]
        # Skip .mov/.avi if an .mp4 version exists (dedup after conversion)
        if name in seen_names:
            continue
        # Prefer .mp4 if both exist
        mp4_path = os.path.join(USER_INPUT, name + '.mp4')
        if os.path.exists(mp4_path):
            f = name + '.mp4'
        seen_names.add(name)
        thumb_path = os.path.join(THUMBS_DIR, name + '.jpg')
        scenes.append({
            "name": name,
            "file": f,
            "has_thumb": os.path.exists(thumb_path),
            "processed": os.path.exists(os.path.join(USER_INPUT, "new_vggt", name)),
        })
    return scenes


def generate_thumbnails():
    os.makedirs(THUMBS_DIR, exist_ok=True)
    for f in os.listdir(USER_INPUT):
        if not f.endswith(('.mp4', '.mov', '.avi', '.MOV')):
            continue
        name = f.rsplit('.', 1)[0]
        thumb = os.path.join(THUMBS_DIR, name + '.jpg')
        if os.path.exists(thumb):
            continue
        src = os.path.join(USER_INPUT, f)
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", src, "-ss", "0.5",
                "-vframes", "1", "-vf", "scale=160:-1", thumb
            ], capture_output=True, timeout=10)
        except Exception:
            pass


def _do_restart_dit(video_file, video_name, status, progress_queue):
    """Restart DiT stream with a scene that's already preprocessed."""
    status("Starting stream with new scene...")
    session_state["active_scene"] = video_file

    # Clean frames
    try:
        subprocess.run(["find", FRAMES_DIR, "-name", "*.jpg", "-delete"],
                       capture_output=True, timeout=30)
    except Exception:
        pass

    write_pose(yaw=0, pitch=0, zoom=1.0, paused=False, stop=False)

    dit_cmd = (
        "cd /workspace/inspatio-world && "
        "TORCH_CUDA_ARCH_LIST=12.1a "
        "TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas "
        "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache "
        "python3 dit_stream.py"
    )
    subprocess.Popen(
        ["docker", "exec", "inspatio-world", "bash", "-c", dit_cmd],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
    )

    status("\u2705 Scene loaded! Model warming up (~1-2 min)...")
    progress_queue.put({"type": "toast", "message": "\u2705 Ready! Warming up model...", "done": True, "error": False})
    session_state["processing_scene"] = None


def process_scene_background(video_file, progress_queue):
    """Run preprocessing pipeline for a new video."""
    SCRIPT_DIR = "/workspace/inspatio-world"
    GEN_H, GEN_W = 240, 416
    video_name = video_file.rsplit('.', 1)[0]
    traj = "x_y_circle_cycle.txt"

    def status(msg):
        print(f"[SCENE] {msg}", flush=True)
        progress_queue.put({"type": "toast", "message": msg})

    try:
        status(f"Processing {video_name}...")
        write_pose(stop=True)
        time.sleep(2)
        # Kill ALL dit_stream processes inside container
        try:
            subprocess.run(["docker", "exec", "inspatio-world", "bash", "-c", "pkill -9 -f dit_stream"],
                           timeout=10, capture_output=True)
        except Exception:
            pass
        # Also kill from host side
        try:
            subprocess.run(["pkill", "-f", "dit_stream.py"], timeout=5, capture_output=True)
        except Exception:
            pass
        time.sleep(3)

        subprocess.run(["docker", "start", "inspatio-world"], timeout=10, capture_output=True)
        time.sleep(1)

        # Check if already preprocessed — skip pipeline if render data exists
        render_dir = os.path.join(USER_INPUT, "new_vggt", video_name, "render")
        if os.path.exists(render_dir) and len(os.listdir(render_dir)) > 0:
            status("Scene already preprocessed — skipping pipeline")
            # Just update new.json to point at this scene
            scene_json = json.dumps([{
                "video_path": f"./user_input/{video_file}",
                "vggt_depth_path": f"./user_input/new_vggt/{video_name}",
                "vggt_extrinsics_path": f"./user_input/new_vggt/{video_name}/extrinsics.txt",
                "radius_ratio": 1,
                "text": "A video scene."
            }], indent=2)
            with open(os.path.join(USER_INPUT, "new.json"), 'w') as f:
                f.write(scene_json)
            # Skip straight to DiT restart (below the pipeline steps)
            # Jump to restart section
            _do_restart_dit(video_file, video_name, status, progress_queue)
            return

        # Step 1: Florence-2 caption
        status("Step 1/3: Captioning with Florence-2...")
        step1_cmd = (
            f"cd {SCRIPT_DIR} && "
            f"CUDA_VISIBLE_DEVICES=0 python scripts/gen_json.py "
            f"--root_dir ./user_input "
            f"--model_path ./checkpoints/Florence-2-large"
        )
        result = subprocess.run(
            ["docker", "exec", "inspatio-world", "bash", "-c", step1_cmd],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            status("Caption step had issues, using fallback...")
            fallback = json.dumps([{
                "video_path": f"./user_input/{video_file}",
                "vggt_depth_path": f"./user_input/new_vggt/{video_name}",
                "vggt_extrinsics_path": f"./user_input/new_vggt/{video_name}/extrinsics.txt",
                "radius_ratio": 1,
                "text": "A video scene with various objects and elements."
            }], indent=2)
            with open(os.path.join(USER_INPUT, "new.json"), 'w') as f:
                f.write(fallback)
        else:
            # Filter JSON to just our video
            json_path = os.path.join(USER_INPUT, "new.json")
            try:
                with open(json_path, 'r') as f:
                    entries = json.load(f)
                target = [e for e in entries if video_file in e.get("video_path", "")]
                if not target:
                    target = entries[-1:]
                with open(json_path, 'w') as f:
                    json.dump(target, f, indent=2)
            except Exception:
                pass

        # Step 2a: DA3 depth
        status("Step 2/3: Depth estimation (~1-2 min)...")
        da3_config = json.dumps({
            "model_path": f"{SCRIPT_DIR}/checkpoints/DA3",
            "fix_resize": True,
            "fix_resize_height": GEN_H,
            "fix_resize_width": GEN_W,
            "num_frames": 1000,
            "save_point_cloud": True
        })
        step2a_cmd = (
            f"cd {SCRIPT_DIR} && "
            f"python scripts/run_da3_parallel.py "
            f"--json_path ./user_input/new.json "
            f"--gpu_list 0 "
            f"--da3_cli {SCRIPT_DIR}/depth/depth_predict_da3_cli.py "
            f"--da3_config '{da3_config}' "
            f"--convert_script {SCRIPT_DIR}/scripts/convert_da3_to_pi3.py"
        )
        result = subprocess.run(
            ["docker", "exec", "inspatio-world", "bash", "-c", step2a_cmd],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout)[-200:]
            status(f"Depth failed: {err}")
            progress_queue.put({"type": "toast", "message": "❌ Depth estimation failed", "done": True, "error": True})
            session_state["processing_scene"] = None
            return

        # Step 2b: Render point clouds
        status("Step 3/3: Rendering point clouds (~1 min)...")
        step2b_cmd = (
            f"cd {SCRIPT_DIR} && "
            f"python scripts/run_render_parallel.py "
            f"--json_path ./user_input/new.json "
            f"--gpu_list 0 "
            f"--render_script {SCRIPT_DIR}/scripts/render_point_cloud.py "
            f"--traj_txt_path ./traj/{traj} "
            f"--width {GEN_W} --height {GEN_H} "
            f"--relative_to_source"
        )
        result = subprocess.run(
            ["docker", "exec", "inspatio-world", "bash", "-c", step2b_cmd],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout)[-200:]
            status(f"Render failed: {err}")
            progress_queue.put({"type": "toast", "message": "❌ Rendering failed", "done": True, "error": True})
            session_state["processing_scene"] = None
            return

        # Restart DiT with the new scene
        _do_restart_dit(video_file, video_name, status, progress_queue)

    except Exception as e:
        print(f"[SCENE] Error: {e}", flush=True)
        progress_queue.put({"type": "toast", "message": f"❌ Error: {str(e)[:80]}", "done": True, "error": True})
        session_state["processing_scene"] = None


# Detect active scene from new.json at startup
try:
    with open(os.path.join(USER_INPUT, "new.json"), 'r') as _f:
        _entries = json.load(_f)
    if _entries:
        _vp = _entries[0].get("video_path", "")
        session_state["active_scene"] = os.path.basename(_vp)
except Exception:
    pass
if not session_state["active_scene"]:
    session_state["active_scene"] = "IMG_7643.mp4"

threading.Thread(target=generate_thumbnails, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════
#  HTML — Full-screen game-style HUD
# ══════════════════════════════════════════════════════════════════════

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no,viewport-fit=cover">
<title>InSpatio</title>
<style>
*{margin:0;padding:0;box-sizing:border-box;-webkit-tap-highlight-color:transparent}

html,body{
  height:100%;width:100%;overflow:hidden;
  background:#000;color:#fff;
  font-family:-apple-system,BlinkMacSystemFont,'SF Pro Display',system-ui,sans-serif;
  -webkit-touch-callout:none;-webkit-user-select:none;user-select:none;
  overscroll-behavior:none;touch-action:manipulation;
}

/* ── Full-screen canvas ── */
.canvas{
  position:fixed;inset:0;
  display:flex;align-items:center;justify-content:center;
  background:radial-gradient(ellipse at center,#0a0a12 0%,#000 70%);
}
.canvas img{
  width:100%;height:100%;
  object-fit:cover;
  display:block;
}
.canvas.dimmed img{opacity:0.35}

/* ── Loading spinner ── */
.loading{
  position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);
  z-index:5;display:none;flex-direction:column;align-items:center;gap:12px;
}
.loading.visible{display:flex}
.spinner{
  width:36px;height:36px;border:3px solid rgba(255,255,255,0.15);
  border-top-color:#51cf66;border-radius:50%;
  animation:spin 0.8s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg)}}
.loading-text{font-size:12px;color:rgba(255,255,255,0.6);letter-spacing:0.5px}

/* ── HUD Overlays ── */
.hud{position:fixed;z-index:10;pointer-events:none}
.hud>*{pointer-events:auto}

/* Top bar */
.top-bar{
  top:0;left:0;right:0;
  display:flex;justify-content:space-between;align-items:flex-start;
  padding:max(env(safe-area-inset-top,8px),8px) 12px 0;
}
.brand{
  font-size:13px;font-weight:700;letter-spacing:1.5px;
  color:rgba(255,255,255,0.85);
  background:rgba(0,0,0,0.4);backdrop-filter:blur(12px);
  padding:6px 12px;border-radius:10px;
  display:flex;align-items:center;gap:6px;
}
.status-chip{
  font-size:11px;font-weight:600;
  background:rgba(0,0,0,0.4);backdrop-filter:blur(12px);
  padding:5px 10px;border-radius:10px;
  display:flex;align-items:center;gap:5px;
}
.dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.dot.live{background:#51cf66;box-shadow:0 0 6px #51cf66;animation:pulse 2s infinite}
.dot.paused{background:#fcc419;animation:pulse 1s infinite}
.dot.off{background:#555}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}

/* Info chips — bottom-left, above dock */
.info-left{
  bottom:200px;left:12px;
  display:flex;flex-direction:column;gap:4px;
}
.chip{
  font-size:10px;font-weight:600;
  background:rgba(0,0,0,0.4);backdrop-filter:blur(8px);
  padding:3px 8px;border-radius:6px;
  color:rgba(255,255,255,0.7);
  width:fit-content;
}
.chip.fps{color:#51cf66}
.chip.rec{background:rgba(255,60,60,0.5);color:#fff;display:none}
.chip.rec.active{display:flex;align-items:center;gap:4px;animation:recblink 1s infinite}
@keyframes recblink{0%,100%{opacity:1}50%{opacity:0.5}}

/* Timer — inline in top-bar, not separate */
.timer-chip{
  font-size:13px;font-weight:700;font-variant-numeric:tabular-nums;
  color:rgba(255,255,255,0.45);
  background:rgba(0,0,0,0.4);backdrop-filter:blur(12px);
  padding:5px 10px;border-radius:10px;
}

/* ── Controls dock (bottom) ── */
.dock{
  position:fixed;bottom:0;left:0;right:0;z-index:20;
  padding:0 8px max(env(safe-area-inset-bottom,8px),8px);
  pointer-events:none;
}
.dock>*{pointer-events:auto}

/* Action bar */
.action-bar{
  display:flex;align-items:center;justify-content:center;
  gap:10px;
  padding:8px 0 6px;
}
.action-bar .spacer{width:8px}
.abtn{
  border:none;border-radius:50%;cursor:pointer;
  display:flex;align-items:center;justify-content:center;
  transition:transform 0.08s;
  -webkit-user-select:none;
  box-shadow:0 2px 12px rgba(0,0,0,0.4);
}
.abtn:active{transform:scale(0.9)}
.abtn-play{
  width:52px;height:52px;
  background:#51cf66;color:#000;font-size:20px;font-weight:bold;
}
.abtn-play.paused{background:#fcc419}
.abtn-play.stopped{background:#444;color:#666}
.abtn-rec{
  width:44px;height:44px;
  background:#e03131;color:#fff;font-size:18px;
}
.abtn-rec.active{
  background:rgba(40,40,40,0.9);border:2px solid #e03131;
  box-shadow:0 0 16px rgba(224,49,49,0.3);
}
.abtn-sm{
  width:40px;height:40px;
  background:rgba(50,50,50,0.8);color:#bbb;font-size:15px;
  backdrop-filter:blur(8px);
}

/* Joystick row */
.joy-row{
  display:flex;align-items:center;justify-content:space-between;
  padding:4px 4px 0;
}

.joy-zone{
  width:110px;height:110px;
  border-radius:50%;
  background:rgba(30,30,40,0.5);
  border:2px solid rgba(60,60,70,0.6);
  backdrop-filter:blur(6px);
  position:relative;
  touch-action:none;
  flex-shrink:0;
}
.joy-zone::before{
  content:'';position:absolute;
  top:50%;left:20%;right:20%;height:1px;
  background:rgba(255,255,255,0.06);
}
.joy-zone::after{
  content:'';position:absolute;
  left:50%;top:20%;bottom:20%;width:1px;
  background:rgba(255,255,255,0.06);
}
.joy-zone.active{
  border-color:rgba(81,207,102,0.6);
  box-shadow:0 0 24px rgba(81,207,102,0.12);
}
.joy-knob{
  width:44px;height:44px;
  border-radius:50%;
  background:radial-gradient(circle at 40% 35%,#555,#383838);
  border:2px solid #666;
  position:absolute;top:50%;left:50%;
  transform:translate(-50%,-50%);
  pointer-events:none;
  will-change:left,top;
  box-shadow:0 2px 8px rgba(0,0,0,0.5);
  z-index:2;
}
.joy-knob.active{
  border-color:#51cf66;
  background:radial-gradient(circle at 40% 35%,#5a6a5a,#3a4a3a);
}
.joy-label{
  position:absolute;top:-14px;left:0;right:0;
  text-align:center;font-size:9px;
  color:rgba(255,255,255,0.3);
  text-transform:uppercase;letter-spacing:1px;
}

/* Center info between joysticks */
.joy-center{
  flex:1;display:flex;flex-direction:column;
  align-items:center;gap:2px;
  padding:0 8px;
}
.joy-center .mini-info{
  font-size:10px;color:rgba(255,255,255,0.35);
}

/* ── Settings drawer ── */
.drawer-handle{
  display:flex;align-items:center;justify-content:center;
  padding:6px;cursor:pointer;
}
.drawer-pill{
  width:36px;height:4px;border-radius:2px;
  background:rgba(255,255,255,0.15);
}
.drawer{
  position:fixed;bottom:0;left:0;right:0;z-index:50;
  background:rgba(18,18,22,0.97);backdrop-filter:blur(20px);
  border-top:1px solid rgba(255,255,255,0.08);
  border-radius:20px 20px 0 0;
  transform:translateY(100%);
  transition:transform 0.3s cubic-bezier(0.32,0.72,0,1);
  max-height:70vh;overflow-y:auto;
  -webkit-overflow-scrolling:touch;
  padding-bottom:max(env(safe-area-inset-bottom,20px),20px);
}
.drawer.open{transform:translateY(0)}
.drawer-header{
  display:flex;align-items:center;justify-content:space-between;
  padding:16px 20px 8px;
  position:sticky;top:0;
  background:rgba(18,18,22,0.97);z-index:1;
}
.drawer-title{font-size:15px;font-weight:700;letter-spacing:0.5px}
.drawer-close{
  font-size:13px;color:#51cf66;font-weight:600;
  cursor:pointer;padding:4px 8px;
}
.drawer-section{padding:4px 20px 14px}
.drawer-label{
  font-size:10px;font-weight:700;
  text-transform:uppercase;letter-spacing:1px;
  color:rgba(255,255,255,0.35);
  margin-bottom:8px;
}
.pills{display:flex;gap:6px;flex-wrap:wrap}
.pill{
  padding:7px 14px;border-radius:10px;
  border:1px solid rgba(255,255,255,0.1);
  background:rgba(255,255,255,0.04);
  color:rgba(255,255,255,0.5);
  font-size:12px;font-weight:600;
  cursor:pointer;transition:all 0.15s;
}
.pill.active{
  background:rgba(81,207,102,0.12);
  border-color:rgba(81,207,102,0.4);
  color:#51cf66;
}
.pill:active{transform:scale(0.96)}

/* Scene strip */
.scene-strip{
  display:flex;gap:8px;overflow-x:auto;
  padding:4px 0;
  -webkit-overflow-scrolling:touch;
}
.scene-strip::-webkit-scrollbar{display:none}
.scene-card{
  width:72px;flex-shrink:0;cursor:pointer;
}
.scene-card .thumb{
  width:72px;height:48px;border-radius:8px;
  overflow:hidden;border:2px solid transparent;
  background:#1a1a1a;
}
.scene-card .thumb.active{border-color:#51cf66}
.scene-card .thumb img{width:100%;height:100%;object-fit:cover}
.scene-card .sname{
  font-size:8px;color:rgba(255,255,255,0.35);
  text-align:center;margin-top:3px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.scene-add{
  width:72px;height:48px;border-radius:8px;
  border:1px dashed rgba(255,255,255,0.15);
  display:flex;align-items:center;justify-content:center;
  font-size:22px;color:rgba(255,255,255,0.2);
  cursor:pointer;flex-shrink:0;
}
.scene-add:active{background:rgba(255,255,255,0.05)}

/* GPU row */
.gpu-row{
  display:flex;align-items:center;justify-content:space-between;
  padding:10px 14px;margin:0 -6px;
  background:rgba(255,255,255,0.03);border-radius:10px;
}
.gpu-row .gi{font-size:11px;color:rgba(255,255,255,0.4)}
.gpu-btn{
  padding:6px 12px;border-radius:8px;border:1px solid rgba(255,255,255,0.1);
  background:transparent;color:rgba(255,255,255,0.5);
  font-size:11px;font-weight:600;cursor:pointer;
}
.gpu-btn:active{background:rgba(255,255,255,0.05)}

/* Toast */
.toast{
  position:fixed;top:60px;left:50%;transform:translateX(-50%);z-index:100;
  background:rgba(20,20,28,0.95);backdrop-filter:blur(16px);
  border:1px solid rgba(255,255,255,0.1);
  border-radius:14px;padding:10px 20px;
  font-size:13px;color:#ddd;
  max-width:85%;text-align:center;
  box-shadow:0 8px 32px rgba(0,0,0,0.5);
  opacity:0;transition:opacity 0.3s;pointer-events:none;
}
.toast.visible{opacity:1}
.toast.error{border-color:rgba(224,49,49,0.4)}
.toast.success{border-color:rgba(81,207,102,0.4)}

/* Upload modal */
.upload-modal{
  position:fixed;inset:0;z-index:200;
  background:rgba(0,0,0,0.85);backdrop-filter:blur(12px);
  display:none;flex-direction:column;align-items:center;justify-content:center;gap:12px;
  padding:20px;
}
.upload-modal.visible{display:flex}
.upload-modal .ubtn{
  width:220px;padding:14px;border-radius:14px;
  border:1px solid rgba(255,255,255,0.1);
  background:rgba(255,255,255,0.06);
  color:#fff;font-size:15px;cursor:pointer;
  display:flex;align-items:center;justify-content:center;gap:8px;
}
.upload-modal .ubtn:active{background:rgba(255,255,255,0.1)}
.upload-modal .ucancel{
  color:rgba(255,255,255,0.4);border-color:transparent;
  font-size:13px;margin-top:4px;
}
.upload-prog{
  width:220px;display:none;margin-top:8px;
}
.upload-prog.visible{display:block}
.upload-prog .utext{font-size:11px;color:rgba(255,255,255,0.5);text-align:center;margin-bottom:4px}
.upload-prog .ubar-bg{height:4px;background:rgba(255,255,255,0.1);border-radius:2px;overflow:hidden}
.upload-prog .ubar{height:100%;background:#51cf66;width:0%;transition:width 0.3s}
</style>
</head>
<body>

<!-- ═══ Video canvas (full screen) ═══ -->
<div class="canvas" id="canvas">
  <img id="vid" src="" alt="">
</div>

<!-- Loading spinner -->
<div class="loading" id="loading">
  <div class="spinner"></div>
  <div class="loading-text" id="loadingText">Warming up model...</div>
</div>

<!-- ═══ HUD overlays ═══ -->

<!-- Top bar -->
<div class="hud top-bar">
  <div class="brand">⟐ INSPATIO</div>
  <div style="display:flex;align-items:center;gap:8px">
    <div class="timer-chip" id="timerChip">60:00</div>
    <div class="status-chip" id="statusChip">
      <div class="dot live" id="dot"></div>
      <span id="statusText">connecting</span>
    </div>
  </div>
</div>

<!-- Info chips (left side) -->
<div class="hud info-left">
  <div class="chip fps" id="fpsChip">-- FPS</div>
  <div class="chip" id="qualChip">240p · 2 steps</div>
  <div class="chip rec" id="recChip">● REC <span id="recTime">0:00</span></div>
</div>

<!-- Toast -->
<div class="toast" id="toast"></div>

<!-- ═══ Controls dock ═══ -->
<div class="dock">
  <!-- Action bar -->
  <div class="action-bar">
    <button class="abtn abtn-sm" id="settingsBtn" onclick="openDrawer()">⚙</button>
    <div class="spacer"></div>
    <button class="abtn abtn-rec" id="recBtn" onclick="toggleRecord()">●</button>
    <button class="abtn abtn-play" id="playBtn" onclick="togglePause()">⏸</button>
    <button class="abtn abtn-sm" onclick="stopSession()">■</button>
    <div class="spacer"></div>
    <button class="abtn abtn-sm" onclick="resetView()">↻</button>
  </div>

  <!-- Joystick row -->
  <div class="joy-row">
    <div class="joy-zone" id="joyL">
      <div class="joy-label">move</div>
      <div class="joy-knob" id="knobL"></div>
    </div>
    <div class="joy-center">
      <div class="mini-info" id="sceneInfo">IMG_7643</div>
      <div class="mini-info" id="frameInfo">Frame 0</div>
    </div>
    <div class="joy-zone" id="joyR">
      <div class="joy-label">look</div>
      <div class="joy-knob" id="knobR"></div>
    </div>
  </div>
</div>

<!-- ═══ Settings drawer ═══ -->
<div class="drawer" id="drawer">
  <div class="drawer-header">
    <div class="drawer-title">Settings</div>
    <div class="drawer-close" onclick="closeDrawer()">Done</div>
  </div>

  <div class="drawer-section">
    <div class="drawer-label">Session Timer</div>
    <div class="pills" id="timerPills">
      <div class="pill" onclick="setTimer(5)" data-t="5">5m</div>
      <div class="pill" onclick="setTimer(15)" data-t="15">15m</div>
      <div class="pill" onclick="setTimer(30)" data-t="30">30m</div>
      <div class="pill active" onclick="setTimer(60)" data-t="60">1h</div>
      <div class="pill" onclick="setTimer(0)" data-t="0">∞</div>
    </div>
  </div>

  <div class="drawer-section">
    <div class="drawer-label">Quality</div>
    <div class="pills" id="qualPills">
      <div class="pill active" onclick="setQuality('scout')" data-q="scout">⚡ Scout 240p</div>
      <div class="pill" onclick="setQuality('draft')" data-q="draft">Draft 360p</div>
      <div class="pill" onclick="setQuality('full')" data-q="full">✨ Full 480p</div>
    </div>
  </div>

  <div class="drawer-section">
    <div class="drawer-label">Denoising Steps</div>
    <div class="pills" id="stepPills">
      <div class="pill active" onclick="setSteps(2)" data-s="2">2 fast</div>
      <div class="pill" onclick="setSteps(3)" data-s="3">3 balanced</div>
      <div class="pill" onclick="setSteps(4)" data-s="4">4 best</div>
    </div>
  </div>

  <div class="drawer-section">
    <div class="drawer-label">Joystick Sensitivity</div>
    <div class="pills" id="sensPills">
      <div class="pill" onclick="setSens(0.5)" data-v="0.5">Gentle</div>
      <div class="pill active" onclick="setSens(1)" data-v="1">Normal</div>
      <div class="pill" onclick="setSens(1.5)" data-v="1.5">Snappy</div>
      <div class="pill" onclick="setSens(2)" data-v="2">Aggressive</div>
    </div>
  </div>

  <div class="drawer-section">
    <div class="drawer-label">Scenes</div>
    <div class="scene-strip" id="sceneStrip"></div>
  </div>

  <div class="drawer-section">
    <div class="gpu-row">
      <div class="gi" id="gpuLabel">GPU: InSpatio active</div>
      <button class="gpu-btn" id="gpuBtn" onclick="toggleGPU()">Free GPU</button>
    </div>
  </div>
</div>

<!-- Drawer backdrop -->
<div id="backdrop" style="position:fixed;inset:0;z-index:45;background:rgba(0,0,0,0.4);display:none" onclick="closeDrawer()"></div>

<!-- Upload modal -->
<div class="upload-modal" id="uploadModal">
  <div style="font-size:17px;font-weight:700;margin-bottom:4px">Add Video</div>
  <button class="ubtn" onclick="pickFile('cam')">📹 Record New</button>
  <button class="ubtn" onclick="pickFile('lib')">📁 From Library</button>
  <button class="ubtn ucancel" onclick="closeUpload()">Cancel</button>
  <div class="upload-prog" id="upProg">
    <div class="utext" id="upText">Uploading...</div>
    <div class="ubar-bg"><div class="ubar" id="upBar"></div></div>
  </div>
</div>

<input type="file" id="fileCam" accept="video/*" capture="" style="display:none" onchange="doUpload(this)">
<input type="file" id="fileLib" accept="video/*,.mp4,.mov,.avi,.mkv" style="display:none" onchange="doUpload(this)">

<script>
// ═══ State ═══
const S={ws:null,paused:false,stopped:false,rec:false,recStart:0,recTimer:null,
  timerMin:60,timerEnd:null,timerInt:null,frames:0,lastFT:Date.now(),fpsSm:0,
  quality:'scout',steps:2,sens:1.0,gotFirstFrame:false,activeScene:'IMG_7643.mp4'};

// ═══ Elements ═══
const $=id=>document.getElementById(id);
const el={
  vid:$('vid'),canvas:$('canvas'),loading:$('loading'),loadingText:$('loadingText'),
  dot:$('dot'),statusText:$('statusText'),statusChip:$('statusChip'),
  fpsChip:$('fpsChip'),qualChip:$('qualChip'),recChip:$('recChip'),recTime:$('recTime'),
  timerChip:$('timerChip'),
  playBtn:$('playBtn'),recBtn:$('recBtn'),
  sceneInfo:$('sceneInfo'),frameInfo:$('frameInfo'),
  drawer:$('drawer'),backdrop:$('backdrop'),sceneStrip:$('sceneStrip'),
  toast:$('toast'),gpuLabel:$('gpuLabel'),gpuBtn:$('gpuBtn'),
  uploadModal:$('uploadModal'),upProg:$('upProg'),upText:$('upText'),upBar:$('upBar'),
};

// ═══ WebSocket ═══
function connect(){
  const p=location.protocol==='https:'?'wss:':'ws:';
  S.ws=new WebSocket(`${p}//${location.host}/ws`);

  S.ws.onopen=()=>{
    setStatus('live','streaming');
    S.stopped=false;
    if(!S.gotFirstFrame){el.loading.classList.add('visible');el.loadingText.textContent='Connecting to stream...';}
    startTimer();
  };

  S.ws.onmessage=ev=>{
    const d=JSON.parse(ev.data);

    if(d.type==='frame'&&!S.paused){
      el.vid.src='data:image/jpeg;base64,'+d.data;
      if(!S.gotFirstFrame){S.gotFirstFrame=true;el.loading.classList.remove('visible');el.canvas.classList.remove('dimmed');}
      S.frames++;
      el.frameInfo.textContent='Frame '+d.frame;
      const now=Date.now(),dt=(now-S.lastFT)/1000;
      if(dt>0&&dt<2){const f=1/dt;S.fpsSm=S.fpsSm*0.85+f*0.15;el.fpsChip.textContent=S.fpsSm.toFixed(1)+' FPS';}
      S.lastFT=now;
    }

    if(d.type==='status'){
      const s=d.status||'unknown';
      if(s==='streaming')setStatus('live','streaming');
      else if(s==='paused')setStatus('paused','paused');
      else if(s.includes('loading')||s.includes('warming')||s.includes('compil')){
        setStatus('off',s);el.loading.classList.add('visible');el.loadingText.textContent=s.replace(/_/g,' ');
      }else setStatus('off',s);
    }

    if(d.type==='download'){
      const a=document.createElement('a');a.href=d.url;a.download='inspatio_recording.mp4';
      document.body.appendChild(a);a.click();document.body.removeChild(a);
    }

    if(d.type==='scenes')renderScenes(d.scenes);
    if(d.type==='toast')showToast(d.message,d.done,d.error);
    if(d.type==='active_scene'){
      S.activeScene=d.scene;
      const name=d.scene.replace(/\.[^.]+$/,'');
      el.sceneInfo.textContent=name;
    }
    if(d.type==='gpu_status'){
      el.gpuLabel.textContent=d.stopped?'GPU: Free':'GPU: InSpatio';
      el.gpuBtn.textContent=d.stopped?'Restart':'Free GPU';
    }
  };

  S.ws.onclose=()=>{
    setStatus('off','disconnected');
    if(!S.stopped)setTimeout(connect,2000);
  };
}

function send(m){if(S.ws&&S.ws.readyState===1)S.ws.send(JSON.stringify(m));}

function setStatus(mode,text){
  el.dot.className='dot '+mode;
  el.statusText.textContent=text;
}

// ═══ Play/Pause ═══
function togglePause(){
  if(S.stopped)return;
  S.paused=!S.paused;
  send({action:S.paused?'pause':'resume'});
  el.playBtn.textContent=S.paused?'▶':'⏸';
  el.playBtn.classList.toggle('paused',S.paused);
  el.canvas.classList.toggle('dimmed',S.paused);
  if(S.paused){el.loading.classList.add('visible');el.loadingText.textContent='Paused';}
  else el.loading.classList.remove('visible');
}

// ═══ Stop ═══
function stopSession(){
  if(S.stopped)return;
  if(!confirm('Stop InSpatio and free GPU?'))return;
  S.stopped=true;S.paused=false;
  send({action:'stop'});
  el.playBtn.textContent='■';el.playBtn.classList.add('stopped');
  el.canvas.classList.add('dimmed');
  el.loading.classList.add('visible');el.loadingText.textContent='Stopped — GPU freed';
  setStatus('off','stopped');
  clearInterval(S.timerInt);el.timerChip.textContent='DONE';
}

// ═══ Record ═══
function toggleRecord(){
  S.rec=!S.rec;
  send({action:S.rec?'start_record':'stop_record'});
  el.recBtn.classList.toggle('active',S.rec);
  el.recBtn.textContent=S.rec?'■':'●';
  el.recChip.classList.toggle('active',S.rec);
  if(S.rec){
    S.recStart=Date.now();
    S.recTimer=setInterval(()=>{
      const s=Math.floor((Date.now()-S.recStart)/1000);
      el.recTime.textContent=Math.floor(s/60)+':'+String(s%60).padStart(2,'0');
    },500);
  }else{clearInterval(S.recTimer);}
}

function resetView(){send({action:'reset'});}

// ═══ Timer ═══
function setTimer(m){
  S.timerMin=m;
  document.querySelectorAll('#timerPills .pill').forEach(p=>p.classList.toggle('active',parseInt(p.dataset.t)===m));
  send({action:'set_timer',minutes:m});
  startTimer();
}
function startTimer(){
  clearInterval(S.timerInt);
  if(!S.timerMin){el.timerChip.textContent='∞';return;}
  S.timerEnd=Date.now()+S.timerMin*60000;
  S.timerInt=setInterval(()=>{
    const left=Math.max(0,S.timerEnd-Date.now());
    if(left<=0){clearInterval(S.timerInt);el.timerChip.textContent='0:00';stopSession();return;}
    const m=Math.floor(left/60000),s=Math.floor((left%60000)/1000);
    el.timerChip.textContent=m+':'+String(s).padStart(2,'0');
  },1000);
}

// ═══ Quality ═══
function setQuality(q){
  S.quality=q;
  document.querySelectorAll('#qualPills .pill').forEach(p=>p.classList.toggle('active',p.dataset.q===q));
  updateQualChip();
  send({action:'set_quality',quality:q,steps:S.steps});
}
function setSteps(n){
  S.steps=n;
  document.querySelectorAll('#stepPills .pill').forEach(p=>p.classList.toggle('active',parseInt(p.dataset.s)===n));
  updateQualChip();
  send({action:'set_quality',quality:S.quality,steps:n});
}
function updateQualChip(){
  const r={scout:'240p',draft:'360p',full:'480p'};
  el.qualChip.textContent=r[S.quality]+' · '+S.steps+' steps';
}

// ═══ Sensitivity ═══
function setSens(v){
  S.sens=v;
  document.querySelectorAll('#sensPills .pill').forEach(p=>p.classList.toggle('active',parseFloat(p.dataset.v)===v));
}

// ═══ Drawer ═══
function openDrawer(){
  el.drawer.classList.add('open');el.backdrop.style.display='block';
  send({action:'get_scenes'});
}
function closeDrawer(){el.drawer.classList.remove('open');el.backdrop.style.display='none';}

// ═══ Toast ═══
let _toastTimer=null;
function showToast(msg,done,error){
  el.toast.textContent=msg;
  el.toast.className='toast visible'+(error?' error':'')+(done&&!error?' success':'');
  clearTimeout(_toastTimer);
  if(done)_toastTimer=setTimeout(()=>{el.toast.classList.remove('visible');},4000);
}

// ═══ Scenes ═══
function renderScenes(scenes){
  el.sceneStrip.innerHTML='';
  scenes.forEach(sc=>{
    const d=document.createElement('div');d.className='scene-card';
    d.onclick=()=>send({action:'load_scene',scene:sc.file});
    const cls=sc.active?'thumb active':'thumb';
    if(sc.has_thumb)d.innerHTML=`<div class="${cls}"><img src="/thumb/${sc.name}.jpg"></div><div class="sname">${sc.name}</div>`;
    else d.innerHTML=`<div class="${cls}" style="display:flex;align-items:center;justify-content:center;color:#444">🎬</div><div class="sname">${sc.name}</div>`;
    el.sceneStrip.appendChild(d);
  });
  const add=document.createElement('div');add.className='scene-add';add.textContent='+';
  add.onclick=()=>{el.uploadModal.classList.add('visible');};
  el.sceneStrip.appendChild(add);
}

// ═══ Upload ═══
function pickFile(mode){
  el.uploadModal.classList.remove('visible');
  (mode==='cam'?$('fileCam'):$('fileLib')).click();
}
function closeUpload(){el.uploadModal.classList.remove('visible');el.upProg.classList.remove('visible');}
function doUpload(input){
  if(!input.files[0])return;
  const file=input.files[0],fd=new FormData();fd.append('file',file);
  el.uploadModal.classList.add('visible');
  el.upProg.classList.add('visible');
  el.upBar.style.width='0%';el.upBar.style.background='#51cf66';
  el.upText.textContent='Uploading '+file.name+'...';
  const xhr=new XMLHttpRequest();
  xhr.open('POST','/upload_video');
  xhr.upload.onprogress=e=>{if(e.lengthComputable){const p=Math.round(e.loaded/e.total*100);el.upBar.style.width=p+'%';el.upText.textContent=p<100?'Uploading... '+p+'%':'Processing...';}};
  xhr.onload=()=>{
    el.upBar.style.width='100%';
    try{const r=JSON.parse(xhr.responseText);el.upText.textContent=xhr.status<400?'✅ '+(r.message||'Done!'):'❌ '+(r.detail||'Failed');}
    catch(e){el.upText.textContent=xhr.status<400?'✅ Uploaded!':'❌ Error';}
    if(xhr.status>=400)el.upBar.style.background='#e03131';
    setTimeout(()=>{el.upProg.classList.remove('visible');el.uploadModal.classList.remove('visible');send({action:'get_scenes'});},1500);
  };
  xhr.onerror=()=>{el.upText.textContent='❌ Network error';el.upBar.style.background='#e03131';setTimeout(closeUpload,3000);};
  xhr.timeout=300000;
  xhr.ontimeout=()=>{el.upText.textContent='❌ Timeout';setTimeout(closeUpload,3000);};
  xhr.send(fd);input.value='';
}
function toggleGPU(){send({action:'toggle_gpu'});}

// ═══ Joystick ═══
class Joystick{
  constructor(zid,kid,cb){
    this.zone=$(zid);this.knob=$(kid);this.cb=cb;
    this.tid=null;this.KR=22;this.x=0;this.y=0;

    this.zone.addEventListener('touchstart',e=>this.ts(e),{passive:false});
    this.zone.addEventListener('touchmove',e=>this.tm(e),{passive:false});
    this.zone.addEventListener('touchend',e=>this.te(e),{passive:false});
    this.zone.addEventListener('touchcancel',e=>this.te(e),{passive:false});
    this.zone.addEventListener('contextmenu',e=>e.preventDefault());

    this.md=false;
    this.zone.addEventListener('mousedown',e=>{this.md=true;this.activate();this.proc(e.clientX,e.clientY);e.preventDefault();});
    document.addEventListener('mousemove',e=>{if(this.md)this.proc(e.clientX,e.clientY);});
    document.addEventListener('mouseup',()=>{if(this.md){this.md=false;this.release();}});
  }
  ft(list){for(let i=0;i<list.length;i++)if(list[i].identifier===this.tid)return list[i];return null;}
  activate(){this.zone.classList.add('active');this.knob.classList.add('active');}
  ts(e){e.preventDefault();e.stopPropagation();if(this.tid!==null)return;const t=e.changedTouches[0];this.tid=t.identifier;this.activate();this.proc(t.clientX,t.clientY);}
  tm(e){e.preventDefault();e.stopPropagation();if(this.tid===null)return;const t=this.ft(e.changedTouches)||this.ft(e.touches);if(t)this.proc(t.clientX,t.clientY);}
  te(e){e.preventDefault();e.stopPropagation();if(this.tid===null)return;if(this.ft(e.changedTouches)||e.touches.length===0)this.release();}

  proc(cx,cy){
    const r=this.zone.getBoundingClientRect(),zr=r.width/2,mt=zr-this.KR-4;
    let dx=cx-(r.left+zr),dy=cy-(r.top+zr);
    const dist=Math.sqrt(dx*dx+dy*dy);
    if(dist>mt){dx=dx/dist*mt;dy=dy/dist*mt;}
    const raw=Math.min(1,dist/mt),curved=Math.min(1,Math.pow(raw,1/S.sens));
    const ang=Math.atan2(dy,dx);
    this.x=curved*Math.cos(ang);this.y=curved*Math.sin(ang);
    this.knob.style.left=(zr+dx-this.KR)+'px';
    this.knob.style.top=(zr+dy-this.KR)+'px';
    this.knob.style.transform='none';
    this.cb(this.x,this.y);
  }
  release(){
    this.tid=null;this.x=0;this.y=0;
    this.knob.style.left='50%';this.knob.style.top='50%';this.knob.style.transform='translate(-50%,-50%)';
    this.zone.classList.remove('active');this.knob.classList.remove('active');
    this.cb(0,0);
  }
}

let _mx=0,_my=0,_lx=0,_ly=0,_pd=false;
setInterval(()=>{if(!_pd)return;_pd=false;send({action:'pose',moveX:Math.round(_mx*100)/100,moveY:Math.round(_my*100)/100,lookX:Math.round(_lx*100)/100,lookY:Math.round(_ly*100)/100});},50);

const jL=new Joystick('joyL','knobL',(x,y)=>{_mx=x;_my=y;_pd=true;});
const jR=new Joystick('joyR','knobR',(x,y)=>{_lx=x;_ly=y;_pd=true;});

// ═══ Init ═══
el.loading.classList.add('visible');
connect();
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════
#  API Endpoints
# ══════════════════════════════════════════════════════════════════════

recording = False
recorded_frames = []


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.get("/thumb/{name}")
async def get_thumbnail(name: str):
    path = os.path.join(THUMBS_DIR, name)
    if os.path.exists(path):
        return FileResponse(path, media_type="image/jpeg")
    return JSONResponse({"error": "not found"}, status_code=404)


@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    try:
        safe_name = file.filename.replace(' ', '_').replace('(', '').replace(')', '')
        dest = os.path.join(USER_INPUT, safe_name)
        content = await file.read()
        with open(dest, 'wb') as f:
            f.write(content)
        size_mb = len(content) / (1024 * 1024)
        print(f"[UPLOAD] {safe_name} ({size_mb:.1f} MB)")
        
        # Convert non-mp4 to mp4 (pipeline only reads .mp4)
        if not safe_name.lower().endswith('.mp4'):
            mp4_name = safe_name.rsplit('.', 1)[0] + '.mp4'
            mp4_dest = os.path.join(USER_INPUT, mp4_name)
            print(f"[UPLOAD] Converting {safe_name} -> {mp4_name}")
            conv = subprocess.run([
                "ffmpeg", "-y", "-i", dest,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-an", mp4_dest
            ], capture_output=True, timeout=120)
            if conv.returncode == 0:
                safe_name = mp4_name
                dest = mp4_dest
                print(f"[UPLOAD] Converted to mp4")
            else:
                print(f"[UPLOAD] Conversion failed, keeping original")
        
        thumb = os.path.join(THUMBS_DIR, safe_name.rsplit('.', 1)[0] + '.jpg')
        try:
            subprocess.run(["ffmpeg", "-y", "-i", dest, "-ss", "0.5",
                            "-vframes", "1", "-vf", "scale=160:-1", thumb],
                           capture_output=True, timeout=10)
        except Exception:
            pass
        return {"message": f"Uploaded {safe_name} ({size_mb:.1f} MB)"}
    except PermissionError:
        return JSONResponse({"detail": "Permission denied"}, status_code=500)
    except Exception as e:
        return JSONResponse({"detail": str(e)[:100]}, status_code=500)


@app.get("/download_recording")
async def download_recording():
    path = os.path.join(IO_DIR, "recording.mp4")
    if os.path.exists(path):
        return FileResponse(path, filename="inspatio_recording.mp4", media_type="video/mp4")
    return JSONResponse({"error": "No recording"}, status_code=404)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global recording, recorded_frames
    await websocket.accept()

    last_frame_idx = -1
    paused = False

    async def receive_commands():
        nonlocal paused
        global recording, recorded_frames
        while True:
            try:
                data = await websocket.receive_text()
                msg = json.loads(data)
                action = msg.get("action", "")

                if action == "pause":
                    paused = True
                    write_pose(paused=True)
                    session_state["paused"] = True

                elif action == "resume":
                    paused = False
                    write_pose(paused=False)
                    session_state["paused"] = False

                elif action == "stop":
                    stop_dit_stream()
                    restore_llama_servers()
                    await websocket.send_json({"type": "status", "status": "stopped"})

                elif action == "start_record":
                    recording = True
                    recorded_frames = []

                elif action == "stop_record":
                    recording = False
                    await stitch_and_send(websocket)

                elif action == "reset":
                    write_pose(yaw=0, pitch=0, zoom=1.0, moveX=0, moveY=0)

                elif action == "set_timer":
                    minutes = msg.get("minutes", 60)
                    session_state["timer_minutes"] = minutes
                    session_state["timer_end"] = time.time() + minutes * 60 if minutes else None

                elif action == "set_quality":
                    cfg = {"quality": msg.get("quality", "scout"), "steps": msg.get("steps", 2)}
                    with open(os.path.join(IO_DIR, "quality.json"), 'w') as f:
                        json.dump(cfg, f)

                elif action == "load_scene":
                    scene_file = msg.get("scene", "")
                    if scene_file == session_state["active_scene"]:
                        await websocket.send_json({"type": "toast", "message": "Already playing this scene", "done": True})
                    elif session_state["processing_scene"]:
                        await websocket.send_json({"type": "toast", "message": "Already processing, please wait..."})
                    else:
                        session_state["processing_scene"] = scene_file
                        scene_name = scene_file.rsplit('.', 1)[0]
                        # Check if already preprocessed
                        render_dir = os.path.join(USER_INPUT, "new_vggt", scene_name, "render")
                        already_done = os.path.exists(render_dir) and len(os.listdir(render_dir)) > 0
                        pq = queue_mod.Queue()
                        threading.Thread(target=process_scene_background, args=(scene_file, pq), daemon=True).start()
                        if already_done:
                            await websocket.send_json({"type": "toast", "message": f"Loading {scene_name}... (already processed, ~1-2 min)"})
                        else:
                            await websocket.send_json({"type": "toast", "message": f"Processing {scene_name}... (~3-5 min)"})

                        async def poll():
                            while True:
                                await asyncio.sleep(1)
                                try:
                                    while not pq.empty():
                                        u = pq.get_nowait()
                                        await websocket.send_json(u)
                                        if u.get("done"):
                                            return
                                except Exception:
                                    return
                        asyncio.create_task(poll())

                elif action == "get_scenes":
                    scenes = get_available_scenes()
                    for sc in scenes:
                        sc["active"] = sc["file"] == session_state.get("active_scene", "IMG_7643.mp4")
                    await websocket.send_json({"type": "scenes", "scenes": scenes})

                elif action == "pose":
                    write_pose(
                        moveX=msg.get("moveX", 0), moveY=msg.get("moveY", 0),
                        lookX=msg.get("lookX", 0), lookY=msg.get("lookY", 0),
                    )

                elif action == "toggle_gpu":
                    if session_state["servers_stopped"]:
                        restore_llama_servers()
                        await websocket.send_json({"type": "gpu_status", "stopped": False})
                    else:
                        stop_llama_servers()
                        await websocket.send_json({"type": "gpu_status", "stopped": True})

            except Exception:
                break

    # Send initial active scene info
    try:
        await websocket.send_json({"type": "active_scene", "scene": session_state.get("active_scene", "IMG_7643.mp4")})
    except Exception:
        pass

    asyncio.create_task(receive_commands())

    # Instead of globbing 1000s of files every tick, probe sequentially
    no_frame_count = 0  # track consecutive misses to detect scene switch
    while True:
        await asyncio.sleep(0.05)

        if paused:
            continue

        # Probe for next frame(s) — skip ahead if we fell behind
        found = False
        probe_idx = last_frame_idx + 1
        attempts = 0
        while attempts < 100:  # skip up to 100 frames if we fell behind
            path = os.path.join(FRAMES_DIR, f"frame_{probe_idx:06d}.jpg")
            if os.path.exists(path):
                found = True
                probe_idx += 1
                attempts = 0  # reset since we found one
            else:
                attempts += 1
                probe_idx += 1

        if not found:
            no_frame_count += 1
            # After 2 seconds of no frames (40 ticks), do a glob to find where frames are
            # This handles: initial connect, scene switches (frames restart at 0), loops
            if no_frame_count >= 40:
                no_frame_count = 0
                try:
                    some = glob.glob(os.path.join(FRAMES_DIR, "frame_*.jpg"))
                    if some:
                        idxs = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in some[-50:]]
                        max_idx = max(idxs)
                        if max_idx != last_frame_idx:
                            probe_idx = max_idx
                            found = True
                except Exception:
                    pass
            if not found:
                continue
        else:
            no_frame_count = 0

        # Send the latest frame we found
        target_idx = probe_idx - 1  # last successful probe
        if target_idx <= last_frame_idx:
            continue

        target_path = os.path.join(FRAMES_DIR, f"frame_{target_idx:06d}.jpg")
        try:
            with open(target_path, 'rb') as f:
                frame_bytes = f.read()
            if len(frame_bytes) < 100:  # incomplete write
                continue
            b64 = base64.b64encode(frame_bytes).decode('ascii')
            await websocket.send_json({"type": "frame", "data": b64, "frame": target_idx})
            last_frame_idx = target_idx
            if recording:
                recorded_frames.append(target_path)
        except Exception:
            continue


async def stitch_and_send(websocket):
    global recorded_frames
    os.makedirs(RECORD_DIR, exist_ok=True)
    for f in os.listdir(RECORD_DIR):
        os.remove(os.path.join(RECORD_DIR, f))
    for i, fp in enumerate(recorded_frames):
        if os.path.exists(fp):
            shutil.copy2(fp, os.path.join(RECORD_DIR, f"frame_{i:06d}.jpg"))
    output = os.path.join(IO_DIR, "recording.mp4")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-framerate", "24",
            "-i", os.path.join(RECORD_DIR, "frame_%06d.jpg"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", output
        ], capture_output=True, timeout=120)
        await websocket.send_json({"type": "download", "url": "/download_recording"})
    except Exception:
        pass


if __name__ == "__main__":
    os.makedirs(THUMBS_DIR, exist_ok=True)
    print(f"\n{'='*50}")
    print(f"  InSpatio-World Viewer v3")
    print(f"  http://100.109.173.109:{PORT}")
    print(f"{'='*50}\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
