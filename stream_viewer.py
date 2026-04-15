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
DIT_PID_FILE = os.path.join(IO_DIR, "dit_stream.pid")
PORT = 7861

app = FastAPI()

# ── State ──
session_state = {
    "paused": False,
    "running": True,
    "timer_end": None,       # epoch timestamp when session expires (server-side)
    "timer_minutes": 60,     # current timer setting
    "timer_started": None,   # epoch when timer was started
    "servers_stopped": True,
    "recording": False,
    "active_scene": None,    # set at startup from new.json
    "processing_scene": None,
    "scene_generation": 0,   # increments on each scene switch to signal frame reset
    "quality": "scout",
    "steps": 2,
}


def start_server_timer(minutes):
    """Start or restart the server-side session timer."""
    session_state["timer_minutes"] = minutes
    if minutes <= 0:
        session_state["timer_end"] = None
        session_state["timer_started"] = None
    else:
        now = time.time()
        session_state["timer_end"] = now + minutes * 60
        session_state["timer_started"] = now


def get_timer_remaining():
    """Get seconds remaining on server timer. Returns -1 for infinite."""
    if session_state["timer_end"] is None:
        return -1
    remaining = session_state["timer_end"] - time.time()
    return max(0, remaining)


def end_session_cleanup():
    """Clean up when session ends (stop or timer expiry). Restores llama servers."""
    print("[SESSION] Ending session, restoring services...", flush=True)
    stop_dit_stream()
    restore_llama_servers()
    session_state["running"] = False
    session_state["timer_end"] = None


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


_liveness_cache = {"checked_at": 0.0, "pid": None, "alive": None}


def write_viewer_status(status, **kwargs):
    data = {"status": status, "timestamp": time.time(), **kwargs}
    try:
        with open(STATUS_FILE, 'w') as f:
            json.dump(data, f)
    except Exception:
        pass
    return data


def dit_process_alive(pid=None, cache_ttl=3.0):
    """Best-effort liveness check for the containerized DiT worker.

    Treat zombie or pid-reused processes as dead so the viewer does not keep
    presenting a crashed stream as still live.
    """
    now = time.time()
    cached_pid = _liveness_cache.get("pid")
    cached_alive = _liveness_cache.get("alive")
    checked_at = _liveness_cache.get("checked_at", 0.0)
    if pid and cached_pid == pid and cached_alive is not None and (now - checked_at) < cache_ttl:
        return cached_alive

    if not pid:
        try:
            if os.path.exists(DIT_PID_FILE):
                with open(DIT_PID_FILE, 'r') as f:
                    pid = f.read().strip()
        except Exception:
            pid = None

    if not pid:
        _liveness_cache.update({"checked_at": now, "pid": None, "alive": False})
        return False

    alive = False
    try:
        check_cmd = (
            f"pid={pid}; "
            f"if ! kill -0 \"$pid\" 2>/dev/null; then exit 1; fi; "
            f"state=$(awk '/^State:/ {{print $2}}' /proc/$pid/status 2>/dev/null || true); "
            f"if [ \"$state\" = \"Z\" ]; then exit 1; fi; "
            f"cmd=$(tr '\\0' ' ' < /proc/$pid/cmdline 2>/dev/null || true); "
            f"case \"$cmd\" in *dit_stream.py*) exit 0 ;; *) exit 1 ;; esac"
        )
        result = subprocess.run(
            ["docker", "exec", "inspatio-world", "bash", "-lc", check_cmd],
            capture_output=True,
            timeout=2,
            check=False,
        )
        alive = (result.returncode == 0)
    except Exception:
        alive = False

    _liveness_cache.update({"checked_at": now, "pid": pid, "alive": alive})
    return alive


def read_status_for_viewer():
    """Return a UI-safe stream status.

    The raw status.json can go stale after a crash or after a scene restart.
    Surface that honestly to connected viewers instead of leaving the client on
    an old optimistic state.
    """
    status = read_status()
    ts = status.get("timestamp")
    state = status.get("status", "unknown")
    previous_state = status.get("previous_status")
    active_states = {"streaming", "loading_scene", "encoding", "warming_up", "camera_ready", "ready"}

    pid = None
    try:
        if os.path.exists(DIT_PID_FILE):
            with open(DIT_PID_FILE, 'r') as f:
                pid = f.read().strip()
    except Exception:
        pid = None

    if state == "stale" and previous_state in active_states and not dit_process_alive(pid):
        return write_viewer_status("crashed", previous_status=previous_state, age_seconds=status.get("age_seconds"))

    if ts:
        age = time.time() - ts
        if age > 15 and state not in ("stopped", "ended", "unknown", "crashed", "stale"):
            if state in active_states and not dit_process_alive(pid):
                status = write_viewer_status("crashed", previous_status=state, age_seconds=round(age, 1))
            else:
                status = write_viewer_status("stale", previous_status=state, age_seconds=round(age, 1))
    return status


def viewer_status_allows_frame_delivery(status=None):
    """Only stream cached frames when the backend says they are still live."""
    if status is None:
        status = read_status_for_viewer()
    return status.get("status") in {"streaming", "looping", "paused"}


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

    pid = None
    try:
        if os.path.exists(DIT_PID_FILE):
            with open(DIT_PID_FILE, 'r') as f:
                pid = f.read().strip()
    except Exception:
        pid = None

    graceful_loops = 40  # allow up to ~20s for NCCL/process-group cleanup
    kill_timeout = 30

    if pid:
        try:
            stop_cmd = (
                f"if kill -0 {pid} 2>/dev/null; then "
                f"kill -TERM {pid} 2>/dev/null || true; "
                f"for _ in $(seq 1 {graceful_loops}); do "
                f"kill -0 {pid} 2>/dev/null || break; "
                f"sleep 0.5; "
                f"done; "
                f"if kill -0 {pid} 2>/dev/null; then echo '[STOP] dit_stream still alive after graceful wait, forcing kill' >&2; kill -KILL {pid} 2>/dev/null || true; fi; "
                f"fi; "
                f"rm -f /workspace/inspatio-world/interactive_io/dit_stream.pid"
            )
            subprocess.run(
                ["docker", "exec", "inspatio-world", "bash", "-lc", stop_cmd],
                timeout=kill_timeout,
                capture_output=True,
            )
        except Exception:
            pass
    else:
        try:
            subprocess.run(
                [
                    "docker", "exec", "inspatio-world", "bash", "-lc",
                    "pkill -TERM -f dit_stream || true; "
                    f"for _ in $(seq 1 {graceful_loops}); do pgrep -f dit_stream >/dev/null || break; sleep 0.5; done; "
                    "if pgrep -f dit_stream >/dev/null; then echo '[STOP] dit_stream still alive after graceful wait, forcing kill' >&2; pkill -KILL -f dit_stream || true; fi"
                ],
                timeout=kill_timeout,
                capture_output=True,
            )
        except Exception:
            pass

    try:
        if os.path.exists(DIT_PID_FILE):
            os.remove(DIT_PID_FILE)
    except Exception:
        pass


def scene_artifact_paths(name: str):
    return {
        "mp4": os.path.join(USER_INPUT, name + '.mp4'),
        "mov": os.path.join(USER_INPUT, name + '.mov'),
        "avi": os.path.join(USER_INPUT, name + '.avi'),
        "MOV": os.path.join(USER_INPUT, name + '.MOV'),
        "thumb": os.path.join(THUMBS_DIR, name + '.jpg'),
        "vggt": os.path.join(USER_INPUT, 'new_vggt', name),
        "tmp": os.path.join(USER_INPUT, 'new_vggt', name + '_da3_tmp'),
    }


def clear_scene_artifacts(name: str, keep_video: bool = False, keep_thumb: bool = False):
    paths = scene_artifact_paths(name)
    for key, path in paths.items():
        if keep_video and key in ('mp4', 'mov', 'avi', 'MOV'):
            continue
        if keep_thumb and key == 'thumb':
            continue
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def sanitize_upload_name(filename: str) -> str:
    base = os.path.basename(filename or 'upload.mp4').strip()
    safe = ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '_' for ch in base)
    safe = safe.replace(' ', '_')
    while '__' in safe:
        safe = safe.replace('__', '_')
    return safe or 'upload.mp4'


def get_available_scenes():
    scenes = []
    seen_names = set()
    for f in sorted(os.listdir(USER_INPUT)):
        if not f.endswith(('.mp4', '.mov', '.avi', '.MOV')):
            continue
        name = f.rsplit('.', 1)[0]
        if name in seen_names:
            continue
        paths = scene_artifact_paths(name)
        if os.path.exists(paths['mp4']):
            f = name + '.mp4'
        seen_names.add(name)
        render_dir = os.path.join(paths['vggt'], 'render')
        has_render = os.path.exists(render_dir) and len(os.listdir(render_dir)) > 0 if os.path.exists(render_dir) else False
        has_vggt = os.path.exists(paths['vggt'])
        has_tmp = os.path.exists(paths['tmp'])
        try:
            size_mb = os.path.getsize(os.path.join(USER_INPUT, f)) / (1024 * 1024)
        except Exception:
            size_mb = 0
        scenes.append({
            'name': name,
            'file': f,
            'has_thumb': os.path.exists(paths['thumb']),
            'processed': has_render,
            'partially_processed': (has_vggt or has_tmp) and not has_render,
            'size_mb': round(size_mb, 1),
            'is_processing': session_state.get('processing_scene') == f,
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


def build_scene_entry(video_file, video_name, text=None):
    return {
        "video_path": f"./user_input/{video_file}",
        "vggt_depth_path": f"./user_input/new_vggt/{video_name}",
        "vggt_extrinsics_path": f"./user_input/new_vggt/{video_name}/extrinsics.txt",
        "radius_ratio": 1,
        "text": text or "A video scene.",
    }


def update_active_scene_json(video_file, text=None):
    video_name = video_file.rsplit('.', 1)[0]
    json_path = os.path.join(USER_INPUT, "new.json")
    existing_text = text
    try:
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                entries = json.load(f)
            for entry in entries:
                vp = os.path.basename(entry.get("video_path", ""))
                if vp == video_file and entry.get("text"):
                    existing_text = entry.get("text")
                    break
    except Exception:
        pass

    entry = build_scene_entry(video_file, video_name, existing_text)
    with open(json_path, 'w') as f:
        json.dump([entry], f, indent=2)
    return entry


def _do_restart_dit(video_file, video_name, status, progress_queue):
    """Restart DiT stream with a scene that's already preprocessed."""
    status("Starting stream with new scene...")
    session_state["active_scene"] = video_file
    session_state["scene_generation"] += 1

    # Clean frames
    try:
        subprocess.run(["find", FRAMES_DIR, "-name", "*.jpg", "-delete"],
                       capture_output=True, timeout=30)
    except Exception:
        pass

    write_pose(yaw=0, pitch=0, zoom=1.0, moveX=0, moveY=0, moveZ=0, lookX=0, lookY=0, paused=False, stop=False, resetToken=time.time())

    # Write current quality settings so dit_stream picks them up
    try:
        quality_path = os.path.join(IO_DIR, "quality.json")
        with open(quality_path, 'w') as f:
            json.dump({"quality": session_state.get("quality", "scout"), "steps": session_state.get("steps", 2)}, f)
    except Exception:
        pass

    dit_cmd = (
        "cd /workspace/inspatio-world && "
        "INSPATIO_USE_TORCH_COMPILE=0 "
        "TORCH_CUDA_ARCH_LIST=12.1a "
        "TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas "
        "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache "
        "python3 dit_stream.py > /workspace/inspatio-world/interactive_io/dit_stream.log 2>&1 & echo $! > /workspace/inspatio-world/interactive_io/dit_stream.pid"
    )
    subprocess.run(
        ["docker", "exec", "inspatio-world", "bash", "-lc", dit_cmd],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
    )

    status("\u2705 Scene loaded! Model warming up (~1-2 min)...")
    progress_queue.put({"type": "toast", "message": "\u2705 Ready! Warming up model...", "done": True, "error": False})
    session_state["processing_scene"] = None


def process_scene_background(video_file, progress_queue):
    """Run preprocessing pipeline for a new video."""
    SCRIPT_DIR = "/workspace/inspatio-world"
    # Use current quality setting for resolution
    quality = session_state.get("quality", "scout")
    res_map = {"scout": (240, 416), "draft": (360, 624), "full": (480, 832)}
    GEN_H, GEN_W = res_map.get(quality, (240, 416))
    video_name = video_file.rsplit('.', 1)[0]
    traj = "x_y_circle_cycle.txt"

    def status(msg):
        print(f"[SCENE] {msg}", flush=True)
        progress_queue.put({"type": "toast", "message": msg})

    try:
        status(f"Processing {video_name}...")
        stop_dit_stream()
        time.sleep(1)

        subprocess.run(["docker", "start", "inspatio-world"], timeout=10, capture_output=True)
        time.sleep(1)

        render_dir = os.path.join(USER_INPUT, "new_vggt", video_name, "render")
        if os.path.exists(render_dir) and len(os.listdir(render_dir)) > 0:
            status("Scene already preprocessed — skipping pipeline")
            update_active_scene_json(video_file)
            _do_restart_dit(video_file, video_name, status, progress_queue)
            return

        # Fresh first-time processing: clear stale partial outputs before starting.
        clear_scene_artifacts(video_name, keep_video=True, keep_thumb=True)

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
            update_active_scene_json(video_file, text="A video scene with various objects and elements.")
        else:
            # Filter JSON to just our video
            json_path = os.path.join(USER_INPUT, "new.json")
            try:
                with open(json_path, 'r') as f:
                    entries = json.load(f)
                target = [e for e in entries if os.path.basename(e.get("video_path", "")) == video_file]
                if target:
                    with open(json_path, 'w') as f:
                        json.dump(target[:1], f, indent=2)
                else:
                    update_active_scene_json(video_file)
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
            clear_scene_artifacts(video_name, keep_video=True, keep_thumb=True)
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
            clear_scene_artifacts(video_name, keep_video=True, keep_thumb=True)
            session_state["processing_scene"] = None
            return

        # Restart DiT with the new scene
        _do_restart_dit(video_file, video_name, status, progress_queue)

    except Exception as e:
        print(f"[SCENE] Error: {e}", flush=True)
        clear_scene_artifacts(video_name, keep_video=True, keep_thumb=True)
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

# Start server-side timer (default 60 min)
start_server_timer(60)


# ══════════════════════════════════════════════════════════════════════
#  HTML — Full-screen game-style HUD
# ══════════════════════════════════════════════════════════════════════

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no,viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>InSpatio</title>
<script>/* Block any external service workers on this origin */
if('serviceWorker' in navigator){navigator.serviceWorker.getRegistrations().then(r=>r.forEach(w=>w.unregister()));}
</script>
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
.lift-controls{
  display:flex;gap:6px;align-items:center;justify-content:center;
  margin-top:6px;
}
.mini-lift-btn{
  width:34px;height:34px;border-radius:999px;border:1px solid rgba(255,255,255,0.12);
  background:rgba(40,40,48,0.9);color:#ddd;font-size:15px;font-weight:700;cursor:pointer;
  backdrop-filter:blur(8px);
}
.mini-lift-btn:active,.mini-lift-btn.active{
  background:rgba(81,207,102,0.18);border-color:rgba(81,207,102,0.45);color:#fff;
}
.home-btn{font-size:14px}

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
  position:relative;
}
.scene-card .thumb{
  width:72px;height:48px;border-radius:8px;
  overflow:hidden;border:2px solid transparent;
  background:#1a1a1a;
  position:relative;
}
.scene-card .thumb.active{border-color:#51cf66}
.scene-card .thumb.processing{border-color:#fcc419;animation:pulse 1s infinite}
.scene-card .thumb img{width:100%;height:100%;object-fit:cover}
.scene-card .sname{
  font-size:8px;color:rgba(255,255,255,0.35);
  text-align:center;margin-top:3px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.scene-card .status-badge{
  position:absolute;top:2px;right:2px;
  font-size:8px;padding:1px 4px;border-radius:4px;
  background:rgba(0,0,0,0.7);backdrop-filter:blur(4px);
}
.scene-card .status-badge.ready{color:#51cf66}
.scene-card .status-badge.pending{color:#fcc419}
.scene-card .status-badge.working{color:#74c0fc;animation:pulse 1s infinite}
.scene-add{
  width:72px;height:48px;border-radius:8px;
  border:1px dashed rgba(255,255,255,0.15);
  display:flex;align-items:center;justify-content:center;
  font-size:22px;color:rgba(255,255,255,0.2);
  cursor:pointer;flex-shrink:0;
}
.scene-add:active{background:rgba(255,255,255,0.05)}

/* Library section */
.lib-item{
  display:flex;align-items:center;gap:10px;
  padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.05);
}
.lib-item:last-child{border-bottom:none}
.lib-thumb{
  width:56px;height:38px;border-radius:6px;overflow:hidden;
  background:#1a1a1a;flex-shrink:0;
}
.lib-thumb img{width:100%;height:100%;object-fit:cover}
.lib-info{flex:1;min-width:0}
.lib-name{font-size:12px;font-weight:600;color:rgba(255,255,255,0.8);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.lib-meta{font-size:10px;color:rgba(255,255,255,0.35);margin-top:1px}
.lib-status{
  font-size:9px;font-weight:600;padding:2px 6px;border-radius:4px;
  white-space:nowrap;
}
.lib-status.ready{background:rgba(81,207,102,0.15);color:#51cf66}
.lib-status.pending{background:rgba(252,196,25,0.15);color:#fcc419}
.lib-status.working{background:rgba(116,192,252,0.15);color:#74c0fc;animation:pulse 1s infinite}
.lib-delete{
  width:28px;height:28px;border-radius:6px;
  border:1px solid rgba(255,80,80,0.2);background:transparent;
  color:rgba(255,80,80,0.5);font-size:12px;
  cursor:pointer;display:flex;align-items:center;justify-content:center;
  flex-shrink:0;
}
.lib-delete:active{background:rgba(255,80,80,0.15)}
.lib-delete.disabled{opacity:0.3;pointer-events:none}

/* Scene loading overlay */
.scene-overlay{
  position:fixed;inset:0;z-index:35;
  background:rgba(0,0,0,0.7);backdrop-filter:blur(8px);
  display:none;flex-direction:column;align-items:center;justify-content:center;gap:16px;
}
.scene-overlay.visible{display:flex}
.scene-overlay .so-spinner{width:40px;height:40px;border:3px solid rgba(255,255,255,0.1);border-top-color:#51cf66;border-radius:50%;animation:spin 0.8s linear infinite}
.scene-overlay .so-title{font-size:16px;font-weight:700;color:#fff}
.scene-overlay .so-detail{font-size:12px;color:rgba(255,255,255,0.5);text-align:center;max-width:250px}
.scene-overlay .so-steps{display:flex;gap:6px;margin-top:4px}
.scene-overlay .so-step{
  font-size:10px;padding:3px 8px;border-radius:6px;
  background:rgba(255,255,255,0.05);color:rgba(255,255,255,0.3);
}
.scene-overlay .so-step.active{background:rgba(81,207,102,0.15);color:#51cf66}
.scene-overlay .so-step.done{color:rgba(81,207,102,0.5)}

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
    <button class="abtn abtn-sm" title="Kill current session" aria-label="Kill current session" onclick="stopSession()">■</button>
    <div class="spacer"></div>
    <button class="abtn abtn-sm" id="resetBtn" title="Hold to reset">↻</button>
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
      <div class="lift-controls">
        <button class="mini-lift-btn" id="upBtn" title="Move up">↑</button>
        <button class="mini-lift-btn home-btn" id="homeBtn" title="Hold to return home">⌂</button>
        <button class="mini-lift-btn" id="downBtn" title="Move down">↓</button>
      </div>
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
    <div style="display:flex;align-items:center;justify-content:space-between">
      <div class="drawer-label" style="margin-bottom:0">Library</div>
      <div style="font-size:10px;color:rgba(255,255,255,0.25);cursor:pointer" onclick="refreshLibrary()">↻ Refresh</div>
    </div>
    <div id="libraryList" style="margin-top:8px"></div>
  </div>

  <div class="drawer-section">
    <div class="gpu-row">
      <div class="gi" id="gpuLabel">GPU: InSpatio active</div>
      <button class="gpu-btn" id="gpuBtn" onclick="toggleGPU()">Free GPU</button>
    </div>
  </div>

  <div class="drawer-section">
    <div class="drawer-label">Safety</div>
    <button class="gpu-btn" style="width:100%;background:#5a1b1b;border-color:rgba(255,120,120,.35);color:#ffd7d7" onclick="stopSession()">Kill current session and free GPU</button>
  </div>
</div>

<!-- Scene loading overlay -->
<div class="scene-overlay" id="sceneOverlay">
  <div class="so-spinner"></div>
  <div class="so-title" id="soTitle">Loading scene...</div>
  <div class="so-detail" id="soDetail">Preparing 3D data</div>
  <div class="so-steps" id="soSteps"></div>
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
  quality:'scout',steps:2,sens:1.0,gotFirstFrame:false,activeScene:'IMG_7643.mp4',resetToken:0,
  resetHoldTimer:null,resetHoldArmed:false};

// ═══ Elements ═══
const $=id=>document.getElementById(id);
const el={
  vid:$('vid'),canvas:$('canvas'),loading:$('loading'),loadingText:$('loadingText'),
  dot:$('dot'),statusText:$('statusText'),statusChip:$('statusChip'),
  fpsChip:$('fpsChip'),qualChip:$('qualChip'),recChip:$('recChip'),recTime:$('recTime'),
  timerChip:$('timerChip'),
  playBtn:$('playBtn'),recBtn:$('recBtn'),resetBtn:$('resetBtn'),homeBtn:$('homeBtn'),upBtn:$('upBtn'),downBtn:$('downBtn'),
  sceneInfo:$('sceneInfo'),frameInfo:$('frameInfo'),
  drawer:$('drawer'),backdrop:$('backdrop'),sceneStrip:$('sceneStrip'),
  toast:$('toast'),gpuLabel:$('gpuLabel'),gpuBtn:$('gpuBtn'),
  uploadModal:$('uploadModal'),upProg:$('upProg'),upText:$('upText'),upBar:$('upBar'),
  sceneOverlay:$('sceneOverlay'),soTitle:$('soTitle'),soDetail:$('soDetail'),soSteps:$('soSteps'),
  libraryList:$('libraryList'),
};

// ═══ WebSocket ═══
function connect(){
  const p=location.protocol==='https:'?'wss:':'ws:';
  S.ws=new WebSocket(`${p}//${location.host}/ws`);

  S.ws.onopen=()=>{
    setStatus('off','connecting');
    // Wait for server status sync before claiming live streaming.
    if(S.gotFirstFrame){
      el.loading.classList.add('visible');el.loadingText.textContent='Resuming...';
      setTimeout(()=>send({action:'resync'}),200);
    } else {
      el.loading.classList.add('visible');el.loadingText.textContent='Connecting to stream...';
    }
    // Timer will sync from server via timer_sync message
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
      // Apply right joystick viewport transform
      applyViewport();
    }

    if(d.type==='status'){
      const s=d.status||'unknown';
      if(s==='streaming'){
        S.stopped=false;
        setStatus('live','streaming');
      }
      else if(s==='paused'){
        S.stopped=false;
        setStatus('paused','paused');
      }
      else if(s.includes('loading')||s.includes('warming')||s.includes('compil')){
        S.stopped=false;
        setStatus('off',s);el.loading.classList.add('visible');el.loadingText.textContent=s.replace(/_/g,' ');
      }else if(s==='stale'){
        S.stopped=false;
        setStatus('off','stream stale');
        if(!S.gotFirstFrame || Date.now()-S.lastFT>5000){
          el.loading.classList.add('visible');
          el.loadingText.textContent='Stream stalled, waiting for fresh frames...';
        }
      }else if(s==='crashed'){
        S.stopped=true;
        setStatus('off','stream crashed');
        el.loading.classList.add('visible');
        el.loadingText.textContent='Stream process stopped. Reload the scene to restart it.';
      }else{
        if(s==='stopped' || s==='ended') S.stopped=true;
        setStatus('off',s);
      }
    }

    if(d.type==='download'){
      const a=document.createElement('a');a.href=d.url;a.download='inspatio_recording.mp4';
      document.body.appendChild(a);a.click();document.body.removeChild(a);
    }

    if(d.type==='scenes'){renderScenes(d.scenes);renderLibrary(d.scenes);}
    if(d.type==='toast')showToast(d.message,d.done,d.error);
    if(d.type==='active_scene'){
      S.activeScene=d.scene;
      const name=d.scene.replace(/\.[^.]+$/,'');
      el.sceneInfo.textContent=name;
      // Hide scene overlay if showing
      el.sceneOverlay.classList.remove('visible');
      S.gotFirstFrame=false; // expect new frames from new scene
      el.loading.classList.add('visible');el.loadingText.textContent='Loading new scene...';
    }
    if(d.type==='scene_loading'){
      // Show scene loading overlay
      el.sceneOverlay.classList.add('visible');
      el.soTitle.textContent=d.message||'Loading scene...';
      el.soDetail.textContent=d.estimated?'Estimated: '+d.estimated:'Please wait...';
      closeDrawer();
    }
    if(d.type==='scene_ready'){
      // Scene finished loading
      el.sceneOverlay.classList.remove('visible');
      S.activeScene=d.scene;
      S.gotFirstFrame=false;
      el.loading.classList.add('visible');el.loadingText.textContent='Model warming up...';
      const name=(d.scene||'').replace(/\.[^.]+$/,'');
      el.sceneInfo.textContent=name;
    }
    if(d.type==='confirm_reload'){
      // Scene already playing - ask to restart
      S._pendingReload=d.scene;
      showToast('Tap scene again to restart it',true,false);
    }
    if(d.type==='timer_sync'){
      syncTimer(d.remaining_seconds, d.minutes_setting);
    }
    if(d.type==='timer_expired'){
      S.stopped=true;S.paused=false;
      el.playBtn.textContent='■';el.playBtn.classList.add('stopped');
      el.canvas.classList.add('dimmed');
      el.loading.classList.add('visible');el.loadingText.textContent='Session ended — GPU freed';
      setStatus('off','ended');
      el.timerChip.textContent='0:00';el.timerChip.style.color='#e03131';
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
  if(!confirm('Kill the current InSpatio session and free GPU memory? You can start it again later.'))return;
  S.stopped=true;S.paused=false;
  send({action:'stop'});
  el.playBtn.textContent='■';el.playBtn.classList.add('stopped');
  el.canvas.classList.add('dimmed');
  el.loading.classList.add('visible');el.loadingText.textContent='Session stopped — GPU freed';
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

function resetView(){
  S.resetToken=Date.now();
  vpZoom=1.0;
  _mx=0;_my=0;_mz=0;_lx=0;_ly=0;_pd=true;
  send({action:'reset',resetToken:S.resetToken});
}

function armResetHold(){
  if(S.resetHoldTimer) clearTimeout(S.resetHoldTimer);
  S.resetHoldArmed=true;
  el.resetBtn.textContent='…';
  S.resetHoldTimer=setTimeout(()=>{
    S.resetHoldTimer=null;
    if(!S.resetHoldArmed) return;
    el.resetBtn.textContent='↻';
    toast('Returned to start', true);
    resetView();
  }, 900);
}

function cancelResetHold(){
  S.resetHoldArmed=false;
  if(S.resetHoldTimer){
    clearTimeout(S.resetHoldTimer);
    S.resetHoldTimer=null;
  }
  if(el.resetBtn) el.resetBtn.textContent='↻';
}

// ═══ Timer (server-synced) ═══
function setTimer(m){
  S.timerMin=m;
  document.querySelectorAll('#timerPills .pill').forEach(p=>p.classList.toggle('active',parseInt(p.dataset.t)===m));
  send({action:'set_timer',minutes:m});
  if(!m){el.timerChip.textContent='∞';S.timerRemaining=-1;}
}
function syncTimer(remainingSec, minutesSetting){
  // Server sends remaining seconds — just display it
  S.timerMin=minutesSetting;
  S.timerRemaining=remainingSec;
  document.querySelectorAll('#timerPills .pill').forEach(p=>p.classList.toggle('active',parseInt(p.dataset.t)===minutesSetting));
  if(remainingSec<0){el.timerChip.textContent='∞';return;}
  updateTimerDisplay(remainingSec);
}
function updateTimerDisplay(sec){
  if(sec<0){el.timerChip.textContent='∞';return;}
  const m=Math.floor(sec/60),s=Math.floor(sec%60);
  el.timerChip.textContent=m+':'+String(s).padStart(2,'0');
  if(sec<60)el.timerChip.style.color='#e03131';
  else if(sec<300)el.timerChip.style.color='#fcc419';
  else el.timerChip.style.color='';
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
    d.onclick=()=>{
      // If pending reload for same scene, force it
      if(S._pendingReload===sc.file){
        S._pendingReload=null;
        send({action:'load_scene',scene:sc.file,force:true});
      } else {
        S._pendingReload=null;
        send({action:'load_scene',scene:sc.file});
      }
    };
    let cls='thumb';
    if(sc.active)cls+=' active';
    if(sc.is_processing)cls+=' processing';
    // Status badge
    let badge='';
    if(sc.is_processing)badge='<div class="status-badge working">↻</div>';
    else if(sc.processed)badge='<div class="status-badge ready">✓</div>';
    else badge='<div class="status-badge pending">○</div>';
    if(sc.has_thumb)d.innerHTML=`<div class="${cls}"><img src="/thumb/${sc.name}.jpg">${badge}</div><div class="sname">${sc.name}</div>`;
    else d.innerHTML=`<div class="${cls}" style="display:flex;align-items:center;justify-content:center;color:#444">🎬${badge}</div><div class="sname">${sc.name}</div>`;
    el.sceneStrip.appendChild(d);
  });
  const add=document.createElement('div');add.className='scene-add';add.textContent='+';
  add.onclick=()=>{el.uploadModal.classList.add('visible');};
  el.sceneStrip.appendChild(add);
}

// ═══ Library ═══
function renderLibrary(scenes){
  if(!scenes)return;
  el.libraryList.innerHTML='';
  if(scenes.length===0){el.libraryList.innerHTML='<div style="font-size:11px;color:rgba(255,255,255,0.3);padding:8px 0">No videos yet. Tap + to add one.</div>';return;}
  scenes.forEach(sc=>{
    const item=document.createElement('div');item.className='lib-item';
    // Status label
    let statusCls='pending',statusTxt='Not processed';
    if(sc.is_processing){statusCls='working';statusTxt='Processing...';}
    else if(sc.processed){statusCls='ready';statusTxt='Ready';}
    else if(sc.partially_processed){statusCls='pending';statusTxt='Partial';}
    // Can delete?
    const canDelete=!sc.active&&!sc.is_processing;
    const thumbHtml=sc.has_thumb?`<img src="/thumb/${sc.name}.jpg">`:'<div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;color:#444;font-size:14px">🎬</div>';
    item.innerHTML=`
      <div class="lib-thumb">${thumbHtml}</div>
      <div class="lib-info">
        <div class="lib-name">${sc.name}${sc.active?' ▶':''}</div>
        <div class="lib-meta">${sc.size_mb} MB · ${sc.file.split('.').pop().toUpperCase()}</div>
      </div>
      <div class="lib-status ${statusCls}">${statusTxt}</div>
      <button class="lib-delete ${canDelete?'':'disabled'}" onclick="deleteVideo('${sc.name}')">🗑</button>
    `;
    // Tap library item to load scene
    item.querySelector('.lib-info').style.cursor='pointer';
    item.querySelector('.lib-info').onclick=()=>send({action:'load_scene',scene:sc.file});
    el.libraryList.appendChild(item);
  });
}

async function deleteVideo(name){
  if(!confirm('Delete "'+name+'" and all its processed data?'))return;
  try{
    const r=await fetch('/api/video/'+encodeURIComponent(name),{method:'DELETE'});
    const data=await r.json();
    if(r.ok){
      showToast('✅ '+data.message,true,false);
      send({action:'get_scenes'}); // refresh
    }else{
      showToast('❌ '+(data.detail||'Failed'),true,true);
    }
  }catch(e){showToast('❌ Network error',true,true);}
}

function refreshLibrary(){send({action:'get_scenes'});}

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

let _mx=0,_my=0,_mz=0,_lx=0,_ly=0,_pd=false;
setInterval(()=>{
  const active = _pd || Math.abs(_mx)>0.01 || Math.abs(_my)>0.01 || Math.abs(_mz)>0.01 || Math.abs(_lx)>0.01 || Math.abs(_ly)>0.01 || Math.abs(vpZoom-1.0)>0.01;
  if(!active)return;
  _pd=false;
  send({
    action:'pose',
    moveX:Math.round(_mx*100)/100,
    moveY:Math.round(_my*100)/100,
    moveZ:Math.round(_mz*100)/100,
    lookX:Math.round(_lx*100)/100,
    lookY:Math.round(_ly*100)/100,
    zoom:Math.round(vpZoom*100)/100,
    resetToken:S.resetToken,
  });
},50);

// Backend-driven camera only. Keep client viewport visually honest.
let vpZoom=1.0;

const jL=new Joystick('joyL','knobL',(x,y)=>{
  _mx=x;_my=y;_pd=true;
});
const jR=new Joystick('joyR','knobR',(x,y)=>{
  _lx=x;_ly=y;_pd=true;
  vpZoom=1.0+(-y*0.5);
  vpZoom=Math.max(0.5,Math.min(2.0,vpZoom));
});

function bindResetButton(){
  const b=el.resetBtn;
  const h=el.homeBtn;
  [b,h].forEach(btn=>{
    if(!btn) return;
    ['mousedown','touchstart','pointerdown'].forEach(ev=>btn.addEventListener(ev, armResetHold, {passive:false}));
    ['mouseup','mouseleave','touchend','touchcancel','pointerup','pointercancel'].forEach(ev=>btn.addEventListener(ev, cancelResetHold, {passive:false}));
  });
}

function bindLiftButton(btn, value){
  if(!btn) return;
  const start = (e)=>{ if(e) e.preventDefault(); _mz=value; _pd=true; btn.classList.add('active'); };
  const end = (e)=>{ if(e) e.preventDefault(); _mz=0; _pd=true; btn.classList.remove('active'); };
  ['mousedown','touchstart','pointerdown'].forEach(ev=>btn.addEventListener(ev, start, {passive:false}));
  ['mouseup','mouseleave','touchend','touchcancel','pointerup','pointercancel'].forEach(ev=>btn.addEventListener(ev, end, {passive:false}));
}

// ═══ Init ═══
el.loading.classList.add('visible');
bindResetButton();
bindLiftButton(el.upBtn, 1);
bindLiftButton(el.downBtn, -1);
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
    dest = None
    try:
        safe_name = sanitize_upload_name(file.filename)
        stem = safe_name.rsplit('.', 1)[0]

        clear_scene_artifacts(stem, keep_video=False, keep_thumb=False)

        dest = os.path.join(USER_INPUT, safe_name)
        total_bytes = 0
        chunk_size = 1024 * 1024
        with open(dest, 'wb') as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                f.write(chunk)

        if total_bytes <= 0:
            try:
                os.remove(dest)
            except Exception:
                pass
            return JSONResponse({"detail": "Empty upload"}, status_code=400)

        size_mb = total_bytes / (1024 * 1024)
        print(f"[UPLOAD] {safe_name} ({size_mb:.1f} MB)")

        # Convert non-mp4 to mp4 (pipeline only reads .mp4)
        if not safe_name.lower().endswith('.mp4'):
            mp4_name = stem + '.mp4'
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
                clear_scene_artifacts(stem, keep_video=False, keep_thumb=False)
                err = (conv.stderr or conv.stdout or b'')[-200:]
                return JSONResponse({"detail": f"Video conversion failed: {err.decode('utf-8', 'ignore')}"}, status_code=500)

        thumb = os.path.join(THUMBS_DIR, stem + '.jpg')
        thumb_ok = False
        try:
            thumb_res = subprocess.run(["ffmpeg", "-y", "-i", dest, "-ss", "0.5",
                                        "-vframes", "1", "-vf", "scale=160:-1", thumb],
                                       capture_output=True, timeout=20)
            thumb_ok = thumb_res.returncode == 0 and os.path.exists(thumb)
        except Exception:
            thumb_ok = False
        return {"message": f"Uploaded {safe_name} ({size_mb:.1f} MB)", "file": safe_name, "thumbnail": thumb_ok}
    except PermissionError:
        if dest and os.path.exists(dest):
            try:
                os.remove(dest)
            except Exception:
                pass
        return JSONResponse({"detail": "Permission denied"}, status_code=500)
    except Exception as e:
        if dest and os.path.exists(dest):
            try:
                os.remove(dest)
            except Exception:
                pass
        return JSONResponse({"detail": str(e)[:200]}, status_code=500)
    finally:
        try:
            await file.close()
        except Exception:
            pass


@app.get("/api/library")
async def get_library():
    """Full library with processing status and sizes."""
    scenes = get_available_scenes()
    for sc in scenes:
        sc["active"] = sc["file"] == session_state.get("active_scene")
    return {"scenes": scenes, "active_scene": session_state.get("active_scene")}


@app.delete("/api/video/{name}")
async def delete_video(name: str):
    """Delete a video and its preprocessed data."""
    if session_state.get("active_scene", "").startswith(name):
        return JSONResponse({"detail": "Cannot delete the active scene. Switch to another first."}, status_code=400)
    if session_state.get("processing_scene", "") and name in session_state["processing_scene"]:
        return JSONResponse({"detail": "Cannot delete while processing."}, status_code=400)
    
    deleted = []
    # Delete video files
    for ext in ('.mp4', '.mov', '.avi', '.MOV'):
        path = os.path.join(USER_INPUT, name + ext)
        if os.path.exists(path):
            os.remove(path)
            deleted.append(name + ext)
    # Delete preprocessed data (full + partial temp)
    vggt_dir = os.path.join(USER_INPUT, "new_vggt", name)
    if os.path.exists(vggt_dir):
        shutil.rmtree(vggt_dir, ignore_errors=True)
        deleted.append(f"new_vggt/{name}/")
    tmp_dir = os.path.join(USER_INPUT, "new_vggt", name + '_da3_tmp')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
        deleted.append(f"new_vggt/{name}_da3_tmp/")
    # Delete thumbnail
    thumb = os.path.join(THUMBS_DIR, name + '.jpg')
    if os.path.exists(thumb):
        os.remove(thumb)
        deleted.append("thumbnail")
    
    if deleted:
        return {"message": f"Deleted {name}", "deleted": deleted}
    return JSONResponse({"detail": f"Video '{name}' not found"}, status_code=404)


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
    frame_speed = 1  # 1=normal, >1=fast forward, <0=reverse
    frame_jump = 0   # instant frame offset from left joystick X

    async def receive_commands():
        nonlocal paused, last_frame_idx, frame_speed, frame_jump
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
                    end_session_cleanup()
                    await websocket.send_json({"type": "status", "status": "stopped"})
                    await websocket.send_json({"type": "toast", "message": "✅ Session ended. GPU freed, llama servers restored.", "done": True})

                elif action == "start_record":
                    recording = True
                    recorded_frames = []

                elif action == "stop_record":
                    recording = False
                    await stitch_and_send(websocket)

                elif action == "reset":
                    write_pose(yaw=0, pitch=0, zoom=1.0, moveX=0, moveY=0, moveZ=0, lookX=0, lookY=0, paused=False, stop=False, resetToken=msg.get("resetToken", time.time()))

                elif action == "set_timer":
                    minutes = msg.get("minutes", 60)
                    start_server_timer(minutes)

                elif action == "set_quality":
                    new_quality = msg.get("quality", session_state["quality"])
                    new_steps = msg.get("steps", session_state["steps"])
                    resolution_changed = new_quality != session_state["quality"]
                    cfg = {"quality": new_quality, "steps": new_steps}
                    with open(os.path.join(IO_DIR, "quality.json"), 'w') as f:
                        json.dump(cfg, f)
                    session_state["quality"] = new_quality
                    session_state["steps"] = new_steps
                    if resolution_changed and session_state["active_scene"] and not session_state["processing_scene"]:
                        # Resolution change requires DiT restart
                        await websocket.send_json({"type": "toast", "message": f"Switching to {new_quality}... restarting stream", "done": False})
                        scene_file = session_state["active_scene"]
                        session_state["processing_scene"] = scene_file
                        pq = queue_mod.Queue()
                        threading.Thread(target=process_scene_background, args=(scene_file, pq), daemon=True).start()
                        async def poll_qual():
                            while True:
                                await asyncio.sleep(1)
                                try:
                                    while not pq.empty():
                                        u = pq.get_nowait()
                                        await websocket.send_json(u)
                                        if u.get("done"):
                                            await websocket.send_json({"type": "scene_ready", "scene": session_state["active_scene"]})
                                            return
                                except Exception:
                                    return
                        asyncio.create_task(poll_qual())
                    else:
                        # Steps change only — dit_stream.py reads quality.json every block
                        step_labels = {2: "fast", 3: "balanced", 4: "best"}
                        await websocket.send_json({"type": "toast", "message": f"Steps → {new_steps} ({step_labels.get(new_steps, '')})", "done": True})

                elif action == "load_scene":
                    scene_file = msg.get("scene", "")
                    force = msg.get("force", False)
                    if session_state["processing_scene"]:
                        await websocket.send_json({"type": "toast", "message": "Already processing a scene, please wait...", "done": True})
                    elif scene_file == session_state["active_scene"] and not force:
                        # Same scene — offer to restart it
                        await websocket.send_json({"type": "confirm_reload", "scene": scene_file, "message": "This scene is already playing. Tap again to restart it."})
                    else:
                        session_state["processing_scene"] = scene_file
                        scene_name = scene_file.rsplit('.', 1)[0]
                        render_dir = os.path.join(USER_INPUT, "new_vggt", scene_name, "render")
                        already_done = os.path.exists(render_dir) and len(os.listdir(render_dir)) > 0
                        pq = queue_mod.Queue()
                        threading.Thread(target=process_scene_background, args=(scene_file, pq), daemon=True).start()
                        # Notify frontend to show loading overlay
                        if already_done:
                            await websocket.send_json({"type": "scene_loading", "scene": scene_name, "message": f"Loading {scene_name}...", "estimated": "~1-2 min"})
                        else:
                            await websocket.send_json({"type": "scene_loading", "scene": scene_name, "message": f"Processing {scene_name}...", "estimated": "~3-5 min"})

                        async def poll():
                            while True:
                                await asyncio.sleep(1)
                                try:
                                    while not pq.empty():
                                        u = pq.get_nowait()
                                        await websocket.send_json(u)
                                        if u.get("done"):
                                            # Signal scene change complete
                                            await websocket.send_json({"type": "scene_ready", "scene": session_state["active_scene"]})
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
                        moveX=msg.get("moveX", 0), moveY=msg.get("moveY", 0), moveZ=msg.get("moveZ", 0),
                        lookX=msg.get("lookX", 0), lookY=msg.get("lookY", 0),
                        zoom=msg.get("zoom", 1.0),
                        resetToken=msg.get("resetToken"),
                    )

                elif action == "speed":
                    # Left joystick controls playback speed/direction
                    nonlocal frame_speed, frame_jump
                    sy = msg.get("speedY", 0)  # -1 to 1: negative=reverse, positive=forward
                    sx = msg.get("speedX", 0)
                    # Map joystick to speed: center=1x, up=5x, down=-2x (reverse)
                    if abs(sy) < 0.1:
                        frame_speed = 1  # normal
                    elif sy > 0:
                        frame_speed = 1 + int(sy * 8)  # 1-9x forward
                    else:
                        frame_speed = min(-1, int(sy * 4))  # -1 to -4x reverse
                    # X axis: instant jump
                    if abs(sx) > 0.5:
                        frame_jump = int(sx * 50)  # jump up to 50 frames
                    else:
                        frame_jump = 0

                elif action == "resync":
                    nonlocal last_frame_idx
                    last_frame_idx = -1

                elif action == "toggle_gpu":
                    if session_state["servers_stopped"]:
                        restore_llama_servers()
                        await websocket.send_json({"type": "gpu_status", "stopped": False})
                    else:
                        stop_llama_servers()
                        await websocket.send_json({"type": "gpu_status", "stopped": True})

            except Exception:
                break

    # Send initial state (scene + timer sync)
    try:
        await websocket.send_json({"type": "active_scene", "scene": session_state.get("active_scene", "IMG_7643.mp4")})
        await websocket.send_json({"type": "status", "status": read_status_for_viewer().get("status", "unknown")})
        # Sync server timer to client
        remaining = get_timer_remaining()
        await websocket.send_json({
            "type": "timer_sync",
            "remaining_seconds": remaining,
            "minutes_setting": session_state["timer_minutes"],
        })
    except Exception:
        pass

    asyncio.create_task(receive_commands())

    # Track scene generation to detect switches
    current_generation = session_state["scene_generation"]
    last_status_sent = None
    last_status_push = 0.0
    last_timer_sync_push = 0.0

    # Frame delivery loop
    while True:
        await asyncio.sleep(0.05)

        now = time.time()
        if now - last_status_push >= 0.5:
            last_status_push = now
            try:
                viewer_status = read_status_for_viewer().get("status", "unknown")
                if viewer_status != last_status_sent:
                    await websocket.send_json({"type": "status", "status": viewer_status})
                    last_status_sent = viewer_status
            except Exception:
                pass

        if paused:
            continue

        # Server-side timer expiry check
        remaining = get_timer_remaining()
        if remaining == 0 and session_state["timer_end"] is not None:
            try:
                await websocket.send_json({"type": "timer_expired"})
                await websocket.send_json({"type": "toast", "message": "⏰ Session timer expired. GPU freed, llama servers restored.", "done": True})
            except Exception:
                pass
            end_session_cleanup()
            break

        # Send timer sync at most once every 5 seconds.
        # The old wall-clock modulo check could emit many duplicate syncs
        # during the same second, which adds avoidable websocket chatter.
        if remaining > 0 and (now - last_timer_sync_push) >= 5.0:
            last_timer_sync_push = now
            try:
                await websocket.send_json({"type": "timer_sync", "remaining_seconds": remaining, "minutes_setting": session_state["timer_minutes"]})
            except Exception:
                pass

        # Check if scene changed (another scene loaded)
        if session_state["scene_generation"] != current_generation:
            current_generation = session_state["scene_generation"]
            last_frame_idx = -1
            frame_speed = 1
            frame_jump = 0
            try:
                await websocket.send_json({"type": "active_scene", "scene": session_state["active_scene"]})
            except Exception:
                pass

        viewer_status = read_status_for_viewer()
        if not viewer_status_allows_frame_delivery(viewer_status):
            last_frame_idx = -1
            continue

        # On first connect or after resync/scene-switch, find the latest frame via status.json
        if last_frame_idx == -1:
            frame_num = viewer_status.get("frame", 0)
            if frame_num > 0:
                last_frame_idx = max(0, frame_num - 3)  # start a few behind latest
            # Fallback: check if frame 0 exists
            if last_frame_idx == -1:
                if os.path.exists(os.path.join(FRAMES_DIR, "frame_000000.jpg")):
                    last_frame_idx = -1  # will probe from 0
                else:
                    continue  # no frames yet

        # Apply speed/jump from left joystick
        step = max(1, abs(frame_speed)) if frame_speed >= 0 else -1
        if frame_jump != 0:
            last_frame_idx = max(0, last_frame_idx + frame_jump)
            frame_jump = 0  # consume the jump

        if frame_speed < 0:
            # Reverse: go backward
            next_idx = max(0, last_frame_idx - abs(frame_speed))
        else:
            next_idx = last_frame_idx + step

        next_path = os.path.join(FRAMES_DIR, f"frame_{next_idx:06d}.jpg")

        if not os.path.exists(next_path):
            # Maybe DiT looped and frames restarted at 0
            if next_idx > 100 and os.path.exists(os.path.join(FRAMES_DIR, "frame_000000.jpg")):
                # Check if status.json shows a lower frame number (loop detected)
                try:
                    with open(STATUS_FILE, 'r') as f:
                        st = json.load(f)
                    current_frame = st.get("frame", next_idx)
                    if current_frame < last_frame_idx - 10:
                        last_frame_idx = max(0, current_frame - 3)
                except Exception:
                    pass
            continue

        # Read and send
        try:
            with open(next_path, 'rb') as f:
                frame_bytes = f.read()
            if len(frame_bytes) < 100:  # incomplete write
                continue
            b64 = base64.b64encode(frame_bytes).decode('ascii')
            await websocket.send_json({"type": "frame", "data": b64, "frame": next_idx})
            last_frame_idx = next_idx
            if recording:
                recorded_frames.append(next_path)
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
    import signal

    os.makedirs(THUMBS_DIR, exist_ok=True)

    # ── Kill any existing viewer instances (prevents zombie accumulation) ──
    my_pid = os.getpid()
    try:
        result = subprocess.run(
            ["pgrep", "-f", "stream_viewer.py"],
            capture_output=True, text=True, timeout=5
        )
        for pid_str in result.stdout.strip().split('\n'):
            if pid_str.strip():
                pid = int(pid_str.strip())
                if pid != my_pid:
                    try:
                        os.kill(pid, signal.SIGKILL)
                        print(f"[CLEANUP] Killed stale viewer PID {pid}")
                    except ProcessLookupError:
                        pass
    except Exception:
        pass

    # Also free the port if something else holds it
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{PORT}"],
            capture_output=True, text=True, timeout=5
        )
        for pid_str in result.stdout.strip().split('\n'):
            if pid_str.strip():
                pid = int(pid_str.strip())
                if pid != my_pid:
                    try:
                        os.kill(pid, signal.SIGKILL)
                        print(f"[CLEANUP] Killed port holder PID {pid}")
                    except ProcessLookupError:
                        pass
    except Exception:
        pass

    time.sleep(0.5)  # let OS release the port

    print(f"\n{'='*50}")
    print(f"  InSpatio-World Viewer v3")
    print(f"  http://100.109.173.109:{PORT}")
    print(f"  PID: {my_pid}")
    print(f"{'='*50}\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
