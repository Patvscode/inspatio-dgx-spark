#!/usr/bin/env python3
"""InSpatio-World Streaming Viewer — live frames from the DiT model.

The world moves. You steer. Record your flythrough.

Start the DiT server first (in another terminal):
    docker exec inspatio-world bash -c "cd /workspace/inspatio-world && \
        TORCH_CUDA_ARCH_LIST=12.1a \
        TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
        TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache \
        python3 dit_stream.py"

Then run this:
    python3 interactive_stream.py
"""

import glob
import json
import os
import shutil
import subprocess
import threading
import time

import numpy as np
import viser
from PIL import Image

HOST_DIR = os.path.expanduser("~/Desktop/AI-apps-workspace/inspatio-world")
IO_DIR = os.path.join(HOST_DIR, "interactive_io")
FRAMES_DIR = os.path.join(IO_DIR, "frames")
POSE_FILE = os.path.join(IO_DIR, "pose.json")
STATUS_FILE = os.path.join(IO_DIR, "status.json")
RECORD_DIR = os.path.join(IO_DIR, "recording")
PORT = 7861


def read_status():
    try:
        with open(STATUS_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {"status": "unknown"}


def write_pose(yaw=0, pitch=0, zoom=1.0, recording=False, stop=False):
    with open(POSE_FILE, 'w') as f:
        json.dump({"yaw": yaw, "pitch": pitch, "zoom": zoom, 
                    "recording": recording, "stop": stop}, f)


def main():
    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(RECORD_DIR, exist_ok=True)

    # ── Viser server ──
    server = viser.ViserServer(port=PORT)
    server.gui.configure_theme(
        control_layout="collapsible",
        control_width="small",
        dark_mode=True,
        show_logo=False,
        show_share_button=False,
        brand_color=(102, 126, 234),
    )
    server.gui.set_panel_label("🎬 Controls")

    # ── Load point cloud as backdrop ──
    try:
        from plyfile import PlyData
        plys = glob.glob(os.path.join(HOST_DIR, "user_input/**/*_da3_tmp/point_cloud.ply"), recursive=True)
        if plys:
            ply = PlyData.read(plys[0])
            v = ply['vertex']
            points = np.stack([v['x'], v['y'], v['z']], axis=-1).astype(np.float32)
            colors = np.stack([v['red'], v['green'], v['blue']], axis=-1).astype(np.uint8)
            server.scene.add_point_cloud("/backdrop", points=points, colors=colors, point_size=0.01)
            print(f"Loaded backdrop: {len(points):,} points")
    except Exception as e:
        print(f"No backdrop point cloud: {e}")

    # ── GUI ──
    gui_status = server.gui.add_markdown("*Waiting for DiT server...*")

    # Recording
    gui_record_btn = server.gui.add_button("🔴 Start Recording")
    gui_stop_record = server.gui.add_button("⏹ Stop & Save Video")
    gui_stop_record.visible = False

    with server.gui.add_folder("Settings", expand_by_default=False):
        gui_point_size = server.gui.add_slider(
            "Backdrop Size", min=0.005, max=0.03, step=0.002, initial_value=0.01
        )
        gui_show_backdrop = server.gui.add_checkbox("Show Point Cloud", initial_value=True)

    # ── State ──
    state = {
        "recording": False,
        "recorded_frames": [],
        "last_frame_idx": -1,
        "frame_count": 0,
    }

    @gui_point_size.on_update
    def _(_):
        if 'points' in dir() and gui_show_backdrop.value:
            server.scene.add_point_cloud("/backdrop", points=points, colors=colors,
                                          point_size=gui_point_size.value)

    @gui_show_backdrop.on_update
    def _(_):
        if not gui_show_backdrop.value:
            server.scene.remove("/backdrop")
        else:
            try:
                server.scene.add_point_cloud("/backdrop", points=points, colors=colors,
                                              point_size=gui_point_size.value)
            except Exception:
                pass

    @gui_record_btn.on_click
    def _(_):
        state["recording"] = True
        state["recorded_frames"] = []
        gui_record_btn.visible = False
        gui_stop_record.visible = True
        gui_status.content = "*🔴 RECORDING — fly around to capture your path*"

    @gui_stop_record.on_click
    def _(_):
        state["recording"] = False
        gui_record_btn.visible = True
        gui_stop_record.visible = False

        n_frames = len(state["recorded_frames"])
        if n_frames < 2:
            gui_status.content = "*⚠️ Not enough frames to save*"
            return

        gui_status.content = f"*Stitching {n_frames} frames to video...*"

        # Save frames and stitch
        for f in os.listdir(RECORD_DIR):
            os.remove(os.path.join(RECORD_DIR, f))

        for i, frame_path in enumerate(state["recorded_frames"]):
            if os.path.exists(frame_path):
                shutil.copy2(frame_path, os.path.join(RECORD_DIR, f"frame_{i:06d}.jpg"))

        # Stitch with ffmpeg
        output_video = os.path.join(IO_DIR, "recording.mp4")
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-framerate", "24",
                "-i", os.path.join(RECORD_DIR, "frame_%06d.jpg"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "18",
                output_video
            ], capture_output=True, timeout=60)
            gui_status.content = f"*✅ Saved! {n_frames} frames → 24fps video*"

            # Offer download
            for _, client in server.get_clients().items():
                with open(output_video, 'rb') as f:
                    client.send_file_download("inspatio_flythrough.mp4", f.read())

        except Exception as e:
            gui_status.content = f"*❌ FFmpeg failed: {str(e)[:50]}*"

    # ── Frame streaming loop ──
    def frame_loop():
        while True:
            time.sleep(0.05)  # Check 20x/sec

            # Find newest frame
            try:
                frames = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_*.jpg")))
            except Exception:
                continue

            if not frames:
                continue

            newest = frames[-1]
            frame_idx = int(os.path.basename(newest).split('_')[1].split('.')[0])

            if frame_idx <= state["last_frame_idx"]:
                continue

            # New frame available!
            try:
                img = np.array(Image.open(newest))
                server.scene.set_background_image(img, format="jpeg")
                state["last_frame_idx"] = frame_idx
                state["frame_count"] += 1

                if state["recording"]:
                    state["recorded_frames"].append(newest)

            except Exception:
                continue  # File might still be writing

            # Update status
            dit_status = read_status()
            fps = dit_status.get("fps", 0)
            block = dit_status.get("block", 0)
            st = dit_status.get("status", "unknown")

            if state["recording"]:
                n_rec = len(state["recorded_frames"])
                gui_status.content = f"*🔴 REC {n_rec} frames | {fps:.1f} FPS | Block {block}*"
            elif st == "streaming":
                gui_status.content = f"*🌍 Live | {fps:.1f} FPS | Frame {state['frame_count']}*"
            elif st == "ready":
                gui_status.content = "*⏳ Model ready — waiting for scene*"
            elif "loading" in st or "warming" in st or "compiling" in st:
                gui_status.content = f"*⏳ {st.replace('_', ' ').title()}...*"
            elif st == "looping":
                gui_status.content = "*🔄 Looping world...*"

    frame_thread = threading.Thread(target=frame_loop, daemon=True)
    frame_thread.start()

    print(f"\n{'='*60}")
    print(f"  InSpatio-World STREAMING Viewer")
    print(f"  Local:     http://localhost:{PORT}")
    print(f"  Tailscale: http://100.109.173.109:{PORT}")
    print(f"  ")
    print(f"  Start DiT server in another terminal:")
    print(f"  docker exec inspatio-world bash -c \\")
    print(f"    \"cd /workspace/inspatio-world && \\")
    print(f"    TORCH_CUDA_ARCH_LIST=12.1a \\")
    print(f"    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \\")
    print(f"    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache \\")
    print(f"    python3 dit_stream.py\"")
    print(f"{'='*60}\n")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        write_pose(stop=True)
        print("\nStopped.")


if __name__ == "__main__":
    main()
