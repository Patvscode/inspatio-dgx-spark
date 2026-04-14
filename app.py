"""InSpatio-World — Gradio Web UI for DGX Spark
Clean, mobile-first UI. Upload → Generate → View.
Resolution presets + Scout/Render workflow.
"""
import gradio as gr
import subprocess
import os
import glob
import shutil
import time
import threading
import atexit

# ── Paths ──
HOST_DIR = os.path.expanduser("~/Desktop/AI-apps-workspace/inspatio-world")
HOST_TRAJ_DIR = os.path.join(HOST_DIR, "traj")
HOST_OUTPUT_DIR = os.path.join(HOST_DIR, "output")
HOST_INPUT_DIR = os.path.join(HOST_DIR, "user_input")
CONTAINER_WORK = "/workspace/inspatio-world"
CONTAINER_CONFIG = "configs/inference_1.3b.yaml"
CONTAINER_NAME = "inspatio-world"

# Human-friendly trajectory names
TRAJECTORIES = {
    "🔄 Orbit Around": "x_y_circle_cycle.txt",
    "🔍 Zoom Through": "zoom_out_in.txt",
}

# Resolution presets: name → (height, width, description)
RESOLUTION_PRESETS = {
    "⚡ Scout (240p)": (240, 416, "~4x faster, grainy preview"),
    "🔄 Draft (360p)": (360, 624, "~2x faster, decent quality"),
    "✨ Full (480p)": (480, 832, "Best quality, ~2 min"),
}

# Quality presets: name → (denoising_steps, description)
QUALITY_PRESETS = {
    "⚡ Fast (2 steps)": ("1000,250", "Quick preview, some artifacts"),
    "🔄 Balanced (3 steps)": ("1000,500,250", "Good quality, faster"),
    "✨ Best (4 steps)": ("1000,750,500,250", "Full quality, slowest"),
}


# ── GPU Memory Manager ──
class GPUMemoryManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.servers_stopped = False
        self.timer = None
        self.timer_end = None
        self._start_watchdog()

    def _start_watchdog(self):
        def watchdog():
            while True:
                time.sleep(60)
                with self.lock:
                    self._sync_actual_state()
                    if self.servers_stopped and self.timer_end and time.time() > self.timer_end + 30:
                        self._do_restore()
        t = threading.Thread(target=watchdog, daemon=True)
        t.start()

    def _sync_actual_state(self):
        try:
            result = subprocess.run(["pgrep", "-c", "llama-server"], capture_output=True, text=True, timeout=5)
            count = int(result.stdout.strip()) if result.returncode == 0 else 0
            self.servers_stopped = (count == 0)
        except Exception:
            pass

    def _do_restore(self):
        try:
            subprocess.run(["systemctl", "--user", "start", "llama-main.service"], timeout=15, capture_output=True)
            time.sleep(2)
            result = subprocess.run(["pgrep", "-af", "llama-server.*18081"], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                subprocess.Popen(
                    ["/home/pmello/llama.cpp-new/build-cuda/bin/llama-server",
                     "-m", "/home/pmello/models/gemma-4/E2B-it/gemma-4-E2B-it-Q8_0.gguf",
                     "--mmproj", "/home/pmello/models/gemma-4/E2B-it/mmproj-BF16.gguf",
                     "--host", "127.0.0.1", "--port", "18081",
                     "--ctx-size", "262144", "--n-gpu-layers", "999",
                     "--threads", "8", "--jinja", "--reasoning-format", "deepseek"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        except Exception:
            pass
        self.servers_stopped = False
        if self.timer:
            self.timer.cancel()
        self.timer = None
        self.timer_end = None

    def stop_servers(self):
        with self.lock:
            self._sync_actual_state()
            if self.servers_stopped:
                return True, "Already free"
            try:
                subprocess.run(["systemctl", "--user", "stop", "llama-main.service"], timeout=10, capture_output=True)
                subprocess.run(["pkill", "-f", "llama-server"], timeout=5, capture_output=True)
                time.sleep(3)
                subprocess.run(["pkill", "-9", "-f", "llama-server"], timeout=5, capture_output=True)
                time.sleep(2)
                self._sync_actual_state()
                return self.servers_stopped, "Freed" if self.servers_stopped else "Some servers still running"
            except Exception as e:
                return False, str(e)

    def restart_servers(self):
        with self.lock:
            self._sync_actual_state()
            if not self.servers_stopped:
                return True, "Already running"
            self._do_restore()
            time.sleep(3)
            self._sync_actual_state()
            return not self.servers_stopped, "Restored" if not self.servers_stopped else "May not have started"

    def set_timer(self, minutes):
        with self.lock:
            if self.timer:
                self.timer.cancel()
            if minutes <= 0:
                self.timer = None
                self.timer_end = None
                return
            minutes = min(60, int(minutes))
            self.timer_end = time.time() + (minutes * 60)
            self.timer = threading.Timer(minutes * 60, self._timer_expired)
            self.timer.daemon = True
            self.timer.start()

    def cancel_timer(self):
        with self.lock:
            if self.timer:
                self.timer.cancel()
            self.timer = None
            self.timer_end = None

    def _timer_expired(self):
        with self.lock:
            self._sync_actual_state()
            if self.servers_stopped:
                self._do_restore()
            self.timer = None
            self.timer_end = None

    def get_status_html(self):
        with self.lock:
            self._sync_actual_state()
            parts = []

            if self.servers_stopped:
                parts.append('<span style="color:#ff6b6b">⬤</span> GPU free for InSpatio')
            else:
                try:
                    result = subprocess.run(["pgrep", "-c", "llama-server"], capture_output=True, text=True, timeout=5)
                    count = int(result.stdout.strip()) if result.returncode == 0 else 0
                    parts.append(f'<span style="color:#51cf66">⬤</span> {count} model(s) active')
                except Exception:
                    parts.append('<span style="color:#51cf66">⬤</span> Models active')

            if self.timer_end:
                remaining = max(0, self.timer_end - time.time())
                mins = int(remaining // 60)
                secs = int(remaining % 60)
                parts.append(f'⏱ {mins}:{secs:02d} until restore')

            try:
                result = subprocess.run(
                    ["docker", "ps", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Status}}"],
                    capture_output=True, text=True, timeout=5)
                if result.stdout.strip():
                    parts.append(f'🐳 {result.stdout.strip()}')
                else:
                    parts.append('🐳 Container stopped')
            except Exception:
                pass

            return " · ".join(parts)

    def cleanup(self):
        with self.lock:
            if self.timer:
                self.timer.cancel()
            if self.servers_stopped:
                self._do_restore()


gpu_mgr = GPUMemoryManager()
atexit.register(gpu_mgr.cleanup)

_pipeline_lock = threading.Lock()


def _check_container():
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=5)
        if CONTAINER_NAME in result.stdout:
            return True, ""
        subprocess.run(["docker", "start", CONTAINER_NAME], timeout=15, capture_output=True)
        time.sleep(2)
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=5)
        if CONTAINER_NAME in result.stdout:
            return True, "Container restarted."
        return False, "Container not found. Run setup.sh first."
    except Exception as e:
        return False, str(e)


def get_trajs():
    trajs = dict(TRAJECTORIES)
    if os.path.isdir(HOST_TRAJ_DIR):
        for f in sorted(glob.glob(os.path.join(HOST_TRAJ_DIR, "*.txt"))):
            basename = os.path.basename(f)
            if basename not in trajs.values():
                name = os.path.splitext(basename)[0].replace("_", " ").title()
                trajs[name] = basename
    return trajs


def _parse_resolution(preset_name):
    """Return (height, width) from preset name."""
    preset = RESOLUTION_PRESETS.get(preset_name)
    if preset:
        return preset[0], preset[1]
    return 480, 832  # default


def _parse_quality(preset_name):
    """Return denoising steps string from preset name."""
    preset = QUALITY_PRESETS.get(preset_name)
    if preset:
        return preset[0]
    return "1000,750,500,250"  # default


def estimate_time(resolution, quality):
    """Rough estimate of generation time based on settings."""
    h, w = _parse_resolution(resolution)
    steps = _parse_quality(quality)
    n_steps = len(steps.split(","))

    # Base: 480x832 at 4 steps ≈ 2 min
    pixel_ratio = (h * w) / (480 * 832)
    step_ratio = n_steps / 4
    est_seconds = 120 * pixel_ratio * step_ratio

    if est_seconds < 60:
        return f"~{int(est_seconds)}s"
    else:
        return f"~{est_seconds/60:.1f} min"


def update_estimate(resolution, quality):
    """Update the time estimate display."""
    est = estimate_time(resolution, quality)
    h, w = _parse_resolution(resolution)
    steps = _parse_quality(quality)
    n_steps = len(steps.split(","))
    return f"**{w}×{h}** · {n_steps} denoising steps · {est} estimated"


def run_pipeline(video_file, trajectory, resolution, quality,
                 keep_loaded, timer_min, compile_dit, use_tae):
    if not _pipeline_lock.acquire(blocking=False):
        return None, "⏳ Already processing a video. Please wait.", gpu_mgr.get_status_html()

    try:
        return _run_inner(video_file, trajectory, resolution, quality,
                          keep_loaded, timer_min, compile_dit, use_tae)
    finally:
        _pipeline_lock.release()


def _run_inner(video_file, trajectory, resolution, quality,
               keep_loaded, timer_min, compile_dit, use_tae):
    if video_file is None:
        return None, "Upload a video first.", gpu_mgr.get_status_html()

    ok, msg = _check_container()
    if not ok:
        return None, f"❌ {msg}", gpu_mgr.get_status_html()

    # Parse settings
    gen_h, gen_w = _parse_resolution(resolution)
    denoising_steps = _parse_quality(quality)

    # Free GPU
    freed_ok, freed_msg = gpu_mgr.stop_servers()
    if not freed_ok and "already" not in freed_msg.lower():
        return None, f"Couldn't free GPU: {freed_msg}", gpu_mgr.get_status_html()
    time.sleep(2)

    # Prep input
    os.makedirs(HOST_INPUT_DIR, exist_ok=True)
    for item in os.listdir(HOST_INPUT_DIR):
        p = os.path.join(HOST_INPUT_DIR, item)
        if os.path.isfile(p):
            os.remove(p)
        elif os.path.isdir(p):
            shutil.rmtree(p)

    video_name = os.path.basename(video_file)
    if not video_name.lower().endswith(".mp4"):
        video_name = os.path.splitext(video_name)[0] + ".mp4"
    dest = os.path.join(HOST_INPUT_DIR, video_name)
    shutil.copy2(video_file, dest)

    # Resolve trajectory
    trajs = get_trajs()
    traj_file = trajs.get(trajectory, "x_y_circle_cycle.txt")
    traj_name = os.path.splitext(traj_file)[0]

    # Clean stale output
    output_dir = os.path.join(HOST_OUTPUT_DIR, "user_input", traj_name)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Build command with resolution params
    cmd = (
        f"cd {CONTAINER_WORK} && "
        f"TORCH_CUDA_ARCH_LIST=12.1a "
        f"TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas "
        f"TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache "
        f"bash run_test_pipeline.sh "
        f"--input_dir ./user_input "
        f"--traj_txt_path ./traj/{traj_file} "
        f"--config_path {CONTAINER_CONFIG} "
        f"--master_port 29515 "
        f"--gen_width {gen_w} "
        f"--gen_height {gen_h} "
        f"--denoising_steps {denoising_steps}"
    )
    if compile_dit:
        cmd += " --compile_dit"
    if use_tae:
        cmd += " --use_tae"

    mode_label = resolution.split("(")[0].strip() if "(" in resolution else resolution
    status_update = f"🔄 Running {mode_label} at {gen_w}×{gen_h}..."

    start = time.time()
    try:
        result = subprocess.run(
            ["docker", "exec", CONTAINER_NAME, "bash", "-c", cmd],
            capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        if not keep_loaded:
            gpu_mgr.restart_servers()
        return None, "Timed out after 30 minutes.", gpu_mgr.get_status_html()
    except Exception as e:
        if not keep_loaded:
            gpu_mgr.restart_servers()
        return None, str(e), gpu_mgr.get_status_html()

    elapsed = time.time() - start
    logs = (result.stdout or "") + "\n" + (result.stderr or "")

    # Restore or keep loaded
    if keep_loaded:
        if timer_min and timer_min > 0:
            gpu_mgr.set_timer(timer_min)
    else:
        gpu_mgr.restart_servers()

    if result.returncode != 0:
        snippet = logs[-800:] if len(logs) > 800 else logs
        return None, f"Pipeline failed:\n{snippet}", gpu_mgr.get_status_html()

    # Find output
    pred_videos = sorted(glob.glob(os.path.join(output_dir, "**/*pred_video*.mp4"), recursive=True))
    pred = pred_videos[0] if pred_videos else None

    if pred is None:
        return None, "Completed but no output video found.", gpu_mgr.get_status_html()

    # Build status
    timing = ""
    for line in logs.split("\n"):
        if "Video 0 timing" in line:
            t = line.strip().replace("[Rank 0] ", "")
            timing = f"\n{t}"
            break

    status = f"✅ {elapsed/60:.1f} min · {gen_w}×{gen_h}{timing}"
    if keep_loaded:
        status += "\n📌 Models paused — timer running"

    return pred, status, gpu_mgr.get_status_html()


def do_free_gpu(timer_min):
    gpu_mgr.stop_servers()
    if timer_min and timer_min > 0:
        gpu_mgr.set_timer(timer_min)
    return gpu_mgr.get_status_html()


def do_restore():
    gpu_mgr.restart_servers()
    gpu_mgr.cancel_timer()
    return gpu_mgr.get_status_html()


# ── Custom CSS ──
CUSTOM_CSS = """
/* Clean, modern look */
.gradio-container {
    max-width: 800px !important;
    margin: auto !important;
}

/* Hero header */
.hero {
    text-align: center;
    padding: 1.5rem 1rem 0.5rem;
}
.hero h1 {
    font-size: 1.8rem;
    margin-bottom: 0.3rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    color: #888;
    font-size: 0.95rem;
    margin: 0;
}

/* Generate button */
.generate-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    padding: 0.8rem !important;
    border-radius: 12px !important;
    color: white !important;
}
.generate-btn:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px);
}

/* Scout button */
.scout-btn {
    background: linear-gradient(135deg, #ffd43b 0%, #fab005 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    color: #333 !important;
}

/* Status bar */
.status-bar {
    font-size: 0.8rem;
    padding: 0.4rem 0.8rem;
    border-radius: 8px;
    background: rgba(102, 126, 234, 0.1);
}

/* Estimate display */
.estimate-display {
    font-size: 0.85rem;
    padding: 0.3rem 0.6rem;
    border-radius: 6px;
    background: rgba(81, 207, 102, 0.1);
    text-align: center;
}

/* Result video */
.result-video {
    border-radius: 12px;
    overflow: hidden;
}

/* Compact accordion content */
.compact-panel .form {
    gap: 0.5rem !important;
}

/* Mobile tweaks */
@media (max-width: 768px) {
    .hero h1 { font-size: 1.4rem; }
    .gradio-container { padding: 0 0.5rem !important; }
}
"""

# ── Build UI ──
with gr.Blocks(css=CUSTOM_CSS, title="InSpatio-World") as app:

    # Hero
    gr.HTML("""
    <div class="hero">
        <h1>InSpatio-World</h1>
        <p>Turn any video into a 3D scene you can explore from new angles</p>
    </div>
    """)

    # Status bar
    status_bar = gr.HTML(
        value=f'<div class="status-bar">{gpu_mgr.get_status_html()}</div>',
        elem_classes=["status-bar"]
    )

    # Upload + trajectory
    video_input = gr.Video(label="Video", format="mp4", height=200)

    trajectory = gr.Radio(
        choices=list(get_trajs().keys()),
        value="🔄 Orbit Around",
        label="Camera Move",
    )

    # Resolution & Quality controls
    with gr.Row():
        resolution = gr.Radio(
            choices=list(RESOLUTION_PRESETS.keys()),
            value="✨ Full (480p)",
            label="Resolution",
        )
        quality = gr.Radio(
            choices=list(QUALITY_PRESETS.keys()),
            value="✨ Best (4 steps)",
            label="Quality",
        )

    # Time estimate
    estimate_display = gr.Markdown(
        value=update_estimate("✨ Full (480p)", "✨ Best (4 steps)"),
        elem_classes=["estimate-display"],
    )

    # Update estimate when settings change
    resolution.change(fn=update_estimate, inputs=[resolution, quality], outputs=[estimate_display])
    quality.change(fn=update_estimate, inputs=[resolution, quality], outputs=[estimate_display])

    # Buttons
    with gr.Row():
        scout_btn = gr.Button(
            "⚡ Scout Preview",
            variant="secondary",
            size="lg",
            elem_classes=["scout-btn"],
        )
        run_btn = gr.Button(
            "✨ Generate",
            variant="primary",
            size="lg",
            elem_classes=["generate-btn"],
        )

    # Result
    pipeline_status = gr.Textbox(label="", lines=2, interactive=False, show_label=False)
    video_output = gr.Video(label="Result", elem_classes=["result-video"], height=300)

    # Advanced: Keep loaded + timer
    with gr.Accordion("⚡ Batch Mode", open=False):
        gr.Markdown("*Keep GPU dedicated to InSpatio for faster back-to-back runs.*", elem_classes=["compact-panel"])
        keep_loaded = gr.Checkbox(
            value=False,
            label="Keep loaded between runs",
            info="Other AI models stay paused until timer expires"
        )
        timer_slider = gr.Slider(
            minimum=0, maximum=60, step=5, value=30,
            label="Auto-restore after (min)",
            info="0 = manual restore only"
        )

    # Advanced: Performance
    with gr.Accordion("⚙️ Performance", open=False):
        compile_dit = gr.Checkbox(value=True, label="Compiled inference (2x faster, slow first run)")
        use_tae = gr.Checkbox(value=True, label="Fast decoder (TAE)")

    # Advanced: GPU controls
    with gr.Accordion("🔧 GPU Control", open=False):
        with gr.Row():
            free_gpu_btn = gr.Button("Free GPU", variant="stop", size="sm")
            restore_btn = gr.Button("Restore Models", variant="primary", size="sm")

    # Example
    example_dir = os.path.join(HOST_DIR, "test", "example")
    if os.path.exists(example_dir):
        examples = [[os.path.join(example_dir, f)] for f in sorted(os.listdir(example_dir)) if f.endswith(".mp4")]
        if examples:
            with gr.Accordion("📎 Examples", open=False):
                gr.Examples(examples=examples, inputs=[video_input])

    # ── Wire events ──

    # Scout button: forces low-res fast settings
    def run_scout(video_file, trajectory, keep_loaded, timer_min, compile_dit, use_tae):
        return run_pipeline(
            video_file, trajectory,
            "⚡ Scout (240p)", "⚡ Fast (2 steps)",
            keep_loaded, timer_min, compile_dit, use_tae
        )

    scout_btn.click(
        fn=run_scout,
        inputs=[video_input, trajectory, keep_loaded, timer_slider, compile_dit, use_tae],
        outputs=[video_output, pipeline_status, status_bar],
    ).then(
        fn=lambda: f'<div class="status-bar">{gpu_mgr.get_status_html()}</div>',
        outputs=[status_bar],
    )

    # Generate button: uses selected resolution/quality
    run_btn.click(
        fn=run_pipeline,
        inputs=[video_input, trajectory, resolution, quality,
                keep_loaded, timer_slider, compile_dit, use_tae],
        outputs=[video_output, pipeline_status, status_bar],
    ).then(
        fn=lambda: f'<div class="status-bar">{gpu_mgr.get_status_html()}</div>',
        outputs=[status_bar],
    )

    free_gpu_btn.click(
        fn=lambda t: f'<div class="status-bar">{do_free_gpu(t)}</div>',
        inputs=[timer_slider],
        outputs=[status_bar],
    )

    restore_btn.click(
        fn=lambda: f'<div class="status-bar">{do_restore()}</div>',
        outputs=[status_bar],
    )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
