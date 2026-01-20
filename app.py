#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mlx>=0.22.0",
#     "numpy>=1.24.0",
#     "safetensors>=0.4.0",
#     "huggingface-hub>=0.20",
#     "tokenizers>=0.15",
#     "soundfile>=0.12",
#     "tqdm>=4.60.0",
#     "psutil>=5.9.0",
#     "gradio>=4.0.0",
# ]
# ///
"""
Gradio UI for HeartMuLa MLX music generation.
Styled to match Suno's interface.
"""

import gc
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr

# Add src to path for development
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

import mlx.core as mx
import numpy as np
import soundfile as sf
from tokenizers import Tokenizer

from heartlib_mlx.heartmula import HeartMuLa
from heartlib_mlx.heartcodec import HeartCodec

# Global model instances
MODEL = None
CODEC = None
TOKENIZER = None

# Suno-style CSS
SUNO_CSS = """
/* ===== SUNO-STYLE DARK THEME ===== */

/* Root variables */
:root {
    --bg-primary: #0a0a0a;
    --bg-secondary: #111111;
    --bg-tertiary: #1a1a1a;
    --bg-input: #1f1f1f;
    --border-color: #2a2a2a;
    --border-hover: #3a3a3a;
    --text-primary: #ffffff;
    --text-secondary: #999999;
    --text-muted: #666666;
    --accent-pink: #e91e8c;
    --accent-pink-dim: #c4177a;
}

/* Global reset */
.gradio-container {
    background: var(--bg-primary) !important;
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

.main, .contain {
    background: var(--bg-primary) !important;
}

/* Hide Gradio footer */
footer { display: none !important; }

/* ===== MAIN LAYOUT ===== */
#main-row {
    gap: 0 !important;
    min-height: 100vh;
}

/* Left sidebar */
#sidebar {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-color) !important;
    padding: 20px 12px !important;
    min-width: 200px !important;
    max-width: 200px !important;
}

#sidebar-logo {
    font-size: 28px;
    font-weight: 800;
    color: var(--text-primary);
    padding: 8px 12px 24px 12px;
    letter-spacing: -1px;
}

.nav-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 12px;
    color: var(--text-secondary) !important;
    font-size: 14px;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.15s ease;
    margin-bottom: 2px;
}

.nav-item:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary) !important;
}

.nav-item.active {
    color: var(--text-primary) !important;
    background: var(--bg-tertiary);
}

.nav-icon {
    width: 20px;
    height: 20px;
    opacity: 0.7;
}

/* Center content area */
#content-area {
    background: var(--bg-primary) !important;
    padding: 24px 32px !important;
    flex: 1;
    max-width: 600px;
    border-right: 1px solid var(--border-color);
}

/* Right library panel */
#library-panel {
    background: var(--bg-primary) !important;
    padding: 24px !important;
    min-width: 400px;
}

/* ===== SECTION STYLING ===== */
.section-container {
    background: transparent !important;
    border: none !important;
    margin-bottom: 24px !important;
}

/* Accordion headers - Suno style */
.accordion {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
}

.accordion > .label-wrap {
    background: transparent !important;
    border: none !important;
    padding: 12px 0 !important;
    cursor: pointer;
}

.accordion > .label-wrap > span {
    color: var(--text-primary) !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    display: flex;
    align-items: center;
    gap: 8px;
}

.accordion > .label-wrap svg {
    width: 18px;
    height: 18px;
    transition: transform 0.2s ease;
}

.accordion[open] > .label-wrap svg {
    transform: rotate(180deg);
}

/* ===== LYRICS INPUT ===== */
#lyrics-box textarea {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    padding: 16px !important;
    min-height: 180px !important;
    resize: vertical;
    transition: border-color 0.15s ease;
}

#lyrics-box textarea::placeholder {
    color: var(--text-muted) !important;
}

#lyrics-box textarea:focus {
    border-color: var(--border-hover) !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Expand button in corner */
.expand-btn {
    position: absolute;
    bottom: 12px;
    right: 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 6px;
    cursor: pointer;
    opacity: 0.6;
    transition: opacity 0.15s;
}

.expand-btn:hover {
    opacity: 1;
}

/* ===== STYLES INPUT ===== */
#styles-box input {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-size: 14px !important;
    padding: 14px 16px !important;
    transition: border-color 0.15s ease;
}

#styles-box input::placeholder {
    color: var(--text-muted) !important;
}

#styles-box input:focus {
    border-color: var(--border-hover) !important;
    outline: none !important;
    box-shadow: none !important;
}

/* ===== TAG PILLS ===== */
#tag-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 16px;
    padding: 4px 0;
}

#tag-row button, .tag-pill {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 20px !important;
    color: var(--text-primary) !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    padding: 8px 14px !important;
    cursor: pointer;
    transition: all 0.15s ease;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

#tag-row button:hover, .tag-pill:hover {
    background: var(--bg-input) !important;
    border-color: var(--border-hover) !important;
}

#tag-row button::before {
    content: "+";
    font-size: 14px;
    opacity: 0.6;
}

/* Equalizer icon button */
.eq-btn {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 20px !important;
    padding: 8px 12px !important;
    min-width: auto !important;
}

/* ===== ADVANCED OPTIONS ===== */
#advanced-section {
    margin-top: 24px;
    border-top: 1px solid var(--border-color);
    padding-top: 16px;
}

/* Slider rows */
.slider-row {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
    margin-bottom: 10px !important;
}

.slider-row label {
    color: var(--text-primary) !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    margin-bottom: 0 !important;
}

/* Custom slider styling */
input[type="range"] {
    -webkit-appearance: none;
    appearance: none;
    background: var(--bg-tertiary) !important;
    border-radius: 4px;
    height: 6px !important;
    cursor: pointer;
}

input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 4px;
    height: 20px;
    background: var(--accent-pink);
    border-radius: 2px;
    cursor: pointer;
    transition: transform 0.1s ease;
}

input[type="range"]::-webkit-slider-thumb:hover {
    transform: scaleY(1.1);
}

/* Slider value display */
.slider-value {
    color: var(--text-secondary);
    font-size: 13px;
    min-width: 40px;
    text-align: right;
}

/* Toggle buttons (Male/Female, Manual/Auto) */
.toggle-group {
    display: flex;
    background: var(--bg-tertiary);
    border-radius: 6px;
    padding: 2px;
}

.toggle-btn {
    padding: 6px 16px;
    border-radius: 4px;
    font-size: 13px;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.15s ease;
}

.toggle-btn.active {
    background: var(--bg-input);
    color: var(--text-primary);
}

/* ===== CREATE BUTTON ===== */
#create-btn {
    background: transparent !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    color: var(--text-muted) !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    padding: 14px 24px !important;
    width: 100%;
    margin-top: 24px;
    cursor: pointer;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
}

#create-btn:hover {
    border-color: var(--border-hover) !important;
    color: var(--text-secondary) !important;
}

#create-btn:active {
    transform: scale(0.98);
}

/* Sparkle icon */
#create-btn::before {
    content: "✦";
    font-size: 14px;
}

/* ===== OUTPUT SECTION ===== */
#output-section {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    margin-top: 20px;
}

/* Audio player */
#audio-player audio {
    width: 100%;
    height: 48px;
    border-radius: 8px;
}

#audio-player audio::-webkit-media-controls-panel {
    background: var(--bg-tertiary);
}

/* Status text */
#status-text {
    color: var(--text-muted) !important;
    font-size: 12px !important;
    font-family: ui-monospace, monospace !important;
    background: transparent !important;
    border: none !important;
    padding: 8px 0 !important;
}

/* ===== LIBRARY PANEL (RIGHT SIDE) ===== */
#library-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}

#library-title {
    color: var(--text-secondary);
    font-size: 14px;
}

.library-item {
    display: flex;
    gap: 12px;
    padding: 12px;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.15s ease;
    margin-bottom: 8px;
}

.library-item:hover {
    background: var(--bg-tertiary);
}

.library-thumb {
    width: 56px;
    height: 56px;
    border-radius: 6px;
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    position: relative;
    overflow: hidden;
}

.library-thumb .duration {
    position: absolute;
    bottom: 4px;
    right: 4px;
    background: rgba(0,0,0,0.7);
    color: white;
    font-size: 10px;
    padding: 2px 4px;
    border-radius: 3px;
}

.library-info {
    flex: 1;
}

.library-info h4 {
    color: var(--text-primary);
    font-size: 14px;
    font-weight: 500;
    margin-bottom: 4px;
}

.library-info p {
    color: var(--text-muted);
    font-size: 12px;
    line-height: 1.4;
}

/* ===== LABELS ===== */
label, .label {
    color: var(--text-secondary) !important;
    font-size: 12px !important;
    font-weight: 400 !important;
    margin-bottom: 6px !important;
}

/* Hide default Gradio elements */
.gr-box, .gr-padded {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

.gr-form {
    background: transparent !important;
    border: none !important;
}

/* Info icons */
.info-icon {
    color: var(--text-muted);
    font-size: 12px;
    cursor: help;
    margin-left: 4px;
}

/* ===== CHECKBOX ===== */
input[type="checkbox"] {
    accent-color: var(--accent-pink);
    width: 16px;
    height: 16px;
}

/* ===== ANIMATIONS ===== */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.2s ease;
}

/* Progress indicator */
.generating-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--text-secondary);
    font-size: 13px;
}

.generating-indicator::before {
    content: "";
    width: 8px;
    height: 8px;
    background: var(--accent-pink);
    border-radius: 50%;
    animation: pulse 1s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ===== MOBILE RESPONSIVE ===== */
@media (max-width: 1024px) {
    #sidebar { display: none !important; }
    #library-panel { display: none !important; }
    #content-area {
        max-width: 100% !important;
        border: none !important;
    }
}
"""


def find_checkpoint():
    """Find the checkpoint directory."""
    candidates = [
        Path(__file__).parent / "ckpt-mlx",
        Path.home() / "Developer/heartlib-mlx/ckpt-mlx",
        Path("./ckpt-mlx"),
    ]
    for c in candidates:
        if c.exists() and (c / "heartmula").exists():
            return str(c)
    return None


def load_models(progress=gr.Progress()):
    """Load models if not already loaded."""
    global MODEL, CODEC, TOKENIZER

    if MODEL is not None:
        return "Models ready"

    ckpt_path = find_checkpoint()
    if ckpt_path is None:
        raise gr.Error("Checkpoint not found.")

    progress(0.2, desc="Loading HeartMuLa...")
    MODEL = HeartMuLa.from_pretrained(f"{ckpt_path}/heartmula")

    progress(0.5, desc="Loading HeartCodec...")
    CODEC = HeartCodec.from_pretrained(f"{ckpt_path}/heartcodec")

    MODEL.set_dtype(mx.bfloat16)
    CODEC.set_dtype(mx.bfloat16)

    progress(0.8, desc="Loading tokenizer...")
    tokenizer_path = Path(ckpt_path).parent / "ckpt" / "tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer_path = Path.home() / "Developer/heartlib/ckpt/tokenizer.json"

    TOKENIZER = Tokenizer.from_file(str(tokenizer_path))
    progress(1.0, desc="Ready")
    return "Models ready"


def add_tag(current_tags, new_tag):
    """Add a tag to the styles input."""
    if not current_tags:
        return new_tag
    tags = [t.strip() for t in current_tags.split(",") if t.strip()]
    if new_tag not in tags:
        tags.append(new_tag)
    return ", ".join(tags)


def generate_music(
    lyrics: str,
    styles: str,
    duration: float,
    cfg_scale: float,
    temperature: float,
    topk: int,
    ignore_eos: bool,
    progress=gr.Progress(),
):
    """Generate music."""
    global MODEL, CODEC, TOKENIZER

    if MODEL is None:
        load_models(progress=progress)

    # Config
    text_bos_id = 128000
    text_eos_id = 128001
    audio_eos_id = 8193
    num_codebooks = 8
    parallel = num_codebooks + 1
    frame_rate = 12.5
    sample_rate = 48000

    # Tokenize
    tags_text = f"<tag>{styles}</tag>"
    tags_ids = TOKENIZER.encode(tags_text.lower()).ids
    if tags_ids[0] != text_bos_id:
        tags_ids = [text_bos_id] + tags_ids
    if tags_ids[-1] != text_eos_id:
        tags_ids = tags_ids + [text_eos_id]

    if lyrics and lyrics.strip():
        lyrics_ids = TOKENIZER.encode(lyrics.lower()).ids
        if lyrics_ids[0] != text_bos_id:
            lyrics_ids = [text_bos_id] + lyrics_ids
        if lyrics_ids[-1] != text_eos_id:
            lyrics_ids = lyrics_ids + [text_eos_id]
        prompt_len = len(tags_ids) + 1 + len(lyrics_ids)
        muq_idx = len(tags_ids)
    else:
        lyrics_ids = []
        prompt_len = len(tags_ids) + 1
        muq_idx = len(tags_ids)

    prompt_tokens = np.zeros((prompt_len, parallel), dtype=np.int64)
    prompt_tokens[:len(tags_ids), -1] = tags_ids
    if lyrics_ids:
        prompt_tokens[len(tags_ids) + 1:, -1] = lyrics_ids

    prompt_mask = np.zeros((prompt_len, parallel), dtype=np.float32)
    prompt_mask[:, -1] = 1.0

    tokens = mx.array(prompt_tokens)[None, :, :]
    tokens = mx.concatenate([tokens, tokens], axis=0)
    mask = mx.array(prompt_mask)[None, :, :]
    mask = mx.concatenate([mask, mask], axis=0)

    muq_embed = mx.zeros((2, MODEL.config.muq_dim))
    pos = mx.broadcast_to(mx.arange(prompt_len)[None, :], (2, prompt_len))

    max_frames = int(duration * frame_rate)
    MODEL.setup_caches(2)

    progress(0, desc="Starting...")
    frames = []

    curr_token = MODEL.generate_frame(
        tokens=tokens,
        tokens_mask=mask,
        input_pos=pos,
        temperature=temperature,
        topk=topk,
        cfg_scale=cfg_scale,
        continuous_segments=muq_embed,
        starts=[muq_idx, muq_idx],
    )
    mx.eval(curr_token)
    frames.append(curr_token[0:1])

    mx.set_memory_limit(32 * 1024**3)
    start_time = time.time()

    for i in range(max_frames - 1):
        elapsed = time.time() - start_time
        fps = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (max_frames - i - 1) / fps if fps > 0 else 0

        progress(
            (i + 1) / max_frames,
            desc=f"Generating {i+2}/{max_frames} • {fps:.1f} f/s • {eta:.0f}s left"
        )

        padded = mx.concatenate([
            curr_token[:, None, :],
            mx.zeros((2, 1, 1), dtype=mx.int32)
        ], axis=-1)
        padded_mask = mx.concatenate([
            mx.ones((2, 1, num_codebooks)),
            mx.zeros((2, 1, 1))
        ], axis=-1)

        curr_token = MODEL.generate_frame(
            tokens=padded,
            tokens_mask=padded_mask,
            input_pos=pos[:, -1:] + i + 1,
            temperature=temperature,
            topk=topk,
            cfg_scale=cfg_scale,
        )
        mx.eval(curr_token)

        if (i + 1) % 25 == 0:
            mx.clear_cache()
            gc.collect()

        if mx.any(curr_token[0] >= audio_eos_id):
            if ignore_eos:
                curr_token = mx.clip(curr_token, 0, audio_eos_id - 1)
            else:
                break

        frames.append(curr_token[0:1])

    progress(0.95, desc="Decoding audio...")
    frames_arr = mx.concatenate(frames, axis=0)[None, :, :]
    mx.eval(frames_arr)

    audio = CODEC.detokenize(frames_arr, duration=len(frames) / frame_rate)
    mx.eval(audio)
    audio_np = np.array(audio.astype(mx.float32)).flatten()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio_np, sample_rate)
        output_path = f.name

    MODEL.reset_caches()
    mx.clear_cache()
    gc.collect()

    progress(1.0, desc="Done")

    actual_duration = len(audio_np) / sample_rate
    total_time = time.time() - start_time
    return output_path, f"✓ {len(frames)} frames ({actual_duration:.1f}s) in {total_time:.1f}s"


def create_ui():
    """Create Suno-style UI."""

    with gr.Blocks(title="HeartMuLa", css=SUNO_CSS) as app:

        with gr.Row(elem_id="main-row"):

            # === LEFT SIDEBAR ===
            with gr.Column(elem_id="sidebar", scale=0, min_width=200):
                gr.HTML('<div id="sidebar-logo">HEART</div>')

                gr.HTML('''
                <div class="nav-item active">
                    <svg class="nav-icon" viewBox="0 0 24 24" fill="currentColor"><path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/></svg>
                    Create
                </div>
                <div class="nav-item">
                    <svg class="nav-icon" viewBox="0 0 24 24" fill="currentColor"><path d="M12 3L2 12h3v8h6v-6h2v6h6v-8h3L12 3z"/></svg>
                    Home
                </div>
                <div class="nav-item">
                    <svg class="nav-icon" viewBox="0 0 24 24" fill="currentColor"><path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z"/></svg>
                    Library
                </div>
                ''')

            # === MAIN CONTENT ===
            with gr.Column(elem_id="content-area", scale=1):

                # Lyrics Section
                with gr.Accordion("Lyrics", open=True, elem_classes=["accordion"]):
                    lyrics_input = gr.Textbox(
                        label="",
                        placeholder="Write some lyrics or a prompt — or leave blank for instrumental",
                        lines=7,
                        elem_id="lyrics-box",
                        show_label=False,
                    )

                # Styles Section
                with gr.Accordion("Styles", open=True, elem_classes=["accordion"]):
                    styles_input = gr.Textbox(
                        label="",
                        placeholder="femenina, augment, cori, émotion, high tension",
                        value="electronic, ambient, instrumental",
                        elem_id="styles-box",
                        show_label=False,
                    )

                    # Tag pills
                    with gr.Row(elem_id="tag-row"):
                        for tag in ["pop", "rock", "jazz", "piano", "acoustic"]:
                            btn = gr.Button(tag, size="sm", scale=0)
                            btn.click(
                                fn=lambda t, tag=tag: add_tag(t, tag),
                                inputs=[styles_input],
                                outputs=[styles_input],
                            )
                        for tag in ["female vocal", "male vocal", "instrumental"]:
                            btn = gr.Button(tag, size="sm", scale=0)
                            btn.click(
                                fn=lambda t, tag=tag: add_tag(t, tag),
                                inputs=[styles_input],
                                outputs=[styles_input],
                            )

                # Advanced Options
                with gr.Accordion("Advanced Options", open=False, elem_classes=["accordion"]):

                    with gr.Row(elem_classes=["slider-row"]):
                        duration_slider = gr.Slider(
                            minimum=5, maximum=120, value=30, step=5,
                            label="Duration (seconds)",
                        )

                    with gr.Row(elem_classes=["slider-row"]):
                        cfg_slider = gr.Slider(
                            minimum=1.0, maximum=4.0, value=1.5, step=0.1,
                            label="Style Influence",
                        )

                    with gr.Row(elem_classes=["slider-row"]):
                        temp_slider = gr.Slider(
                            minimum=0.5, maximum=1.5, value=1.0, step=0.05,
                            label="Creativity",
                        )

                    with gr.Row(elem_classes=["slider-row"]):
                        topk_slider = gr.Slider(
                            minimum=10, maximum=100, value=50, step=10,
                            label="Top-K",
                        )

                    ignore_eos = gr.Checkbox(
                        label="Force full duration",
                        value=False,
                    )

                # Create Button
                create_btn = gr.Button("Create", elem_id="create-btn")

                # Output Section
                with gr.Group(elem_id="output-section"):
                    audio_output = gr.Audio(
                        label="",
                        type="filepath",
                        elem_id="audio-player",
                        show_label=False,
                    )
                    status_output = gr.Textbox(
                        label="",
                        value="Ready",
                        interactive=False,
                        elem_id="status-text",
                        show_label=False,
                    )

            # === RIGHT PANEL (Library placeholder) ===
            with gr.Column(elem_id="library-panel", scale=0, min_width=300, visible=False):
                gr.HTML('<div id="library-header"><span id="library-title">My Workspace</span></div>')

        # Connect generate
        create_btn.click(
            fn=generate_music,
            inputs=[
                lyrics_input,
                styles_input,
                duration_slider,
                cfg_slider,
                temp_slider,
                topk_slider,
                ignore_eos,
            ],
            outputs=[audio_output, status_output],
        )

    return app


if __name__ == "__main__":
    print("🎵 HeartMuLa")
    print("Loading models...")

    try:
        load_models()
        print("✓ Models ready")
    except Exception as e:
        print(f"⚠ {e}")

    app = create_ui()
    print("\n→ http://localhost:7860\n")

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
