#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fastapi>=0.109.0",
#     "uvicorn>=0.27.0",
#     "mlx>=0.22.0",
#     "numpy>=1.24.0",
#     "safetensors>=0.4.0",
#     "huggingface-hub>=0.20",
#     "tokenizers>=0.15",
#     "soundfile>=0.12",
#     "psutil>=5.9.0",
#     "mflux>=0.15.0",
# ]
#
# [tool.uv]
# prerelease = "allow"
# ///
"""
FastAPI server for HeartMuLa MLX music generation.

Usage:
    uv run server.py
"""

import asyncio
import gc
import json
import sys
import time
import uuid
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add src to path
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

app = FastAPI(title="HeartMuLa MLX API")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
model = None
codec = None
tokenizer = None
generation_state = {
    "progress": 0.0,
    "message": "Ready",
    "is_generating": False,
    "cancelled": False,  # Flag to cancel generation
    "current_request": None,  # Store current generation params
    "started_at": None,
}

# Album art generation state
art_model = None
art_thread = None
art_generation_state = {
    "is_generating": False,
    "current_song": None,
    "pending_count": 0,
    "paused": False,  # Pause when music is generating
}

# Surprise me lyrics templates
SURPRISE_LYRICS = [
    {
        "title": "Digital Dreams",
        "styles": "electronic, synthwave, dreamy",
        "lyrics": """[verse]
Neon lights are calling me tonight
Through the digital rain I find my way
Binary stars align in perfect sight
In this virtual world I want to stay

[chorus]
Digital dreams, electric streams
Nothing is ever what it seems
Lost in the code, finding my soul
In this machine I'm finally whole"""
    },
    {
        "title": "Midnight Rain",
        "styles": "indie, acoustic, melancholic",
        "lyrics": """[verse]
Watching raindrops race down my window pane
Every drop a memory of you
The streetlights flicker like a fading flame
Painting shadows in shades of blue

[chorus]
Midnight rain keeps falling down
On this quiet sleeping town
And I wonder where you are tonight
Under the same moon, the same starlight"""
    },
    {
        "title": "Rise Up",
        "styles": "pop, upbeat, empowering",
        "lyrics": """[verse]
They told me I would never make it through
Said my dreams were way too high to reach
But I've got fire burning in my soul
And there's still so much this heart can teach

[chorus]
Rise up, stand tall
We were born to have it all
Break the chains, touch the sky
Spread your wings and learn to fly"""
    },
    {
        "title": "Ocean Heart",
        "styles": "ambient, chill, atmospheric",
        "lyrics": """[verse]
Waves are crashing on forgotten shores
Carrying secrets from the deep
The tide pulls back what time ignores
Promises the ocean couldn't keep

[bridge]
Salt and sand and endless blue
A horizon line that fades from view

[chorus]
My ocean heart keeps drifting far
Chasing the light of a distant star"""
    },
    {
        "title": "City Lights",
        "styles": "jazz, soul, smooth",
        "lyrics": """[verse]
Walking through these crowded streets at night
A million stories passing by
Every window holds a different light
Every corner hears a different cry

[chorus]
City lights, they shine so bright
Hiding all our fears from sight
In the noise I find my peace
In the chaos, sweet release"""
    },
    {
        "title": "Starfire",
        "styles": "rock, epic, powerful",
        "lyrics": """[verse]
From the ashes we will rise again
Burning brighter than before
Every ending brings a new begin
We are what we're fighting for

[chorus]
Starfire in my veins tonight
We're the spark that starts the light
Can't contain this wild flame
Nothing's ever gonna be the same"""
    },
    {
        "title": "Velvet Sky",
        "styles": "r&b, smooth, romantic",
        "lyrics": """[verse]
Underneath the velvet sky we lay
Counting stars until the dawn
Your whisper takes my breath away
In your arms is where I belong

[chorus]
Velvet sky above us two
Every star reminds me of you
In this moment, time stands still
And I know it always will"""
    },
    {
        "title": "Ghost Town",
        "styles": "alternative, dark, moody",
        "lyrics": """[verse]
Empty streets and hollow sounds
Echoes of what used to be
Walking through these haunted grounds
Searching for a memory

[chorus]
Welcome to this ghost town heart
Where everything falls apart
The silence screams your name
Nothing here will be the same"""
    },
    {
        "title": "Golden Hour",
        "styles": "folk, warm, nostalgic",
        "lyrics": """[verse]
Sunlight dripping through the trees
Honey colored afternoon
Dancing shadows in the breeze
Summer ending way too soon

[chorus]
Golden hour, fading light
Hold me close before the night
In this moment, pure and true
All I ever need is you"""
    },
    {
        "title": "Neon Heart",
        "styles": "electropop, dance, energetic",
        "lyrics": """[verse]
Flashing lights and pounding bass
Lost inside the sound
Every beat picks up the pace
Feet don't touch the ground

[chorus]
Neon heart is beating fast
Make this moment last
Dancing through the night
Everything feels right"""
    },
]

# Random art prompts for songs without lyrics
RANDOM_ART_PROMPTS = [
    "Abstract geometric shapes with vibrant colors, album cover art",
    "Cosmic nebula and stars, space aesthetic album artwork",
    "Neon city lights reflected on wet streets, cyberpunk album cover",
    "Flowing liquid metal and chrome, futuristic album art",
    "Surreal floating islands in clouds, dreamlike album cover",
    "Digital glitch art with bright colors, electronic music aesthetic",
    "Ocean waves at sunset, ambient music album artwork",
    "Dense forest with rays of light, nature album cover",
    "Vintage vinyl records and retro equipment, classic album art",
    "Crystal formations and prismatic light, ethereal album cover",
    "Mountain peaks above clouds at dawn, epic album artwork",
    "Abstract paint splashes and brush strokes, artistic album cover",
]


def load_art_model():
    """Load the MFLUX image generation model."""
    global art_model
    if art_model is not None:
        return art_model

    # Try Z-Image Turbo first (fastest, best quality)
    try:
        from mflux.models.z_image import ZImageTurbo
        print("Loading Z-Image Turbo model for album art...")
        art_model = ZImageTurbo(quantize=8)
        print("Album art model loaded (Z-Image Turbo)!")
        return art_model
    except ImportError:
        pass
    except Exception as e:
        print(f"Z-Image Turbo failed: {e}")

    # Fallback to Flux1 Schnell (smaller, faster)
    try:
        from mflux import Flux1
        print("Loading Flux1 Schnell model for album art...")
        art_model = Flux1(model_alias="schnell", quantize=8)
        print("Album art model loaded (Flux1 Schnell)!")
        return art_model
    except ImportError:
        pass
    except Exception as e:
        print(f"Flux1 Schnell failed: {e}")

    print("No album art model available - mflux not properly installed")
    return None


def generate_album_art(filename: str, styles: str, lyrics: str, title: str):
    """Generate album art for a song."""
    global art_generation_state

    model = load_art_model()
    if model is None:
        print(f"Skipping album art for {filename} - model not available")
        return False

    art_generation_state["is_generating"] = True
    art_generation_state["current_song"] = filename

    try:
        # Build prompt based on available metadata
        if lyrics and lyrics.strip():
            # Use song info for prompt
            style_part = styles if styles else "music"
            lyrics_part = lyrics[:100]
            prompt = f"Album cover art, {style_part} aesthetic, mood: {lyrics_part}, artistic, professional album artwork"
        else:
            # No lyrics - use random artistic prompt
            import random
            seed = hash(filename) % len(RANDOM_ART_PROMPTS)
            prompt = RANDOM_ART_PROMPTS[seed]
            if styles:
                prompt = f"{styles} style, {prompt}"

        if title:
            prompt = f"'{title}' - {prompt}"

        print(f"Generating album art for {filename}: {prompt[:80]}...")

        # Generate image
        image = model.generate_image(
            prompt=prompt,
            seed=hash(filename) % (2**32),
            num_inference_steps=9,
            width=512,
            height=512,
        )

        # Save image
        art_path = OUTPUT_DIR / f"{Path(filename).stem}.png"
        image.save(str(art_path))
        print(f"Album art saved: {art_path}")

        # Update metadata
        meta_path = OUTPUT_DIR / f"{Path(filename).stem}.json"
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
            metadata["album_art"] = f"{Path(filename).stem}.png"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

        return True

    except Exception as e:
        print(f"Album art generation failed for {filename}: {e}")
        return False
    finally:
        art_generation_state["is_generating"] = False
        art_generation_state["current_song"] = None


def find_songs_without_art():
    """Find all songs that don't have album art."""
    missing = []
    for wav_path in OUTPUT_DIR.glob("*.wav"):
        art_path = OUTPUT_DIR / f"{wav_path.stem}.png"
        if not art_path.exists():
            missing.append(wav_path.name)
    return missing


def art_scanner_thread():
    """Background thread that scans for songs without album art."""
    global art_generation_state
    print("Album art scanner started")

    while True:
        try:
            # Wait while music is generating
            while art_generation_state["paused"] or generation_state["is_generating"]:
                time.sleep(2)

            # Find songs without art
            missing = find_songs_without_art()
            art_generation_state["pending_count"] = len(missing)

            if not missing:
                # Nothing to do, sleep and check again
                time.sleep(5)
                continue

            # Process one song at a time
            filename = missing[0]
            print(f"Found {len(missing)} songs without album art, processing {filename}...")

            # Load metadata to get song info
            meta_path = OUTPUT_DIR / f"{Path(filename).stem}.json"
            styles, lyrics, title = "", "", ""
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        metadata = json.load(f)
                    styles = metadata.get("styles", "")
                    lyrics = metadata.get("lyrics", "")
                    title = metadata.get("title", "")
                except:
                    pass

            # Generate art (will skip if paused during generation)
            if not art_generation_state["paused"]:
                generate_album_art(filename, styles, lyrics, title)

            # Small delay before next
            time.sleep(1)

        except Exception as e:
            print(f"Art scanner error: {e}")
            time.sleep(10)


def start_art_scanner():
    """Start the background album art scanner thread."""
    global art_thread
    if art_thread is None or not art_thread.is_alive():
        art_thread = threading.Thread(target=art_scanner_thread, daemon=True)
        art_thread.start()


def pause_art_generation():
    """Pause art generation when music generation starts."""
    global art_generation_state
    art_generation_state["paused"] = True


def resume_art_generation():
    """Resume art generation after music generation completes."""
    global art_generation_state
    art_generation_state["paused"] = False

# Output directory
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


class GenerateRequest(BaseModel):
    lyrics: str = ""
    styles: str = "electronic, ambient, instrumental"
    duration: int = 10
    cfg_scale: float = 1.5
    temperature: float = 1.0
    topk: int = 50
    title: str = ""  # Optional title


class GenerateResponse(BaseModel):
    filename: str
    frames: int
    duration: float
    time: float


class ProgressResponse(BaseModel):
    progress: float
    message: str
    is_generating: bool
    current_request: dict | None = None
    started_at: float | None = None


class SongInfo(BaseModel):
    filename: str
    title: str
    styles: str
    lyrics: str
    duration: float
    created_at: float
    album_art: str | None = None


class SongsResponse(BaseModel):
    songs: list[SongInfo]


def load_models():
    """Load models on startup."""
    global model, codec, tokenizer

    if model is not None:
        return

    import mlx.core as mx
    from tokenizers import Tokenizer
    from heartlib_mlx.heartmula import HeartMuLa
    from heartlib_mlx.heartcodec import HeartCodec

    # Find checkpoint
    ckpt_path = None
    candidates = [
        Path(__file__).parent / "ckpt-mlx",
        Path.home() / "Developer/heartlib-mlx/ckpt-mlx",
        Path("./ckpt-mlx"),
    ]
    for c in candidates:
        if c.exists() and (c / "heartmula").exists():
            ckpt_path = str(c)
            break

    if ckpt_path is None:
        raise RuntimeError("Checkpoint not found")

    print(f"Loading models from {ckpt_path}...")
    model = HeartMuLa.from_pretrained(f"{ckpt_path}/heartmula")
    codec = HeartCodec.from_pretrained(f"{ckpt_path}/heartcodec")

    # Use bfloat16 for memory efficiency
    print("Converting to bfloat16...")
    model.set_dtype(mx.bfloat16)
    codec.set_dtype(mx.bfloat16)

    # Load tokenizer
    tokenizer_path = Path(ckpt_path).parent / "ckpt" / "tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer_path = Path.home() / "Developer/heartlib/ckpt/tokenizer.json"
    if not tokenizer_path.exists():
        raise RuntimeError(f"Tokenizer not found at {tokenizer_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print("Models loaded!")


def save_song_metadata(filename: str, title: str, styles: str, lyrics: str, duration: float):
    """Save metadata JSON alongside the audio file."""
    meta_path = OUTPUT_DIR / f"{Path(filename).stem}.json"
    metadata = {
        "filename": filename,
        "title": title or filename,
        "styles": styles,
        "lyrics": lyrics,
        "duration": duration,
        "created_at": time.time(),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_song_metadata(wav_path: Path) -> SongInfo | None:
    """Load metadata for a WAV file, or create basic info from file."""
    meta_path = wav_path.with_suffix(".json")

    if meta_path.exists():
        try:
            with open(meta_path) as f:
                data = json.load(f)
                return SongInfo(**data)
        except Exception:
            pass

    # Fallback: create basic info from file
    try:
        import soundfile as sf
        info = sf.info(str(wav_path))
        duration = info.duration
    except Exception:
        duration = 0.0

    # Check if album art exists
    art_path = wav_path.with_suffix(".png")
    album_art = art_path.name if art_path.exists() else None

    return SongInfo(
        filename=wav_path.name,
        title=wav_path.stem,
        styles="",
        lyrics="",
        duration=duration,
        created_at=wav_path.stat().st_mtime,
        album_art=album_art,
    )


def generate_music_sync(
    lyrics: str,
    styles: str,
    duration: int,
    cfg_scale: float,
    temperature: float,
    topk: int,
    title: str,
) -> tuple[str, int, float, float]:
    """Synchronous music generation."""
    global model, codec, tokenizer, generation_state

    # Pause album art generation while music is generating
    pause_art_generation()

    # Reset cancelled flag
    generation_state["cancelled"] = False

    import mlx.core as mx
    import numpy as np
    import soundfile as sf

    # Config
    text_bos_id = 128000
    text_eos_id = 128001
    audio_eos_id = 8193
    num_codebooks = 8
    parallel = num_codebooks + 1
    frame_rate = 12.5
    sample_rate = 48000

    generation_state["progress"] = 0.05
    generation_state["message"] = "Tokenizing..."

    # Tokenize
    tags_text = f"<tag>{styles}</tag>"
    tags_ids = tokenizer.encode(tags_text.lower()).ids
    if tags_ids[0] != text_bos_id:
        tags_ids = [text_bos_id] + tags_ids
    if tags_ids[-1] != text_eos_id:
        tags_ids = tags_ids + [text_eos_id]

    if lyrics:
        lyrics_ids = tokenizer.encode(lyrics.lower()).ids
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

    # Build prompt
    prompt_tokens = np.zeros((prompt_len, parallel), dtype=np.int64)
    prompt_tokens[:len(tags_ids), -1] = tags_ids
    if lyrics_ids:
        prompt_tokens[len(tags_ids) + 1:, -1] = lyrics_ids

    prompt_mask = np.zeros((prompt_len, parallel), dtype=np.float32)
    prompt_mask[:, -1] = 1.0

    # Setup for CFG (batch=2)
    tokens = mx.array(prompt_tokens)[None, :, :]
    tokens = mx.concatenate([tokens, tokens], axis=0)
    mask = mx.array(prompt_mask)[None, :, :]
    mask = mx.concatenate([mask, mask], axis=0)

    muq_embed = mx.zeros((2, model.config.muq_dim))
    pos = mx.broadcast_to(mx.arange(prompt_len)[None, :], (2, prompt_len))

    max_frames = int(duration * frame_rate)
    model.setup_caches(2)

    generation_state["progress"] = 0.1
    generation_state["message"] = f"Generating audio (0/{max_frames} frames)..."

    # Generate
    start_time = time.time()
    frames = []

    curr_token = model.generate_frame(
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

    for i in range(max_frames - 1):
        padded = mx.concatenate([
            curr_token[:, None, :],
            mx.zeros((2, 1, 1), dtype=mx.int32)
        ], axis=-1)
        padded_mask = mx.concatenate([
            mx.ones((2, 1, num_codebooks)),
            mx.zeros((2, 1, 1))
        ], axis=-1)

        curr_token = model.generate_frame(
            tokens=padded,
            tokens_mask=padded_mask,
            input_pos=pos[:, -1:] + i + 1,
            temperature=temperature,
            topk=topk,
            cfg_scale=cfg_scale,
        )
        mx.eval(curr_token)

        # Clear cache periodically
        if (i + 1) % 25 == 0:
            mx.clear_cache()
            gc.collect()

        # Update progress
        progress = 0.1 + 0.7 * (i + 1) / max_frames
        elapsed = time.time() - start_time
        fps = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (max_frames - i - 1) / fps if fps > 0 else 0

        generation_state["progress"] = progress
        generation_state["message"] = f"Generating ({i + 2}/{max_frames} frames, {fps:.1f} f/s, ETA {eta:.0f}s)"

        # Check for cancellation
        if generation_state["cancelled"]:
            generation_state["message"] = "Cancelled"
            resume_art_generation()
            raise Exception("Generation cancelled by user")

        if mx.any(curr_token[0] >= audio_eos_id):
            generation_state["message"] = f"Audio EOS at frame {i + 2}"
            break
        frames.append(curr_token[0:1])

    generation_state["progress"] = 0.85
    generation_state["message"] = "Decoding audio..."

    # Decode
    frames_arr = mx.concatenate(frames, axis=0)[None, :, :]
    mx.eval(frames_arr)

    audio = codec.detokenize(frames_arr, duration=len(frames) / frame_rate)
    mx.eval(audio)
    audio_np = np.array(audio.astype(mx.float32)).flatten()

    generation_state["progress"] = 0.95
    generation_state["message"] = "Saving audio..."

    # Save audio
    filename = f"generation_{uuid.uuid4().hex[:8]}.wav"
    output_path = OUTPUT_DIR / filename
    sf.write(str(output_path), audio_np, sample_rate)

    actual_duration = len(audio_np) / sample_rate

    # Save metadata
    save_song_metadata(filename, title, styles, lyrics, actual_duration)

    # Art scanner will automatically pick up this song

    elapsed = time.time() - start_time

    generation_state["progress"] = 1.0
    generation_state["message"] = "Complete!"

    # Resume album art generation now that music is done
    resume_art_generation()

    return filename, len(frames), actual_duration, elapsed


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    load_models()
    start_art_scanner()


@app.get("/favicon.ico")
async def get_favicon():
    """Serve favicon."""
    favicon_path = Path(__file__).parent / "web" / "favicon.png"
    if not favicon_path.exists():
        raise HTTPException(status_code=404, detail="Favicon not found")
    return FileResponse(favicon_path, media_type="image/png")


@app.get("/surprise-lyrics")
async def get_surprise_lyrics():
    """Get random lyrics for 'Surprise Me' feature."""
    import random
    song = random.choice(SURPRISE_LYRICS)
    return {
        "title": song["title"],
        "styles": song["styles"],
        "lyrics": song["lyrics"].strip(),
    }


@app.get("/status")
async def get_status():
    """Check API status."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "is_generating": generation_state["is_generating"],
    }


@app.get("/progress")
async def get_progress() -> ProgressResponse:
    """Get generation progress."""
    return ProgressResponse(
        progress=generation_state["progress"],
        message=generation_state["message"],
        is_generating=generation_state["is_generating"],
        current_request=generation_state["current_request"],
        started_at=generation_state["started_at"],
    )


@app.post("/cancel")
async def cancel_generation():
    """Cancel ongoing music generation."""
    global generation_state
    if not generation_state["is_generating"]:
        return {"status": "no_generation", "message": "No generation in progress"}

    generation_state["cancelled"] = True
    generation_state["message"] = "Cancelling..."
    return {"status": "cancelling", "message": "Generation will be cancelled"}


@app.get("/songs")
async def get_songs() -> SongsResponse:
    """List all songs in the outputs folder."""
    songs = []
    for wav_path in sorted(OUTPUT_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True):
        song_info = load_song_metadata(wav_path)
        if song_info:
            songs.append(song_info)
    return SongsResponse(songs=songs)


@app.post("/generate")
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Generate music."""
    global generation_state

    if generation_state["is_generating"]:
        raise HTTPException(status_code=409, detail="Generation already in progress")

    generation_state["is_generating"] = True
    generation_state["progress"] = 0.0
    generation_state["message"] = "Starting..."
    generation_state["current_request"] = {
        "title": request.title,
        "styles": request.styles,
        "lyrics": request.lyrics[:100] + "..." if len(request.lyrics) > 100 else request.lyrics,
        "duration": request.duration,
    }
    generation_state["started_at"] = time.time()

    try:
        # Run generation in thread pool to not block
        loop = asyncio.get_event_loop()
        filename, frames, duration, elapsed = await loop.run_in_executor(
            None,
            generate_music_sync,
            request.lyrics,
            request.styles,
            request.duration,
            request.cfg_scale,
            request.temperature,
            request.topk,
            request.title,
        )

        return GenerateResponse(
            filename=filename,
            frames=frames,
            duration=duration,
            time=elapsed,
        )
    finally:
        generation_state["is_generating"] = False
        generation_state["current_request"] = None
        generation_state["started_at"] = None


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve generated audio file."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path, media_type="audio/wav")


@app.get("/art/{filename}")
async def get_art(filename: str):
    """Serve album art image."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Album art not found")
    return FileResponse(file_path, media_type="image/png")


@app.get("/art-status")
async def get_art_status():
    """Get album art generation status."""
    return {
        "is_generating": art_generation_state["is_generating"],
        "current_song": art_generation_state["current_song"],
        "pending_count": art_generation_state["pending_count"],
    }


# Serve static files from web/ directory
web_dir = Path(__file__).parent / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
