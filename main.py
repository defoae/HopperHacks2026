"""
Meeting Inclusion Analyzer - FastAPI backend.
Analyzes meeting audio: transcription, diarization, speaking time, interruptions, inclusion score.
"""
import os
import shutil
import ssl
import uuid
import warnings
from pathlib import Path

# Load .env from project root if present so HUGGINGFACE_TOKEN can be set there
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv is optional; if not installed the app will continue and rely on real env vars
    pass

# Optional: skip SSL verification for model downloads (corporate proxy / self-signed cert)
if os.environ.get("MEETING_ANALYZER_DISABLE_SSL_VERIFY", "").lower() in ("1", "true", "yes"):
    ssl._create_default_https_context = ssl._create_unverified_context

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from huggingface_hub import login
import requests

import whisper
from pyannote.audio import Pipeline
from pydub import AudioSegment

# Authenticate to Hugging Face at startup
_token = os.environ.get('HUGGINGFACE_TOKEN', '').strip().strip('"\'')
print(f"[INFO] HUGGINGFACE_TOKEN found: {bool(_token)}")
if _token:
    try:
        print(f"[DEBUG] Token value (first 20 chars): {_token[:20]}...")
        print(f"[DEBUG] Token length: {len(_token)}")
        
        # Test token validity
        print("[DEBUG] Testing connection to Hugging Face Hub...")
        headers = {"Authorization": f"Bearer {_token}"}
        resp = requests.head("https://huggingface.co/api/user", headers=headers, timeout=10)
        if resp.status_code == 200:
            print("[DEBUG] ✓ Token is valid")
        elif resp.status_code == 401:
            print("[ERROR] ✗ Token is invalid or expired")
        else:
            print(f"[WARNING] Hub returned status {resp.status_code}")
        
        # Authenticate globally
        print("[DEBUG] Calling login()...")
        login(token=_token)
        print("[DEBUG] ✓ login() successful")
        
    except Exception as e:
        print(f"[ERROR] Authentication failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("[ERROR] HUGGINGFACE_TOKEN environment variable not set!")

# Suppress pydub warning when ffprobe is missing; we check explicitly and raise a clear error
warnings.filterwarnings("ignore", message="Couldn't find ffprobe", category=RuntimeWarning, module="pydub")

app = FastAPI(title="Meeting Inclusion Analyzer")

UPLOADS_DIR = Path(__file__).resolve().parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a"}
WHISPER_MODEL = None
DIARIZATION_PIPELINE = None


def get_whisper_model():
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        WHISPER_MODEL = whisper.load_model("base")
    return WHISPER_MODEL


def get_diarization_pipeline():
    global DIARIZATION_PIPELINE
    if DIARIZATION_PIPELINE is None:
        token = os.environ.get('HUGGINGFACE_TOKEN', '').strip().strip('"\'')
        if not token:
            raise HTTPException(
                status_code=500,
                detail="HUGGINGFACE_TOKEN environment variable is not set. "
                       "Get a token at https://huggingface.co/settings/tokens and accept "
                       "pyannote/speaker-diarization-3.1 conditions.",
            )
        
        # Ensure HF_TOKEN is set in environment for all subprocesses
        os.environ['HF_TOKEN'] = token
        
        try:
            print(f"[DEBUG] Loading diarization pipeline (HF_TOKEN set in environment)...")
            DIARIZATION_PIPELINE = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
            )
            print("[DEBUG] ✓ Pipeline loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to load pipeline: {e}")
            print(f"[ERROR] To fix this:")
            print(f"[ERROR] 1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1")
            print(f"[ERROR] 2. Confirm you see 'You have been granted access to this model'")
            print(f"[ERROR] 3. Verify your token at https://huggingface.co/settings/tokens")
            print(f"[ERROR] 4. Try deleting ~/.cache/huggingface and retry")
            raise HTTPException(
                status_code=500,
                detail="Could not load pyannote model. Please verify: "
                "(1) You accepted the model at https://huggingface.co/pyannote/speaker-diarization-3.1, "
                "(2) Your token is valid, "
                "(3) You have internet connection. "
                f"Error: {str(e)}"
            )
    return DIARIZATION_PIPELINE


def _ffprobe_available() -> bool:
    """True if ffprobe (from ffmpeg) is on PATH, needed for MP3/M4A conversion."""
    return shutil.which("ffprobe") is not None


def ensure_wav(audio_path: Path) -> Path:
    """Convert to 16kHz mono WAV if needed (for consistent processing)."""
    suf = audio_path.suffix.lower()
    if suf == ".wav":
        return audio_path
    if not _ffprobe_available():
        raise HTTPException(
            status_code=500,
            detail="Processing MP3/M4A requires ffmpeg. Install it (e.g. brew install ffmpeg on Mac) or upload a WAV file.",
        )
    try:
        seg = AudioSegment.from_file(str(audio_path), format=suf[1:])
        seg = seg.set_channels(1)
        seg = seg.set_frame_rate(16000)
        wav_path = audio_path.with_suffix(".wav")
        seg.export(str(wav_path), format="wav")
        return wav_path
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Audio conversion failed: {e}. Install ffmpeg (e.g. brew install ffmpeg) or upload a WAV file.",
        )


def transcribe(audio_path: Path) -> list[dict]:
    """Return list of {start, end, text} from Whisper."""
    model = get_whisper_model()
    result = model.transcribe(str(audio_path), word_timestamps=False)
    segments = []
    for s in result.get("segments", []):
        segments.append({
            "start": s["start"],
            "end": s["end"],
            "text": (s.get("text") or "").strip(),
        })
    return segments


def diarize(audio_path: Path) -> list[dict]:
    """Return list of {start, end, speaker} with Speaker 1, Speaker 2, ..."""
    import librosa
    
    pipeline = get_diarization_pipeline()
    
    # Load audio as waveform dict instead of file path to avoid AudioDecoder import issue
    print(f"[DEBUG] Loading audio with librosa...")
    waveform, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)
    
    # Convert to tensor and reshape to (channels, time)
    import torch
    waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)  # Add channel dim
    
    audio_dict = {
        "waveform": waveform_tensor,
        "sample_rate": sample_rate
    }
    
    print(f"[DEBUG] Running diarization on loaded audio...")
    diar = pipeline(audio_dict)
    
    # Map SPEAKER_00 -> Speaker 1, etc.
    speaker_ids = {}
    turns = []
    for segment, _, label in diar.itertracks(yield_label=True):
        if label not in speaker_ids:
            speaker_ids[label] = f"Speaker {len(speaker_ids) + 1}"
        turns.append({
            "start": segment.start,
            "end": segment.end,
            "speaker": speaker_ids[label],
        })
    return sorted(turns, key=lambda x: x["start"])


def align_segments(whisper_segments: list[dict], diar_turns: list[dict]) -> list[dict]:
    """Combine diarization with transcript: each segment has start, end, speaker, text."""
    if not diar_turns:
        return []
    combined = []
    for turn in diar_turns:
        t_start, t_end, speaker = turn["start"], turn["end"], turn["speaker"]
        texts = []
        for ws in whisper_segments:
            # Overlap: whisper segment overlaps with this diarization turn
            w_start, w_end = ws["start"], ws["end"]
            overlap_start = max(w_start, t_start)
            overlap_end = min(w_end, t_end)
            if overlap_end > overlap_start and ws["text"]:
                texts.append(ws["text"])
        text = " ".join(texts).strip() if texts else ""
        combined.append({
            "start_time": round(t_start, 2),
            "end_time": round(t_end, 2),
            "speaker": speaker,
            "text": text,
        })
    return combined


def compute_speaking_time(segments: list[dict]) -> dict[str, float]:
    """Total speaking time in seconds per speaker."""
    total = {}
    for s in segments:
        sp = s["speaker"]
        dur = s["end_time"] - s["start_time"]
        total[sp] = round(total.get(sp, 0) + dur, 2)
    return total


def compute_interruptions(segments: list[dict]) -> list[dict]:
    """Interruption = B starts before A ends (overlap). Return list of {interrupter, target, count}."""
    counts = {}
    for i, seg_a in enumerate(segments):
        for seg_b in segments[i + 1 :]:
            # B starts before A ends => B interrupted A
            if seg_b["start_time"] < seg_a["end_time"] and seg_a["speaker"] != seg_b["speaker"]:
                key = (seg_b["speaker"], seg_a["speaker"])
                counts[key] = counts.get(key, 0) + 1
    return [
        {"interrupter": inter, "target": target, "count": c}
        for (inter, target), c in sorted(counts.items())
    ]


def compute_analytics(speaking_time: dict, interruptions: list[dict]) -> tuple[str, float]:
    """Returns (dominant_speaker, inclusion_score 0-100)."""
    if not speaking_time:
        return "", 0.0
    total_sec = sum(speaking_time.values())
    if total_sec <= 0:
        return max(speaking_time, key=speaking_time.get), 0.0
    n = len(speaking_time)
    fractions = [speaking_time[sp] / total_sec for sp in speaking_time]
    # Participation balance: 100 when perfectly even (each 1/n), 0 when one has all
    min_frac = min(fractions)
    participation_balance = 100 * n * min_frac
    total_interruptions = sum(x["count"] for x in interruptions)
    interruption_penalty = min(50, total_interruptions * 5)
    inclusion_score = max(0.0, min(100.0, round(participation_balance - interruption_penalty, 1)))
    dominant_speaker = max(speaking_time, key=speaking_time.get)
    return dominant_speaker, inclusion_score


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Upload audio, transcribe, diarize, compute speaking time & interruptions, return analytics."""
    suf = Path(file.filename or "").suffix.lower()
    if suf not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}")
    save_path = UPLOADS_DIR / f"{uuid.uuid4().hex}{suf}"
    try:
        contents = await file.read()
        save_path.write_bytes(contents)
        print(f"[DEBUG] File saved to {save_path}")
    except Exception as e:
        raise HTTPException(500, f"Failed to save file: {e}")

    try:
        print(f"[DEBUG] Converting audio to WAV...")
        wav_path = ensure_wav(save_path)
        print(f"[DEBUG] ✓ Audio converted to {wav_path}")
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Audio conversion failed: {e}")

    try:
        print(f"[DEBUG] Transcribing audio...")
        whisper_segments = transcribe(wav_path)
        print(f"[DEBUG] ✓ Transcription complete: {len(whisper_segments)} segments")
        
        print(f"[DEBUG] Diarizing speakers...")
        diar_turns = diarize(wav_path)
        print(f"[DEBUG] ✓ Diarization complete: {len(diar_turns)} turns")
        
        print(f"[DEBUG] Aligning segments...")
        segments = align_segments(whisper_segments, diar_turns)
        print(f"[DEBUG] ✓ Alignment complete: {len(segments)} aligned segments")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        save_path.unlink(missing_ok=True)
        if wav_path != save_path:
            wav_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Analysis failed: {e}")

    speaking_time = compute_speaking_time(segments)
    interruptions = compute_interruptions(segments)
    dominant_speaker, inclusion_score = compute_analytics(speaking_time, interruptions)

    # Clean up temp wav if we created one
    if wav_path != save_path:
        wav_path.unlink(missing_ok=True)

    return {
        "speaking_time": speaking_time,
        "interruptions": interruptions,
        "dominant_speaker": dominant_speaker,
        "inclusion_score": inclusion_score,
        "segments": segments,
    }


@app.get("/")
async def index():
    return FileResponse(Path(__file__).resolve().parent / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
