# Meeting Inclusion Analyzer

Analyze a meeting audio recording and measure inclusivity by tracking **speaking time** and **interruptions**. Built for HopperHacks 2026.

- **Backend:** Python FastAPI  
- **Speech-to-text:** local OpenAI Whisper  
- **Speaker diarization:** pyannote.audio  
- **Frontend:** plain HTML + minimal JavaScript  

---

## Quick start (one command)

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. (Optional) For MP3/M4A uploads, install ffmpeg. Skip if you only use WAV.
#    Mac: brew install ffmpeg
#    Ubuntu: sudo apt install ffmpeg

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your HuggingFace token (required for pyannote)
export HUGGINGFACE_TOKEN="hf_tokenhere

# 5. Run the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in your browser, upload an audio file (WAV, MP3, or M4A), and click **Analyze Meeting**.

---

## Where to put `HUGGINGFACE_TOKEN`

1. **Get a token:** [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)  
2. **Accept model terms:** Visit [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and accept the conditions.  
3. **Set the variable** (pick one):

   - **Shell (current session):**  
     `export HUGGINGFACE_TOKEN="hf_xxxxxxxx"`
   - **`.env` file in project root:**  
     `echo 'HUGGINGFACE_TOKEN=hf_xxxxxxxx' >> .env`  
     Then run with: `source .env && uvicorn main:app --reload --port 8000`
   - **Windows CMD:**  
     `set HUGGINGFACE_TOKEN=hf_xxxxxxxx`
   - **Windows PowerShell:**  
     `$env:HUGGINGFACE_TOKEN="hf_xxxxxxxx"`

---

## Example `curl` request

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@/path/to/your/meeting.wav"
```

Response (JSON):

```json
{
  "speaking_time": { "Speaker 1": 45.2, "Speaker 2": 120.1 },
  "interruptions": [
    { "interrupter": "Speaker 2", "target": "Speaker 1", "count": 3 }
  ],
  "dominant_speaker": "Speaker 2",
  "inclusion_score": 72.5,
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 5.2,
      "speaker": "Speaker 1",
      "text": "So today we'll discuss..."
    }
  ]
}
```

---

## Project layout

```
HopperHacks26/
├── main.py           # FastAPI app: /analyze + serves index.html
├── index.html        # Upload form + results
├── requirements.txt  # Python dependencies
├── uploads/          # Uploaded audio files (created at first run)
└── README.md
```

---

## Features

| Feature | Description |
|--------|-------------|
| **Upload** | POST `/analyze` accepts WAV, MP3, M4A and saves to `uploads/`. |
| **Transcription** | Local Whisper produces full transcript. |
| **Diarization** | pyannote.audio labels Speaker 1, 2, 3… and aligns with transcript. |
| **Speaking time** | Total seconds per speaker. |
| **Interruptions** | Overlapping speech: who interrupted whom and how many times. |
| **Inclusion score** | 0–100 from participation balance minus interruption penalty. |

---

## Tips

- **MP3/M4A:** Requires ffmpeg on your PATH (`brew install ffmpeg` on Mac). WAV uploads work without it.
- **First run** downloads Whisper and pyannote models; allow a few minutes.
- **SSL error** (`CERTIFICATE_VERIFY_FAILED` / self-signed certificate): Common on corporate networks or some Mac setups. Either install/trust your system certs, or run with SSL verification disabled for model downloads only:  
  `MEETING_ANALYZER_DISABLE_SSL_VERIFY=1 uvicorn main:app --reload --port 8000`  
- **CPU only:** Default `pip install torch` is usually CPU on Mac. For GPU, install the right [PyTorch build](https://pytorch.org/get-started/locally/) then `pip install -r requirements.txt`.  
- **Short meetings** (1–3 min) work best for a quick demo.
