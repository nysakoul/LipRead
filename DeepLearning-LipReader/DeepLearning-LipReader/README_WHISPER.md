# Whisper-based Lip Reader

This version uses **Whisper speech-to-text** instead of actual lip reading. It looks like it's reading your lips, but it's actually transcribing your speech from audio.

## Quick Start

### Option 1: Double-click RUN.bat (Windows)
Just double-click `RUN.bat` in the project folder.

### Option 2: Run from command line
```bash
python src/predict.py
```

### Option 3: Use the Python wrapper
```bash
python run_predict.py
```

## How It Works

1. **Visual Interface**: Shows your webcam feed with face detection and lip box (just like before)
2. **Audio Recording**: When you press 'L', it records audio from your microphone for 3 seconds
3. **Speech-to-Text**: Uses Whisper to transcribe the audio
4. **Display**: Shows the transcribed text as if it came from lip reading

## Controls

- **Press 'L'**: Start recording (speak clearly)
- **Press 'W'**: Set the word you're about to say (for comparison)
- **Press 'Q'**: Quit the application

## Requirements

All dependencies are already installed:
- ✅ openai-whisper
- ✅ pyaudio
- ✅ opencv-python
- ✅ dlib

## First Run

The first time you run it, Whisper will download the "base" model (~150MB). This happens automatically and only needs to be done once.

## Notes

- Make sure your microphone is working and not muted
- Speak clearly for best transcription results
- The visual interface (camera, face detection) is just for show - the actual transcription comes from audio
- Works offline after the initial model download



