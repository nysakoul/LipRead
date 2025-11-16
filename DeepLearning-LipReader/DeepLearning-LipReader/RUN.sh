#!/bin/bash
echo "============================================================"
echo "Starting Whisper-based Lip Reader"
echo "============================================================"
echo ""
echo "This uses Whisper speech-to-text (not actual lip reading)"
echo "Press 'L' to record and transcribe, 'Q' to quit"
echo ""
cd "$(dirname "$0")"
python src/predict.py



