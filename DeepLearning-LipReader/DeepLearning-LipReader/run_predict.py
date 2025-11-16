"""
Simple script to run the Whisper-based lip reader prediction.
This script makes it easy to run the prediction without navigating directories.
"""
import os
import sys
import subprocess

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICT_SCRIPT = os.path.join(SCRIPT_DIR, "src", "predict.py")

if __name__ == "__main__":
    print("="*60)
    print("Starting Lip Reader")
    print("="*60)
    print("Press 'L' to record and transcribe, 'Q' to quit\n")
    
    # Change to the project directory and run the script
    os.chdir(SCRIPT_DIR)
    try:
        subprocess.run([sys.executable, PREDICT_SCRIPT], check=True)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Script exited with error: {e}")
        sys.exit(1)

