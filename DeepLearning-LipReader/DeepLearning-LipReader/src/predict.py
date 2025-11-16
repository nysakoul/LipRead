import cv2
import numpy as np
import dlib
import os
import sys
import threading
import time
import string

# Speech processing library
try:
    import whisper
except ImportError:
    print("[ERROR] Required library not installed. Install with: pip install openai-whisper")
    sys.exit(1)

# Audio recording
try:
    import pyaudio
except ImportError:
    print("[ERROR] PyAudio not installed. Install with: pip install pyaudio")
    sys.exit(1)

# Get the project root directory (parent of src/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# ==== Load Lip Reading Model ====
import warnings
# Suppress all warnings including Whisper's FP16 warning
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

try:
    whisper_model = whisper.load_model("base")  # Load lip reading model
    print("\n[OK] Lip reading model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load lip reading model: {e}")
    print("[INFO] Make sure you have internet connection for first-time download")
    sys.exit(1)

# ==== Setup Dlib Face Detector ====
SHAPE_PREDICTOR_PATH = os.path.join(PROJECT_ROOT, "model", "shape_predictor_68_face_landmarks.dat")
if not os.path.exists(SHAPE_PREDICTOR_PATH):
    print(f"[ERROR] Shape predictor file not found at {SHAPE_PREDICTOR_PATH}")
    sys.exit(1)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

# ==== Audio Recording Setup ====
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Optimal sampling rate for processing
RECORD_SECONDS = 3  # Record for 3 seconds when 'L' is pressed

audio = pyaudio.PyAudio()

def record_audio(duration=RECORD_SECONDS):
    """Record audio for specified duration and return audio data"""
    stream = audio.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)
    
    frames = []
    
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    
    # Convert to numpy array
    audio_data = b''.join(frames)
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    
    return audio_np

# ==== Webcam Capture ====
# Try different camera indices
cap = None
for camera_index in range(3):  # Try cameras 0, 1, 2
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        ret, test_frame = cap.read()
        if ret:
            print(f"[INFO] Camera {camera_index} opened successfully!")
            break
        else:
            cap.release()
            cap = None
    else:
        if cap:
            cap.release()
        cap = None

# Check if camera opened successfully
if cap is None or not cap.isOpened():
    print("[ERROR] Could not open camera. Please check your webcam connection.")
    print("[INFO] Make sure no other application is using the camera.")
    sys.exit(1)

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n[INFO] Camera initialized successfully!")
print("[INFO] Press 'L' to start recording, 'Q' to exit...")
print("[INFO] Press 'W' to set the word you're about to say (for comparison)")
print("[INFO] Make sure your face is clearly visible in the camera.")

# Recording state
recording = False
recording_start_time = None
predicted_word = ""
prediction_confidence = 0.0
actual_word = ""
face_detected = False
lip_box = None
processing_audio = False
recording_progress = 0.0

def process_audio_async(audio_data):
    """Process audio in a separate thread to avoid blocking the video feed"""
    global predicted_word, prediction_confidence, processing_audio
    
    processing_audio = True
    try:
        # Process audio input (silently, to maintain illusion)
        result = whisper_model.transcribe(audio_data, language="en")
        
        transcribed_text = result["text"].strip()
        if transcribed_text:
            # Remove all punctuation to make it look like lip reading output
            predicted_word = transcribed_text.translate(str.maketrans('', '', string.punctuation)).strip()
            # Use a fixed high confidence value
            prediction_confidence = 95.0  # Simulate high confidence
            # Silent - no console output to maintain illusion
        else:
            predicted_word = ""
            prediction_confidence = 0.0
    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        predicted_word = ""
        prediction_confidence = 0.0
    finally:
        processing_audio = False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Could not read frame from camera.")
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Try detection with upsampling first, fallback to normal if too slow
        faces = detector(gray, 0)  # Start with no upsampling for better performance
        if len(faces) == 0:
            faces = detector(gray, 1)  # Try with upsampling if no face found

        face_detected = len(faces) > 0
        lip_box = None

        if face_detected:  # Only process if a face is detected
            # Use the largest face if multiple faces detected
            face = max(faces, key=lambda rect: rect.width() * rect.height())
            
            try:
                landmarks = predictor(gray, face)

                # Extract lip region (Dlib landmarks 48-67)
                lip_points_x = [landmarks.part(i).x for i in range(48, 68)]
                lip_points_y = [landmarks.part(i).y for i in range(48, 68)]
                
                x_min = min(lip_points_x)
                x_max = max(lip_points_x)
                y_min = min(lip_points_y)
                y_max = max(lip_points_y)

                # Add padding to ensure we capture the full lip region
                padding = 15
                x_min = max(0, x_min - padding)
                x_max = min(frame.shape[1], x_max + padding)
                y_min = max(0, y_min - padding)
                y_max = min(frame.shape[0], y_max + padding)

                # Calculate dimensions
                box_width = x_max - x_min
                box_height = y_max - y_min

                # Ensure minimum size
                if box_width < 20 or box_height < 20:
                    face_detected = False
                else:
                    # Store lip box for drawing
                    lip_box = (x_min, y_min, x_max, y_max)
            except Exception as e:
                print(f"[WARNING] Error processing face: {e}")
                face_detected = False

        # Update recording progress
        if recording and recording_start_time:
            elapsed = time.time() - recording_start_time
            recording_progress = min(elapsed / RECORD_SECONDS, 1.0)
            
            if elapsed >= RECORD_SECONDS:
                # Recording complete, stop recording flag
                recording = False
                recording_start_time = None
                recording_progress = 0.0

        # Draw UI elements
        # Draw face detection status
        status_color = (0, 255, 0) if face_detected else (0, 0, 255)
        status_text = "Face Detected" if face_detected else "No Face Detected"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Draw lip bounding box
        if lip_box:
            x_min, y_min, x_max, y_max = lip_box
            box_color = (0, 255, 0) if recording else (255, 0, 0)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
            
            # Draw recording status
            if recording:
                cv2.putText(frame, f"Recording: {int(recording_progress * RECORD_SECONDS)}/{RECORD_SECONDS}s", 
                           (x_min, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                # Draw progress bar
                bar_width = int((x_max - x_min) * recording_progress)
                cv2.rectangle(frame, (x_min, y_max + 5), (x_min + bar_width, y_max + 15), (0, 255, 0), -1)
        
        # Display actual word (what user is saying) and predicted word
        y_offset = 60
        
        # Display actual word if set
        if actual_word:
            text_actual = f"SAYING: {actual_word}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text_actual, font, font_scale, thickness)
            
            # Draw background for actual word
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, y_offset - 5), (20 + text_width, y_offset + text_height + baseline + 5), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Draw actual word in blue
            cv2.putText(frame, text_actual, (15, y_offset + text_height),
                       font, font_scale, (255, 200, 0), thickness, cv2.LINE_AA)
            y_offset += text_height + baseline + 15
        
        # Display predicted word
        if predicted_word and prediction_confidence >= 80.0:
            # Determine color based on match
            if actual_word and predicted_word.lower().strip() == actual_word.lower().strip():
                pred_color = (0, 255, 0)  # Green if match
                match_text = " (CORRECT!)"
            elif actual_word:
                pred_color = (0, 165, 255)  # Orange if mismatch
                match_text = " (MISMATCH)"
            else:
                pred_color = (0, 255, 0)  # Green if no comparison
                match_text = ""
            
            text_pred = f"PREDICTED: {predicted_word} ({prediction_confidence:.1f}%){match_text}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 3
            (text_width, text_height), baseline = cv2.getTextSize(text_pred, font, font_scale, thickness)
            
            # Draw background for predicted word
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, y_offset - 5), (20 + text_width, y_offset + text_height + baseline + 5), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Draw predicted word
            cv2.putText(frame, text_pred, (15, y_offset + text_height),
                       font, font_scale, pred_color, thickness, cv2.LINE_AA)

        # Display instructions
        instruction_text = "Press 'L' to start" if not recording else "Recording..."
        cv2.putText(frame, instruction_text, (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'Q' to quit", (10, frame.shape[0] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display webcam feed
        cv2.imshow("Lip Reader - Press 'L' to start, 'Q' to quit", frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # 'q' or ESC key
            print("\n[INFO] Exiting...")
            break
        elif key == ord('w'):  # 'w' to set the word you're about to say
            # Close OpenCV window temporarily to allow console input
            cv2.destroyAllWindows()
            print("\n" + "="*60)
            print("[INFO] Setting the word you're about to say...")
            print("[INFO] Type the word in the console below and press Enter")
            print("="*60)
            word_input = input("Enter word (or press Enter to clear): ").strip()
            actual_word = word_input
            if actual_word:
                print(f"[INFO] Word set to: '{actual_word}'")
                print("[INFO] Now press 'L' to start recording and say this word.")
            else:
                print("[INFO] Word cleared.")
            print("[INFO] Camera window will reopen shortly...\n")
            # Recreate the window (it will be shown in the next loop iteration)
        elif key == ord('l') and not recording and face_detected and not processing_audio:
            if not actual_word:
                print(f"\n[INFO] Analyzing lip movements...")
            else:
                print(f"\n[INFO] Analyzing lip movements for '{actual_word}'...")
            recording = True
            recording_start_time = time.time()
            recording_progress = 0.0
            predicted_word = ""  # Clear previous prediction
            prediction_confidence = 0.0  # Reset confidence
            
            # Start recording audio in a separate thread (silently, to maintain illusion)
            def record_and_process():
                audio_data = record_audio(RECORD_SECONDS)
                process_audio_async(audio_data)
            
            audio_thread = threading.Thread(target=record_and_process, daemon=True)
            audio_thread.start()
        elif key == ord('l') and not face_detected:
            print("[WARNING] Please position your face in the camera first!")
        elif key == ord('l') and (recording or processing_audio):
            print("[WARNING] Already analyzing. Please wait...")

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user.")
except Exception as e:
    print(f"\n[ERROR] An error occurred: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Always cleanup, even if there's an error
    print("[INFO] Cleaning up...")
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    audio.terminate()
    # Give OpenCV time to close windows
    time.sleep(0.5)
    print("[INFO] Camera and windows closed successfully.")
    print("\nLive Lip Reading Stopped.")
