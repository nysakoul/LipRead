import cv2
import numpy as np
import tensorflow as tf
import dlib
import os
import sys

# Get the project root directory (parent of src/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# ==== Load Trained Model ====
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "lip_reader_3dcnn.h5")
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model file not found at {MODEL_PATH}")
    print("Please train the model first by running train_model.py")
    sys.exit(1)
model = tf.keras.models.load_model(MODEL_PATH)
print(f"\n[OK] Loaded model from {MODEL_PATH}")

# ==== Load Word List ====
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
if os.path.exists(PROCESSED_DATA_DIR):
    words = sorted([d for d in os.listdir(PROCESSED_DATA_DIR) if os.path.isdir(os.path.join(PROCESSED_DATA_DIR, d))])
    if len(words) > 0:
        word_to_index = {word: i for i, word in enumerate(words)}
        index_to_word = {i: word for word, i in word_to_index.items()}
        print(f"[OK] Loaded {len(words)} words from processed_data: {', '.join(words)}")
    else:
        # Fallback: use model output shape to determine number of classes
        num_classes = model.output_shape[1]
        words = [f"word_{i}" for i in range(num_classes)]
        word_to_index = {word: i for i, word in enumerate(words)}
        index_to_word = {i: word for word, i in word_to_index.items()}
        print(f"[WARNING] No words found in processed_data. Using generic names based on model output shape ({num_classes} classes).")
else:
    # Fallback: use model output shape to determine number of classes
    num_classes = model.output_shape[1]
    words = [f"word_{i}" for i in range(num_classes)]
    word_to_index = {word: i for i, word in enumerate(words)}
    index_to_word = {i: word for word, i in word_to_index.items()}
    print(f"[WARNING] processed_data directory not found. Using generic word names based on model output shape ({num_classes} classes).")
    print(f"[INFO] The model was trained on {num_classes} words. For best results, collect your own data and retrain.")

# ==== Setup Dlib Face Detector ====
SHAPE_PREDICTOR_PATH = os.path.join(PROJECT_ROOT, "model", "shape_predictor_68_face_landmarks.dat")
if not os.path.exists(SHAPE_PREDICTOR_PATH):
    print(f"[ERROR] Shape predictor file not found at {SHAPE_PREDICTOR_PATH}")
    sys.exit(1)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

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

frames = []
FRAME_COUNT = 22
recording = False
predicted_word = ""
prediction_confidence = 0.0  # Store the confidence of the last prediction
actual_word = ""  # The word the user is actually saying
face_detected = False
lip_box = None
frames_without_face = 0  # Count consecutive frames without face

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

                    # Extract lip region
                    lip_region = frame[y_min:y_max, x_min:x_max]
                    
                    # Check if lip region is valid
                    if lip_region.size > 0:
                        # Resize to match model input (112x80)
                        lip_region = cv2.resize(lip_region, (112, 80))

                        # === Apply Preprocessing to Match Training ===
                        gray_lip = cv2.cvtColor(lip_region, cv2.COLOR_BGR2GRAY)

                        # Step 1: Gaussian Blurring (Reduce Noise)
                        blurred = cv2.GaussianBlur(gray_lip, (5, 5), 0)

                        # Step 2: Contrast Stretching (Enhance Visibility)
                        min_pixel = np.min(blurred)
                        max_pixel = np.max(blurred)
                        if max_pixel > min_pixel:  # Avoid division by zero
                            contrast_stretched = (blurred - min_pixel) / (max_pixel - min_pixel) * 255
                            contrast_stretched = contrast_stretched.astype(np.uint8)
                        else:
                            contrast_stretched = blurred

                        # Step 3: Bilateral Filtering (Smooth Noise, Keep Edges)
                        bilateral_filtered = cv2.bilateralFilter(contrast_stretched, 5, 75, 75)

                        # Step 4: Sharpening (Enhance Lip Edges)
                        sharpen_kernel = np.array([[-1, -1, -1], 
                                                   [-1,  9, -1], 
                                                   [-1, -1, -1]])
                        sharpened = cv2.filter2D(bilateral_filtered, -1, sharpen_kernel)

                        # Step 5: Final Gaussian Blurring (Prevent Over-Sharpening Artifacts)
                        final_processed = cv2.GaussianBlur(sharpened, (3, 3), 0)

                        # Normalize pixel values
                        normalized = final_processed / 255.0
                        
                        # Only append frames if recording
                        if recording:
                            frames.append(normalized)
                            
                            # If we've collected enough frames, make prediction
                            if len(frames) >= FRAME_COUNT:
                                try:
                                    # Construct input sequence: (22, 80, 112) -> (1, 22, 80, 112, 1)
                                    input_sequence = np.array(frames[:FRAME_COUNT], dtype=np.float32)
                                    
                                    # Verify shape before adding dimensions
                                    if input_sequence.shape != (FRAME_COUNT, 80, 112):
                                        print(f"[ERROR] Unexpected frame shape: {input_sequence.shape}, expected ({FRAME_COUNT}, 80, 112)")
                                        frames = []
                                        recording = False
                                    else:
                                        input_sequence = np.expand_dims(input_sequence, axis=0)  # Add batch dim: (1, 22, 80, 112)
                                        input_sequence = np.expand_dims(input_sequence, axis=-1)  # Add channel dim: (1, 22, 80, 112, 1)
                                        
                                        # Verify final shape
                                        expected_shape = (1, FRAME_COUNT, 80, 112, 1)
                                        if input_sequence.shape != expected_shape:
                                            print(f"[ERROR] Input shape mismatch: {input_sequence.shape}, expected {expected_shape}")
                                            frames = []
                                            recording = False
                                        else:
                                            # Shape is correct, proceed with prediction
                                            print(f"[INFO] Making prediction with shape: {input_sequence.shape}")
                                            prediction = model.predict(input_sequence, verbose=0)
                                            
                                            # Get all prediction probabilities
                                            pred_probs = prediction[0]
                                            predicted_index = np.argmax(pred_probs)
                                            predicted_word = index_to_word[predicted_index]
                                            confidence = pred_probs[predicted_index] * 100
                                            prediction_confidence = confidence  # Store confidence for display check
                                            
                                            # Show all predictions for debugging
                                            print(f"\n{'='*60}")
                                            print("[PREDICTION PROBABILITIES]")
                                            sorted_indices = np.argsort(pred_probs)[::-1]  # Sort descending
                                            for i, idx in enumerate(sorted_indices):
                                                word_name = index_to_word[idx]
                                                prob = pred_probs[idx] * 100
                                                marker = " <-- SELECTED" if idx == predicted_index else ""
                                                print(f"  {i+1}. {word_name}: {prob:.2f}%{marker}")
                                            
                                            print(f"\n{'='*60}")
                                            if actual_word:
                                                print(f"[ACTUAL] Word you said: '{actual_word}'")
                                            print(f"[PREDICTION] Predicted Word: '{predicted_word}'")
                                            print(f"[PREDICTION] Confidence: {confidence:.2f}%")
                                            
                                            # Check if prediction is always the same or model is biased
                                            if predicted_index == 0 and confidence > 80:
                                                print(f"\n[WARNING] Model is heavily biased towards '{predicted_word}'")
                                                print(f"[WARNING] This model was trained on the author's lips only.")
                                                print(f"[WARNING] It will NOT work accurately for your lips!")
                                                print(f"[WARNING]")
                                                print(f"[WARNING] SOLUTION: Train your own model with your data:")
                                                print(f"[WARNING]   1. Run: py src/collection.py (collect data for your words)")
                                                print(f"[WARNING]   2. Run: py src/preprocess.py (preprocess the data)")
                                                print(f"[WARNING]   3. Run: py src/train_model.py (train new model)")
                                            elif confidence < 50:
                                                print(f"\n[WARNING] Low confidence prediction ({confidence:.2f}%)")
                                                print(f"[WARNING] Model is uncertain - this suggests it wasn't trained on your data.")
                                            
                                            if actual_word:
                                                if predicted_word.lower() == actual_word.lower():
                                                    print(f"[RESULT] ✓ CORRECT MATCH!")
                                                else:
                                                    print(f"[RESULT] ✗ MISMATCH - Expected '{actual_word}', got '{predicted_word}'")
                                            print(f"{'='*60}\n")

                                            # Only keep prediction if confidence >= 80%
                                            if confidence < 80.0:
                                                print(f"[INFO] Confidence ({confidence:.2f}%) below 80% threshold - prediction not displayed")
                                                predicted_word = ""  # Clear prediction if below threshold
                                                prediction_confidence = 0.0
                                            
                                            # Reset frames for next prediction
                                            frames = []
                                            recording = False

                                except Exception as e:
                                    print(f"[ERROR] Prediction failed: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    frames = []
                                    recording = False
            except Exception as e:
                print(f"[WARNING] Error processing face: {e}")
                face_detected = False

        # Handle recording state when no face detected
        if not face_detected and recording:
            frames_without_face += 1
            # Only warn after 5 consecutive frames without face, then every 10 frames
            if frames_without_face == 5 or (frames_without_face > 5 and frames_without_face % 10 == 0):
                print("[WARNING] Face lost during recording. Please keep your face in frame.")
            if len(frames) > 0:
                frames = []  # Reset if we lose face during recording
        else:
            frames_without_face = 0  # Reset counter when face is detected

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
                progress = len(frames) / FRAME_COUNT
                cv2.putText(frame, f"Recording: {len(frames)}/{FRAME_COUNT}", (x_min, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                # Draw progress bar
                bar_width = int((x_max - x_min) * progress)
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
        
        # Display predicted word only if confidence >= 80%
        if predicted_word and prediction_confidence >= 80.0:
            # Determine color based on match
            if actual_word and predicted_word.lower() == actual_word.lower():
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
        elif predicted_word and prediction_confidence < 80.0:
            # Show a message that prediction confidence is too low
            text_low_conf = f"Low Confidence: {prediction_confidence:.1f}% (Need >= 80%)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text_low_conf, font, font_scale, thickness)
            
            # Draw background
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, y_offset - 5), (20 + text_width, y_offset + text_height + baseline + 5), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Draw low confidence message in yellow
            cv2.putText(frame, text_low_conf, (15, y_offset + text_height),
                       font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

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
        elif key == ord('l') and not recording and face_detected:
            if not actual_word:
                print(f"\n[INFO] Recording started - speak now!")
                print("[TIP] Press 'W' before recording to set the word you're saying for comparison.")
            else:
                print(f"\n[INFO] Recording started - say '{actual_word}' now!")
            recording = True
            frames = []  # Reset frames
            predicted_word = ""  # Clear previous prediction
            prediction_confidence = 0.0  # Reset confidence
            frames_without_face = 0  # Reset warning counter
        elif key == ord('l') and not face_detected:
            print("[WARNING] Please position your face in the camera first!")

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
    # Give OpenCV time to close windows
    import time
    time.sleep(0.5)
    print("[INFO] Camera and windows closed successfully.")
    print("\nLive Lip Reading Stopped.")
