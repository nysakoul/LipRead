import cv2
import dlib
import os
import numpy as np
import sys

# Get the project root directory (parent of src/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Load Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
SHAPE_PREDICTOR_PATH = os.path.join(PROJECT_ROOT, "model", "shape_predictor_68_face_landmarks.dat")
if not os.path.exists(SHAPE_PREDICTOR_PATH):
    print(f"[ERROR] Shape predictor file not found at {SHAPE_PREDICTOR_PATH}")
    sys.exit(1)
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

# Constants
VIDEOS_DIR = os.path.join(PROJECT_ROOT, "videos")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data")
FRAMES_PER_WORD = 22  # Fixed number of frames per take

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Check if videos directory exists
if not os.path.exists(VIDEOS_DIR):
    print(f"[ERROR] Videos directory not found at {VIDEOS_DIR}")
    sys.exit(1)

print(f"\nProcessing videos from: {VIDEOS_DIR}")
print(f"Output directory: {OUTPUT_DIR}\n")

# Get all word folders
words = sorted([d for d in os.listdir(VIDEOS_DIR) if os.path.isdir(os.path.join(VIDEOS_DIR, d))])

if len(words) == 0:
    print(f"[ERROR] No word folders found in {VIDEOS_DIR}")
    sys.exit(1)

print(f"Found {len(words)} words: {', '.join(words)}\n")

# Process each word
for word in words:
    word_video_dir = os.path.join(VIDEOS_DIR, word)
    word_output_dir = os.path.join(OUTPUT_DIR, word)
    
    if not os.path.exists(word_output_dir):
        os.makedirs(word_output_dir)
    
    # Get all video files for this word
    video_files = sorted([f for f in os.listdir(word_video_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))])
    
    if len(video_files) == 0:
        print(f"[WARNING] No video files found in {word_video_dir}")
        continue
    
    print(f"Processing word: '{word}' ({len(video_files)} videos)")
    
    take_number = 1
    
    # Process each video
    for video_file in video_files:
        video_path = os.path.join(word_video_dir, video_file)
        take_dir = os.path.join(word_output_dir, f"take_{take_number}")
        
        if not os.path.exists(take_dir):
            os.makedirs(take_dir)
        
        print(f"  -> Processing: {video_file}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"    [ERROR] Could not open video: {video_path}")
            continue
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame interval to get exactly FRAMES_PER_WORD frames
        if total_frames < FRAMES_PER_WORD:
            print(f"    [WARNING] Video has only {total_frames} frames, need {FRAMES_PER_WORD}. Using all frames.")
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, FRAMES_PER_WORD, dtype=int)
        
        frames_collected = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if this frame should be extracted
            if frame_count in frame_indices:
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = detector(gray, 0)
                if len(faces) == 0:
                    faces = detector(gray, 1)  # Try with upsampling
                
                if len(faces) > 0:
                    # Use the largest face
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
                        
                        # Add padding
                        padding = 15
                        x_min = max(0, x_min - padding)
                        x_max = min(frame.shape[1], x_max + padding)
                        y_min = max(0, y_min - padding)
                        y_max = min(frame.shape[0], y_max + padding)
                        
                        # Extract and resize lip region
                        lip_region = frame[y_min:y_max, x_min:x_max]
                        
                        if lip_region.size > 0:
                            lip_region = cv2.resize(lip_region, (112, 80))
                            frames_collected.append((frame_count, lip_region))
                    except Exception as e:
                        print(f"    [WARNING] Error processing frame {frame_count}: {e}")
            
            frame_count += 1
        
        cap.release()
        
        # Save collected frames
        if len(frames_collected) >= FRAMES_PER_WORD:
            # Sort by frame number and take first FRAMES_PER_WORD
            frames_collected.sort(key=lambda x: x[0])
            frames_collected = frames_collected[:FRAMES_PER_WORD]
            
            for idx, (frame_num, lip_frame) in enumerate(frames_collected):
                frame_path = os.path.join(take_dir, f"frame_{idx}.png")
                cv2.imwrite(frame_path, lip_frame)
            
            print(f"    [OK] Saved {len(frames_collected)} frames to {take_dir}")
        else:
            print(f"    [WARNING] Only collected {len(frames_collected)} frames, need {FRAMES_PER_WORD}")
            # Still save what we have
            for idx, (frame_num, lip_frame) in enumerate(frames_collected):
                frame_path = os.path.join(take_dir, f"frame_{idx}.png")
                cv2.imwrite(frame_path, lip_frame)
        
        take_number += 1

print(f"\n[OK] Video processing complete! Data saved in '{OUTPUT_DIR}'")
print(f"Next step: Run 'python src/preprocess.py' to preprocess the data")

