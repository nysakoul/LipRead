"""
Test script to verify the setup is working correctly
"""
import os
import sys

# Get the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
sys.path.insert(0, PROJECT_ROOT)

print("=" * 60)
print("Testing DeepLearning-LipReader Setup")
print("=" * 60)

# Test 1: Import all required modules
print("\n[1/5] Testing module imports...")
try:
    import cv2
    import dlib
    import tensorflow as tf
    import numpy as np
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    print("[OK] All modules imported successfully!")
    print(f"   - TensorFlow: {tf.__version__}")
    print(f"   - OpenCV: {cv2.__version__}")
    print(f"   - NumPy: {np.__version__}")
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    sys.exit(1)

# Test 2: Check if model files exist
print("\n[2/5] Checking model files...")
model_path = os.path.join(PROJECT_ROOT, "model", "lip_reader_3dcnn.h5")
shape_predictor_path = os.path.join(PROJECT_ROOT, "model", "shape_predictor_68_face_landmarks.dat")

if os.path.exists(model_path):
    print(f"[OK] Model file found: {model_path}")
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"   - Size: {file_size:.2f} MB")
else:
    print(f"[WARNING] Model file not found: {model_path}")
    print("   (This is OK if you haven't trained yet)")

if os.path.exists(shape_predictor_path):
    print(f"[OK] Shape predictor found: {shape_predictor_path}")
    file_size = os.path.getsize(shape_predictor_path) / (1024 * 1024)  # MB
    print(f"   - Size: {file_size:.2f} MB")
else:
    print(f"[ERROR] Shape predictor not found: {shape_predictor_path}")
    print("   This file is required for face detection!")

# Test 3: Try loading the model
print("\n[3/5] Testing model loading...")
if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("[OK] Model loaded successfully!")
        print(f"   - Input shape: {model.input_shape}")
        print(f"   - Output shape: {model.output_shape}")
        print(f"   - Number of parameters: {model.count_params():,}")
    except Exception as e:
        print(f"[WARNING] Could not load model: {e}")
        print("   (This might be OK if the model was trained on different data)")
else:
    print("[WARNING] Skipping model load test (model file not found)")

# Test 4: Check data directories
print("\n[4/5] Checking data directories...")
data_dir = os.path.join(PROJECT_ROOT, "data")
processed_dir = os.path.join(PROJECT_ROOT, "processed_data")

if os.path.exists(data_dir):
    words = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if words:
        print(f"[OK] Data directory found with {len(words)} words: {', '.join(words)}")
    else:
        print(f"[WARNING] Data directory exists but is empty")
else:
    print(f"[WARNING] Data directory not found (will be created when you run collection.py)")

if os.path.exists(processed_dir):
    words = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    if words:
        print(f"[OK] Processed data directory found with {len(words)} words")
    else:
        print(f"[WARNING] Processed data directory exists but is empty")
else:
    print(f"[WARNING] Processed data directory not found (will be created when you run preprocess.py)")

# Test 5: Test dlib face detector
print("\n[5/5] Testing dlib face detector...")
if os.path.exists(shape_predictor_path):
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(shape_predictor_path)
        print("[OK] Dlib face detector initialized successfully!")
    except Exception as e:
        print(f"[ERROR] Error initializing dlib: {e}")
else:
    print("[WARNING] Skipping dlib test (shape predictor not found)")

print("\n" + "=" * 60)
print("Setup Test Complete!")
print("=" * 60)
print("\nNext steps:")
print("1. Run 'py src/collection.py' to collect lip reading data")
print("2. Run 'py src/preprocess.py' to preprocess the data")
print("3. Run 'py src/train_model.py' to train the model")
print("4. Run 'py src/predict.py' to test real-time lip reading")
print("\nNote: The existing model was trained on the author's lips only.")
print("      You'll need to collect your own data and retrain for best results.")

