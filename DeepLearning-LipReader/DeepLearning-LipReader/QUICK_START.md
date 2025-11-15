# Quick Start Guide

## Setup Verification

Run the test script to verify everything is working:
```bash
py test_setup.py
```

## Running the Project

### 1. Data Collection
Collect lip reading data using your webcam:
```bash
py src/collection.py
```
- Edit `WORD = "panda"` in `src/collection.py` to change the word you're recording
- Press 'L' to start recording when you're ready to speak
- Press 'Q' to quit
- Collect multiple takes (the script automatically increments take numbers)

### 2. Data Preprocessing
Preprocess the collected data:
```bash
py src/preprocess.py
```
This will:
- Convert images to grayscale
- Apply noise reduction and enhancement filters
- Normalize pixel values
- Save processed data as .npy files

### 3. Model Training
Train the 3D CNN model:
```bash
py src/train_model.py
```
This will:
- Load preprocessed data
- Split into train/validation sets (80/20)
- Train the model for 20 epochs
- Save the trained model to `model/lip_reader_3dcnn.h5`
- Display training metrics and plots

### 4. Real-Time Prediction
Run live lip reading:
```bash
py src/predict.py
```
- Press 'L' to start recording a 22-frame sequence
- The model will predict the word after collecting frames
- Press 'Q' to quit

## Important Notes

1. **Model Compatibility**: The included model (`model/lip_reader_3dcnn.h5`) was trained on the author's lips only. For best results, you should:
   - Collect your own data (multiple takes per word)
   - Retrain the model with your data

2. **Data Collection Tips**:
   - Record in good lighting
   - Keep your face centered in the frame
   - Speak clearly and consistently
   - Collect at least 50-100 takes per word for good accuracy

3. **Webcam Index**: If your webcam doesn't work, try changing the camera index in the scripts:
   - `cap = cv2.VideoCapture(0)` → try `1`, `2`, etc.

## Troubleshooting

- **Import errors**: Make sure all dependencies are installed
- **Model won't load**: The model might be incompatible with your TensorFlow version
- **No face detected**: Check lighting and camera position
- **Poor accuracy**: Collect more training data and retrain

## Project Structure

```
DeepLearning-LipReader/
├── src/
│   ├── collection.py      # Data collection script
│   ├── preprocess.py      # Data preprocessing script
│   ├── train_model.py     # Model training script
│   └── predict.py         # Real-time prediction script
├── model/
│   ├── lip_reader_3dcnn.h5                    # Trained model
│   └── shape_predictor_68_face_landmarks.dat  # Face landmark detector
├── data/                   # Raw collected data (created when running collection.py)
├── processed_data/         # Preprocessed data (created when running preprocess.py)
└── test_setup.py          # Setup verification script
```

