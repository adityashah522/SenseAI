# SenseAI

A native iOS accessibility application that uses on-device machine learning to break communication barriers for deaf, hard-of-hearing, and visually impaired individuals. Built for the TSA Virginia State Conference — Software Development category, 2025/2026.

---

## The Problem

Eleven million Americans live with significant hearing loss. One million use American Sign Language as their primary language. For these individuals, everyday experiences that hearing people take for granted — knowing when a fire alarm goes off, having a conversation with someone who does not know ASL, feeling the rhythm of a song — are either inaccessible or require expensive accommodations.

A professional ASL interpreter costs over one hundred dollars per hour. Smoke alarms offer no warning to someone who cannot hear them. Music, one of the most universally cited cultural losses among the deaf community, remains largely closed off to those who cannot experience it through sound alone.

SenseAI is an attempt to address these gaps directly, using the machine learning and sensor capabilities already present in a modern iPhone.

---

## What SenseAI Does

SenseAI is a unified accessibility hub with three independent AI-powered modules. Users select the module they need from a central home screen. All inference runs entirely on-device using Apple Core ML and the Neural Engine — no internet connection is required, and no audio or camera data ever leaves the phone.

### BridgeAI

BridgeAI translates American Sign Language into text in real time using the iPhone camera. It uses Apple's Vision framework to detect 21 hand landmarks per frame, normalizes those coordinates relative to the wrist to achieve position and scale invariance, and passes the resulting 63-dimensional feature vector into a trained Multi-Layer Perceptron that classifies all 26 letters of the ASL alphabet. The model was trained on the ASL Alphabet dataset from Kaggle using PyTorch and Google Colab Pro, then exported to Core ML using coremltools. A smoothing buffer requiring consistent predictions across multiple frames prevents jitter and false positives. The module supports both hands by detecting hand chirality through Vision and conditionally applying a mirror transform before normalization.

### QuietAlert

QuietAlert listens for critical environmental sounds and alerts the user through haptic feedback and visual banners. It uses AVAudioEngine to capture live microphone audio, which is passed directly into a Core ML model that has the mel spectrogram computation baked into the model graph. The underlying classifier is a ResNet-style convolutional neural network trained on mel spectrograms from the ESC-50 and UrbanSound8K datasets across nine sound categories including fire alarms, sirens, dog barking, glass breaking, crying babies, and door knocks. Urgent sounds trigger four rapid haptic pulses through Core Haptics. Non-urgent sounds trigger a single softer pulse. All inference runs locally with no latency from network requests.

### HarmoniAI

HarmoniAI allows deaf and hard-of-hearing users to experience music through synchronized visuals and haptics. It uses Demucs, Meta AI Research's state-of-the-art source separation model, to isolate the drums, bass, vocals, and melody stems of any song. A Google Colab pipeline computes frame-accurate RMS energy envelopes at 43 frames per second for each stem and detects strong onsets using librosa, exporting the results as a structured JSON file. The iOS app reads this JSON and drives a real-time visual engine built in SwiftUI Canvas that renders comets, shockwaves, aurora streaks, and ambient plasma effects synchronized to the music. Drum hits trigger shockwave explosions. Vocal energy spawns comets that shoot across the screen. Bass energy generates sweeping aurora ribbons. The background color shifts based on the detected emotional mood of the song. Haptics are triggered on strong onset frames using UIImpactFeedbackGenerator with intensity mapped to the stem that fired the event.

---

## Technical Architecture

| Layer | Technology |
|-------|-----------|
| Application | Swift, SwiftUI |
| ML Inference | Core ML, Apple Neural Engine |
| Audio Capture | AVFoundation, AVAudioEngine |
| Camera and Hand Detection | AVFoundation, Vision framework |
| Haptics | Core Haptics, UIImpactFeedbackGenerator |
| Visual Rendering | SwiftUI Canvas, GraphicsContext |
| Signal Processing | Accelerate framework, vDSP |
| Model Training | PyTorch, torchaudio, librosa, MediaPipe |
| Model Export | coremltools |
| Training Environment | Google Colab Pro |
| Datasets | ESC-50, UrbanSound8K, ASL Alphabet (Kaggle) |
| Stem Separation | Demucs (Meta AI Research) |

---

## Model Details

### QuietAlert Classifier

- Architecture: ResNet-style CNN
- Input: Raw audio samples, shape [1, 110250] — spectrogram baked into model graph
- Output: Class logits, shape [1, 9]
- Sample rate: 22,050 Hz
- Duration: 5 seconds per inference window
- Classes: crackling fire, siren, dog, clock alarm, glass breaking, crying baby, vacuum cleaner, hand saw, door knock
- Training: PyTorch, Google Colab Pro, ESC-50 and UrbanSound8K
- Export: coremltools, mlprogram format, iOS 16 minimum deployment target

### BridgeAI ASL Classifier

- Architecture: Multi-Layer Perceptron
- Input: Normalized hand landmarks, shape [1, 63]
- Output: Class logits, shape [1, 26]
- Feature extraction: MediaPipe Hands, 21 landmarks x (x, y, z)
- Normalization: Wrist-relative subtraction, max-absolute scaling
- Layers: 63 -> 256 -> 256 -> 128 -> 26
- Training: PyTorch, Google Colab, ASL Alphabet dataset (Kaggle)
- Export: coremltools, mlprogram format

### HarmonAI Classifer

- Architecture: CNN-based emotion regression model
- Input: Grayscale Mel spectrogram (made from real time audio data), shape `[1, 224, 224, 1]`
- Output: Valence + arousal values, shape `[1, 2]` -> Classifed into an emotion depnding on the values
- Preprocessing:
  * Audio → Mel spectrogram
  * Resize to `224 × 224`
  * Normalize to `[0,1]`
- Predictions:
  * Valence = emotional positivity
  * Arousal = emotional intensity
- Training: TensorFlow / Keras, Python
- Model format: `.keras`
- Usage: Real-time music visualization control

---

## Key Engineering Challenges

### The Training-Inference Contract

The most significant technical challenge on this project was ensuring that the preprocessing applied to audio during training matched exactly what the iOS app applied during inference. When QuietAlert was first deployed, the model detected nothing despite achieving over 90 percent validation accuracy in Python. The root cause was a parameter mismatch between the mel filterbank implementation in librosa and the equivalent computation in Swift. The fix was to bake the spectrogram computation directly into the Core ML model graph using coremltools, eliminating the Swift preprocessing pipeline entirely and guaranteeing that training and inference would always see identical representations.

### Coordinate System Correction in BridgeAI

Apple's Vision framework reports hand landmark coordinates with the Y-axis inverted relative to the coordinate system used in the ASL training images. The front-facing camera also mirrors the image left-to-right. Without correcting both transformations, the model produced incorrect predictions consistently. The fix required three sequential transforms applied before the wrist-relative normalization step: flipping the Y coordinate, detecting hand chirality from the Vision observation, and conditionally mirroring the X coordinate for left hands. The order of these operations is critical — applying them in the wrong sequence produces incorrect results even though each individual transform is correct.

### Frame-Accurate Audio-Visual Sync in HarmoniAI

Synchronizing haptics and visuals to pre-computed stem data required converting continuous playback time into discrete frame indices on a 43 fps grid. A CADisplayLink fires at 60 fps and computes the current frame as the integer floor of the product of current playback time and the stem data frame rate. Onset lookups use Swift Set for O(1) lookup time per frame, avoiding any per-frame linear search across potentially thousands of onset indices.

---

## Project Structure

```
SenseAI/
├── SenseAIApp.swift
├── ContentView.swift
├── Managers/
│   ├── AppState.swift
│   ├── PermissionsManager.swift
│   ├── QuietAlertEngine.swift
│   ├── BridgeAIEngine.swift
│   └── HarmoniAIEngine.swift
├── Views/
│   ├── HomeView.swift
│   ├── HistoryView.swift
│   ├── SettingsView.swift
│   ├── ProfileView.swift
│   ├── Onboarding/
│   │   └── OnboardingFlowView.swift
│   └── Modules/
│       ├── BridgeAIView.swift
│       ├── HarmoniAIView.swift
│       ├── HarmoniAIImporter.swift
│       └── QuietAlertView.swift
├── Models/
│   └── SenseModule.swift
├── Components/
│   ├── ModuleCardView.swift
│   └── ModulePlaceholderView.swift
└── TrainedModels/
    ├── ASLClassifier.mlpackage
    ├── QuietAlertClassifier.mlpackage
    ├── asl_label_map.json
    └── label_map.json
```

---

## Requirements

- iOS 17.0 or later
- iPhone with Neural Engine (iPhone 12 or later recommended)
- Xcode 15 or later
- Microphone permission for QuietAlert
- Camera permission for BridgeAI

---

## Running the Project

1. Clone the repository
2. Open SenseAI.xcodeproj in Xcode
3. Select your iPhone as the build target
4. Build and run — all models are included in the repository and require no additional setup
5. Grant microphone and camera permissions when prompted on first launch

For HarmoniAI, process a song using the provided Google Colab notebook to generate a stem data JSON file, then import it through the in-app file picker.

---

## Team

This project was built by the following members of the TSA Software Development team:

- Vibhun Naredla
- Aryan Mathur
- Mayukh Aduru
- Aditya Shah
- Krish Shah
- Ronav Gopal

---

## Acknowledgments

- ESC-50 dataset: Karol Piczak, 2015
- UrbanSound8K dataset: Salamon et al., 2014
- ASL Alphabet dataset: Kaggle, grassknoted
- Demucs source separation model: Meta AI Research
- MediaPipe Hands: Google

---

TSA Virginia State Conference — Software Development
2025/2026
