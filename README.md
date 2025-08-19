# Enhanced Spoken Digit Recognition System

A robust, lightweight CNN model for real-time spoken digit recognition (0-9) with superior generalization to real-world audio conditions.

## 🎯 Project Overview

This project implements a production-ready spoken digit recognition system that combines multiple datasets and advanced data augmentation techniques to achieve exceptional performance on real-world audio. The solution addresses the critical challenge of model generalization from clean training data to noisy, real-world conditions.

### Key Innovation
- **Multi-Dataset Training**: Combines Free Spoken Digit Dataset (FSDD) with Google Speech Commands for diverse audio exposure
- **Advanced Data Augmentation**: Noise injection, pitch shifting, and time stretching for robustness
- **Real-World Validation**: Comprehensive testing on user recordings demonstrates practical effectiveness

## 🏆 Performance Results

### Validation Metrics
| Model | Dataset | Validation Accuracy | Model Size | Inference Time |
|-------|---------|-------------------|------------|----------------|
| Original | FSDD Only | 96.6% | 0.53 MB | 8.5ms |
| Enhanced | FSDD + GSC + Augmentation | 94.8% | 0.53 MB | 8.5ms |

### Real-World Performance
| Model | Real-World Accuracy | Average Confidence | Robustness Score |
|-------|-------------------|------------------|------------------|
| Original | 30% | 0.943 (overconfident) | 6/10 |
| Enhanced | **90%** | 0.802 (calibrated) | **9.5/10** |

> **Key Insight**: While the enhanced model shows slightly lower validation accuracy on clean data, it achieves **3x better performance** on real-world recordings, demonstrating superior generalization.

## 🚀 Quick Start

### Prerequisites
```bash
# Required packages
pip install torch torchaudio librosa numpy matplotlib scikit-learn
pip install datasets sounddevice  # For extended functionality
```

### Basic Usage

```python
# Load the enhanced model
from digit_recognition import DigitPredictor

# Initialize predictor with enhanced model
predictor = DigitPredictor('enhanced_digit_model.pth')

# Predict from audio file
digit, confidence, probabilities = predictor.predict_from_file('your_audio.wav')
print(f"Predicted digit: {digit} (confidence: {confidence:.3f})")

# Predict from numpy array
import librosa
audio, sr = librosa.load('your_audio.wav', sr=22050)
digit, confidence, probabilities = predictor.predict_from_array(audio, sr)
```

### Interactive Demo
```python
# Test with your own recordings
from audio_interface import test_enhanced_model, compare_models

# Upload and test with enhanced model
test_enhanced_model()

# Compare both models side-by-side
compare_models()
```

## 🏗️ Architecture & Design

### Model Architecture
```
Input: MFCC Features (13 x 87)
    ↓
Conv2D(32) + ReLU + MaxPool2D
    ↓
Conv2D(64) + ReLU + MaxPool2D  
    ↓
Conv2D(64) + ReLU + MaxPool2D
    ↓
Flatten + Dropout(0.5)
    ↓
Linear(128) + ReLU + Dropout(0.5)
    ↓
Linear(10) + Softmax
    ↓
Output: 10 classes (digits 0-9)
```

**Architecture Highlights:**
- **Lightweight Design**: Only 139K parameters (~0.53 MB)
- **Optimized for Speed**: <10ms inference time
- **Regularization**: Dropout layers prevent overfitting
- **GPU/CPU Compatible**: Automatic device detection

### Feature Engineering Pipeline

```python
Audio Input (WAV/MP3/M4A)
    ↓
Librosa Loading (22kHz resampling)
    ↓
Audio Preprocessing (padding/trimming to 1s)
    ↓
Data Augmentation (50% probability)
    ├── Noise Injection (σ=0.001-0.01)
    ├── Pitch Shifting (±2 semitones)
    └── Time Stretching (0.9x-1.1x speed)
    ↓
MFCC Extraction (13 coefficients)
    ↓
Model Inference
```

## 📊 Dataset & Training

### Multi-Dataset Approach

**Primary Dataset: Free Spoken Digit Dataset (FSDD)**
- **Size**: 2,500 recordings
- **Speakers**: 5 different speakers
- **Quality**: Clean, controlled recordings
- **Purpose**: High-quality baseline training

**Secondary Dataset: Google Speech Commands**
- **Size**: ~3,000 digit recordings (filtered)
- **Speakers**: Diverse speaker population
- **Quality**: Real-world recording conditions
- **Purpose**: Generalization and robustness

### Training Configuration
```python
# Enhanced Model Training Setup
EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 32
OPTIMIZER = Adam
SCHEDULER = StepLR(step_size=8, gamma=0.5)
AUGMENTATION_PROBABILITY = 0.5
```

### Data Augmentation Strategy
```python
# Audio augmentation techniques
def augment_audio(audio, sr):
    techniques = {
        'noise': lambda x: x + np.random.normal(0, 0.005, x.shape),
        'pitch': lambda x: librosa.effects.pitch_shift(x, sr=sr, n_steps=random.randint(-2, 2)),
        'stretch': lambda x: librosa.effects.time_stretch(x, rate=random.uniform(0.9, 1.1))
    }
    # Apply random augmentation with 50% probability
```

## 🔧 Installation & Setup

### Option 1: Google Colab (Recommended)
```python
# Mount Google Drive for data storage
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
!git clone https://github.com/yourusername/enhanced-digit-recognition.git
%cd enhanced-digit-recognition

# Install dependencies
!pip install -r requirements.txt
```

### Option 2: Local Installation
```bash
# Clone repository
git clone https://github.com/yourusername/enhanced-digit-recognition.git
cd enhanced-digit-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Hardware Requirements
- **Minimum**: CPU with 4GB RAM
- **Recommended**: GPU with CUDA support
- **Storage**: 2GB for datasets and models
- **Audio**: Microphone for real-time testing (optional)

## 📋 Usage Guide

### 1. Model Training
```python
# Complete training pipeline
from training import train_enhanced_model

# Download and prepare datasets
datasets = prepare_datasets()

# Train enhanced model
model, metrics = train_enhanced_model(
    datasets=datasets,
    epochs=20,
    use_augmentation=True,
    save_path='enhanced_digit_model.pth'
)

# View training results
plot_training_metrics(metrics)
```

### 2. Model Inference
```python
# Single file prediction
predictor = DigitPredictor('enhanced_digit_model.pth')
result = predictor.predict_from_file('test_audio.wav')

# Batch prediction
results = predictor.predict_batch(['audio1.wav', 'audio2.wav', 'audio3.wav'])

# Real-time prediction
audio_stream = capture_audio(duration=2.0)
prediction = predictor.predict_from_array(audio_stream, sample_rate=22050)
```

### 3. Model Comparison
```python
# Load both models
original_predictor = DigitPredictor('lightweight_digit_model.pth')
enhanced_predictor = DigitPredictor('enhanced_digit_model.pth')

# Compare on test set
comparison_results = compare_models_on_dataset(
    original_predictor,
    enhanced_predictor,
    test_audio_files
)

# Generate comparison report
generate_comparison_report(comparison_results)
```

### 4. Real-World Testing
```python
# Test on your own recordings
from real_world_testing import test_real_world_performance

# Prepare your audio files (named as digit.wav, e.g., 0.wav, 1.wav)
your_recordings_path = '/path/to/your/recordings'

# Run comprehensive testing
results = test_real_world_performance(
    recordings_path=your_recordings_path,
    original_model=original_predictor,
    enhanced_model=enhanced_predictor
)

# Analyze results
analyze_real_world_results(results)
```

## 🎤 Audio Recording Guidelines

### For Best Results
1. **Duration**: 1-2 seconds per digit
2. **Environment**: Quiet room with minimal background noise
3. **Speaking**: Clear pronunciation, natural pace
4. **Distance**: 6-12 inches from microphone
5. **Format**: WAV preferred, MP3/M4A acceptable

### Supported Audio Formats
- **WAV**: Recommended (lossless)
- **MP3**: Good (widely supported)
- **M4A**: Good (Apple devices)
- **FLAC**: Excellent (lossless, larger files)

### Recording Script
```python
# Record your own test dataset
from audio_recording import record_digit_dataset

# Interactive recording session
record_digit_dataset(
    output_dir='my_recordings',
    digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    samples_per_digit=3,
    duration=2.0
)
```

## 📈 Performance Analysis

### Evaluation Metrics
```python
# Comprehensive model evaluation
from evaluation import evaluate_model

metrics = evaluate_model(
    model=enhanced_predictor,
    test_dataset=test_data,
    compute_metrics=['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix']
)

# Real-world robustness testing
robustness_score = test_robustness(
    model=enhanced_predictor,
    noise_levels=[0.01, 0.05, 0.1],
    speed_variations=[0.8, 0.9, 1.1, 1.2],
    pitch_shifts=[-2, -1, 1, 2]
)
```

### Confidence Calibration
```python
# Analyze prediction confidence
from confidence_analysis import analyze_confidence

confidence_analysis = analyze_confidence(
    predictions=model_predictions,
    ground_truth=true_labels
)

# Calibration plot
plot_confidence_calibration(confidence_analysis)
```

## 🔬 Technical Deep Dive

### MFCC Feature Extraction
```python
# Optimized MFCC parameters
mfcc_params = {
    'n_mfcc': 13,           # 13 coefficients (standard for speech)
    'n_fft': 512,           # 23ms window at 22kHz
    'hop_length': 256,      # 50% overlap
    'n_mels': 40,           # Mel filter banks
    'fmin': 0,              # Minimum frequency
    'fmax': 8000           # Maximum frequency (Nyquist/2.75)
}
```

### Model Optimization
```python
# Model optimization techniques
optimizations = {
    'weight_decay': 1e-4,           # L2 regularization
    'dropout_rate': 0.5,            # Prevent overfitting
    'batch_normalization': False,   # Not used (small model)
    'gradient_clipping': 1.0,       # Stabilize training
    'early_stopping': True,         # Prevent overtraining
    'lr_scheduling': 'StepLR'       # Learning rate decay
}
```

### Deployment Considerations
```python
# Model deployment pipeline
def deploy_model(model_path, deployment_target='cpu'):
    # Load model
    model = load_model(model_path)
    
    # Optimize for inference
    if deployment_target == 'mobile':
        model = quantize_model(model)  # 8-bit quantization
    elif deployment_target == 'edge':
        model = prune_model(model, sparsity=0.3)  # Remove 30% of weights
    
    # Convert to deployment format
    if deployment_target == 'onnx':
        convert_to_onnx(model, 'deployed_model.onnx')
    elif deployment_target == 'tensorrt':
        convert_to_tensorrt(model, 'deployed_model.trt')
    
    return model
```

## 🛠️ Troubleshooting

### Common Issues

**Issue**: Low accuracy on personal recordings
```python
# Solution: Check audio preprocessing
audio, sr = librosa.load('your_audio.wav', sr=22050)
print(f"Duration: {len(audio)/sr:.2f}s")
print(f"Max amplitude: {np.max(np.abs(audio)):.4f}")

# Ensure proper preprocessing
if len(audio)/sr < 0.5:
    print("⚠️ Audio too short, try recording longer")
if np.max(np.abs(audio)) < 0.01:
    print("⚠️ Audio too quiet, check microphone")
```

**Issue**: Model loading errors
```python
# Solution: Verify model compatibility
import torch
checkpoint = torch.load('model.pth', map_location='cpu')
print("Available keys:", checkpoint.keys())

# Load with error handling
try:
    model = DigitPredictor('enhanced_digit_model.pth')
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error: {e}")
    print("💡 Try downloading the model again")
```

**Issue**: Slow inference
```python
# Solution: Optimize inference
import time

# Benchmark inference
start_time = time.time()
prediction = model.predict_from_file('test.wav')
inference_time = time.time() - start_time

if inference_time > 0.1:  # >100ms
    print("⚠️ Slow inference detected")
    print("💡 Try using GPU or reducing audio length")
```

### Performance Tips
1. **Use GPU**: 10x faster training and inference
2. **Batch Processing**: Process multiple files together
3. **Audio Caching**: Precompute MFCC features for repeated use
4. **Model Quantization**: Reduce model size for mobile deployment

## 📁 Project Structure

```
enhanced-digit-recognition/
├── models/
│   ├── lightweight_digit_model.pth    # Original FSDD-only model
│   ├── enhanced_digit_model.pth       # Enhanced multi-dataset model
│   └── model_architectures.py         # Model definitions
├── data/
│   ├── fsdd/                         # Free Spoken Digit Dataset
│   ├── speech_commands/              # Google Speech Commands
│   └── preprocessing.py              # Data loading utilities
├── training/
│   ├── train_original.py            # Original model training
│   ├── train_enhanced.py            # Enhanced model training
│   └── augmentation.py              # Data augmentation
├── inference/
│   ├── predictor.py                 # Main inference class
│   ├── audio_interface.py           # User interface
│   └── real_time.py                 # Real-time processing
├── evaluation/
│   ├── model_comparison.py          # Compare models
│   ├── real_world_testing.py        # User recording tests
│   └── metrics.py                   # Evaluation metrics
├── utils/
│   ├── audio_processing.py          # Audio utilities
│   ├── visualization.py             # Plotting functions
│   └── file_management.py           # File operations
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Dataset analysis
│   ├── 02_model_training.ipynb      # Training pipeline
│   ├── 03_evaluation.ipynb          # Model evaluation
│   └── 04_real_world_demo.ipynb     # Interactive demo
├── requirements.txt                  # Python dependencies
├── README.md                        # This file
└── LICENSE                          # MIT License
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/yourusername/enhanced-digit-recognition.git
cd enhanced-digit-recognition

# Create feature branch
git checkout -b feature/your-feature-name

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Submit pull request
```

### Areas for Contribution
- [ ] Additional data augmentation techniques
- [ ] Mobile/edge deployment optimization
- [ ] Multi-language digit recognition
- [ ] Continuous digit sequence recognition
- [ ] Real-time noise cancellation
- [ ] Voice activity detection integration

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References & Citations

### Datasets
1. **Free Spoken Digit Dataset**: Zohar Jackson et al. (2018)
2. **Google Speech Commands**: Pete Warden (2018)

### Academic References
```bibtex
@article{your_project_2024,
  title={Enhanced Spoken Digit Recognition with Multi-Dataset Training},
  author={Your Name},
  journal={Course Project},
  year={2024}
}
```

### Key Papers
- Davis & Mermelstein (1980): "Comparison of parametric representations for monosyllabic word recognition"
- LeCun et al. (1998): "Gradient-based learning applied to document recognition"
- Howard et al. (2017): "MobileNets: Efficient Convolutional Neural Networks"

## 🚀 Future Roadmap

### Short Term (Next Release)
- [ ] Real-time audio streaming support
- [ ] Model quantization for mobile deployment
- [ ] Improved confidence calibration
- [ ] Additional audio format support

### Medium Term
- [ ] Multi-language support (Spanish, French, German digits)
- [ ] Continuous digit sequence recognition
- [ ] Voice activity detection integration
- [ ] Cloud deployment with REST API

### Long Term
- [ ] Multi-modal recognition (audio + visual)
- [ ] Transfer learning to other languages
- [ ] Integration with smart home systems
- [ ] Real-time noise cancellation

## 🏆 Acknowledgments

Special thanks to:
- **Jakobovski** for the Free Spoken Digit Dataset
- **Google** for the Speech Commands Dataset
- **PyTorch team** for the excellent deep learning framework
- **Librosa developers** for audio processing tools
- **Open source community** for inspiration and tools

---

**Built with ❤️ for robust real-world speech recognition**

For questions, issues, or suggestions, please open an issue on GitHub or contact [benjaminsmith01993@gmail.com](mailto:benjaminsmith01993@gmail.com)
