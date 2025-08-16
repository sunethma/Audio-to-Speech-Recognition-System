# Wav2Vec2 Fine-tuning on TIMIT Dataset

This project demonstrates how to fine-tune Facebook's Wav2Vec2 model for Automatic Speech Recognition (ASR) using the TIMIT dataset. The model is trained to transcribe speech to text using Connectionist Temporal Classification (CTC).

## 🎯 Project Overview

Model: Facebook's Wav2Vec2-base

Dataset: TIMIT ASR dataset

Task: Speech-to-Text transcription

Framework: Hugging Face Transformers

Platform: Google Colab with GPU support

## 📋 Requirements

```
numpy==1.23.5
scikit-learn
datasets==1.18.3
transformers==4.17.0
jiwer
librosa
soundfile
```

## 🚀 Getting Started

1. Environment Setup
The notebook automatically checks for GPU availability and mounts Google Drive for model persistence:

```
# Check GPU
gpu_info = !nvidia-smi

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

2. Install Dependencies

```
!pip uninstall -y albumentations bigframes chex jax jaxlib pymc scikit-image xarray
!pip install numpy==1.23.5
!pip install scikit-learn datasets==1.18.3 transformers==4.17.0 jiwer
```

3. Run the Training
   Execute the notebook cells sequentially to:

Load and preprocess the TIMIT dataset

Create vocabulary and tokenizer

Fine-tune the Wav2Vec2 model

Evaluate model performance

---

## 📊 Dataset Information

### TIMIT Dataset:

Contains phonetically balanced speech corpus

Includes audio files with corresponding transcriptions

Cleaned to remove special characters and normalize text

Filtered to sequences ≤ 4 seconds for memory efficiency

---

## 🏗️ Model Architecture

Base Model: ```facebook/wav2vec2-base```

Task Head: CTC (Connectionist Temporal Classification)

Vocabulary: Custom vocabulary extracted from TIMIT transcriptions

Feature Extraction: Frozen during fine-tuning

Sampling Rate: 16 kHz

## 🔧 Training Configuration

```bash
training_args = TrainingArguments(
    output_dir="location of the saved model checkpoints",
    save_strategy="epoch",
    group_by_length=True,
    per_device_train_batch_size=7,
    gradient_accumulation_steps=7,
    evaluation_strategy="steps",
    num_train_epochs=7,
    fp16=True,
    gradient_checkpointing=True,
    learning_rate=1e-4,
    weight_decay=0.005,
    warmup_steps=1000,
    save_total_limit=2
)
```

## 📈 Results

Test WER: ~0.36 (36%)

Performance: Acoustically similar transcriptions with some spelling/grammatical errors

Improvement Potential: Additional training epochs, language model integration, data augmentation


---

## 🎵 Testing with Custom Audio
The notebook includes functionality to test the trained model on custom audio files:

```bash
# Load custom audio file
audio_file = "path to the audio file"
audio_input, sampling_rate = sf.read(audio_file)

# Process and transcribe
input_values = processor(audio_input, return_tensors="pt", sampling_rate=sampling_rate).input_values.cuda()
transcription = processor.decode(predicted_ids[0])
```

---

## 📊 Evaluation Metrics

Primary Metric: Word Error Rate (WER)

Evaluation Strategy: Step-based evaluation every 500 steps

Prediction Analysis: Random sample inspection for error analysis

---

## 🚨 Important Notes

Requires GPU for training (CUDA support)

Google Drive is used for model persistence

Training time varies based on available GPU resources

Model performance can be improved with additional training epochs

📚 References

[Wav2Vec2 Paper](https://arxiv.org/pdf/2006.11477)

[Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

[TIMIT Dataset](https://catalog.ldc.upenn.edu/LDC93S1)

---

📄 License
This project is for educational purposes. Please check the licenses of the underlying models and datasets used.
