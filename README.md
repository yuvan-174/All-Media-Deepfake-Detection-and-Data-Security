# All-Media-Deepfake-Detection-and-Data-Security

# DeepDetect - Advanced Deepfake & PII Detection

A comprehensive web-based solution for detecting AI-generated content (Deepfakes) across images, videos, and text, along with Sensitive Data (PII) protection.

## About
DeepDetect is a project designed to combat the rising threat of misinformation and privacy violations caused by Generative AI. As deepfakes become more realistic, traditional verification methods fail. This project integrates state-of-the-art Deep Learning models (Vision Transformers, MTCNN) and Natural Language Processing (NLP) to analyze media files and text. It features a modern, glassmorphism-based UI to provide users with an intuitive interface for verifying digital content and redacting sensitive information.

## Features
* **Multi-Modal Detection:** Capable of analyzing Images, Videos, and Text documents.
* **Advanced Deep Learning:** Uses `prithivMLmods/Deep-Fake-Detector-v2-Model` and Vision Transformers for high-accuracy deepfake detection.
* **Frame-by-Frame Video Analysis:** Extracts and analyzes individual video frames to detect temporal anomalies.
* **PII Protection:** Automatically detects and redacts Personally Identifiable Information (PII) using Microsoft Presidio.
* **Secure Data Handling:** Implements AES-256 encryption for all uploaded files to ensure user privacy.
* **Modern UI/UX:** Responsive Glassmorphism design with real-time progress tracking and dynamic result cards.

## Requirements
* **Operating System:** Requires a 64-bit OS (Windows 10/11, Ubuntu, or macOS) for compatibility with PyTorch/TensorFlow.
* **Development Environment:** Python 3.8 or later is necessary for the Flask backend.
* **Deep Learning Frameworks:** PyTorch and Transformers (Hugging Face) for model inference.
* **Image Processing Libraries:** OpenCV (cv2) and MTCNN for facial feature extraction and preprocessing.
* **NLP Libraries:** Presidio Analyzer for PII detection and various Transformer models for text analysis.
* **Web Framework:** Flask (Python) for the backend API and serving HTML templates.
* **Additional Dependencies:** `cryptography` for file encryption, `werkzeug` for secure file handling, and `Pillow` for image manipulation.

## System Architecture
_(Placeholder for your architecture diagram)_
* **Frontend:** HTML5, CSS3 (Glassmorphism), JavaScript (Fetch API).
* **Backend:** Flask (REST API).
* **AI Engine:** Hugging Face Transformers, MTCNN, Presidio.
* **Storage:** AES-Encrypted local storage for temporary processing.

## Output

### Output 1 - Image Detection
_Detects if a face in an uploaded image is Real or Fake._
![Image Detection Screenshot](https://via.placeholder.com/600x400?text=Image+Detection+Result+Card)

### Output 2 - Video Analysis
_Scans video files frame-by-frame to identify manipulation._
![Video Detection Screenshot](https://via.placeholder.com/600x400?text=Video+Analysis+Result)

### Output 3 - PII Redaction
_Identifies sensitive entities (Names, Phones, Emails) and encrypts the file._
![PII Detection Screenshot](https://via.placeholder.com/600x400?text=PII+Redaction+Result)

**Detection Accuracy:** ~96% (Varies by model and input quality)

## Results and Impact
DeepDetect empowers individuals and organizations to verify the authenticity of digital media in real-time. By integrating PII protection, it also serves as a dual-purpose tool for security and privacy. The project showcases the potential of combining Computer Vision and NLP to create a safer digital environment, mitigating the risks of identity theft and disinformation campaigns.

## References / Models Used
1.  **Image Model:** [prithivMLmods/Deep-Fake-Detector-v2-Model](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model)
2.  **Face Detection:** MTCNN (Multi-task Cascaded Convolutional Networks)
3.  **PII Detection:** Microsoft Presidio
4.  **Backend:** Flask Documentation
