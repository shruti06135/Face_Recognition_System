# Real-Time Face Recognition and Authentication System

A real-time biometric authentication system that uses facial recognition and blink-based liveness detection for secure user verification. The system combines OpenCV, DeepFace (FaceNet), and dlib facial landmarks to provide reliable and spoof-resistant authentication.

## Features

- Real-time face detection using OpenCV
- Face recognition using DeepFace (FaceNet)
- Blink-based liveness detection using Eye Aspect Ratio (EAR)
- Multi-frame verification for improved authentication accuracy
- User registration and secure login workflow
- CLAHE-based image enhancement for varying lighting conditions
- Anti-spoofing protection against photo attacks

## Tech Stack

- Python
- OpenCV
- DeepFace (FaceNet)
- dlib
- NumPy
- SciPy

## Project Workflow

### Registration

- User enters a username
- Multiple face images are captured
- Images are enhanced using CLAHE preprocessing
- User dataset is stored locally

### Authentication

- User enters registered username
- Blink-based liveness detection is performed
- Multiple verification frames are captured
- FaceNet embeddings are compared using DeepFace
- Authentication is granted only if the verification threshold is met

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd face-recognition-authentication
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download Landmark Model

Place the file below in the project root directory:

```text
shape_predictor_68_face_landmarks.dat
```

## Run Project

```bash
python app.py
```

## Project Structure

```text
face-recognition-authentication/
│
├── venv/
├── .gitignore
├── app.py
├── README.md
├── requirements.txt
└── shape_predictor_68_face_landmarks.dat
```

## Future Improvements

- Database integration
- Web dashboard
- Attendance management system
- Cloud deployment
- Advanced anti-spoofing techniques

## License

This project is developed for educational and learning purposes.