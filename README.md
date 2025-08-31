# Face Recognition Authentication System

A simple face recognition-based authentication system built using Python, OpenCV, and DeepFace.

## Features

- User registration using webcam
- Captures multiple face images during signup
- Face quality validation before registration
- Face verification during login
- Automatic image preprocessing for better recognition
- Secure user authentication using DeepFace (FaceNet)

## Tech Stack

- Python
- OpenCV
- DeepFace
- NumPy

## How It Works

1. User registers by providing a username.
2. The system captures multiple face images.
3. Images are validated and stored locally.
4. During login, the system captures new images.
5. DeepFace compares the captured images with the registered images and verifies the user.

## Run

```bash
python main.py