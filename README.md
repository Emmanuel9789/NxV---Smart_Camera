# SMART CAMERA PROJECT

## Overview

This project is a Raspberry Pi–based intelligent camera system designed to perform real-time computer vision tasks such as motion detection, face recognition, and smart event logging. The system is built using Python and OpenCV and is designed for extensibility, allowing additional AI-based modules to be integrated over time.

---

## Key Features

* Real-time camera feed processing
* Motion detection using frame differencing
* Face detection and recognition support
* Modular structure for AI feature expansion
* Logging of detected events
* Designed for Raspberry Pi deployment

---

## Requirements

Make sure you have the following installed on your Raspberry Pi:

* Python 3.7+
* OpenCV
* face_recognition
* numpy
* dlib (required by face_recognition)

Install dependencies:

```bash
pip install opencv-python face_recognition numpy dlib
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/USERNAME/NxV---Smart_Camera.git
```

2. Navigate into the project folder:

```bash
cd camera_project
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

Run the main application:

```bash
python3 main.py
```

The system will:

* Start the camera feed
* Detect motion or faces depending on configuration
* Display live output window

Press `q` to exit the program.

---

## How It Works

### 1. Camera Input

The Raspberry Pi camera captures real-time video frames.

### 2. Preprocessing

Frames are resized and converted to grayscale for efficient processing.

### 3. Detection Modules

* Motion detection compares frame differences
* Face detection uses Haar cascades or deep learning models

### 4. Recognition (Optional)

Known faces are matched using precomputed encodings stored in the `encodings/` folder.

### 5. Output

Detected events are displayed live and optionally logged.

---

## Future Improvements

* Weapon detection module
* Behavior analysis (suspicious activity detection)
* Cloud backup of recorded events
* Mobile alert system
* Improved AI-based object detection (YOLO integration)

---

## Author

Developed as part of a personal AI surveillance and computer vision project using Raspberry Pi.

---

## License

This project is propietary software 
