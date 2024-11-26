# Face Recognition Attendance System

## Overview
This project is a **Face Recognition Attendance System** built with Python, OpenCV, and the `face_recognition` library. It automates the attendance marking process by detecting faces through a camera feed and recording attendance in a CSV file.

---

## Features
- Real-time face recognition using `dlib` and `face_recognition` libraries.
- Attendance marking in a CSV file with lecture-specific details.
- GUI for entering lecture information using `Tkinter`.
- Optimized face recognition with:
  - Scaled image processing for faster execution.
  - Rolling buffer system to reduce false positives.
  - Selective frame processing for performance improvement.
- Auto-detection of new or removed training images and encoding updates.

---

