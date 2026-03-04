# AI Driver Monitoring System

A real-time computer vision system that detects driver drowsiness and distractions using OpenCV and YOLOv8.

## Features

- Face detection
- Phone usage detection
- Eye closure detection
- Multi-frame confirmation for alerts
- Face-phone overlap validation
- Real-time attention score monitoring

## Technologies Used

- Python
- OpenCV
- YOLOv8
- NumPy

## How it Works

1. Camera captures driver video.
2. YOLOv8 detects objects like phone and face.
3. Eye closure and head pose are analyzed.
4. Multi-frame confirmation ensures accurate alerts.
5. Warning is triggered when unsafe behavior is detected.

## Installation

### Python Version
This project was developed using:

Python 3.10+

Check your version:

```bash
python --version

## Clone the Repository

git clone https://github.com/vishtodev/AI-Driver-Monitoring-System.git
cd AI-Driver-Monitoring-System

##Install Dependencies

pip install -r requirements.txt

**Download yolov8s.pt from Ultralytics official release. If unable to download from git.

##Run the Project

```bash
python main.py
