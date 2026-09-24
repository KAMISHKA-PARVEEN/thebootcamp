# Day 04: Computer Vision & Natural Language Processing (NLP)

This folder contains all the Computer Vision and Natural Language Processing projects developed during **Day 04** of the AI Bootcamp.

---

## File Descriptions

| File Name | Topic / Domain | Description |
| :--- | :--- | :--- |
| **`day_04.py`** | OpenCV Image Processing | Basic image manipulation using OpenCV: reading image files, resizing dimensions, applying horizontal/vertical flips, and cropping Regions of Interest (ROI). |
| **`draw_canvas.py`** | Computer Vision (Air Canvas) | Real-time virtual drawing board using OpenCV webcam color tracking (HSV masking, morphological filtering, and contour centroid tracking). |
| **`face_detection.py`** | Computer Vision (Biometrics) | Real-time webcam face detection using OpenCV's pre-trained Haar Cascade Classifier (`haarcascade_frontalface_default.xml`). |
| **`object_detection.py`** | Deep Learning (YOLOv8) | Real-time multi-object detection and live object counting on webcam streams using the lightweight YOLOv8 Nano model (`yolov8n.pt`). |
| **`yolov8n.pt`** | Model Weights | Pre-trained weights for the YOLOv8 Nano object detection model (~6.5 MB). |
| **`sentiment_analysis.py`** | NLP (Sentiment Analysis) | Interactive real-time sentiment analysis CLI tool built with TextBlob to compute polarity (-1.0 to +1.0) and subjectivity scores. |
| **`word_cloud.py`** | NLP & Visualization | Text processing pipeline using NLTK: tokenization, stopword removal, word frequency counting, and visual WordCloud generation with Matplotlib. |

---

## How to Run

Navigate into the `day_04` folder or execute from the repository root:

```powershell
python day_04/day_04.py
python day_04/draw_canvas.py
python day_04/face_detection.py
python day_04/object_detection.py
python day_04/sentiment_analysis.py
python day_04/word_cloud.py
```
