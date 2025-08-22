# MindScope--Facial-Emotion-Detection
Face-based emotion recognition with PyTorch and OpenCV

A real-time emotion detection project using Deep Learning and Computer Vision.
The model recognizes facial expressions from a webcam and classifies them into:
😠 Angry | 😄 Happy | 😐 Neutral | 😢 Sad | 😲 Surprise

📌 Features

Detects faces in real-time

Classifies 5 basic emotions

Displays both labels and emojis on faces

⚙️ How it Works

Captures video from webcam using OpenCV

Detects faces with Haar Cascade Classifier

Extracted face is preprocessed (grayscale, resize, tensor conversion)

A trained CNN model (PyTorch) predicts the emotion

Result is shown on the face with a colored box, label, and emoji

🛠️ Tech Used

Python

PyTorch – for building and training the CNN model

OpenCV – for real-time face detection and video processing

we can upload the data of the human faces making various expressions to train this model.

Torchvision & PIL – for image preprocessing

Haar Cascade – for face detection


