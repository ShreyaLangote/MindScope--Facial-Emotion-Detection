import cv2
import torch
import torchvision.transforms as transforms
from models.cnn import EmotionCNN
from PIL import Image

# Load the trained model
model = EmotionCNN()
model.load_state_dict(torch.load("emotion_model.pth", map_location=torch.device('cpu')))
model.eval()

# Emotion classes, emojis, and colors
classes = ['angry', 'happy', 'neutral', 'sad', 'surprise']
emojis = {
    'angry': "😠",
    'happy': "😄",
    'neutral': "😐",
    'sad': "😢",
    'surprise': "😲"
}
colors = {
    'angry': (0, 0, 255),
    'happy': (0, 255, 0),
    'neutral': (255, 255, 0),
    'sad': (255, 0, 0),
    'surprise': (255, 0, 255)
}

# Face detection using Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        try:
            input_tensor = transform(face_img).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                prediction = torch.argmax(output, 1).item()
                emotion = classes[prediction]
                emoji = emojis[emotion]
                color = colors[emotion]

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Put emoji above the face
            emoji_pos = (x + w//2 - 20, y - 20)
            cv2.putText(frame, emoji, emoji_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)

            # Show label below the face
            label_pos = (x, y + h + 30)
            cv2.putText(frame, emotion.upper(), label_pos, cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)
        except Exception as e:
            print("Error in processing face:", e)

    # Display frame
    cv2.imshow("🧠 Emotion Detector", frame)

    # Quit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
