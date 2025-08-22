import torch
from torchvision import transforms
from PIL import Image
from models.cnn import EmotionCNN
from termcolor import colored

# Load model
model = EmotionCNN()
model.load_state_dict(torch.load("emotion_model.pth", map_location=torch.device('cpu')))
model.eval()

# Emotion classes and emojis
classes = ['angry', 'happy', 'neutral', 'sad', 'surprise']
emojis = {
    'angry': "😠",
    'happy': "😊",
    'neutral': "😐",
    'sad': "😢",
    'surprise': "😲"
}

# Load and transform the image
image_path = "test.jpg"
image = Image.open(image_path).convert('L')  # grayscale
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])
image = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)
    emotion = classes[predicted.item()]
    emoji = emojis[emotion]

# Color mapping
color_map = {
    'angry': 'red',
    'happy': 'green',
    'neutral': 'yellow',
    'sad': 'blue',
    'surprise': 'magenta'
}

# Print engaging output
print(colored(f"\nDetected Emotion: {emotion.upper()} {emoji}\n", color_map[emotion], attrs=['bold']))
