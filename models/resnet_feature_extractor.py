import torch
from torchvision import models, transforms
from PIL import Image


class ResnetFeatureExtractor:
    def __init__(self):
        self.model = models.resnet18(pretrained=True)  # Load the pre-trained model
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))  # Remove the last layer
        self.model.eval()
        self.transform = transforms.Compose([  # Define the image transformations
            transforms.Resize((224, 224)),  # Resize the image
            transforms.ToTensor(),  # Convert to PyTorch tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize the image
        ])

    def extract_features(self, image_path):
        image = Image.open(image_path)  # Load the image
        image = self.transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():  # Disable gradient calculation
            features = self.model(image)  # Extract features
        return features.squeeze().numpy()  # Remove batch dimension and convert to NumPy array for CatBoost