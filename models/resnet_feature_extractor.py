import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


class ResNetWithTransformer:
    def __init__(self):
        self.model = models.resnet18(pretrained=True)  # Load the pre-trained model
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))  # Remove the last layer

        # Define a transformer for further processing the extracted features
        self.transformer = nn.Transformer(nhead=4, num_encoder_layers=2)  # Basic transformer setup
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

        # Transform the features using the transformer
        features = features.view(1, -1, features.size(1))  # Reshape for transformer input
        transformed_features = self.transformer(features, features)  # Self-attention mechanism

        return transformed_features.flatten().detach().numpy()  # Detach from graph and convert to NumPy

