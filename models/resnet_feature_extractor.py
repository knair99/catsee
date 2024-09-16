import os

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
        # Load and preprocess the image (convert to RGB to avoid alpha channel issues)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)  # Add batch dimension

        # Extract features using ResNet
        with torch.no_grad():
            features = self.model(image)
        return features.flatten().numpy()

    def extract_features_with_transformer(self, image_path):
        image = Image.open(image_path).convert('RGB')  # Load the image
        image = self.transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():  # Disable gradient calculation
            features = self.model(image)  # Extract features

        # Transform the features using the transformer
        features = features.view(1, -1, features.size(1))  # Reshape for transformer input
        transformed_features = self.transformer(features, features)  # Self-attention mechanism

        return transformed_features.flatten().detach().numpy()  # Detach from graph and convert to NumPy

    def extract_features_with_transformer_spatial_only(self, image_path):
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)

        # Extract spatial features from ResNet
        with torch.no_grad():
            spatial_features = self.model(image)  # Keep spatial dimensions (e.g., 7x7 grid)

        # Reshape spatial features for the transformer (sequence_length, batch_size, embedding_size)
        B, C, H, W = spatial_features.size()  # Batch, Channels, Height, Width
        spatial_features = spatial_features.view(B, C, -1).permute(2, 0, 1)  # (H*W, B, C)

        # Apply transformer
        transformed_features = self.transformer(spatial_features, spatial_features)

        return transformed_features.flatten().detach().numpy()  # Flatten and return


def retrieve_oxford_feature_dataset():
    # Instantiate the feature extractor
    resnet_feature_extractor_instance = ResNetWithTransformer()

    # Directory containing your images
    IMAGE_DIR = '/Users/kprasad/repos/catsee/static/training_corpus/images'

    # Example to extract features from all images in the dataset
    image_features = []
    image_labels = []

    # Read the annotation file
    with open('/Users/kprasad/repos/catsee/static/training_corpus/annotations/list.txt', 'r') as f:
        for line_number, line in enumerate(f.readlines()):

            # Skip the header row
            if line_number < 6:
                continue

            # Each line is structured as: image_name class_id species breed_id
            parts = line.strip().split()
            image_name = parts[0]
            species_id = int(parts[2])  # You can decide to use class_id, species, or breed_id

            # Extract features for each image
            image_path = os.path.join(IMAGE_DIR, image_name)
            image_path = image_path + '.jpg'  # Append the file extension
            features = resnet_feature_extractor_instance.extract_features(image_path)

            # Append features and labels
            image_features.append(features)
            image_labels.append(species_id)

    return image_features, image_labels  # Return the extracted features (X) and labels (y


if __name__ == '__main__':
    retrieve_oxford_feature_dataset()
