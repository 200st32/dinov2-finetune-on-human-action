import argparse
import json
import os

import cv2
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.transforms import Resize, ConvertImageDtype, Normalize
from functools import partial
import myutils

import sys
sys.path.append('dinov2')

from dinov2.eval.linear import create_linear_input
from dinov2.eval.linear import LinearClassifier
from dinov2.eval.utils import ModelWithIntermediateLayers
from dinov2.models.vision_transformer import vit_small 


def preprocess_image(image_path: str, device: torch.device) -> torch.Tensor:
    image = Image.open(image_path)
    
    data_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    tensor = data_transforms(image)
    tensor = tensor.unsqueeze(0)
    # Transfer tensor channel image format data to CUDA device
    tensor = tensor.to(device=device, memory_format=torch.channels_last, non_blocking=True)

    return tensor


def main():
    # Get the label name corresponding to the drawing
    class_label_map = [
        "calling",
        "clapping",
        "cycling",
        "dancing",
        "drinking",
        "eating",
        "fighting",
        "hugging",
        "laughing",
        "listening_to_music",
        "running",
        "sitting",
        "sleeping",
        "texting",
        "using_laptop"         
    ]

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")

    # Initialize the model
    model = myutils.load_best_model()
    # Load model weights
    model.load_state_dict(torch.load(args.model_weights_path, weights_only=True))

    # Start the verification mode of the model.
    model.eval()

    input_img = preprocess_image(args.image_path, device)

    # Inference
    with torch.no_grad():
        features = model.feature_model(input_img)
        output = model.classifier(features)

    # Calculate the highest classification probability
    prediction_class_index = torch.topk(output, k=3).indices.squeeze(0).tolist()

    # Print classification results
    for class_index in prediction_class_index:
        prediction_class_label = class_label_map[class_index]
        prediction_class_prob = torch.softmax(output, dim=1)[0, class_index].item()
        print(f"{prediction_class_label}: {prediction_class_prob * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_weights_path", type=str, default="./myoutput/40e_best_model.pth")
    parser.add_argument("--image_path", type=str, default="/home/cap6411.student1/CVsystem/assignment/hw5/human-action-recognition-dataset/Structured/test/clapping/Image_10614.jpg")
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    main()
