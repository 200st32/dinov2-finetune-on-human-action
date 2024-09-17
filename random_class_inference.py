import argparse
import json
import os

import cv2
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.transforms import Resize, ConvertImageDtype, Normalize

import random
from matplotlib import pyplot as plt
import myutils

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

def my_plot_img(pridiction_list, class_name, class_path ,random_path):
    img_list = []
    for i in random_path:
        img_list.append(Image.open(class_path + i))
    fig = plt.figure(figsize=(10, 7))
    plt.title(f'class: {class_name}')
    rows = 2
    columns = 2
    
    fig.add_subplot(rows, columns, 1)
    plt.imshow(img_list[0]) 
    plt.axis('off') 
    plt.title(f"pridiction: {pridiction_list[0]}")

    fig.add_subplot(rows, columns, 2)
    plt.imshow(img_list[1])
    plt.axis('off')
    plt.title(f"pridiction: {pridiction_list[1]}")

    
    fig.add_subplot(rows, columns, 3)
    plt.imshow(img_list[2])
    plt.axis('off')
    plt.title(f"pridiction: {pridiction_list[2]}")

    
    fig.add_subplot(rows, columns, 4)
    plt.imshow(img_list[3])
    plt.axis('off')
    plt.title(f"pridiction: {pridiction_list[3]}")

    plt.savefig(f"./myoutput/class_result/{class_name}.png")
    plt.close()
    

def main(args):
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

    for class_name in class_label_map:
        class_path = args.image_path + class_name + '/'
        pridiction_list=[]
        all_path=[]
        for class_img in os.listdir(class_path):
            all_path.append(class_img)
        random_path =  random.sample(all_path, 4)
        for class_img in random_path:
            input_img = preprocess_image(class_path+class_img, device)
            # Inference
            with torch.no_grad():
                output = model(input_img)
            # Calculate the highest classification probability
            prediction_class_index = torch.topk(output, k=1).indices.squeeze(0).tolist()
            # Print classification results
            for class_index in prediction_class_index:
                prediction_class_label = class_label_map[class_index]
            pridiction_list.append(prediction_class_label)
        my_plot_img(pridiction_list, class_name, class_path ,random_path)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="torchvision")
    parser.add_argument("--model_mean_parameters", type=list, default=[0.485, 0.456, 0.406])
    parser.add_argument("--model_std_parameters", type=list, default=[0.229, 0.224, 0.225])
    parser.add_argument("--model_weights_path", type=str, default="./myoutput/40e_best_model.pth")
    parser.add_argument("--image_path", type=str, default="/home/cap6411.student1/CVsystem/assignment/hw5/human-action-recognition-dataset/Structured/test/")
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()
    main(args)






