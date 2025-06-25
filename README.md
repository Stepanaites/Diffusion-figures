# Generative Diffusion Model for Image Generation

This project involves the training and inference of a generative diffusion model designed for image generation tasks. The model is based on the Conditional U-Net architecture and has been trained on a custom dataset. The project includes two main scripts for training and generating images from the model.

## Project Overview

The goal of this project is to create a generative model capable of generating high-quality images conditioned on specific class labels. The model uses a diffusion process to iteratively denoise random noise into a meaningful image representation. This is achieved through the training of a Conditional U-Net, where class labels are embedded into the model architecture to guide the generation process.

## Features

* Training: The model is trained on a custom dataset to learn class-specific image generation. It uses a Conditional U-Net architecture to incorporate class information.

* Image Generation: After training, the model can generate images by starting from random noise and progressively refining the image through the denoising process.

* Diffusion Process: The model implements a multi-step denoising process that gradually transforms noisy input into a clear image, guided by the provided class label.

## Requirements

* Python 3.x
* PyTorch (with CUDA support if available)
* torchvision
* matplotlib
* tqdm

Install the required dependencies using:

pip install torch torchvision matplotlib tqdm

## Usage

### 1. Training

The training script (diffus_train.py) trains the Conditional U-Net on a custom dataset. The dataset should be organized in directories representing different classes, and the images should be resized to 28x28 pixels.

To start training the model, simply run:

python diffus_train.py

The script will save the model checkpoints after every epoch, so you can resume training or perform inference at any point. It also saves a plot of the training loss.

### 2. Image Generation

Once the model is trained, you can use the diffus_test.py script to generate images. The model will take a class label as input and produce an image of that class.

To generate an image, run:

python diffus_test.py

The script will ask for a class index (0-3) and generate an image based on the trained model. The generated image will be displayed using matplotlib.

### 3. Model Checkpoints

During training, the model's state (weights) and optimizer state are saved in a checkpoint file (diffusion_model_epoch_X.pth). You can load the checkpoint at any point to resume training or perform inference.

## Architecture

The model uses a Conditional U-Net architecture, which consists of an encoder-decoder structure with skip connections. The model incorporates class-specific embeddings to guide the image generation process. The input is noisy, and the model progressively refines it using a denoising process.

## Dataset

The model was trained on a custom dataset where the images are organized into different class directories. The dataset was preprocessed with the following transformations:

* Grayscale conversion
* Resizing to 28x28 pixels
* Conversion to tensor format

## Conclusion
This project demonstrates how to build a generative diffusion model for class-conditioned image generation using a custom dataset. The combination of diffusion and class embeddings allows for the generation of images with specific characteristics based on class labels.