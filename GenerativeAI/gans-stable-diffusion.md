# Awesome GANs & Stable Diffusion

A comprehensive collection of resources, papers, implementations, and tools for Generative Adversarial Networks (GANs) and Stable Diffusion models.

## Table of Contents
- [Introduction](#introduction)
- [GANs Resources](#gans-resources)
- [Stable Diffusion Resources](#stable-diffusion-resources)
- [Research Papers](#research-papers)
- [Implementations](#implementations)
- [Tools & Applications](#tools--applications)
- [Tutorials & Courses](#tutorials--courses)
- [Datasets](#datasets)
- [Community & Discussion](#community--discussion)

## Introduction

This repository curates the best resources for understanding and working with GANs and Stable Diffusion models. Whether you're a researcher, practitioner, or enthusiast, you'll find valuable materials to advance your knowledge and projects.

## GANs Resources

### Foundational Papers
- **Generative Adversarial Networks** (Goodfellow et al., 2014) - The original GAN paper
- **DCGAN** - Deep Convolutional GANs
- **WGAN** - Wasserstein GAN for improved training stability
- **StyleGAN** - High-quality image synthesis with style control
- **CycleGAN** - Unpaired image-to-image translation
- **BigGAN** - Large scale GAN training

### Popular Implementations
- **PyTorch-GAN** - Collection of PyTorch implementations
- **TensorFlow-GAN** - TF-GAN library
- **StyleGAN2-ADA** - Official NVIDIA implementation
- **Progressive GAN** - Growing GANs progressively

### Architecture Variants
- Conditional GAN (cGAN)
- Pix2Pix
- InfoGAN
- SRGAN (Super-Resolution)
- ProGAN
- StyleGAN3

## Stable Diffusion Resources

### Core Models
- **Stable Diffusion v1.5** - The classic model
- **Stable Diffusion v2.1** - Improved architecture
- **SDXL** - Stable Diffusion XL for higher resolution
- **SDXL Turbo** - Faster inference
- **Stable Diffusion 3** - Latest iteration

### Key Components
- **CLIP** - Text encoder for prompt understanding
- **VAE** - Variational Autoencoder for latent space
- **U-Net** - Denoising architecture
- **Scheduler** - Sampling algorithms (DDIM, DPM++, etc.)

### Popular Interfaces
- **AUTOMATIC1111 WebUI** - Most popular web interface
- **ComfyUI** - Node-based workflow interface
- **InvokeAI** - Professional creative tool
- **Diffusers** - Hugging Face library

## Research Papers

### Seminal Works
1. **Denoising Diffusion Probabilistic Models** (2020)
2. **High-Resolution Image Synthesis with Latent Diffusion Models** (2022)
3. **Photorealistic Text-to-Image Diffusion Models** (2022)
4. **Classifier-Free Diffusion Guidance** (2021)

### Recent Advances
- ControlNet - Adding spatial control
- LoRA - Low-Rank Adaptation for fine-tuning
- DreamBooth - Personalization techniques
- InstructPix2Pix - Instruction-based editing

## Implementations

### Python Libraries
```python
# Diffusers (Hugging Face)
from diffusers import StableDiffusionPipeline

# PyTorch implementations
import torch
from torchvision import transforms

# TensorFlow/Keras
import tensorflow as tf
```

### Pre-trained Models
- Hugging Face Model Hub
- Civitai - Community models
- RunwayML models
- CompVis checkpoints

## Tools & Applications

### Image Generation
- **Text-to-Image** - Generate images from text prompts
- **Image-to-Image** - Modify existing images
- **Inpainting** - Fill in missing parts
- **Outpainting** - Extend images beyond borders

### Advanced Techniques
- **Fine-tuning Methods**
  - LoRA/LyCORIS
  - Textual Inversion
  - Hypernetworks
  - DreamBooth

- **Control Methods**
  - ControlNet
  - T2I-Adapter
  - IP-Adapter
  - InstantID

### Optimization Tools
- **xFormers** - Memory-efficient attention
- **TensorRT** - NVIDIA optimization
- **ONNX** - Cross-platform deployment
- **Core ML** - Apple Silicon optimization

## Tutorials & Courses

### Beginner Resources
- Understanding GANs - Interactive visualization
- Stable Diffusion explained
- Prompt engineering guide
- Basic fine-tuning tutorials

### Advanced Topics
- Custom model training
- Architecture modifications
- Multi-GPU training strategies
- Production deployment

### Video Courses
- Fast.ai Diffusion course
- Stanford CS236 Deep Generative Models
- Two Minute Papers explanations

## Datasets

### Common Training Datasets
- **LAION-5B** - Large-scale image-text pairs
- **COCO** - Common Objects in Context
- **ImageNet** - Classification dataset
- **CelebA** - Celebrity faces
- **FFHQ** - High-quality faces

### Specialized Datasets
- Art collections
- Medical imaging
- Satellite imagery
- 3D renders

## Community & Discussion

### Forums & Communities
- **r/StableDiffusion** - Reddit community
- **Hugging Face Discord**
- **LAION Discord**
- **GitHub Discussions**

### Conferences & Workshops
- NeurIPS
- CVPR
- ICCV
- SIGGRAPH

### Notable Researchers
- Ian Goodfellow (GANs)
- Robin Rombach (Stable Diffusion)
- Emad Mostaque (Stability AI)
- Various teams at OpenAI, Google, Meta

---

*Originally from umitkacar/awesome-GANs-Stable-Diffusion repository*