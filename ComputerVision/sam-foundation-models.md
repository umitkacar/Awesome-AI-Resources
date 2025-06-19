# SAM & Foundation Models

A comprehensive collection of resources for Segment Anything Model (SAM) and vision foundation models, including implementations, applications, and research materials.

## Table of Contents
- [Overview](#overview)
- [Segment Anything Model (SAM)](#segment-anything-model-sam)
- [Vision Foundation Models](#vision-foundation-models)
- [Research Papers](#research-papers)
- [Implementations & Code](#implementations--code)
- [Applications](#applications)
- [Datasets & Benchmarks](#datasets--benchmarks)
- [Tools & Extensions](#tools--extensions)
- [Tutorials & Learning Resources](#tutorials--learning-resources)
- [Related Projects](#related-projects)

## Overview

Foundation models in computer vision represent a paradigm shift in how we approach visual understanding tasks. These models, pre-trained on massive datasets, can be adapted to various downstream tasks with minimal fine-tuning.

## Segment Anything Model (SAM)

### Core Components
- **Image Encoder** - Vision Transformer (ViT) based architecture
- **Prompt Encoder** - Handles points, boxes, and masks
- **Mask Decoder** - Lightweight transformer for mask generation
- **Data Engine** - Iterative dataset creation process

### Key Features
- Zero-shot segmentation capability
- Multiple prompt types support
- Real-time inference
- High-quality mask generation
- Domain agnostic performance

### Official Resources
- **SAM Paper** - "Segment Anything" by Meta AI Research
- **GitHub Repository** - facebook/segment-anything
- **Model Checkpoints** - ViT-B, ViT-L, ViT-H variants
- **SA-1B Dataset** - 1 billion masks on 11 million images

### Model Variants
```python
# SAM model sizes
sam_vit_b: 91M parameters  # Base model
sam_vit_l: 308M parameters # Large model
sam_vit_h: 636M parameters # Huge model
```

## Vision Foundation Models

### Major Models

#### CLIP (Contrastive Language-Image Pre-training)
- Multi-modal understanding
- Zero-shot classification
- Text-image alignment
- OpenAI's breakthrough model

#### DINO (Self-Distillation with No Labels)
- Self-supervised learning
- Strong visual features
- DINOv2 improvements
- Meta AI Research

#### MAE (Masked Autoencoders)
- Reconstruction-based pre-training
- Efficient visual representation learning
- Scalable architecture

#### Florence
- Microsoft's unified vision model
- Multi-task capabilities
- Object detection, segmentation, captioning

#### ALIGN
- Google's large-scale vision-language model
- Noisy text supervision
- Billion-scale training

### Emerging Models
- **EVA** - Explore Visual Attention
- **BEiT** - BERT Pre-training of Image Transformers
- **SimMIM** - Simple Masked Image Modeling
- **ConvNeXt** - Modernized ConvNets
- **Swin Transformer** - Hierarchical vision transformer

## Research Papers

### Foundational Papers
1. **"An Image is Worth 16x16 Words"** - Vision Transformer (ViT)
2. **"Segment Anything"** - SAM introduction
3. **"Learning Transferable Visual Models"** - CLIP
4. **"Masked Autoencoders Are Scalable Vision Learners"** - MAE

### Recent Advances
- **SAM-HQ** - High-quality segmentation
- **MobileSAM** - Efficient mobile deployment
- **FastSAM** - Real-time segmentation
- **EfficientSAM** - Lightweight variants
- **PerSAM** - Personalized SAM

### Application Papers
- Medical image segmentation with SAM
- Remote sensing applications
- Video object segmentation
- 3D scene understanding
- Robotic perception

## Implementations & Code

### Official Implementations
```python
# SAM installation
pip install segment-anything

# Basic usage
from segment_anything import SamPredictor, sam_model_registry

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)
```

### Community Implementations
- **SAM-Track** - Video object tracking
- **Grounded-SAM** - Combining with language models
- **SAM-3D** - 3D segmentation
- **Medical-SAM** - Medical imaging adapter
- **Remote-SAM** - Satellite imagery

### Framework Support
- PyTorch (official)
- TensorFlow ports
- ONNX exports
- CoreML conversions
- TensorRT optimization

## Applications

### Computer Vision Tasks
- **Instance Segmentation** - Object-level masks
- **Semantic Segmentation** - Pixel classification
- **Panoptic Segmentation** - Combined approach
- **Interactive Segmentation** - User-guided
- **Video Segmentation** - Temporal consistency

### Domain-Specific Applications

#### Medical Imaging
- Organ segmentation
- Tumor detection
- Cell counting
- Surgical planning

#### Remote Sensing
- Building extraction
- Land cover mapping
- Change detection
- Disaster assessment

#### Autonomous Systems
- Object detection
- Path planning
- Obstacle avoidance
- Scene understanding

#### Creative Applications
- Image editing
- Background removal
- Content creation
- AR/VR applications

## Datasets & Benchmarks

### Training Datasets
- **SA-1B** - Segment Anything 1 Billion masks
- **COCO** - Common Objects in Context
- **ADE20K** - Scene parsing
- **Cityscapes** - Urban scene understanding
- **LVIS** - Large Vocabulary Instance Segmentation

### Evaluation Benchmarks
- Instance segmentation metrics (AP, AR)
- Semantic segmentation (mIoU)
- Boundary accuracy (F-score)
- Efficiency metrics (FPS, memory)

### Domain-Specific Datasets
- Medical: ISIC, BraTS, Kvasir
- Remote Sensing: SpaceNet, DOTA
- Autonomous Driving: nuScenes, Waymo

## Tools & Extensions

### Web Applications
- **Segment Anything Web Demo** - Browser-based interface
- **Roboflow SAM** - Integration platform
- **Label Studio SAM** - Annotation tool

### Development Tools
- **SAM-PT** - PyTorch helpers
- **SAM-ONNX** - ONNX conversion tools
- **SAM-Mobile** - Mobile deployment
- **SAM-Server** - API deployment

### Visualization Tools
- Mask overlay utilities
- Interactive notebooks
- Debugging visualizers
- Performance profilers

## Tutorials & Learning Resources

### Getting Started
- Official SAM tutorial
- Vision transformer basics
- Prompt engineering for SAM
- Fine-tuning strategies

### Advanced Topics
- Multi-GPU training
- Model compression
- Custom dataset preparation
- Production deployment

### Video Tutorials
- YouTube walkthroughs
- Conference presentations
- Technical deep dives
- Application showcases

### Courses & Workshops
- Computer Vision with Foundation Models
- Practical SAM Applications
- Vision Transformers Explained
- Deployment Best Practices

## Related Projects

### Integration Projects
- **LangSAM** - Language-guided segmentation
- **SAM-CLIP** - Combined model
- **Open-Vocabulary SAM** - Class-agnostic detection

### Enhanced Versions
- **HQ-SAM** - Higher quality masks
- **Lite-SAM** - Lightweight version
- **Speed-SAM** - Optimized inference
- **Robust-SAM** - Improved robustness

### Application Frameworks
- **Detectron2** - Facebook AI Research
- **MMSegmentation** - OpenMMLab
- **Transformers** - Hugging Face
- **Supervision** - Roboflow

---

*Originally from umitkacar/SAM-Foundation-Models repository*