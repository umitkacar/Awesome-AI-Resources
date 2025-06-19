# AI Edge Computing & Tiny/Embedded Systems

A comprehensive guide to deploying AI models on edge devices, embedded systems, and resource-constrained environments. From smartphones to microcontrollers, this resource covers frameworks, tools, and best practices.

## Table of Contents
- [Introduction](#introduction)
- [Hardware Platforms](#hardware-platforms)
- [Frameworks & Tools](#frameworks--tools)
- [Model Optimization Techniques](#model-optimization-techniques)
- [Deployment Strategies](#deployment-strategies)
- [Use Cases & Applications](#use-cases--applications)
- [Performance Benchmarks](#performance-benchmarks)
- [Learning Resources](#learning-resources)
- [Community & Projects](#community--projects)

## Introduction

Edge AI represents the frontier of bringing intelligence directly to devices where data is generated. This eliminates cloud dependency, reduces latency, preserves privacy, and enables real-time decision making.

### Key Benefits
- **Low Latency** - Immediate inference without network delays
- **Privacy** - Data stays on device
- **Reliability** - Works offline
- **Cost Efficiency** - No cloud compute costs
- **Energy Efficiency** - Optimized for battery-powered devices

## Hardware Platforms

### Mobile Processors
- **Qualcomm Snapdragon** - Neural Processing Units (NPU)
- **Apple Silicon** - A-series with Neural Engine
- **MediaTek Dimensity** - APU for AI acceleration
- **Samsung Exynos** - Built-in NPU
- **Google Tensor** - Custom AI accelerator

### Edge AI Accelerators
- **NVIDIA Jetson Series**
  - Jetson Nano (5-10W)
  - Jetson Xavier NX (10-20W)
  - Jetson AGX Orin (15-60W)
- **Google Coral**
  - Edge TPU USB Accelerator
  - Dev Board Mini
  - System-on-Module
- **Intel Neural Compute Stick** - USB-based inference
- **Hailo-8** - 26 TOPS AI processor

### Embedded Platforms
- **Raspberry Pi** - Popular SBC with AI capabilities
- **Arduino Nano 33 BLE Sense** - TinyML platform
- **ESP32** - WiFi/BLE with AI support
- **STM32** - ARM Cortex-M with AI extensions
- **Nordic nRF52/53** - Bluetooth LE with Edge AI

### Microcontrollers for TinyML
```
Platform         | RAM      | Flash    | AI Framework Support
-----------------|----------|----------|---------------------
Arduino Nano 33  | 256KB    | 1MB      | TensorFlow Lite Micro
ESP32           | 520KB    | 4MB      | TensorFlow Lite, ESP-DL
STM32F7         | 512KB    | 2MB      | X-CUBE-AI, TFLite Micro
nRF52840        | 256KB    | 1MB      | TensorFlow Lite Micro
```

## Frameworks & Tools

### Mobile AI Frameworks

#### TensorFlow Lite
```python
# Convert model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```
- Supports Android, iOS, embedded Linux
- Hardware acceleration via delegates
- Model optimization toolkit

#### Core ML (iOS)
```swift
// Load Core ML model
let model = try VNCoreMLModel(for: YourModel().model)
```
- Apple's framework for on-device inference
- Automatic hardware optimization
- Create ML for training

#### ONNX Runtime Mobile
- Cross-platform deployment
- Multiple execution providers
- Reduced binary size

#### PyTorch Mobile
```python
# Export to mobile
scripted_model = torch.jit.script(model)
scripted_model.save("model.pt")
```
- Android and iOS support
- Selective build for size reduction

### Embedded AI Frameworks

#### TensorFlow Lite Micro
```cpp
// TFLite Micro inference
TfLiteStatus invoke_status = interpreter->Invoke();
```
- Runs on MCUs with KB of memory
- No dynamic memory allocation
- Subset of TFLite ops

#### Edge Impulse
- End-to-end platform for embedded ML
- Data collection to deployment
- Supports various MCU families

#### X-CUBE-AI (STM32)
- STM's AI expansion pack
- Optimized for STM32 MCUs
- CubeIDE integration

#### ESP-DL (Espressif)
- Optimized for ESP32 series
- Supports common DL operations
- Low memory footprint

### Model Conversion Tools
- **ONNX** - Universal model format
- **OpenVINO** - Intel's toolkit
- **Apache TVM** - Deep learning compiler
- **MLIR** - Multi-level IR

## Model Optimization Techniques

### Quantization
```python
# Post-training quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_types = [tf.int8]
```

#### Types of Quantization
- **Dynamic Quantization** - Weights only
- **Static Quantization** - Weights and activations
- **QAT (Quantization Aware Training)** - Train with quantization

### Pruning
```python
# Magnitude-based pruning
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0, final_sparsity=0.5)
}
model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
```

### Knowledge Distillation
- Train smaller student model from larger teacher
- Maintain accuracy with fewer parameters
- Effective for edge deployment

### Neural Architecture Search (NAS)
- **MobileNet** family - Efficient convolutions
- **EfficientNet** - Compound scaling
- **MnasNet** - Mobile NAS
- **FBNet** - Facebook's mobile models

### Model Compression Techniques
1. **Weight Sharing** - Cluster similar weights
2. **Low-Rank Factorization** - Decompose matrices
3. **Binary/Ternary Networks** - Extreme quantization
4. **Depthwise Separable Convolutions** - Reduce parameters

## Deployment Strategies

### Mobile Deployment

#### Android
```java
// TensorFlow Lite Android
Interpreter tflite = new Interpreter(modelBuffer);
tflite.run(inputBuffer, outputBuffer);
```

#### iOS
```swift
// Core ML iOS
let prediction = try model.prediction(input: modelInput)
```

### Embedded Deployment

#### Memory Management
- Static allocation preferred
- Memory pooling strategies
- Avoid heap fragmentation

#### Power Optimization
- Dynamic voltage/frequency scaling
- Sleep modes between inferences
- Batch processing when possible

### Edge Server Deployment
- Docker containers for edge
- Kubernetes edge distributions (K3s)
- AWS IoT Greengrass
- Azure IoT Edge

## Use Cases & Applications

### Computer Vision
- **Object Detection** - Person/vehicle detection
- **Face Recognition** - Biometric authentication
- **Image Classification** - Product recognition
- **Pose Estimation** - Fitness/gesture apps
- **OCR** - Text extraction

### Audio Processing
- **Wake Word Detection** - "Hey Siri" style
- **Speech Recognition** - Voice commands
- **Sound Classification** - Environmental monitoring
- **Voice Activity Detection** - Smart speakers

### Sensor Fusion & IoT
- **Predictive Maintenance** - Vibration analysis
- **Anomaly Detection** - Industrial monitoring
- **Activity Recognition** - Wearables
- **Environmental Sensing** - Air quality

### Healthcare
- **ECG Analysis** - Arrhythmia detection
- **Fall Detection** - Elderly care
- **Medication Adherence** - Smart pillboxes
- **Vital Signs Monitoring** - Continuous tracking

## Performance Benchmarks

### Inference Metrics
```
Model         | Platform      | Latency | Power  | Accuracy
--------------|---------------|---------|--------|----------
MobileNetV2   | Snapdragon 8  | 5ms     | 0.2W   | 71.8%
YOLOv5n       | Jetson Nano   | 20ms    | 5W     | 45.7 mAP
TinyBERT      | Raspberry Pi4 | 50ms    | 2.5W   | 82.3%
KeywordSpot   | Arduino Nano  | 40ms    | 0.02W  | 94.2%
```

### Optimization Impact
- Quantization: 4x model size reduction, 2-3x speedup
- Pruning: 10x compression possible with <1% accuracy loss
- Knowledge Distillation: 50% size reduction, minimal accuracy impact

## Learning Resources

### Books
- "TinyML" by Pete Warden & Daniel Situnayake
- "AI at the Edge" by Daniel Situnayake
- "Efficient Processing of Deep Neural Networks" by Vivienne Sze et al.

### Online Courses
- Coursera - "Introduction to Embedded Machine Learning"
- edX - "Applications of TinyML"
- Udacity - "Edge AI for IoT Developers"

### Tutorials & Workshops
- TensorFlow Lite tutorials
- Edge Impulse university
- NVIDIA DLI edge courses
- ARM's ML tutorials

### Research Papers
- "MobileNets: Efficient Convolutional Neural Networks"
- "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"
- "The Lottery Ticket Hypothesis"
- "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters"

## Community & Projects

### Open Source Projects
- **TensorFlow Lite Examples** - Reference implementations
- **NCNN** - Tencent's mobile framework
- **MNN** - Alibaba's mobile neural network
- **Paddle Lite** - Baidu's mobile inference

### Forums & Communities
- TinyML Foundation
- Edge AI and Vision Alliance
- r/embedded
- TensorFlow Lite Discord

### Competitions & Challenges
- TinyML Contest
- Edge AI Challenge
- Low-Power Computer Vision Challenge

### Industry Initiatives
- MLCommons Mobile/Edge benchmarks
- ONNX Edge Working Group
- Khronos NNEF standard

---

*Originally from umitkacar/ai-edge-computing-tiny-embedded repository*