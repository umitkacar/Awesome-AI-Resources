# NCNN Mobile AI Framework

A comprehensive collection of NCNN (Neural Network Computing) framework resources for mobile and embedded AI deployment.

**Last Updated:** 2025-06-19

## Table of Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
- [Pre-trained Models](#pre-trained-models)
- [Model Conversion](#model-conversion)
- [Platform-Specific Guides](#platform-specific-guides)
- [Optimization Techniques](#optimization-techniques)
- [Example Projects](#example-projects)
- [Performance Benchmarks](#performance-benchmarks)

## Overview

NCNN is a high-performance neural network inference framework optimized for mobile platforms, developed by Tencent. It's designed to be lightweight, fast, and energy-efficient.

### Key Features
- **Ultra Lightweight**: Minimal memory footprint
- **High Performance**: Optimized for ARM NEON and GPU
- **Cross-Platform**: iOS, Android, Windows, Linux, MacOS
- **No Third-Party Dependencies**: Self-contained
- **Production Ready**: Used in 40+ Tencent apps

## Getting Started

### Installation
**[Official NCNN Repository](https://github.com/Tencent/ncnn)** - Main framework
- 🆓 Open Source
- C++ implementation
- Vulkan GPU support
- INT8 quantization

**[NCNN Android Studio](https://github.com/nihui/ncnn-android-studio-project)** - Android template
- Ready-to-use project
- JNI examples
- Build configurations

**[NCNN iOS](https://github.com/nihui/ncnn-ios-benchmark)** - iOS benchmark app
- Performance testing
- Model comparison
- Device profiling

### Quick Start Guides
**[NCNN Wiki](https://github.com/Tencent/ncnn/wiki)** - Official documentation
- 🟢 Beginner friendly
- Step-by-step tutorials
- API reference
- FAQ section

**[How to Build](https://github.com/Tencent/ncnn/wiki/how-to-build)** - Platform build guides
- Android NDK setup
- iOS toolchain
- Cross-compilation
- CMake configuration

## Pre-trained Models

### Model Zoo
**[NCNN Model Zoo](https://github.com/nihui/ncnn-assets)** - Pre-converted models
- 🆓 Free to use
- 100+ models
- Ready for deployment
- Optimized weights

### Popular Models
**Classification Models:**
- MobileNet V1/V2/V3
- ShuffleNet V1/V2
- SqueezeNet
- EfficientNet-Lite
- ResNet18/34/50

**Detection Models:**
- YOLOv5/v7/v8 - **[ncnn-yolov5](https://github.com/nihui/ncnn-yolov5)**
- NanoDet - **[ncnn-nanodet](https://github.com/nihui/ncnn-nanodet)**
- YOLOX - **[ncnn-yolox](https://github.com/FeiGeChuanShu/ncnn-yolox)**
- MobileNet-SSD

**Segmentation Models:**
- U-Net variants
- BiSeNet
- FastSCNN
- Mobile-SAM adaptations

**Face Models:**
- RetinaFace - **[ncnn-retinaface](https://github.com/nihui/ncnn-retinaface)**
- MTCNN
- Face landmarks
- Face recognition

## Model Conversion

### From PyTorch
**[PNNX](https://github.com/pnnx/pnnx)** - PyTorch to NCNN converter
- 🟢 Recommended tool
- Direct conversion
- Operator support
- Model optimization

**Conversion Steps:**
```bash
# 1. Export PyTorch to TorchScript
# 2. Convert with PNNX
# 3. Optimize for mobile
# 4. Quantize (optional)
```

### From ONNX
**[onnx2ncnn](https://github.com/Tencent/ncnn/tree/master/tools/onnx)** - ONNX converter
- Built-in tool
- Wide operator support
- Custom layer guide
- Debug options

### From TensorFlow
**[tensorflow2ncnn](https://github.com/hanzy88/tensorflow2ncnn)** - TF converter
- TFLite support
- Keras models
- SavedModel format

### From Other Frameworks
- **Caffe**: caffe2ncnn tool
- **MXNet**: mxnet2ncnn
- **DarkNet**: darknet2ncnn

## Platform-Specific Guides

### Android Development
**[NCNN Android Examples](https://github.com/nihui/ncnn-android-examples)** - Sample apps
- Camera integration
- Real-time inference
- UI examples
- Performance tips

**Best Practices:**
- Use native C++ for performance
- Implement JNI carefully
- Handle memory management
- Profile on target devices

### iOS Development
**[NCNN iOS Camera](https://github.com/zchrissirhcz/ncnn-ios-cam)** - Camera example
- Swift integration
- Metal GPU support
- CoreML comparison
- Battery optimization

**Integration Tips:**
- Use Objective-C++ wrapper
- Enable GPU acceleration
- Handle background mode
- Memory warnings

### Embedded Linux
**[NCNN Raspberry Pi](https://github.com/nihui/ncnn-raspberry-pi)** - RPi examples
- ARM optimization
- GPIO integration
- Power efficiency
- Heat management

## Optimization Techniques

### Quantization
**INT8 Quantization:**
- 4x model size reduction
- 2-4x speedup
- Minimal accuracy loss
- **[Quantization Guide](https://github.com/Tencent/ncnn/wiki/quantized-int8-inference)**

### Model Pruning
**Channel Pruning:**
- Remove redundant channels
- Maintain accuracy
- Further size reduction
- Speed improvements

### Custom Operators
**[Custom Layer Guide](https://github.com/Tencent/ncnn/wiki/how-to-implement-custom-layer-step-by-step)** - Extend NCNN
- 🔴 Advanced
- C++ implementation
- GPU kernels
- Registration process

## Example Projects

### Real-World Applications
**[Real-ESRGAN-ncnn](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan)** - Image enhancement
- Super resolution
- Cross-platform
- GPU accelerated
- Production ready

**[AnimeGAN-ncnn](https://github.com/nihui/animegan-ncnn-vulkan)** - Style transfer
- Real-time performance
- Artistic filters
- Mobile optimized

**[DAIN-ncnn](https://github.com/nihui/dain-ncnn-vulkan)** - Video interpolation
- Frame interpolation
- Smooth slow motion
- GPU required

### Community Projects
**[ncnn-webassembly](https://github.com/nihui/ncnn-webassembly)** - Browser deployment
- WebAssembly port
- JavaScript API
- Demo applications

**[Flutter-ncnn](https://github.com/KoheiKanagu/flutter_ncnn)** - Flutter plugin
- Cross-platform UI
- Dart bindings
- Example apps

## Performance Benchmarks

### Device Comparisons
**[NCNN Benchmark](https://github.com/nihui/ncnn-benchmark)** - Performance data
- 500+ devices tested
- CPU vs GPU
- Power consumption
- Temperature data

### Optimization Results
**Typical Improvements:**
- FP32 → INT8: 3-4x speedup
- CPU → GPU: 2-10x speedup
- Model pruning: 30-50% faster
- Custom ops: Case dependent

## Best Practices

### Development Tips
1. **Profile First**: Measure before optimizing
2. **Target Devices**: Test on actual hardware
3. **Memory Management**: Avoid leaks, reuse buffers
4. **Thread Safety**: Handle multi-threading carefully
5. **Error Handling**: Graceful degradation

### Deployment Checklist
- [ ] Model converted successfully
- [ ] Accuracy verified
- [ ] Performance acceptable
- [ ] Memory usage optimized
- [ ] Battery impact measured
- [ ] Edge cases handled
- [ ] Crash reporting added

## Resources & Community

### Learning Materials
**[NCNN Tutorial Series](https://zhuanlan.zhihu.com/p/534123169)** - Chinese tutorials
- Detailed explanations
- Code walkthroughs
- Architecture deep dive

**[Mobile AI Workshop](https://github.com/mobile-ai-workshop)** - Workshop materials
- Hands-on exercises
- Slides and videos
- Sample solutions

### Community
**[NCNN Discord](https://discord.gg/ncnn)** - Community chat
- Technical support
- Project showcase
- Job opportunities

**[NCNN QQ Group](https://github.com/Tencent/ncnn#qq-group)** - Chinese community
- 1000+ members
- Daily discussions
- Expert answers

## Related Frameworks
- **[MNN](https://github.com/alibaba/MNN)** - Alibaba's framework
- **[TNN](https://github.com/Tencent/TNN)** - Tencent's newer framework
- **[Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite)** - Baidu's solution
- **[MediaPipe](https://mediapipe.dev/)** - Google's framework