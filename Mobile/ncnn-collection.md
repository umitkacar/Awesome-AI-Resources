# 🚀 Awesome NCNN Collection

**Last Updated:** 2025-06-19

## Overview
NCNN is a high-performance neural network inference framework optimized for mobile platforms, developed by Tencent. This collection covers everything from basic usage to advanced optimization techniques.

## 🎯 What is NCNN?

### Key Features
- **Ultra Lightweight**: Minimal memory footprint
- **Platform Support**: Android, iOS, Windows, Linux, macOS
- **No Third-party Dependencies**: Pure C++ implementation
- **ARM Optimization**: NEON and advanced SIMD support
- **Vulkan Support**: GPU acceleration on mobile
- **Model Security**: Encrypted model support

### Performance Highlights
- **Speed**: 2-4x faster than TensorFlow Lite on ARM
- **Size**: ~500KB binary size
- **Memory**: Efficient memory management
- **Power**: Low power consumption
- **Precision**: FP32, FP16, INT8 support

## 🔧 Getting Started

### Installation
```bash
# Clone repository
git clone https://github.com/Tencent/ncnn.git
cd ncnn

# Build for different platforms
mkdir build && cd build

# Android build
cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI="arm64-v8a" \
      -DANDROID_PLATFORM=android-21 \
      -DNCNN_VULKAN=ON \
      ..

# iOS build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/ios.toolchain.cmake \
      -DPLATFORM=OS64 \
      -DNCNN_VULKAN=ON \
      ..

make -j8
```

### Basic Usage
```cpp
#include "net.h"

int main() {
    ncnn::Net net;
    
    // Load model
    net.load_param("model.param");
    net.load_model("model.bin");
    
    // Prepare input
    ncnn::Mat in(224, 224, 3);
    // ... fill input data ...
    
    // Create extractor
    ncnn::Extractor ex = net.create_extractor();
    
    // Set input
    ex.input("data", in);
    
    // Forward
    ncnn::Mat out;
    ex.extract("output", out);
    
    // Process output
    // ... process results ...
    
    return 0;
}
```

## 🤖 Model Conversion

### From Different Frameworks
```python
# PyTorch to NCNN
import torch
import torchvision

# Export to ONNX first
model = torchvision.models.resnet18(pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "resnet18.onnx")

# Then use onnx2ncnn tool
# ./onnx2ncnn resnet18.onnx resnet18.param resnet18.bin
```

### Model Optimization
```bash
# Optimize for mobile
./ncnnoptimize model.param model.bin \
    model_opt.param model_opt.bin \
    65536  # FP16 inference

# INT8 quantization
./ncnn2table model_opt.param model_opt.bin \
    imagelist.txt model.table

./ncnn2int8 model_opt.param model_opt.bin \
    model_int8.param model_int8.bin \
    model.table
```

## 📱 Mobile Integration

### Android Implementation
```java
public class NCNNWrapper {
    static {
        System.loadLibrary("ncnn");
        System.loadLibrary("ncnn_jni");
    }
    
    private long nativePtr = 0;
    
    public native boolean loadModel(
        AssetManager assetManager, 
        String paramPath, 
        String binPath
    );
    
    public native float[] detect(Bitmap bitmap);
    
    public native void release();
}
```

```cpp
// JNI Implementation
JNIEXPORT jboolean JNICALL
Java_com_example_NCNNWrapper_loadModel(
    JNIEnv *env, 
    jobject thiz,
    jobject assetManager, 
    jstring paramPath, 
    jstring binPath
) {
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    
    const char* param_path = env->GetStringUTFChars(paramPath, nullptr);
    const char* bin_path = env->GetStringUTFChars(binPath, nullptr);
    
    ncnn::Net* net = new ncnn::Net();
    net->opt.use_vulkan_compute = true;
    
    int ret = net->load_param(mgr, param_path);
    ret |= net->load_model(mgr, bin_path);
    
    env->ReleaseStringUTFChars(paramPath, param_path);
    env->ReleaseStringUTFChars(binPath, bin_path);
    
    // Store pointer
    jclass clazz = env->GetObjectClass(thiz);
    jfieldID fid = env->GetFieldID(clazz, "nativePtr", "J");
    env->SetLongField(thiz, fid, (jlong)net);
    
    return ret == 0;
}
```

### iOS Implementation
```objc
// Objective-C++ Wrapper
@interface NCNNModel : NSObject
- (BOOL)loadModel:(NSString*)paramPath binPath:(NSString*)binPath;
- (NSArray*)predict:(UIImage*)image;
@end

@implementation NCNNModel {
    ncnn::Net* net;
}

- (BOOL)loadModel:(NSString*)paramPath binPath:(NSString*)binPath {
    net = new ncnn::Net();
    
    net->opt.use_vulkan_compute = ncnn::get_gpu_count() > 0;
    
    int ret = net->load_param([paramPath UTF8String]);
    ret |= net->load_model([binPath UTF8String]);
    
    return ret == 0;
}

- (NSArray*)predict:(UIImage*)image {
    // Convert UIImage to ncnn::Mat
    ncnn::Mat in = [self imageToMat:image];
    
    // Inference
    ncnn::Extractor ex = net->create_extractor();
    ex.input("data", in);
    
    ncnn::Mat out;
    ex.extract("output", out);
    
    // Convert output to NSArray
    return [self matToArray:out];
}

@end
```

## 🚀 Advanced Features

### Custom Layers
```cpp
class CustomLayer : public ncnn::Layer {
public:
    CustomLayer() {
        one_blob_only = true;
        support_inplace = true;
    }
    
    virtual int load_param(const ncnn::ParamDict& pd) {
        scale = pd.get(0, 1.f);
        return 0;
    }
    
    virtual int forward_inplace(ncnn::Mat& bottom_top_blob, 
                               const ncnn::Option& opt) const {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int c = 0; c < channels; c++) {
            float* ptr = bottom_top_blob.channel(c);
            for (int i = 0; i < w * h; i++) {
                ptr[i] *= scale;
            }
        }
        
        return 0;
    }
    
private:
    float scale;
};

DEFINE_LAYER_CREATOR(CustomLayer)
```

### Vulkan Compute Shaders
```glsl
#version 450

layout (binding = 0) readonly buffer bottom_blob { float bottom_data[]; };
layout (binding = 1) writeonly buffer top_blob { float top_data[]; };

layout (push_constant) uniform parameter {
    int w;
    int h;
    int c;
    int cstep;
} p;

void main() {
    int gx = int(gl_GlobalInvocationID.x);
    int gy = int(gl_GlobalInvocationID.y);
    int gz = int(gl_GlobalInvocationID.z);
    
    if (gx >= p.w || gy >= p.h || gz >= p.c)
        return;
    
    int offset = gz * p.cstep + gy * p.w + gx;
    
    float v = bottom_data[offset];
    top_data[offset] = max(v, 0.0f); // ReLU
}
```

## 📊 Performance Optimization

### ARM NEON Optimization
```cpp
// Optimized convolution
void conv3x3s1_neon(const float* bottom, float* top, 
                    const float* kernel, int w, int h) {
    for (int i = 0; i < h - 2; i++) {
        for (int j = 0; j < w - 2; j += 4) {
            float32x4_t sum = vdupq_n_f32(0.f);
            
            for (int m = 0; m < 3; m++) {
                for (int n = 0; n < 3; n++) {
                    float32x4_t a = vld1q_f32(bottom + (i+m)*w + j+n);
                    float32x4_t k = vdupq_n_f32(kernel[m*3+n]);
                    sum = vmlaq_f32(sum, a, k);
                }
            }
            
            vst1q_f32(top + i*w + j, sum);
        }
    }
}
```

### Memory Optimization
```cpp
// Custom allocator for memory pool
class PoolAllocator : public ncnn::Allocator {
public:
    PoolAllocator() {
        // Pre-allocate memory pool
        pool_size = 64 * 1024 * 1024; // 64MB
        pool = (unsigned char*)ncnn::fastMalloc(pool_size);
    }
    
    virtual void* fastMalloc(size_t size) {
        // Allocate from pool
        if (current + size <= pool_size) {
            void* ptr = pool + current;
            current += align_size(size, 16);
            return ptr;
        }
        return ncnn::fastMalloc(size);
    }
    
    virtual void fastFree(void* ptr) {
        // Check if in pool
        if (ptr >= pool && ptr < pool + pool_size) {
            // Do nothing - pool memory
            return;
        }
        ncnn::fastFree(ptr);
    }
    
private:
    unsigned char* pool;
    size_t pool_size;
    size_t current = 0;
};
```

## 💡 Best Practices

### Model Design
1. **Quantization-aware Training**: Train with quantization in mind
2. **Architecture Choice**: Use mobile-friendly architectures
3. **Layer Fusion**: Fuse operations when possible
4. **Pruning**: Remove unnecessary connections

### Deployment
1. **Model Encryption**: Protect intellectual property
2. **Error Handling**: Graceful fallbacks
3. **Resource Management**: Monitor memory usage
4. **Testing**: Test on various devices

## 🔗 Resources

### Official Resources
- **GitHub**: github.com/Tencent/ncnn
- **Documentation**: github.com/Tencent/ncnn/wiki
- **Model Zoo**: github.com/nihui/ncnn-assets
- **Benchmark**: github.com/nihui/ncnn-benchmark

### Community Projects
- **ncnn-android-yolov5**: YOLO v5 on Android
- **ncnn-ios-blazeface**: Face detection on iOS
- **ncnn-webassembly**: NCNN in browser
- **Real-ESRGAN-ncnn**: Super resolution

### Learning Materials
- **Tutorials**: Step-by-step guides
- **Examples**: Sample applications
- **Blog Posts**: Optimization techniques
- **Videos**: Implementation walkthroughs

## 🎓 Advanced Topics

### Research Areas
1. **Neural Architecture Search**: Auto-optimize for NCNN
2. **Dynamic Networks**: Runtime graph modification
3. **Federated Learning**: On-device training
4. **Edge-Cloud Hybrid**: Distributed inference

### Future Directions
- **8-bit Training**: On-device fine-tuning
- **Sparse Networks**: Structured sparsity
- **Hardware Acceleration**: NPU support
- **AutoML Integration**: Automated optimization

---

*High-performance neural network inference for the mobile AI era* 🚀📱