# ⚡ ONNX Runtime & TensorRT Optimization

**Last Updated:** 2025-06-19

## Overview
Comprehensive guide for optimizing AI models using ONNX Runtime and NVIDIA TensorRT for maximum inference performance across different platforms.

## 🎯 ONNX Runtime

### What is ONNX Runtime?
- **Cross-platform** inference acceleration
- **Hardware agnostic** optimization
- **Multiple execution providers** (CPU, CUDA, TensorRT, DirectML, etc.)
- **Production ready** with enterprise support
- **Language bindings** for Python, C++, C#, Java, JavaScript

### Installation & Setup
```bash
# Python installation
pip install onnxruntime       # CPU version
pip install onnxruntime-gpu   # GPU version with CUDA

# For specific providers
pip install onnxruntime-directml   # Windows DirectML
pip install onnxruntime-openvino   # Intel OpenVINO
```

### Basic Usage
```python
import onnxruntime as ort
import numpy as np

# Create inference session
providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession("model.onnx", providers=providers)

# Get input/output names and shapes
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Prepare input
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run inference
results = session.run([output_name], {input_name: input_data})
```

### Advanced Configuration
```python
# Session options for optimization
session_options = ort.SessionOptions()

# Enable graph optimization
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Enable profiling
session_options.enable_profiling = True

# Memory optimization
session_options.enable_cpu_mem_arena = True
session_options.enable_mem_pattern = True
session_options.enable_mem_reuse = True

# Execution mode
session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

# Create session with options
session = ort.InferenceSession(
    "model.onnx", 
    sess_options=session_options,
    providers=providers
)
```

## 🚀 TensorRT Optimization

### What is TensorRT?
- **NVIDIA's inference optimizer** for deep learning
- **Layer fusion** and kernel auto-tuning
- **Precision calibration** (FP32, FP16, INT8)
- **Dynamic shape support**
- **Custom layer plugins**

### TensorRT with ONNX
```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def build_engine_from_onnx(onnx_file_path, engine_file_path):
    # Logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # Builder
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    
    # Parse ONNX
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Configuration
    config.max_workspace_size = 1 << 30  # 1GB
    
    # Enable FP16
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # Build engine
    engine = builder.build_engine(network, config)
    
    # Save engine
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())
    
    return engine
```

### INT8 Quantization
```python
class Int8EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_loader, cache_file):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.data_loader = data_loader
        self.cache_file = cache_file
        self.current_index = 0
        
    def get_batch_size(self):
        return self.data_loader.batch_size
    
    def get_batch(self, names):
        if self.current_index < len(self.data_loader):
            batch = next(iter(self.data_loader))
            self.current_index += 1
            return [int(batch.data_ptr())]
        return None
    
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# Use calibrator in config
config.int8_calibrator = Int8EntropyCalibrator(
    calibration_loader, 
    "calibration.cache"
)
config.set_flag(trt.BuilderFlag.INT8)
```

## 🔧 Model Optimization Techniques

### Graph Optimization
```python
# ONNX model optimization
import onnx
from onnxruntime.transformers import optimizer

# Load model
model = onnx.load("model.onnx")

# Optimize
optimized_model = optimizer.optimize_model(
    model,
    model_type='bert',  # or 'gpt2', 'vit', etc.
    num_heads=12,
    hidden_size=768,
    use_gpu=True,
    opt_level=2,
    float16=True
)

# Save optimized model
optimized_model.save_model_to_file("model_optimized.onnx")
```

### Dynamic Shape Support
```python
# TensorRT dynamic shapes
profile = builder.create_optimization_profile()

# Set dynamic dimensions [min, opt, max]
profile.set_shape(
    "input", 
    min=(1, 3, 224, 224),
    opt=(8, 3, 224, 224),
    max=(32, 3, 224, 224)
)

config.add_optimization_profile(profile)
```

### Custom Plugins
```cpp
// Custom TensorRT plugin
class CustomPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    // Implementation of custom operations
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override {
        return new CustomPlugin(*this);
    }
    
    int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                    const nvinfer1::PluginTensorDesc* outputDesc,
                    const void* const* inputs,
                    void* const* outputs,
                    void* workspace,
                    cudaStream_t stream) noexcept override {
        // CUDA kernel implementation
        return 0;
    }
};
```

## 📊 Performance Comparison

### Benchmarking Script
```python
import time
import numpy as np

def benchmark_inference(session_or_engine, input_data, num_runs=100):
    # Warmup
    for _ in range(10):
        _ = run_inference(session_or_engine, input_data)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = run_inference(session_or_engine, input_data)
        end = time.perf_counter()
        times.append(end - start)
    
    times = np.array(times) * 1000  # Convert to ms
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'p99': np.percentile(times, 99)
    }
```

### Multi-provider Comparison
```python
providers_to_test = [
    ['CPUExecutionProvider'],
    ['CUDAExecutionProvider'],
    ['TensorrtExecutionProvider'],
]

results = {}
for providers in providers_to_test:
    session = ort.InferenceSession("model.onnx", providers=providers)
    results[providers[0]] = benchmark_inference(session, input_data)
    
# Print comparison
for provider, metrics in results.items():
    print(f"{provider}: {metrics['mean']:.2f} ms (±{metrics['std']:.2f})")
```

## 🛠️ Deployment Strategies

### Edge Deployment
```python
# ONNX Runtime Mobile/Edge
import onnxruntime as ort

# Create lightweight session for edge
options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
options.add_session_config_entry("session.use_ort_model_bytes_directly", "1")

# Use mobile-optimized providers
providers = ['CPUExecutionProvider']
session = ort.InferenceSession(
    "model_quantized.onnx", 
    sess_options=options,
    providers=providers
)
```

### Server Deployment
```python
# TensorRT Inference Server
import tritonclient.grpc as grpcclient

# Create client
triton_client = grpcclient.InferenceServerClient(
    url='localhost:8001',
    verbose=False
)

# Prepare input
input = grpcclient.InferInput('input', [1, 3, 224, 224], "FP32")
input.set_data_from_numpy(input_data)

# Prepare output
output = grpcclient.InferRequestedOutput('output')

# Inference request
response = triton_client.infer(
    model_name='resnet50',
    inputs=[input],
    outputs=[output]
)

result = response.as_numpy('output')
```

## 💡 Best Practices

### Model Preparation
1. **Simplify ONNX graph** before optimization
2. **Use dynamic axes** for batch flexibility
3. **Profile different batch sizes**
4. **Test precision trade-offs**

### Memory Management
```python
# Efficient memory usage
class InferenceEngine:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Pre-allocate buffers
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_shape = self.session.get_outputs()[0].shape
        
    def infer_batch(self, batch_data):
        # Reuse session for multiple inferences
        return self.session.run(
            [self.output_name], 
            {self.input_name: batch_data}
        )[0]
```

### Error Handling
```python
try:
    session = ort.InferenceSession(
        model_path,
        providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider']
    )
except Exception as e:
    print(f"Failed to create session with GPU: {e}")
    # Fallback to CPU
    session = ort.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider']
    )
```

## 🔗 Resources

### Documentation
- **ONNX Runtime**: onnxruntime.ai
- **TensorRT**: developer.nvidia.com/tensorrt
- **Model Zoo**: github.com/onnx/models
- **Optimization Guide**: onnxruntime.ai/docs/performance

### Tools
- **Netron**: Model visualization
- **onnx-simplifier**: Graph simplification
- **trtexec**: TensorRT profiling
- **polygraphy**: Model debugging

### Community
- **ONNX Gitter**: Technical discussions
- **TensorRT Forum**: NVIDIA developer forum
- **GitHub Issues**: Bug reports and features
- **Stack Overflow**: Q&A platform

---

*Accelerating AI inference with industry-leading optimization frameworks* ⚡🚀