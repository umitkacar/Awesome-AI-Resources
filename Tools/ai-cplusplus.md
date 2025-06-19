# 🔷 AI with C++

**Last Updated:** 2025-06-19

## Overview
High-performance AI and machine learning implementation in C++, covering frameworks, optimization techniques, and deployment strategies for production systems.

## 🎯 Why C++ for AI?

### Advantages
- **Performance**: Maximum speed and efficiency
- **Memory Control**: Fine-grained memory management
- **Hardware Access**: Direct GPU/TPU programming
- **Real-time**: Predictable latency for embedded systems
- **Portability**: Cross-platform deployment
- **Integration**: Easy integration with existing systems

### Use Cases
- High-frequency trading algorithms
- Autonomous vehicle systems
- Real-time computer vision
- Embedded AI on edge devices
- Game AI engines
- Scientific computing

## 🚀 Popular C++ AI Frameworks

### DLib
```cpp
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>

using namespace dlib;

// Define a ResNet architecture
template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using res = add_prev1<block<N,bn_con,1,tag1<SUBNET>>>;
template <int N, typename SUBNET> using ares = relu<res<N,SUBNET>>;

using net_type = loss_multiclass_log<
    fc<10,
    avg_pool_everything<
    res<512,res<512,
    res<256,res<256,res<256,
    res<128,res<128,res<128,
    res<64,res<64,
    max_pool<3,3,2,2,relu<bn_con<con<64,7,7,2,2,
    input_rgb_image
    >>>>>>>>>>>>>>>>;

// Train a CNN
void train_network() {
    // Load training data
    std::vector<matrix<rgb_pixel>> images;
    std::vector<unsigned long> labels;
    load_image_dataset(images, labels, "path/to/dataset");

    // Create network and trainer
    net_type net;
    dnn_trainer<net_type> trainer(net);
    trainer.set_learning_rate(0.01);
    trainer.set_min_learning_rate(0.00001);
    trainer.set_mini_batch_size(128);
    trainer.be_verbose();

    // Train the network
    trainer.train(images, labels);

    // Save the trained model
    serialize("model.dat") << net;
}
```

### OpenCV DNN Module
```cpp
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace cv::dnn;

class ObjectDetector {
private:
    Net net;
    std::vector<std::string> classes;
    
public:
    ObjectDetector(const std::string& model_path, 
                   const std::string& config_path,
                   const std::string& classes_path) {
        // Load network
        net = readNet(model_path, config_path);
        net.setPreferableBackend(DNN_BACKEND_CUDA);
        net.setPreferableTarget(DNN_TARGET_CUDA);
        
        // Load class names
        std::ifstream ifs(classes_path);
        std::string line;
        while (getline(ifs, line)) {
            classes.push_back(line);
        }
    }
    
    std::vector<Detection> detect(const Mat& image) {
        // Create blob from image
        Mat blob = blobFromImage(image, 1/255.0, Size(416, 416), 
                                 Scalar(0,0,0), true, false);
        
        // Set input
        net.setInput(blob);
        
        // Forward pass
        std::vector<Mat> outputs;
        net.forward(outputs, getOutputNames(net));
        
        // Post-process detections
        return postprocess(image, outputs);
    }
    
private:
    std::vector<Detection> postprocess(const Mat& frame, 
                                      const std::vector<Mat>& outputs) {
        std::vector<Detection> detections;
        
        for (const auto& output : outputs) {
            float* data = (float*)output.data;
            
            for (int i = 0; i < output.rows; ++i, data += output.cols) {
                Mat scores = output.row(i).colRange(5, output.cols);
                Point classIdPoint;
                double confidence;
                
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                
                if (confidence > 0.5) {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    
                    Detection det;
                    det.classId = classIdPoint.x;
                    det.confidence = confidence;
                    det.box = Rect(centerX - width/2, centerY - height/2, 
                                  width, height);
                    det.className = classes[det.classId];
                    
                    detections.push_back(det);
                }
            }
        }
        
        // Apply NMS
        return applyNMS(detections);
    }
};
```

### Caffe2 C++ API
```cpp
#include <caffe2/core/init.h>
#include <caffe2/core/predictor.h>
#include <caffe2/utils/proto_utils.h>

class Caffe2Model {
private:
    std::unique_ptr<caffe2::Predictor> predictor;
    
public:
    void loadModel(const std::string& init_net_path,
                   const std::string& predict_net_path) {
        // Load protobuf files
        caffe2::NetDef init_net, predict_net;
        CAFFE_ENFORCE(ReadProtoFromFile(init_net_path, &init_net));
        CAFFE_ENFORCE(ReadProtoFromFile(predict_net_path, &predict_net));
        
        // Create predictor
        predictor = std::make_unique<caffe2::Predictor>(init_net, predict_net);
    }
    
    std::vector<float> predict(const std::vector<float>& input_data,
                              const std::vector<int64_t>& input_shape) {
        // Create input tensor
        caffe2::Tensor input(caffe2::CPU);
        input.Resize(input_shape);
        float* input_ptr = input.mutable_data<float>();
        std::copy(input_data.begin(), input_data.end(), input_ptr);
        
        // Run prediction
        caffe2::Predictor::TensorList input_vec{input};
        caffe2::Predictor::TensorList output_vec;
        (*predictor)(input_vec, &output_vec);
        
        // Extract results
        auto& output = output_vec[0];
        const float* output_ptr = output.data<float>();
        std::vector<float> results(output_ptr, 
                                  output_ptr + output.numel());
        
        return results;
    }
};
```

## 🔧 Custom Neural Network Implementation

### Basic Neural Network from Scratch
```cpp
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

class NeuralNetwork {
private:
    struct Layer {
        std::vector<std::vector<double>> weights;
        std::vector<double> biases;
        std::vector<double> activations;
        std::vector<double> z_values;
        
        Layer(int input_size, int output_size) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<> d(0, sqrt(2.0 / input_size));
            
            weights.resize(output_size, std::vector<double>(input_size));
            biases.resize(output_size, 0.0);
            activations.resize(output_size);
            z_values.resize(output_size);
            
            // Xavier initialization
            for (auto& row : weights) {
                for (auto& w : row) {
                    w = d(gen);
                }
            }
        }
    };
    
    std::vector<Layer> layers;
    double learning_rate;
    
    // Activation functions
    double relu(double x) { return std::max(0.0, x); }
    double relu_derivative(double x) { return x > 0 ? 1.0 : 0.0; }
    
    double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
    double sigmoid_derivative(double x) { 
        double s = sigmoid(x);
        return s * (1.0 - s);
    }
    
public:
    NeuralNetwork(const std::vector<int>& architecture, 
                  double lr = 0.01) : learning_rate(lr) {
        for (size_t i = 1; i < architecture.size(); ++i) {
            layers.emplace_back(architecture[i-1], architecture[i]);
        }
    }
    
    std::vector<double> forward(const std::vector<double>& input) {
        std::vector<double> current = input;
        
        for (size_t l = 0; l < layers.size(); ++l) {
            auto& layer = layers[l];
            
            // Compute z = Wx + b
            for (size_t i = 0; i < layer.weights.size(); ++i) {
                layer.z_values[i] = layer.biases[i];
                for (size_t j = 0; j < current.size(); ++j) {
                    layer.z_values[i] += layer.weights[i][j] * current[j];
                }
            }
            
            // Apply activation function
            for (size_t i = 0; i < layer.z_values.size(); ++i) {
                if (l < layers.size() - 1) {
                    layer.activations[i] = relu(layer.z_values[i]);
                } else {
                    layer.activations[i] = sigmoid(layer.z_values[i]);
                }
            }
            
            current = layer.activations;
        }
        
        return current;
    }
    
    void backward(const std::vector<double>& input,
                  const std::vector<double>& target) {
        // Compute output layer gradients
        std::vector<double> delta(layers.back().activations.size());
        for (size_t i = 0; i < delta.size(); ++i) {
            double output = layers.back().activations[i];
            delta[i] = (output - target[i]) * sigmoid_derivative(layers.back().z_values[i]);
        }
        
        // Backpropagate through layers
        std::vector<double> prev_activations = input;
        
        for (int l = layers.size() - 1; l >= 0; --l) {
            auto& layer = layers[l];
            
            // Update weights and biases
            for (size_t i = 0; i < layer.weights.size(); ++i) {
                for (size_t j = 0; j < layer.weights[i].size(); ++j) {
                    layer.weights[i][j] -= learning_rate * delta[i] * 
                                          (l > 0 ? layers[l-1].activations[j] : prev_activations[j]);
                }
                layer.biases[i] -= learning_rate * delta[i];
            }
            
            // Compute gradients for previous layer
            if (l > 0) {
                std::vector<double> new_delta(layers[l-1].activations.size(), 0.0);
                for (size_t j = 0; j < new_delta.size(); ++j) {
                    for (size_t i = 0; i < delta.size(); ++i) {
                        new_delta[j] += delta[i] * layer.weights[i][j];
                    }
                    new_delta[j] *= relu_derivative(layers[l-1].z_values[j]);
                }
                delta = new_delta;
            }
        }
    }
    
    void train(const std::vector<std::vector<double>>& X,
               const std::vector<std::vector<double>>& y,
               int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_loss = 0.0;
            
            for (size_t i = 0; i < X.size(); ++i) {
                // Forward pass
                auto output = forward(X[i]);
                
                // Compute loss
                double loss = 0.0;
                for (size_t j = 0; j < output.size(); ++j) {
                    loss += pow(output[j] - y[i][j], 2);
                }
                total_loss += loss;
                
                // Backward pass
                backward(X[i], y[i]);
            }
            
            if (epoch % 100 == 0) {
                std::cout << "Epoch " << epoch << ", Loss: " 
                         << total_loss / X.size() << std::endl;
            }
        }
    }
};
```

## 🚀 GPU Acceleration

### CUDA Neural Network Kernels
```cpp
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

// Matrix multiplication kernel
__global__ void matmul_kernel(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ReLU activation kernel
__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// Convolution using cuDNN
class CudnnConvolution {
private:
    cudnnHandle_t cudnn;
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnActivationDescriptor_t activation_desc;
    
public:
    CudnnConvolution() {
        cudnnCreate(&cudnn);
        cudnnCreateTensorDescriptor(&input_desc);
        cudnnCreateTensorDescriptor(&output_desc);
        cudnnCreateTensorDescriptor(&bias_desc);
        cudnnCreateFilterDescriptor(&filter_desc);
        cudnnCreateConvolutionDescriptor(&conv_desc);
        cudnnCreateActivationDescriptor(&activation_desc);
    }
    
    void forward(const float* input, const float* filter, 
                 const float* bias, float* output,
                 int batch_size, int channels_in, int channels_out,
                 int height, int width, int filter_size) {
        // Set descriptors
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                  CUDNN_DATA_FLOAT, batch_size,
                                  channels_in, height, width);
        
        cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT,
                                  CUDNN_TENSOR_NCHW, channels_out,
                                  channels_in, filter_size, filter_size);
        
        cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1,
                                       CUDNN_CROSS_CORRELATION,
                                       CUDNN_DATA_FLOAT);
        
        // Get output dimensions
        int out_n, out_c, out_h, out_w;
        cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc,
                                             filter_desc, &out_n, &out_c,
                                             &out_h, &out_w);
        
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
                                  CUDNN_DATA_FLOAT, out_n, out_c,
                                  out_h, out_w);
        
        // Find best algorithm
        cudnnConvolutionFwdAlgo_t algo;
        cudnnGetConvolutionForwardAlgorithm(
            cudnn, input_desc, filter_desc, conv_desc, output_desc,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
        
        // Get workspace size
        size_t workspace_size;
        cudnnGetConvolutionForwardWorkspaceSize(
            cudnn, input_desc, filter_desc, conv_desc, output_desc,
            algo, &workspace_size);
        
        void* workspace;
        cudaMalloc(&workspace, workspace_size);
        
        // Perform convolution
        float alpha = 1.0f, beta = 0.0f;
        cudnnConvolutionForward(cudnn, &alpha, input_desc, input,
                               filter_desc, filter, conv_desc, algo,
                               workspace, workspace_size, &beta,
                               output_desc, output);
        
        // Add bias with ReLU activation
        cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW,
                                  CUDNN_DATA_FLOAT, 1, out_c, 1, 1);
        
        cudnnSetActivationDescriptor(activation_desc, 
                                    CUDNN_ACTIVATION_RELU,
                                    CUDNN_NOT_PROPAGATE_NAN, 0.0);
        
        cudnnConvolutionBiasActivationForward(
            cudnn, &alpha, input_desc, input, filter_desc, filter,
            conv_desc, algo, workspace, workspace_size,
            &beta, output_desc, output, bias_desc, bias,
            activation_desc, output_desc, output);
        
        cudaFree(workspace);
    }
    
    ~CudnnConvolution() {
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyTensorDescriptor(bias_desc);
        cudnnDestroyFilterDescriptor(filter_desc);
        cudnnDestroyConvolutionDescriptor(conv_desc);
        cudnnDestroyActivationDescriptor(activation_desc);
        cudnnDestroy(cudnn);
    }
};
```

## 📊 Optimization Techniques

### SIMD Optimization
```cpp
#include <immintrin.h>

// Vectorized dot product using AVX2
float dot_product_avx2(const float* a, const float* b, int n) {
    __m256 sum = _mm256_setzero_ps();
    
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    
    // Horizontal sum
    __m128 sum_high = _mm256_extractf128_ps(sum, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum);
    __m128 sum_128 = _mm_add_ps(sum_high, sum_low);
    
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    
    float result = _mm_cvtss_f32(sum_128);
    
    // Handle remaining elements
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

// Vectorized ReLU activation
void relu_avx2(float* data, int n) {
    __m256 zero = _mm256_setzero_ps();
    
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 x = _mm256_loadu_ps(&data[i]);
        x = _mm256_max_ps(x, zero);
        _mm256_storeu_ps(&data[i], x);
    }
    
    // Handle remaining elements
    for (; i < n; ++i) {
        data[i] = std::max(0.0f, data[i]);
    }
}
```

### Memory Pool for Tensor Allocation
```cpp
template<typename T>
class TensorMemoryPool {
private:
    struct Block {
        T* ptr;
        size_t size;
        bool in_use;
    };
    
    std::vector<Block> blocks;
    size_t total_allocated = 0;
    const size_t alignment = 64; // Cache line size
    
public:
    T* allocate(size_t size) {
        // Try to find a free block
        for (auto& block : blocks) {
            if (!block.in_use && block.size >= size) {
                block.in_use = true;
                return block.ptr;
            }
        }
        
        // Allocate new block
        size_t aligned_size = ((size * sizeof(T) + alignment - 1) 
                              / alignment) * alignment;
        T* ptr = static_cast<T*>(aligned_alloc(alignment, aligned_size));
        
        blocks.push_back({ptr, size, true});
        total_allocated += aligned_size;
        
        return ptr;
    }
    
    void deallocate(T* ptr) {
        for (auto& block : blocks) {
            if (block.ptr == ptr) {
                block.in_use = false;
                return;
            }
        }
    }
    
    ~TensorMemoryPool() {
        for (const auto& block : blocks) {
            free(block.ptr);
        }
    }
};
```

## 🔧 Model Deployment

### TensorRT Integration
```cpp
#include <NvInfer.h>
#include <NvOnnxParser.h>

class TensorRTEngine {
private:
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    
public:
    void buildEngine(const std::string& onnx_path) {
        auto builder = nvinfer1::createInferBuilder(gLogger);
        auto network = builder->createNetworkV2(
            1U << static_cast<uint32_t>(
                nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        
        auto parser = nvonnxparser::createParser(*network, gLogger);
        parser->parseFromFile(onnx_path.c_str(), 
                             static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
        
        auto config = builder->createBuilderConfig();
        config->setMaxWorkspaceSize(1 << 30); // 1GB
        
        // Enable FP16
        if (builder->platformHasFastFp16()) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        
        engine = builder->buildEngineWithConfig(*network, *config);
        context = engine->createExecutionContext();
    }
    
    std::vector<float> infer(const std::vector<float>& input) {
        // Get binding indices
        int input_idx = engine->getBindingIndex("input");
        int output_idx = engine->getBindingIndex("output");
        
        // Allocate GPU memory
        void* buffers[2];
        cudaMalloc(&buffers[input_idx], input.size() * sizeof(float));
        cudaMalloc(&buffers[output_idx], output_size * sizeof(float));
        
        // Copy input to GPU
        cudaMemcpy(buffers[input_idx], input.data(), 
                   input.size() * sizeof(float), 
                   cudaMemcpyHostToDevice);
        
        // Run inference
        context->executeV2(buffers);
        
        // Copy output from GPU
        std::vector<float> output(output_size);
        cudaMemcpy(output.data(), buffers[output_idx], 
                   output_size * sizeof(float), 
                   cudaMemcpyDeviceToHost);
        
        // Clean up
        cudaFree(buffers[input_idx]);
        cudaFree(buffers[output_idx]);
        
        return output;
    }
};
```

## 💡 Best Practices

### Thread-Safe Model Serving
```cpp
#include <mutex>
#include <queue>
#include <thread>
#include <condition_variable>

template<typename T>
class ThreadSafeModelServer {
private:
    std::queue<std::pair<std::vector<T>, std::promise<std::vector<T>>>> requests;
    std::mutex queue_mutex;
    std::condition_variable cv;
    bool stop = false;
    std::thread worker_thread;
    
    void worker() {
        while (true) {
            std::unique_lock<std::mutex> lock(queue_mutex);
            cv.wait(lock, [this] { return !requests.empty() || stop; });
            
            if (stop && requests.empty()) break;
            
            auto [input, promise] = std::move(requests.front());
            requests.pop();
            lock.unlock();
            
            // Process request
            auto output = model->infer(input);
            promise.set_value(output);
        }
    }
    
public:
    ThreadSafeModelServer() : worker_thread(&ThreadSafeModelServer::worker, this) {}
    
    std::future<std::vector<T>> predict(const std::vector<T>& input) {
        std::promise<std::vector<T>> promise;
        auto future = promise.get_future();
        
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            requests.emplace(input, std::move(promise));
        }
        cv.notify_one();
        
        return future;
    }
    
    ~ThreadSafeModelServer() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            stop = true;
        }
        cv.notify_all();
        worker_thread.join();
    }
};
```

---

*High-performance AI development with the power of C++* 🔷🚀