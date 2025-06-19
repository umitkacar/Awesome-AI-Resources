# 🟨 AI with JavaScript

**Last Updated:** 2025-06-19

## Overview
Comprehensive guide for implementing AI and machine learning in JavaScript, covering browser-based inference, Node.js backends, and modern frameworks.

## 🎯 Why JavaScript for AI?

### Advantages
- **Browser-native**: Run AI directly in web browsers
- **No installation**: Zero setup for end users
- **Cross-platform**: Works on any device with a browser
- **Real-time**: Client-side inference without server calls
- **Privacy**: Data stays on user's device
- **Progressive**: Graceful degradation for older browsers

### Use Cases
- Real-time image processing
- Natural language processing
- Recommendation systems
- Computer vision in browser
- Voice recognition
- Generative AI applications

## 🚀 TensorFlow.js

### Getting Started
```javascript
// Browser
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>

// Node.js
npm install @tensorflow/tfjs-node

// With GPU support
npm install @tensorflow/tfjs-node-gpu
```

### Image Classification Example
```javascript
// Load and use a pre-trained model
async function classifyImage() {
  // Load MobileNet model
  const model = await tf.loadLayersModel(
    'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/classification/3/default/1',
    { fromTFHub: true }
  );

  // Get image from canvas or img element
  const img = document.getElementById('img');
  
  // Preprocess the image
  const tensor = tf.browser.fromPixels(img)
    .resizeBilinear([224, 224])
    .expandDims(0)
    .div(255.0);

  // Make prediction
  const predictions = await model.predict(tensor).data();
  
  // Get top 5 predictions
  const top5 = Array.from(predictions)
    .map((p, i) => ({ probability: p, className: IMAGENET_CLASSES[i] }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, 5);

  return top5;
}
```

### Training in Browser
```javascript
// Create a simple neural network
function createModel() {
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ inputShape: [784], units: 128, activation: 'relu' }),
      tf.layers.dropout({ rate: 0.2 }),
      tf.layers.dense({ units: 10, activation: 'softmax' })
    ]
  });

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return model;
}

// Train the model
async function trainModel(model, data) {
  const BATCH_SIZE = 128;
  const EPOCHS = 10;

  // Convert data to tensors
  const xs = tf.tensor2d(data.images, [data.images.length, 784]);
  const ys = tf.tensor2d(data.labels, [data.labels.length, 10]);

  // Train with real-time UI updates
  await model.fit(xs, ys, {
    batchSize: BATCH_SIZE,
    epochs: EPOCHS,
    shuffle: true,
    validationSplit: 0.1,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(`Epoch ${epoch}: loss = ${logs.loss.toFixed(4)}`);
        updateTrainingChart(epoch, logs);
        await tf.nextFrame(); // Keep UI responsive
      }
    }
  });

  // Save model to browser storage
  await model.save('localstorage://my-model');
}
```

### Custom Layers
```javascript
// Define a custom layer
class AttentionLayer extends tf.layers.Layer {
  constructor(config) {
    super(config);
    this.units = config.units;
  }

  build(inputShape) {
    this.w = this.addWeight(
      'attention_weight',
      [inputShape[inputShape.length - 1], this.units],
      'float32',
      tf.initializers.glorotUniform()
    );
  }

  call(inputs, kwargs) {
    return tf.tidy(() => {
      const dotProduct = tf.matMul(inputs[0], this.w.read());
      const attention = tf.softmax(dotProduct);
      return tf.matMul(attention, inputs[0]);
    });
  }

  getConfig() {
    const config = super.getConfig();
    Object.assign(config, { units: this.units });
    return config;
  }

  static get className() {
    return 'AttentionLayer';
  }
}

// Register the custom layer
tf.serialization.registerClass(AttentionLayer);
```

## 🧠 Brain.js

### Neural Network Training
```javascript
const brain = require('brain.js');

// Create a neural network
const net = new brain.NeuralNetwork({
  hiddenLayers: [4, 5], // 2 hidden layers
  activation: 'sigmoid'
});

// Training data for XOR
const trainingData = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] }
];

// Train the network
net.train(trainingData, {
  iterations: 20000,
  log: true,
  logPeriod: 1000,
  errorThresh: 0.005
});

// Make predictions
console.log(net.run([0, 0])); // ~0
console.log(net.run([1, 0])); // ~1
```

### LSTM for Text Generation
```javascript
const net = new brain.recurrent.LSTM();

// Train on text data
const trainingData = [
  'JavaScript is awesome',
  'Machine learning in JavaScript',
  'Neural networks are powerful'
];

net.train(trainingData, {
  iterations: 1500,
  errorThresh: 0.011
});

// Generate text
const output = net.run('JavaScript');
console.log(output); // Generates continuation
```

## 🤖 ML5.js

### Friendly ML for Creative Coding
```javascript
// Image classifier with ML5
let classifier;
let img;

function preload() {
  classifier = ml5.imageClassifier('MobileNet');
  img = loadImage('cat.jpg');
}

function setup() {
  createCanvas(400, 400);
  classifier.classify(img, gotResult);
}

function gotResult(error, results) {
  if (error) {
    console.error(error);
    return;
  }
  
  // Display results
  results.forEach((result, i) => {
    text(`${result.label}: ${nf(result.confidence, 0, 2)}`, 10, 20 + i * 20);
  });
}
```

### Pose Detection
```javascript
let video;
let poseNet;
let poses = [];

function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.hide();
  
  // Create poseNet
  poseNet = ml5.poseNet(video, modelReady);
  poseNet.on('pose', (results) => {
    poses = results;
  });
}

function modelReady() {
  console.log('Model Loaded!');
}

function draw() {
  image(video, 0, 0);
  
  // Draw keypoints
  for (let pose of poses) {
    for (let keypoint of pose.pose.keypoints) {
      if (keypoint.score > 0.2) {
        fill(255, 0, 0);
        noStroke();
        ellipse(keypoint.position.x, keypoint.position.y, 10, 10);
      }
    }
  }
}
```

## 🔧 ONNX.js

### Running ONNX Models
```javascript
// Load and run ONNX model
const onnx = require('onnxjs');

async function runONNXModel() {
  // Create session
  const session = new onnx.InferenceSession();
  await session.loadModel('./model.onnx');

  // Prepare input
  const input = new Float32Array(1 * 3 * 224 * 224);
  // ... fill input data ...

  // Create tensor
  const tensor = new onnx.Tensor(input, 'float32', [1, 3, 224, 224]);

  // Run inference
  const outputMap = await session.run([tensor]);
  const outputTensor = outputMap.values().next().value;

  // Process results
  const predictions = outputTensor.data;
  return predictions;
}
```

## 📊 Natural Language Processing

### Sentiment Analysis
```javascript
const Sentiment = require('sentiment');
const sentiment = new Sentiment();

function analyzeSentiment(text) {
  const result = sentiment.analyze(text);
  
  return {
    score: result.score,
    comparative: result.comparative,
    positive: result.positive,
    negative: result.negative,
    tokens: result.tokens
  };
}

// Example usage
const analysis = analyzeSentiment('JavaScript is absolutely fantastic!');
console.log(analysis);
// { score: 4, comparative: 0.8, positive: ['fantastic'], ... }
```

### Named Entity Recognition with compromise
```javascript
const nlp = require('compromise');

function extractEntities(text) {
  const doc = nlp(text);
  
  return {
    people: doc.people().out('array'),
    places: doc.places().out('array'),
    organizations: doc.organizations().out('array'),
    dates: doc.dates().out('array'),
    values: doc.values().out('array')
  };
}

// Example
const entities = extractEntities(
  'Apple Inc. was founded by Steve Jobs in Cupertino on April 1, 1976.'
);
```

## 🎨 Generative AI

### Text Generation with Transformers.js
```javascript
import { pipeline } from '@xenova/transformers';

// Initialize text generation pipeline
const generator = await pipeline(
  'text-generation',
  'Xenova/gpt2'
);

// Generate text
async function generateText(prompt) {
  const output = await generator(prompt, {
    max_length: 100,
    temperature: 0.8,
    do_sample: true,
    top_p: 0.9
  });
  
  return output[0].generated_text;
}

// Example usage
const story = await generateText('Once upon a time in JavaScript land...');
```

### Image Generation with Stable Diffusion
```javascript
// Using Replicate API for Stable Diffusion
const Replicate = require('replicate');

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN,
});

async function generateImage(prompt) {
  const output = await replicate.run(
    "stability-ai/stable-diffusion:db21e45d",
    {
      input: {
        prompt: prompt,
        width: 512,
        height: 512,
        num_outputs: 1
      }
    }
  );
  
  return output[0];
}
```

## 🚀 Performance Optimization

### Web Workers for Heavy Computation
```javascript
// main.js
const worker = new Worker('ai-worker.js');

// Send data to worker
worker.postMessage({
  command: 'train',
  data: trainingData
});

// Receive results
worker.onmessage = (event) => {
  if (event.data.type === 'progress') {
    updateProgressBar(event.data.progress);
  } else if (event.data.type === 'complete') {
    console.log('Training complete:', event.data.model);
  }
};

// ai-worker.js
self.onmessage = async (event) => {
  if (event.data.command === 'train') {
    const model = await trainModel(event.data.data);
    
    self.postMessage({
      type: 'complete',
      model: model
    });
  }
};
```

### WebGL Acceleration
```javascript
// Enable WebGL backend for TensorFlow.js
await tf.setBackend('webgl');

// Check backend
console.log('Backend:', tf.getBackend());

// Optimize for mobile
tf.env().set('WEBGL_PACK', false);
tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
```

## 💡 Best Practices

### Memory Management
```javascript
// Always dispose tensors
function processBatch(images) {
  return tf.tidy(() => {
    const batch = tf.stack(images);
    const predictions = model.predict(batch);
    
    // Extract data before disposing
    const results = predictions.arraySync();
    
    // Tensors are automatically disposed when tidy ends
    return results;
  });
}

// Manual disposal
const tensor = tf.tensor([1, 2, 3]);
// ... use tensor ...
tensor.dispose();

// Monitor memory
console.log(tf.memory());
```

### Model Loading Strategies
```javascript
// Progressive loading with fallbacks
async function loadModel() {
  try {
    // Try loading from IndexedDB first
    model = await tf.loadLayersModel('indexeddb://my-model');
  } catch (e) {
    try {
      // Fallback to server
      model = await tf.loadLayersModel('/models/my-model.json');
      // Save to IndexedDB for next time
      await model.save('indexeddb://my-model');
    } catch (e) {
      // Final fallback
      console.error('Failed to load model:', e);
      useDefaultModel();
    }
  }
}
```

## 🔗 Integration Examples

### React + TensorFlow.js
```jsx
import { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';

function ImageClassifier() {
  const [model, setModel] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const imageRef = useRef(null);

  useEffect(() => {
    loadModel();
  }, []);

  const loadModel = async () => {
    const loadedModel = await tf.loadLayersModel('/model.json');
    setModel(loadedModel);
  };

  const classifyImage = async () => {
    if (!model || !imageRef.current) return;

    const predictions = tf.tidy(() => {
      const img = tf.browser.fromPixels(imageRef.current);
      const resized = tf.image.resizeBilinear(img, [224, 224]);
      const batched = resized.expandDims(0);
      const normalized = batched.div(255.0);
      
      return model.predict(normalized);
    });

    const values = await predictions.data();
    predictions.dispose();

    setPredictions(processResults(values));
  };

  return (
    <div>
      <img ref={imageRef} src="/image.jpg" />
      <button onClick={classifyImage}>Classify</button>
      {predictions.map((p, i) => (
        <div key={i}>{p.class}: {p.probability.toFixed(3)}</div>
      ))}
    </div>
  );
}
```

## 🎓 Learning Resources

### Courses & Tutorials
- **TensorFlow.js Course**: Official Google course
- **ML5.js Tutorials**: Creative coding with ML
- **Brain.js Examples**: Neural networks made easy
- **WebML Demos**: Browser-based ML examples

### Books & Documentation
- "Deep Learning with JavaScript" - Manning
- TensorFlow.js API Documentation
- ML5.js Reference
- JavaScript AI Cookbook

---

*Bringing the power of AI to every JavaScript developer* 🟨🤖