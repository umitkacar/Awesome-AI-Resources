# 🌐 Web Application Development for AI/ML

## Overview
Complete guide for building modern web applications that integrate AI/ML capabilities, from frontend frameworks to full-stack architectures.

## 🎯 Architecture Overview

### Modern AI Web App Stack
```
┌─────────────────────────────────────┐
│         Frontend (React/Vue)         │
├─────────────────────────────────────┤
│      API Gateway (Kong/Nginx)       │
├─────────────────────────────────────┤
│   Backend Services (FastAPI/Node)   │
├─────────────────────────────────────┤
│    ML Services (TensorFlow/PyTorch) │
├─────────────────────────────────────┤
│   Data Layer (PostgreSQL/MongoDB)   │
└─────────────────────────────────────┘
```

## 🚀 Frontend Development

### React + AI Integration
```jsx
// AIImageClassifier.jsx
import React, { useState, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Line } from 'react-chartjs-2';

const AIImageClassifier = () => {
  const [model, setModel] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [confidence, setConfidence] = useState([]);
  const fileInputRef = useRef(null);
  const canvasRef = useRef(null);

  // Load model on component mount
  useEffect(() => {
    loadModel();
  }, []);

  const loadModel = async () => {
    try {
      setIsLoading(true);
      const loadedModel = await tf.loadLayersModel('/models/mobilenet/model.json');
      setModel(loadedModel);
      console.log('Model loaded successfully');
    } catch (error) {
      console.error('Error loading model:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const preprocessImage = (imageElement) => {
    // Convert image to tensor
    const tensor = tf.browser.fromPixels(imageElement)
      .resizeNearestNeighbor([224, 224])
      .expandDims(0)
      .toFloat()
      .div(tf.scalar(127.5))
      .sub(tf.scalar(1));
    return tensor;
  };

  const classifyImage = async (imageFile) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = async (e) => {
        const img = new Image();
        img.onload = async () => {
          // Display image on canvas
          const ctx = canvasRef.current.getContext('2d');
          ctx.drawImage(img, 0, 0, 300, 300);

          // Preprocess and predict
          const tensor = preprocessImage(img);
          const predictions = await model.predict(tensor).data();
          
          // Get top 5 predictions
          const top5 = Array.from(predictions)
            .map((p, i) => ({ probability: p, className: IMAGENET_CLASSES[i] }))
            .sort((a, b) => b.probability - a.probability)
            .slice(0, 5);

          setPredictions(top5);
          setConfidence(top5.map(p => p.probability * 100));
          
          // Clean up
          tensor.dispose();
          resolve(top5);
        };
        img.src = e.target.result;
      };
      reader.readAsDataURL(imageFile);
    });
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (file && model) {
      setIsLoading(true);
      try {
        await classifyImage(file);
      } catch (error) {
        console.error('Classification error:', error);
      } finally {
        setIsLoading(false);
      }
    }
  };

  return (
    <div className="ai-classifier-container">
      <h2>AI Image Classifier</h2>
      
      {/* File Upload */}
      <div className="upload-section">
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileUpload}
          accept="image/*"
          style={{ display: 'none' }}
        />
        <button 
          onClick={() => fileInputRef.current.click()}
          disabled={!model || isLoading}
          className="upload-button"
        >
          {isLoading ? 'Processing...' : 'Upload Image'}
        </button>
      </div>

      {/* Image Display */}
      <div className="image-display">
        <canvas ref={canvasRef} width={300} height={300} />
      </div>

      {/* Predictions */}
      {predictions.length > 0 && (
        <div className="predictions-section">
          <h3>Top 5 Predictions:</h3>
          {predictions.map((pred, index) => (
            <div key={index} className="prediction-item">
              <span className="class-name">{pred.className}</span>
              <div className="confidence-bar">
                <div 
                  className="confidence-fill"
                  style={{ width: `${pred.probability * 100}%` }}
                />
              </div>
              <span className="confidence-text">
                {(pred.probability * 100).toFixed(2)}%
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Confidence Chart */}
      {confidence.length > 0 && (
        <div className="chart-section">
          <Line
            data={{
              labels: predictions.map(p => p.className.substring(0, 15)),
              datasets: [{
                label: 'Confidence %',
                data: confidence,
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 2
              }]
            }}
            options={{
              scales: {
                y: {
                  beginAtZero: true,
                  max: 100
                }
              }
            }}
          />
        </div>
      )}
    </div>
  );
};

export default AIImageClassifier;
```

### Vue.js AI Chat Interface
```vue
<!-- AIChatbot.vue -->
<template>
  <div class="ai-chatbot">
    <div class="chat-header">
      <h3>AI Assistant</h3>
      <span :class="['status', { 'online': isConnected }]">
        {{ isConnected ? 'Online' : 'Offline' }}
      </span>
    </div>

    <div class="chat-messages" ref="messagesContainer">
      <div 
        v-for="(message, index) in messages" 
        :key="index"
        :class="['message', message.type]"
      >
        <div class="message-content">
          <div class="author">{{ message.author }}</div>
          <div class="text" v-html="renderMessage(message.text)"></div>
          <div class="timestamp">{{ formatTime(message.timestamp) }}</div>
        </div>
        
        <!-- Typing indicator -->
        <div v-if="message.isTyping" class="typing-indicator">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>
    </div>

    <div class="chat-input">
      <textarea
        v-model="inputMessage"
        @keypress.enter.shift.prevent="sendMessage"
        placeholder="Type your message..."
        :disabled="!isConnected || isProcessing"
      ></textarea>
      
      <button 
        @click="sendMessage" 
        :disabled="!inputMessage.trim() || !isConnected || isProcessing"
      >
        <i class="fas fa-paper-plane"></i>
      </button>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, nextTick, computed } from 'vue';
import axios from 'axios';
import io from 'socket.io-client';
import MarkdownIt from 'markdown-it';
import hljs from 'highlight.js';

export default {
  name: 'AIChatbot',
  setup() {
    const messages = ref([]);
    const inputMessage = ref('');
    const isConnected = ref(false);
    const isProcessing = ref(false);
    const socket = ref(null);
    const messagesContainer = ref(null);
    
    const md = new MarkdownIt({
      highlight: function (str, lang) {
        if (lang && hljs.getLanguage(lang)) {
          try {
            return hljs.highlight(str, { language: lang }).value;
          } catch (__) {}
        }
        return '';
      }
    });

    const connectWebSocket = () => {
      socket.value = io(process.env.VUE_APP_WS_URL || 'ws://localhost:3000');
      
      socket.value.on('connect', () => {
        isConnected.value = true;
        addSystemMessage('Connected to AI assistant');
      });

      socket.value.on('disconnect', () => {
        isConnected.value = false;
        addSystemMessage('Disconnected from AI assistant');
      });

      socket.value.on('ai_response', (data) => {
        handleAIResponse(data);
      });

      socket.value.on('ai_error', (error) => {
        addMessage('AI', `Error: ${error.message}`, 'error');
        isProcessing.value = false;
      });
    };

    const sendMessage = async () => {
      if (!inputMessage.value.trim() || !isConnected.value || isProcessing.value) {
        return;
      }

      const userMessage = inputMessage.value;
      inputMessage.value = '';
      
      // Add user message
      addMessage('You', userMessage, 'user');
      
      // Show typing indicator
      const typingMessage = {
        author: 'AI',
        text: '',
        type: 'ai',
        isTyping: true,
        timestamp: new Date()
      };
      messages.value.push(typingMessage);
      
      isProcessing.value = true;

      try {
        // Send via WebSocket for streaming response
        socket.value.emit('chat_message', {
          message: userMessage,
          context: getConversationContext()
        });
      } catch (error) {
        console.error('Error sending message:', error);
        messages.value = messages.value.filter(m => !m.isTyping);
        addMessage('System', 'Failed to send message', 'error');
        isProcessing.value = false;
      }
    };

    const handleAIResponse = (data) => {
      // Remove typing indicator
      messages.value = messages.value.filter(m => !m.isTyping);
      
      if (data.streaming) {
        // Handle streaming response
        const lastMessage = messages.value[messages.value.length - 1];
        if (lastMessage && lastMessage.type === 'ai' && lastMessage.isStreaming) {
          lastMessage.text += data.chunk;
        } else {
          messages.value.push({
            author: 'AI',
            text: data.chunk,
            type: 'ai',
            isStreaming: true,
            timestamp: new Date()
          });
        }
      } else {
        // Complete response
        addMessage('AI', data.message, 'ai');
        isProcessing.value = false;
      }
      
      scrollToBottom();
    };

    const addMessage = (author, text, type) => {
      messages.value.push({
        author,
        text,
        type,
        timestamp: new Date()
      });
      scrollToBottom();
    };

    const addSystemMessage = (text) => {
      addMessage('System', text, 'system');
    };

    const renderMessage = (text) => {
      return md.render(text);
    };

    const formatTime = (timestamp) => {
      return new Intl.DateTimeFormat('en-US', {
        hour: 'numeric',
        minute: 'numeric',
        hour12: true
      }).format(timestamp);
    };

    const scrollToBottom = () => {
      nextTick(() => {
        if (messagesContainer.value) {
          messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight;
        }
      });
    };

    const getConversationContext = () => {
      // Get last 10 messages for context
      return messages.value
        .slice(-10)
        .filter(m => !m.isTyping && !m.isStreaming)
        .map(m => ({
          role: m.type === 'user' ? 'user' : 'assistant',
          content: m.text
        }));
    };

    onMounted(() => {
      connectWebSocket();
      addSystemMessage('Welcome! How can I assist you today?');
    });

    return {
      messages,
      inputMessage,
      isConnected,
      isProcessing,
      messagesContainer,
      sendMessage,
      renderMessage,
      formatTime
    };
  }
};
</script>

<style scoped>
.ai-chatbot {
  display: flex;
  flex-direction: column;
  height: 600px;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  overflow: hidden;
}

.chat-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px;
  background: #2196f3;
  color: white;
}

.status {
  display: flex;
  align-items: center;
  font-size: 14px;
}

.status::before {
  content: '';
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #ff5252;
  margin-right: 8px;
}

.status.online::before {
  background: #4caf50;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  background: #f5f5f5;
}

.message {
  margin-bottom: 16px;
  display: flex;
  align-items: flex-start;
}

.message.user {
  justify-content: flex-end;
}

.message-content {
  max-width: 70%;
  padding: 12px;
  border-radius: 8px;
  background: white;
  box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

.message.user .message-content {
  background: #2196f3;
  color: white;
}

.typing-indicator {
  display: flex;
  align-items: center;
  margin-left: 8px;
}

.typing-indicator span {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #999;
  margin: 0 2px;
  animation: typing 1.4s infinite;
}

@keyframes typing {
  0%, 60%, 100% {
    opacity: 0.3;
  }
  30% {
    opacity: 1;
  }
}
</style>
```

## 🔧 Backend Integration

### Node.js WebSocket Server
```javascript
// server.js
const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors');
const { OpenAI } = require('openai');
const { createClient } = require('redis');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: process.env.CLIENT_URL || "http://localhost:3000",
    methods: ["GET", "POST"]
  }
});

// Initialize OpenAI
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Initialize Redis for session storage
const redis = createClient({
  url: process.env.REDIS_URL || 'redis://localhost:6379'
});

redis.on('error', (err) => console.log('Redis Client Error', err));
redis.connect();

// Middleware
app.use(cors());
app.use(express.json());

// REST API endpoints
app.post('/api/classify', async (req, res) => {
  try {
    const { image, modelType = 'vision' } = req.body;
    
    // Process image classification
    const result = await classifyImage(image, modelType);
    
    // Store in database
    await storeClassification(result);
    
    res.json({ success: true, result });
  } catch (error) {
    console.error('Classification error:', error);
    res.status(500).json({ 
      success: false, 
      error: error.message 
    });
  }
});

// WebSocket handling
io.on('connection', (socket) => {
  console.log('New client connected:', socket.id);
  
  // Handle chat messages
  socket.on('chat_message', async (data) => {
    try {
      const { message, context } = data;
      
      // Stream AI response
      const stream = await openai.chat.completions.create({
        model: "gpt-4",
        messages: [
          { role: "system", content: "You are a helpful AI assistant." },
          ...context,
          { role: "user", content: message }
        ],
        stream: true,
      });
      
      for await (const chunk of stream) {
        const content = chunk.choices[0]?.delta?.content || '';
        if (content) {
          socket.emit('ai_response', {
            streaming: true,
            chunk: content
          });
        }
      }
      
      // Send completion signal
      socket.emit('ai_response', {
        streaming: false,
        message: 'Response complete'
      });
      
    } catch (error) {
      console.error('Chat error:', error);
      socket.emit('ai_error', {
        message: 'Failed to process message'
      });
    }
  });
  
  // Handle image analysis
  socket.on('analyze_image', async (data) => {
    try {
      const { imageBase64, analysisType } = data;
      
      // Process image
      const analysis = await analyzeImage(imageBase64, analysisType);
      
      socket.emit('analysis_result', {
        success: true,
        analysis
      });
      
    } catch (error) {
      socket.emit('analysis_error', {
        message: 'Failed to analyze image'
      });
    }
  });
  
  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

// Helper functions
async function classifyImage(imageData, modelType) {
  // Implementation for image classification
  // Could use TensorFlow.js, call Python service, etc.
}

async function analyzeImage(imageBase64, analysisType) {
  // Implementation for image analysis
}

// Start server
const PORT = process.env.PORT || 3001;
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```

## 📊 Real-time Dashboard

### AI Metrics Dashboard
```jsx
// AIDashboard.jsx
import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, BarChart, Bar, PieChart, Pie, 
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Cell 
} from 'recharts';
import { Grid, Paper, Typography, Box } from '@mui/material';

const AIDashboard = () => {
  const [metrics, setMetrics] = useState({
    modelPerformance: [],
    apiUsage: [],
    errorRates: [],
    userDistribution: []
  });
  const [realTimeData, setRealTimeData] = useState([]);

  useEffect(() => {
    // Connect to metrics WebSocket
    const ws = new WebSocket('ws://localhost:3001/metrics');
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      updateMetrics(data);
    };

    // Fetch historical data
    fetchHistoricalMetrics();

    return () => ws.close();
  }, []);

  const updateMetrics = (newData) => {
    setRealTimeData(prev => [...prev.slice(-20), newData]);
    
    // Update other metrics
    setMetrics(prev => ({
      ...prev,
      modelPerformance: updatePerformance(prev.modelPerformance, newData),
      apiUsage: updateUsage(prev.apiUsage, newData),
      errorRates: updateErrors(prev.errorRates, newData)
    }));
  };

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h4" gutterBottom>
        AI System Dashboard
      </Typography>
      
      <Grid container spacing={3}>
        {/* Real-time Predictions */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Real-time Predictions
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={realTimeData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="predictions" 
                  stroke="#8884d8" 
                  strokeWidth={2}
                />
                <Line 
                  type="monotone" 
                  dataKey="confidence" 
                  stroke="#82ca9d" 
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Model Performance */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Model Performance
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={metrics.modelPerformance}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={renderCustomizedLabel}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {metrics.modelPerformance.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* API Usage */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              API Usage (Last 24h)
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={metrics.apiUsage}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="hour" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="requests" fill="#8884d8" />
                <Bar dataKey="successful" fill="#82ca9d" />
                <Bar dataKey="failed" fill="#ff6b6b" />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* System Health */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <SystemHealth />
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};
```

## 🔒 Security Implementation

### API Security Middleware
```javascript
// security.js
const rateLimit = require('express-rate-limit');
const helmet = require('helmet');
const mongoSanitize = require('express-mongo-sanitize');
const xss = require('xss-clean');

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again later.',
  standardHeaders: true,
  legacyHeaders: false,
});

// API key validation
const validateApiKey = (req, res, next) => {
  const apiKey = req.headers['x-api-key'];
  
  if (!apiKey) {
    return res.status(401).json({ error: 'API key required' });
  }
  
  // Validate against database
  if (!isValidApiKey(apiKey)) {
    return res.status(403).json({ error: 'Invalid API key' });
  }
  
  next();
};

// Input validation
const validateInput = (schema) => {
  return (req, res, next) => {
    const { error } = schema.validate(req.body);
    if (error) {
      return res.status(400).json({ 
        error: 'Invalid input', 
        details: error.details 
      });
    }
    next();
  };
};

// Apply security middleware
app.use(helmet());
app.use(mongoSanitize());
app.use(xss());
app.use('/api/', limiter);
```

## 🚀 Deployment

### Docker Compose Setup
```yaml
# docker-compose.yml
version: '3.8'

services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://backend:8000
    depends_on:
      - backend

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/aiapp
      - REDIS_URL=redis://redis:6379
      - MODEL_PATH=/models
    volumes:
      - ./models:/models
    depends_on:
      - db
      - redis

  ml-service:
    build:
      context: ./ml-service
      dockerfile: Dockerfile
    ports:
      - "5000:5000"
    environment:
      - MODEL_CACHE=/cache
    volumes:
      - model-cache:/cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  db:
    image: postgres:14
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=aiapp
    volumes:
      - postgres-data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - frontend
      - backend

volumes:
  postgres-data:
  redis-data:
  model-cache:
```

---

*Building modern, scalable web applications powered by artificial intelligence* 🌐🤖