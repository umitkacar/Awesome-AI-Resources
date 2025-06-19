# Speech Recognition & Speech-to-Text Resources

A comprehensive collection of speech recognition, automatic speech recognition (ASR), and speech-to-text resources.

**Last Updated:** 2025-06-19

## Table of Contents
- [Open Source Models](#open-source-models)
- [Commercial APIs](#commercial-apis)
- [Frameworks & Libraries](#frameworks--libraries)
- [Datasets](#datasets)
- [Research Papers](#research-papers)
- [Tools & Applications](#tools--applications)
- [Tutorials & Courses](#tutorials--courses)

## Open Source Models

### Whisper (OpenAI)
**[Whisper](https://github.com/openai/whisper)** - Robust speech recognition model with multilingual support
- 🟢 Beginner friendly
- 🆓 Free & Open Source
- Supports 99+ languages
- Multiple model sizes (tiny to large)

**[Faster Whisper](https://github.com/guillaumekln/faster-whisper)** - CTranslate2 implementation of Whisper
- 4x faster than original
- Lower memory usage
- Same accuracy

**[WhisperX](https://github.com/m-bain/whisperX)** - Whisper with word-level timestamps
- Time-accurate transcription
- Speaker diarization
- VAD preprocessing

### Wav2Vec2 (Meta/Facebook)
**[Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h)** - Self-supervised speech recognition
- Pre-trained on unlabeled data
- Fine-tunable for specific languages
- Available on HuggingFace

### SpeechBrain
**[SpeechBrain](https://speechbrain.github.io/)** - All-in-one speech toolkit
- ASR, speaker recognition, speech enhancement
- PyTorch-based
- Extensive documentation

## Commercial APIs

### Cloud Services
**[Google Cloud Speech-to-Text](https://cloud.google.com/speech-to-text)** - Google's speech recognition API
- 💰 Pay-per-use
- Real-time and batch processing
- 125+ languages

**[Azure Speech Services](https://azure.microsoft.com/services/cognitive-services/speech-to-text/)** - Microsoft's speech API
- Custom model training
- Real-time transcription
- Speaker identification

**[AWS Transcribe](https://aws.amazon.com/transcribe/)** - Amazon's automatic speech recognition
- Medical & call center variants
- Custom vocabulary
- Streaming transcription

**[AssemblyAI](https://www.assemblyai.com/)** - Modern speech-to-text API
- 🔄 Freemium
- Speaker diarization
- Sentiment analysis
- Topic detection

## Frameworks & Libraries

### Python Libraries
**[SpeechRecognition](https://github.com/Uberi/speech_recognition)** - Simple Python library
- 🟢 Beginner friendly
- Multiple engine support
- Microphone input support

**[Vosk](https://alphacephei.com/vosk/)** - Offline speech recognition
- 🆓 Free
- Supports 20+ languages
- Mobile & embedded devices

**[Silero](https://github.com/snakers4/silero-models)** - Pre-trained STT models
- Lightweight models
- PyTorch & ONNX
- Multiple languages

### Deep Learning Frameworks
**[ESPnet](https://github.com/espnet/espnet)** - End-to-end speech processing
- 🔴 Advanced
- Research-oriented
- Multi-task learning

**[NeMo](https://github.com/NVIDIA/NeMo)** - NVIDIA's conversational AI toolkit
- GPU optimized
- Production ready
- Extensive model zoo

## Datasets

### English Datasets
**[LibriSpeech](http://www.openslr.org/12/)** - 1000 hours of English speech
- 🆓 Free
- Clean & other versions
- Standard benchmark

**[Common Voice](https://commonvoice.mozilla.org/)** - Mozilla's multilingual dataset
- Community contributed
- 100+ languages
- Diverse accents

**[VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)** - Celebrity speech dataset
- Speaker verification
- 7000+ speakers
- YouTube sourced

### Multilingual Datasets
**[MLS (Multilingual LibriSpeech)](https://www.openslr.org/94/)** - 8 languages
- 50k+ hours total
- Read speech
- Aligned transcripts

**[FLEURS](https://huggingface.co/datasets/google/fleurs)** - 102 languages
- Few-shot learning
- Sentence-level data
- Google's dataset

## Research Papers

### Foundational Papers
**[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** - Transformer architecture
- Basis for modern ASR
- Self-attention mechanism

**[Conformer](https://arxiv.org/abs/2005.08100)** - Convolution-augmented Transformer
- State-of-the-art ASR
- Combines CNN & Transformer

**[Whisper Paper](https://arxiv.org/abs/2212.04356)** - Robust speech recognition
- Weak supervision at scale
- Multilingual approach

## Tools & Applications

### Desktop Applications
**[Transcriber](https://github.com/cognitivecomputations/transcriber)** - Offline transcription tool
- 🆓 Free
- Privacy focused
- Multiple formats

**[MacWhisper](https://goodsnooze.gumroad.com/l/macwhisper)** - macOS transcription app
- 💰 Paid
- Native Mac app
- Batch processing

### Web Applications
**[Otter.ai](https://otter.ai/)** - Meeting transcription
- 🔄 Freemium
- Real-time collaboration
- Speaker identification

**[Rev](https://www.rev.com/)** - Professional transcription
- 💰 Paid
- Human + AI hybrid
- 99% accuracy

## Tutorials & Courses

### Getting Started
**[Speech Recognition in Python](https://realpython.com/python-speech-recognition/)** - Beginner tutorial
- 🟢 Beginner
- Step-by-step guide
- Code examples

**[Building an ASR System](https://www.assemblyai.com/blog/pytorch-speech-recognition/)** - PyTorch tutorial
- 🟡 Intermediate
- End-to-end implementation
- Custom model training

### Advanced Topics
**[Fine-tuning Whisper](https://huggingface.co/blog/fine-tune-whisper)** - HuggingFace guide
- 🔴 Advanced
- Custom language adaptation
- Performance optimization

**[Streaming ASR Tutorial](https://nvidia.github.io/NeMo/blogs/2023/08/17/streaming-asr-tutorial.html)** - Real-time recognition
- NVIDIA NeMo
- Low-latency systems
- Production deployment

## Best Practices

### Model Selection
1. **Accuracy vs Speed**: Whisper Large for accuracy, Silero for speed
2. **Language Support**: Check model's language coverage
3. **Deployment**: Consider model size and inference requirements

### Data Preparation
1. **Audio Quality**: 16kHz minimum, mono preferred
2. **Noise Reduction**: Pre-process for better results
3. **Segmentation**: Split long audio into chunks

### Performance Optimization
1. **Quantization**: Reduce model size with minimal accuracy loss
2. **Batching**: Process multiple audio files together
3. **GPU Acceleration**: Use CUDA for faster inference