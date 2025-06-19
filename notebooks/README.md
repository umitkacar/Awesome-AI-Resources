# 📓 Interactive Notebooks

Hands-on Jupyter notebooks for learning AI/ML concepts with Google Colab support.

## 🚀 Quick Start

Click any [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) badge to run notebooks directly in your browser!

## 📚 Notebook Categories

### 🟢 Beginner Level
Perfect for getting started with AI/ML

| Notebook | Description | Topics |
|----------|-------------|---------|
| [AutoML Quick Start](./beginner/automl_quickstart.ipynb) | Build ML models without coding | AutoGluon, PyCaret, H2O |
| [Computer Vision Basics](./beginner/cv_basics.ipynb) | Image classification made easy | CNN, Transfer Learning |
| [NLP Getting Started](./beginner/nlp_intro.ipynb) | Text processing fundamentals | Tokenization, Embeddings |

### 🟡 Intermediate Level
Dive deeper into AI concepts

| Notebook | Description | Topics |
|----------|-------------|---------|
| [Quantum ML Introduction](./intermediate/quantum_ml_intro.ipynb) | Quantum computing meets ML | Qiskit, PennyLane, VQC |
| [Advanced Computer Vision](./intermediate/advanced_cv.ipynb) | Segmentation & Detection | YOLO, SAM, GroundingDINO |
| [LLM Fine-tuning](./intermediate/llm_finetuning.ipynb) | Customize language models | LoRA, QLoRA, PEFT |

### 🔴 Advanced Level
State-of-the-art techniques

| Notebook | Description | Topics |
|----------|-------------|---------|
| [Neural Architecture Search](./advanced/nas_demo.ipynb) | Design neural networks automatically | DARTS, ENAS |
| [Multimodal AI](./advanced/multimodal_ai.ipynb) | Combine vision and language | CLIP, DALL-E, Flamingo |
| [Production ML Pipeline](./advanced/ml_production.ipynb) | Deploy models at scale | MLflow, Docker, K8s |

## 🛠️ Running Locally

### Prerequisites
```bash
# Create virtual environment
python -m venv ai-env
source ai-env/bin/activate  # On Windows: ai-env\Scripts\activate

# Install Jupyter
pip install jupyter notebook ipykernel
```

### Clone and Run
```bash
# Clone repository
git clone https://github.com/umitkacar/Awesome-AI-Resources.git
cd Awesome-AI-Resources/notebooks

# Start Jupyter
jupyter notebook
```

## 📝 Contributing Notebooks

We welcome notebook contributions! Please ensure:

1. **Clear Structure**: Introduction → Theory → Code → Results → Exercises
2. **Colab Compatible**: Test on Google Colab before submitting
3. **Dependencies**: List all required packages at the beginning
4. **Documentation**: Explain each code block thoroughly
5. **Exercises**: Include practice problems with solutions

### Notebook Template
```python
# Cell 1: Title and Badges
"""
# Your Notebook Title
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK)

Brief description of what this notebook covers.
"""

# Cell 2: Install Dependencies
!pip install -q required-packages

# Cell 3: Imports
import necessary_modules

# ... Your content ...

# Last Cell: Exercises and Resources
"""
## 🎯 Exercises
1. Exercise description
2. Another exercise

## 📚 Further Reading
- Resource links
- Documentation
"""
```

## 🏷️ Tags

- `beginner` - No prior ML knowledge required
- `intermediate` - Basic ML understanding needed
- `advanced` - Deep technical knowledge required
- `theory` - Focuses on mathematical concepts
- `practical` - Hands-on implementation
- `gpu-required` - Needs GPU for reasonable performance

## 🤝 Community

- 💬 [Discussions](https://github.com/umitkacar/Awesome-AI-Resources/discussions) - Ask questions
- 🐛 [Issues](https://github.com/umitkacar/Awesome-AI-Resources/issues) - Report problems
- 🌟 [Show & Tell](https://github.com/umitkacar/Awesome-AI-Resources/discussions/categories/show-and-tell) - Share your projects