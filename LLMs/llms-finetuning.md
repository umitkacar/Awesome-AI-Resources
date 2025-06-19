# LLMs Fine-tuning Guide

A comprehensive guide to fine-tuning Large Language Models with practical examples, best practices, and optimization techniques.

**Last Updated:** 2025-06-19

## 📚 Table of Contents
- [Introduction](#introduction)
- [Fine-tuning Methods](#fine-tuning-methods)
- [Data Preparation](#data-preparation)
- [Training Strategies](#training-strategies)
- [Hardware Requirements](#hardware-requirements)
- [Code Examples](#code-examples)
- [Best Practices](#best-practices)
- [Common Pitfalls](#common-pitfalls)

## Introduction

Fine-tuning LLMs allows you to adapt pre-trained models to specific tasks or domains while leveraging their existing knowledge.

### Why Fine-tune?
- **Domain Adaptation**: Specialize models for specific industries
- **Task Specialization**: Optimize for particular use cases
- **Style Alignment**: Match specific writing styles or formats
- **Performance**: Better results than zero-shot or few-shot prompting

## Fine-tuning Methods

### 1. Full Fine-tuning
- Updates all model parameters
- Requires significant computational resources
- Best for substantial domain shifts

```python
# Full fine-tuning example
from transformers import AutoModelForCausalLM, Trainer

model = AutoModelForCausalLM.from_pretrained("base-model")
# All parameters are trainable
for param in model.parameters():
    param.requires_grad = True
```

### 2. Parameter-Efficient Fine-Tuning (PEFT)

#### LoRA (Low-Rank Adaptation)
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)
model = get_peft_model(model, lora_config)
```

#### QLoRA (Quantized LoRA)
- 4-bit quantization + LoRA
- Dramatically reduces memory usage
- Minimal performance loss

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
```

#### Prefix Tuning
- Adds trainable prefix tokens
- Keeps original model frozen
- Good for task-specific adaptation

### 3. Instruction Tuning
- Fine-tuning on instruction-following datasets
- Improves zero-shot task generalization
- Popular datasets: Alpaca, Dolly, FLAN

## Data Preparation

### Dataset Formats

#### 1. Instruction Format
```json
{
    "instruction": "Summarize the following text",
    "input": "Long text to summarize...",
    "output": "Summary of the text"
}
```

#### 2. Chat Format
```json
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "User query"},
        {"role": "assistant", "content": "Model response"}
    ]
}
```

### Data Quality Checklist
- ✅ Remove duplicates
- ✅ Balance dataset categories
- ✅ Validate format consistency
- ✅ Check for bias and toxicity
- ✅ Ensure sufficient diversity

### Data Augmentation Techniques
- **Paraphrasing**: Generate variations of existing examples
- **Back-translation**: Translate and translate back
- **Token replacement**: Synonym substitution
- **Instruction variation**: Rephrase instructions

## Training Strategies

### Hyperparameter Optimization

#### Learning Rate Scheduling
```python
from transformers import get_linear_schedule_with_warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=1000
)
```

#### Key Hyperparameters
| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| Learning Rate | 1e-5 to 5e-4 | Lower for larger models |
| Batch Size | 4-32 | Limited by GPU memory |
| Gradient Accumulation | 1-8 | Simulate larger batches |
| Warmup Steps | 3-10% of total | Prevents instability |

### Training Techniques

#### Mixed Precision Training
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    fp16=True,  # or bf16=True for newer GPUs
    gradient_checkpointing=True,
)
```

#### Gradient Accumulation
```python
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 16
)
```

## Hardware Requirements

### GPU Memory Estimation

| Model Size | Full Fine-tuning | LoRA | QLoRA |
|------------|-----------------|------|-------|
| 7B params | 28-56 GB | 16-24 GB | 6-12 GB |
| 13B params | 52-104 GB | 24-32 GB | 10-16 GB |
| 30B params | 120-240 GB | 48-64 GB | 20-30 GB |

### Multi-GPU Strategies
- **Data Parallel**: Split batch across GPUs
- **Model Parallel**: Split model across GPUs
- **Pipeline Parallel**: Split layers across GPUs
- **ZeRO**: Optimizer state sharding

## Code Examples

### Complete Fine-tuning Script
```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import torch

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Prepare dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

dataset = load_dataset("your_dataset")
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=True,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# Train
trainer.train()
```

### LoRA Fine-tuning Example
```python
from peft import LoraConfig, get_peft_model, TaskType

# Configure LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# Apply LoRA
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062%
```

## Best Practices

### 1. Data Quality Over Quantity
- Clean, diverse data > large, noisy datasets
- Manual review of sample outputs
- Iterative dataset refinement

### 2. Evaluation Strategies
- **Automated Metrics**: Perplexity, BLEU, ROUGE
- **Human Evaluation**: Quality ratings, A/B testing
- **Task-specific Metrics**: Accuracy, F1, etc.
- **Safety Evaluation**: Bias, toxicity checks

### 3. Monitoring and Logging
```python
# Weights & Biases integration
import wandb

wandb.init(project="llm-finetuning")
training_args = TrainingArguments(
    report_to="wandb",
    run_name="experiment-1"
)
```

### 4. Checkpoint Management
- Save checkpoints regularly
- Keep best N checkpoints
- Version control configurations
- Document hyperparameters

## Common Pitfalls

### 1. Overfitting
- **Symptoms**: High training accuracy, poor generalization
- **Solutions**: 
  - Increase dropout
  - Reduce training epochs
  - Add more diverse data
  - Use validation-based early stopping

### 2. Catastrophic Forgetting
- **Symptoms**: Model loses general capabilities
- **Solutions**:
  - Lower learning rates
  - Mix general data with task data
  - Use regularization techniques
  - Consider LoRA instead of full fine-tuning

### 3. Memory Issues
- **Solutions**:
  - Use gradient accumulation
  - Enable gradient checkpointing
  - Try QLoRA or other quantization
  - Use DeepSpeed ZeRO

### 4. Training Instability
- **Solutions**:
  - Proper learning rate warmup
  - Clip gradients
  - Use stable optimizers (AdamW)
  - Monitor gradient norms

## Advanced Techniques

### 1. Multi-Task Fine-tuning
Train on multiple tasks simultaneously:
```python
# Mix multiple datasets
dataset = concatenate_datasets([
    dataset1.map(lambda x: {"task": "summarization", **x}),
    dataset2.map(lambda x: {"task": "translation", **x}),
])
```

### 2. Continued Pre-training
Domain adaptation before task fine-tuning:
```python
# First: continued pre-training on domain data
model.train_on_domain_data()
# Then: task-specific fine-tuning
model.finetune_on_task()
```

### 3. Reinforcement Learning from Human Feedback (RLHF)
- Train reward model
- Use PPO for policy optimization
- Frameworks: TRL, DeepSpeed-Chat

### 4. Direct Preference Optimization (DPO)
Simpler alternative to RLHF:
```python
from trl import DPOTrainer

dpo_trainer = DPOTrainer(
    model,
    ref_model,
    beta=0.1,
    train_dataset=preference_dataset,
)
```

## Resources and Tools

### Frameworks
- **Hugging Face Transformers**: Standard library for LLMs
- **PEFT**: Parameter-efficient fine-tuning
- **TRL**: Transformer Reinforcement Learning
- **Axolotl**: Fine-tuning framework with YAML configs
- **LLaMA-Factory**: All-in-one LLM fine-tuning

### Datasets
- **Alpaca**: Instruction-following dataset (52k examples)
- **Dolly**: High-quality instruction dataset (15k examples)
- **OpenAssistant**: Multi-turn conversations
- **ShareGPT**: Real ChatGPT conversations
- **LIMA**: Less Is More for Alignment (1k examples)

### Compute Resources
- **Google Colab**: Free tier with T4 GPU
- **Kaggle**: Free P100 GPUs
- **Lambda Labs**: Cost-effective GPU cloud
- **RunPod**: Flexible GPU rentals
- **vast.ai**: Marketplace for GPU compute

---

*Originally from umitkacar/LLMs-Finetuning repository*