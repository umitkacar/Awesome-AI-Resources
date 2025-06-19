# LLMs Tricks & Optimization Techniques

Advanced tricks, optimizations, and techniques for working with Large Language Models effectively.

**Last Updated:** 2025-06-19

## 📚 Table of Contents
- [Prompt Engineering Tricks](#prompt-engineering-tricks)
- [Inference Optimization](#inference-optimization)
- [Memory Optimization](#memory-optimization)
- [Cost Optimization](#cost-optimization)
- [Performance Tricks](#performance-tricks)
- [API Optimization](#api-optimization)
- [Advanced Techniques](#advanced-techniques)
- [Debugging & Troubleshooting](#debugging--troubleshooting)

## Prompt Engineering Tricks

### 1. Chain-of-Thought (CoT) Prompting
```python
# Standard prompt
prompt = "What is 235 * 89?"

# CoT prompt
cot_prompt = """What is 235 * 89? 
Let's think step by step:
1. First, I'll break this down...
2. 235 × 89 = 235 × (80 + 9)
3. = 235 × 80 + 235 × 9
4. = 18,800 + 2,115
5. = 20,915

So 235 * 89 = """
```

### 2. Few-Shot Learning Optimization
```python
# Optimize token usage with compressed examples
few_shot_template = """
Task: Extract key info
Ex1: "John, 25, Engineer" → Name:John|Age:25|Job:Engineer
Ex2: "Sarah, 30, Doctor" → Name:Sarah|Age:30|Job:Doctor
Now: "{input}" →"""

# Instead of verbose examples
verbose_template = """
Example 1:
Input: John is 25 years old and works as an Engineer
Output: The person's name is John, their age is 25, and their occupation is Engineer

Example 2:
...
"""
```

### 3. System Prompt Optimization
```python
# Concise, effective system prompts
system_prompts = {
    "coder": "You are an expert programmer. Be concise. Use comments.",
    "analyst": "Analyze data precisely. Format: Insight|Evidence|Impact",
    "teacher": "Explain simply. Use analogies. Check understanding."
}
```

### 4. Prompt Caching
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def get_cached_response(prompt_hash, model):
    return model.generate(prompt_hash)

def smart_prompt(prompt, model):
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    return get_cached_response(prompt_hash, model)
```

## Inference Optimization

### 1. Batch Processing
```python
# Instead of sequential calls
responses = []
for prompt in prompts:
    responses.append(model.generate(prompt))

# Use batch processing
responses = model.generate_batch(prompts, batch_size=32)
```

### 2. Streaming Responses
```python
# Stream tokens for better UX
async def stream_response(prompt):
    async for token in model.astream(prompt):
        yield token
        # Process/display token immediately
```

### 3. Speculative Decoding
```python
# Use smaller model to predict, larger to verify
class SpeculativeDecoder:
    def __init__(self, draft_model, target_model):
        self.draft = draft_model
        self.target = target_model
    
    def generate(self, prompt, k=4):
        # Draft model generates k tokens
        draft_tokens = self.draft.generate(prompt, max_tokens=k)
        # Target model verifies in single pass
        verified = self.target.verify(prompt + draft_tokens)
        return verified
```

### 4. KV-Cache Optimization
```python
# Reuse attention cache for similar prompts
class KVCacheManager:
    def __init__(self):
        self.cache = {}
    
    def get_or_compute(self, prefix, suffix):
        if prefix in self.cache:
            # Reuse cached KV for prefix
            return self.cache[prefix].extend(suffix)
        else:
            result = model.generate(prefix + suffix)
            self.cache[prefix] = result.kv_cache
            return result
```

## Memory Optimization

### 1. Dynamic Quantization
```python
import torch

# Quantize model on-the-fly
def dynamic_quantize(model):
    quantized = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    return quantized
```

### 2. Gradient Checkpointing
```python
# Trade compute for memory
model.gradient_checkpointing_enable()

# Custom checkpointing for specific layers
def custom_checkpoint(module, inputs):
    if training:
        return checkpoint(module, inputs)
    return module(inputs)
```

### 3. Offloading Strategies
```python
# CPU offloading for large models
from accelerate import cpu_offload

model = cpu_offload(
    model,
    execution_device="cuda",
    offload_buffers=True
)

# Disk offloading for extreme cases
from accelerate import disk_offload

model = disk_offload(
    model,
    offload_folder="./offload",
    execution_device="cuda"
)
```

### 4. Memory-Efficient Attention
```python
# Flash Attention implementation
from flash_attn import flash_attn_func

def efficient_attention(q, k, v):
    return flash_attn_func(q, k, v, causal=True)

# Sliding window attention
def sliding_window_attention(q, k, v, window_size=512):
    # Only attend to recent tokens
    k = k[:, -window_size:]
    v = v[:, -window_size:]
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)
```

## Cost Optimization

### 1. Token Usage Optimization
```python
# Compress prompts without losing information
def compress_prompt(prompt):
    # Remove redundant spaces
    prompt = " ".join(prompt.split())
    # Use abbreviations for common terms
    replacements = {
        "for example": "e.g.",
        "that is": "i.e.",
        "et cetera": "etc."
    }
    for long, short in replacements.items():
        prompt = prompt.replace(long, short)
    return prompt
```

### 2. Intelligent Caching
```python
import redis
import json

class LLMCache:
    def __init__(self):
        self.redis_client = redis.Redis()
        
    def get_or_generate(self, prompt, model, ttl=3600):
        cache_key = f"llm:{hashlib.md5(prompt.encode()).hexdigest()}"
        
        # Check cache
        cached = self.redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Generate and cache
        response = model.generate(prompt)
        self.redis_client.setex(
            cache_key, 
            ttl, 
            json.dumps(response)
        )
        return response
```

### 3. Model Routing
```python
# Route to appropriate model based on complexity
class ModelRouter:
    def __init__(self):
        self.models = {
            "simple": "gpt-3.5-turbo",
            "complex": "gpt-4",
            "code": "code-davinci-002"
        }
    
    def route(self, prompt, task_type=None):
        if task_type:
            return self.models.get(task_type, "simple")
        
        # Auto-detect complexity
        complexity = self.estimate_complexity(prompt)
        if complexity > 0.8:
            return self.models["complex"]
        return self.models["simple"]
```

## Performance Tricks

### 1. Parallel Generation
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def parallel_generate(prompts, model):
    with ThreadPoolExecutor() as executor:
        tasks = [
            asyncio.get_event_loop().run_in_executor(
                executor, model.generate, prompt
            )
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)
```

### 2. Response Validation
```python
# Automatic retry with validation
def generate_with_validation(prompt, model, validator, max_retries=3):
    for i in range(max_retries):
        response = model.generate(prompt)
        if validator(response):
            return response
        # Modify prompt for retry
        prompt = f"{prompt}\n(Please ensure the response is valid)"
    raise ValueError("Failed to generate valid response")
```

### 3. Output Parsing
```python
import json
import re

class SmartParser:
    @staticmethod
    def extract_json(text):
        # Find JSON in mixed text
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, text)
        for match in matches:
            try:
                return json.loads(match)
            except:
                continue
        return None
    
    @staticmethod
    def extract_code(text, language="python"):
        # Extract code blocks
        pattern = rf'```{language}\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        return matches
```

### 4. Adaptive Temperature
```python
# Adjust temperature based on output quality
class AdaptiveTemperature:
    def __init__(self, initial_temp=0.7):
        self.temp = initial_temp
        self.history = []
    
    def generate(self, prompt, model, quality_scorer):
        response = model.generate(prompt, temperature=self.temp)
        score = quality_scorer(response)
        
        # Adjust temperature
        if score < 0.5:
            self.temp = min(1.0, self.temp + 0.1)
        elif score > 0.8:
            self.temp = max(0.1, self.temp - 0.1)
        
        self.history.append((self.temp, score))
        return response
```

## API Optimization

### 1. Request Batching
```python
class BatchedAPIClient:
    def __init__(self, api_client, batch_size=10, wait_time=0.1):
        self.client = api_client
        self.batch_size = batch_size
        self.wait_time = wait_time
        self.queue = []
        
    async def request(self, prompt):
        future = asyncio.Future()
        self.queue.append((prompt, future))
        
        if len(self.queue) >= self.batch_size:
            await self._process_batch()
        else:
            asyncio.create_task(self._wait_and_process())
            
        return await future
    
    async def _process_batch(self):
        batch = self.queue[:self.batch_size]
        self.queue = self.queue[self.batch_size:]
        
        prompts = [p for p, _ in batch]
        responses = await self.client.batch_complete(prompts)
        
        for (_, future), response in zip(batch, responses):
            future.set_result(response)
```

### 2. Retry Logic
```python
import backoff

@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=5,
    max_time=300
)
def robust_api_call(prompt, model):
    return model.generate(prompt)
```

### 3. Rate Limiting
```python
from typing import Optional
import time

class RateLimiter:
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    def wait_if_needed(self):
        now = time.time()
        # Remove calls older than 1 minute
        self.calls = [t for t in self.calls if now - t < 60]
        
        if len(self.calls) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.calls[0]) + 0.1
            time.sleep(sleep_time)
            
        self.calls.append(time.time())
```

## Advanced Techniques

### 1. Mixture of Experts (MoE) Routing
```python
class MoERouter:
    def __init__(self, experts):
        self.experts = experts  # {"math": model1, "code": model2, ...}
        self.router = load_router_model()
    
    def generate(self, prompt):
        # Router determines best expert
        expert_scores = self.router.predict(prompt)
        best_expert = max(expert_scores, key=expert_scores.get)
        
        # Use weighted combination for uncertainty
        if expert_scores[best_expert] < 0.8:
            responses = {}
            for expert, model in self.experts.items():
                if expert_scores[expert] > 0.2:
                    responses[expert] = model.generate(prompt)
            return self.combine_responses(responses, expert_scores)
        
        return self.experts[best_expert].generate(prompt)
```

### 2. Constrained Generation
```python
# Force specific output format
class ConstrainedGenerator:
    def __init__(self, model, grammar):
        self.model = model
        self.grammar = grammar
    
    def generate(self, prompt):
        # Use grammar to constrain token selection
        def logits_processor(input_ids, scores):
            valid_tokens = self.grammar.get_valid_tokens(input_ids)
            mask = torch.ones_like(scores) * -float('inf')
            mask[valid_tokens] = 0
            return scores + mask
        
        return self.model.generate(
            prompt,
            logits_processor=logits_processor
        )
```

### 3. Self-Consistency
```python
# Generate multiple outputs and select best
def self_consistency_generate(prompt, model, n=5):
    responses = [
        model.generate(prompt, temperature=0.7)
        for _ in range(n)
    ]
    
    # For factual questions, choose most common
    if is_factual(prompt):
        return max(set(responses), key=responses.count)
    
    # For creative tasks, score and select
    scores = [score_response(r) for r in responses]
    return responses[scores.index(max(scores))]
```

### 4. Retrieval-Augmented Generation (RAG) Optimization
```python
class OptimizedRAG:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator
        self.cache = {}
    
    def generate(self, query):
        # Semantic caching
        similar_query = self.find_similar_cached(query)
        if similar_query:
            return self.cache[similar_query]
        
        # Adaptive retrieval
        docs = self.retriever.search(query, k=3)
        
        # Rerank documents
        reranked = self.rerank_docs(query, docs)
        
        # Generate with citations
        context = "\n".join([d.text for d in reranked])
        response = self.generator.generate(
            f"Context: {context}\n\nQuery: {query}\n\nAnswer:"
        )
        
        self.cache[query] = response
        return response
```

## Debugging & Troubleshooting

### 1. Token Analysis
```python
def analyze_tokens(text, tokenizer):
    tokens = tokenizer.encode(text)
    print(f"Token count: {len(tokens)}")
    print(f"Estimated cost: ${len(tokens) * 0.00002:.4f}")
    
    # Find expensive tokens
    token_lengths = [len(tokenizer.decode([t])) for t in tokens]
    inefficient = [
        (i, tokenizer.decode([tokens[i]]))
        for i, l in enumerate(token_lengths)
        if l == 1 and i < len(tokens)-1
    ]
    print(f"Single-char tokens: {len(inefficient)}")
```

### 2. Response Quality Monitoring
```python
class QualityMonitor:
    def __init__(self):
        self.metrics = {
            "coherence": [],
            "relevance": [],
            "completeness": []
        }
    
    def evaluate(self, prompt, response):
        scores = {
            "coherence": self.check_coherence(response),
            "relevance": self.check_relevance(prompt, response),
            "completeness": self.check_completeness(response)
        }
        
        for metric, score in scores.items():
            self.metrics[metric].append(score)
            
        # Alert on quality degradation
        if any(score < 0.5 for score in scores.values()):
            self.alert_quality_issue(prompt, response, scores)
        
        return scores
```

### 3. Error Pattern Detection
```python
class ErrorDetector:
    def __init__(self):
        self.error_patterns = {
            "repetition": r'(.{10,})\1{2,}',
            "incomplete": r'[^.!?]\s*$',
            "hallucination": r'\b(definitely|certainly|always|never)\b',
            "refusal": r"(I can't|I cannot|I'm unable)",
        }
    
    def check(self, response):
        issues = []
        for error_type, pattern in self.error_patterns.items():
            if re.search(pattern, response):
                issues.append(error_type)
        return issues
```

## Pro Tips

### 1. **Temperature Scheduling**
Start with higher temperature for creativity, decrease for refinement:
```python
temps = [0.9, 0.7, 0.5, 0.3]
for temp in temps:
    response = model.generate(prompt, temperature=temp)
    prompt = f"Improve this: {response}"
```

### 2. **Context Window Management**
```python
def sliding_context(messages, max_tokens=4000):
    # Keep system message + recent context
    system = messages[0]
    recent = messages[1:]
    
    while count_tokens(recent) > max_tokens:
        recent = recent[1:]  # Remove oldest
    
    return [system] + recent
```

### 3. **Prompt Templates Library**
```python
TEMPLATES = {
    "analysis": "Analyze {subject}:\n1. Key points\n2. Evidence\n3. Implications\n\n",
    "code_review": "Review this {language} code:\n```{code}```\nFocus on: {aspects}\n",
    "summarize": "Summarize in {length} words, focusing on {focus}:\n{text}\n"
}
```

### 4. **Output Validation Pipeline**
```python
validators = [
    check_length,
    check_format,
    check_content_safety,
    check_factual_accuracy
]

response = model.generate(prompt)
for validator in validators:
    response = validator(response)
```

---

*Originally from umitkacar/LLMs-tricks repository*