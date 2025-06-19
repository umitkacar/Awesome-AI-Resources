# Image Quality Assessment with LLMs

Leveraging Large Language Models and Vision-Language Models for advanced Image Quality Assessment (IQA) tasks.

## 📚 Table of Contents
- [Introduction](#introduction)
- [Vision-Language Models for IQA](#vision-language-models-for-iqa)
- [Quality Metrics](#quality-metrics)
- [Implementation Approaches](#implementation-approaches)
- [Dataset and Benchmarks](#dataset-and-benchmarks)
- [Code Examples](#code-examples)
- [Advanced Techniques](#advanced-techniques)
- [Applications](#applications)

## Introduction

Image Quality Assessment (IQA) using LLMs represents a paradigm shift from traditional metrics to more human-aligned quality evaluation. By combining vision models with language understanding, we can assess both technical quality and semantic content.

### Why LLMs for IQA?

1. **Semantic Understanding**: Beyond pixel-level metrics
2. **Natural Language Feedback**: Descriptive quality assessment
3. **Multi-aspect Evaluation**: Technical + aesthetic + content quality
4. **Zero-shot Capability**: No need for quality-specific training
5. **Contextual Assessment**: Understanding image purpose and context

## Vision-Language Models for IQA

### 1. CLIP-based Assessment
```python
import torch
import clip
from PIL import Image

class CLIPQualityAssessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        
        # Quality descriptors
        self.quality_prompts = [
            "a high quality photo",
            "a low quality photo",
            "a blurry photo",
            "a sharp clear photo",
            "a well-exposed photo",
            "an overexposed photo",
            "a professional photo",
            "an amateur photo"
        ]
    
    def assess_quality(self, image_path):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        
        # Encode text prompts
        text_tokens = clip.tokenize(self.quality_prompts).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text_tokens)
            
            # Calculate similarities
            similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
        # Create quality report
        quality_scores = {
            prompt: float(score) 
            for prompt, score in zip(self.quality_prompts, similarities[0])
        }
        
        return quality_scores
```

### 2. BLIP-2 for Detailed Assessment
```python
from transformers import Blip2Processor, Blip2ForConditionalGeneration

class BLIP2QualityAnalyzer:
    def __init__(self):
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b"
        )
    
    def analyze_quality(self, image_path):
        image = Image.open(image_path)
        
        # Multiple quality-focused questions
        questions = [
            "What is the overall quality of this image?",
            "Describe any quality issues in this image.",
            "Is this image sharp or blurry?",
            "How is the lighting in this image?",
            "What technical improvements could be made to this image?"
        ]
        
        responses = {}
        for question in questions:
            inputs = self.processor(image, question, return_tensors="pt")
            out = self.model.generate(**inputs, max_length=100)
            answer = self.processor.decode(out[0], skip_special_tokens=True)
            responses[question] = answer
            
        return responses
```

### 3. LLaVA for Comprehensive Analysis
```python
class LLaVAQualityExpert:
    def __init__(self, model_path="liuhaotian/llava-v1.5-13b"):
        # Initialize LLaVA model
        self.model = load_llava_model(model_path)
        
    def comprehensive_assessment(self, image_path):
        prompt = """Please analyze this image and provide a detailed quality assessment covering:

1. **Technical Quality**:
   - Sharpness/Focus
   - Noise levels
   - Exposure
   - Color accuracy
   - Resolution adequacy

2. **Composition Quality**:
   - Framing
   - Balance
   - Rule of thirds
   - Leading lines

3. **Aesthetic Quality**:
   - Visual appeal
   - Mood/atmosphere
   - Color harmony
   - Overall impression

4. **Content Quality**:
   - Subject clarity
   - Relevance
   - Completeness

Please rate each aspect on a scale of 1-10 and provide specific feedback."""
        
        return self.model.generate(image_path, prompt)
```

## Quality Metrics

### Traditional Metrics Enhanced with LLMs

#### 1. Perceptual Quality Metrics
```python
class LLMEnhancedMetrics:
    def __init__(self):
        self.traditional_metrics = {
            'psnr': self.calculate_psnr,
            'ssim': self.calculate_ssim,
            'lpips': self.calculate_lpips
        }
        self.llm_assessor = VisionLanguageAssessor()
    
    def hybrid_assessment(self, image, reference=None):
        # Traditional metrics
        technical_scores = {}
        if reference:
            for metric_name, metric_func in self.traditional_metrics.items():
                technical_scores[metric_name] = metric_func(image, reference)
        
        # LLM assessment
        semantic_assessment = self.llm_assessor.assess(image)
        
        # Combine assessments
        return {
            'technical': technical_scores,
            'semantic': semantic_assessment,
            'overall': self.compute_overall_score(technical_scores, semantic_assessment)
        }
```

#### 2. No-Reference Quality Assessment
```python
class NoReferenceIQA:
    def __init__(self):
        self.quality_aspects = [
            "sharpness",
            "contrast",
            "colorfulness",
            "naturalness",
            "noisiness",
            "artifacts"
        ]
    
    def assess_without_reference(self, image_path):
        # Use VLM to assess each aspect
        assessments = {}
        
        for aspect in self.quality_aspects:
            prompt = f"Rate the {aspect} of this image on a scale of 1-10. Provide reasoning."
            response = self.vlm.analyze(image_path, prompt)
            
            # Parse score and reasoning
            score, reasoning = self.parse_response(response)
            assessments[aspect] = {
                'score': score,
                'reasoning': reasoning
            }
        
        return assessments
```

### Custom Quality Dimensions

```python
class DomainSpecificIQA:
    def __init__(self, domain="general"):
        self.domain_criteria = {
            "medical": [
                "diagnostic clarity",
                "anatomical visibility",
                "contrast resolution",
                "artifact presence"
            ],
            "satellite": [
                "ground resolution",
                "atmospheric clarity",
                "geometric accuracy",
                "radiometric quality"
            ],
            "portrait": [
                "skin tone accuracy",
                "eye sharpness",
                "background blur quality",
                "lighting flattery"
            ]
        }
        
    def assess_domain_quality(self, image_path):
        criteria = self.domain_criteria.get(self.domain, [])
        domain_assessment = {}
        
        for criterion in criteria:
            prompt = f"Evaluate the {criterion} of this {self.domain} image. Provide detailed feedback."
            assessment = self.vlm.analyze(image_path, prompt)
            domain_assessment[criterion] = assessment
            
        return domain_assessment
```

## Implementation Approaches

### 1. Multi-Model Ensemble
```python
class EnsembleIQA:
    def __init__(self):
        self.models = {
            'clip': CLIPQualityAssessor(),
            'blip2': BLIP2QualityAnalyzer(),
            'llava': LLaVAQualityExpert()
        }
        self.weights = {'clip': 0.3, 'blip2': 0.3, 'llava': 0.4}
    
    def ensemble_assessment(self, image_path):
        all_assessments = {}
        
        # Collect assessments from all models
        for model_name, model in self.models.items():
            all_assessments[model_name] = model.assess(image_path)
        
        # Weighted combination
        final_score = self.weighted_combine(all_assessments)
        
        # Consistency check
        consistency = self.check_consistency(all_assessments)
        
        return {
            'final_score': final_score,
            'individual_assessments': all_assessments,
            'consistency': consistency
        }
```

### 2. Progressive Refinement
```python
class ProgressiveIQA:
    def __init__(self):
        self.stages = [
            self.quick_assessment,
            self.detailed_technical,
            self.aesthetic_analysis,
            self.contextual_evaluation
        ]
    
    def assess_progressively(self, image_path, depth='full'):
        results = {}
        
        depth_map = {
            'quick': 1,
            'technical': 2,
            'aesthetic': 3,
            'full': 4
        }
        
        stages_to_run = self.stages[:depth_map.get(depth, 4)]
        
        for stage in stages_to_run:
            stage_result = stage(image_path, previous_results=results)
            results.update(stage_result)
            
            # Early stopping on low quality
            if results.get('quality_score', 10) < 3:
                results['early_stopped'] = True
                break
                
        return results
```

### 3. Interactive Quality Assessment
```python
class InteractiveIQA:
    def __init__(self):
        self.vlm = VisionLanguageModel()
        self.conversation_history = []
    
    def start_assessment(self, image_path):
        initial_prompt = """I'll help you assess the quality of this image. 
        Let me start with a general overview, then we can dive into specific aspects 
        you're interested in."""
        
        initial_assessment = self.vlm.analyze(image_path, initial_prompt)
        self.conversation_history.append({
            'role': 'assistant',
            'content': initial_assessment
        })
        
        return initial_assessment
    
    def ask_followup(self, question):
        self.conversation_history.append({
            'role': 'user',
            'content': question
        })
        
        response = self.vlm.continue_conversation(
            self.conversation_history
        )
        
        self.conversation_history.append({
            'role': 'assistant',
            'content': response
        })
        
        return response
```

## Dataset and Benchmarks

### IQA Datasets for VLM Training

```python
class IQADatasetBuilder:
    def __init__(self):
        self.datasets = {
            'tid2013': self.load_tid2013,
            'live': self.load_live,
            'koniq10k': self.load_koniq10k,
            'ava': self.load_ava  # Aesthetic Visual Analysis
        }
    
    def create_vlm_training_data(self, dataset_name):
        loader = self.datasets[dataset_name]
        raw_data = loader()
        
        vlm_data = []
        for item in raw_data:
            # Convert numerical scores to natural language
            quality_description = self.score_to_description(item['mos'])
            
            # Create multiple training examples
            examples = [
                {
                    'image': item['image_path'],
                    'question': "What is the quality of this image?",
                    'answer': quality_description
                },
                {
                    'image': item['image_path'],
                    'question': "Rate this image quality from 1-10",
                    'answer': f"I would rate this image {item['mos']}/10"
                },
                {
                    'image': item['image_path'],
                    'question': "Describe any quality issues in this image",
                    'answer': self.generate_issue_description(item)
                }
            ]
            
            vlm_data.extend(examples)
            
        return vlm_data
```

### Benchmark Evaluation

```python
class IQABenchmark:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        
    def evaluate(self):
        predictions = []
        ground_truth = []
        
        for sample in self.dataset:
            # Get model prediction
            pred_quality = self.model.predict_quality(sample['image'])
            predictions.append(pred_quality)
            ground_truth.append(sample['mos'])
        
        # Calculate metrics
        metrics = {
            'plcc': self.pearson_correlation(predictions, ground_truth),
            'srcc': self.spearman_correlation(predictions, ground_truth),
            'rmse': self.root_mean_square_error(predictions, ground_truth)
        }
        
        return metrics
```

## Code Examples

### Complete IQA Pipeline
```python
import torch
from PIL import Image
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class QualityReport:
    overall_score: float
    technical_scores: Dict[str, float]
    aesthetic_scores: Dict[str, float]
    semantic_feedback: str
    improvement_suggestions: List[str]

class ComprehensiveIQA:
    def __init__(self):
        self.initialize_models()
        
    def initialize_models(self):
        # Initialize various models
        self.clip_model = self.load_clip()
        self.aesthetic_model = self.load_aesthetic_predictor()
        self.vlm = self.load_vlm()
        
    def analyze_image(self, image_path: str) -> QualityReport:
        image = Image.open(image_path)
        
        # Technical analysis
        technical_scores = self.technical_analysis(image)
        
        # Aesthetic analysis
        aesthetic_scores = self.aesthetic_analysis(image)
        
        # Semantic analysis using VLM
        semantic_feedback = self.semantic_analysis(image_path)
        
        # Generate improvement suggestions
        suggestions = self.generate_suggestions(
            technical_scores, 
            aesthetic_scores, 
            semantic_feedback
        )
        
        # Calculate overall score
        overall_score = self.calculate_overall_score(
            technical_scores,
            aesthetic_scores
        )
        
        return QualityReport(
            overall_score=overall_score,
            technical_scores=technical_scores,
            aesthetic_scores=aesthetic_scores,
            semantic_feedback=semantic_feedback,
            improvement_suggestions=suggestions
        )
    
    def technical_analysis(self, image: Image) -> Dict[str, float]:
        return {
            'sharpness': self.measure_sharpness(image),
            'noise': self.measure_noise(image),
            'contrast': self.measure_contrast(image),
            'exposure': self.measure_exposure(image)
        }
    
    def aesthetic_analysis(self, image: Image) -> Dict[str, float]:
        # Use aesthetic prediction model
        features = self.extract_aesthetic_features(image)
        return {
            'composition': self.aesthetic_model.predict_composition(features),
            'color_harmony': self.aesthetic_model.predict_color_harmony(features),
            'interesting_content': self.aesthetic_model.predict_interest(features)
        }
    
    def semantic_analysis(self, image_path: str) -> str:
        prompt = """Analyze this image and provide:
        1. Description of what you see
        2. Assessment of image quality
        3. Any notable issues or strengths
        4. Context-appropriate quality evaluation
        """
        
        return self.vlm.generate(image_path, prompt)
    
    def generate_suggestions(self, technical, aesthetic, semantic):
        suggestions = []
        
        # Technical improvements
        if technical['sharpness'] < 0.5:
            suggestions.append("Increase sharpness or check focus")
        if technical['noise'] > 0.7:
            suggestions.append("Reduce noise through denoising")
        
        # Aesthetic improvements
        if aesthetic['composition'] < 0.5:
            suggestions.append("Improve composition using rule of thirds")
            
        # Parse semantic feedback for issues
        if "blur" in semantic.lower():
            suggestions.append("Address motion blur or focus issues")
            
        return suggestions
```

### Real-time Quality Monitoring
```python
class RealtimeQualityMonitor:
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.quality_history = []
        self.alert_callback = None
        
    def monitor_stream(self, video_stream):
        frame_count = 0
        
        for frame in video_stream:
            if frame_count % 30 == 0:  # Check every 30 frames
                quality = self.quick_quality_check(frame)
                self.quality_history.append({
                    'frame': frame_count,
                    'quality': quality,
                    'timestamp': time.time()
                })
                
                if quality < self.threshold:
                    self.trigger_alert(frame, quality)
                    
            frame_count += 1
    
    def quick_quality_check(self, frame):
        # Fast quality assessment
        return self.clip_scorer.score(frame)
    
    def trigger_alert(self, frame, quality):
        if self.alert_callback:
            detailed_analysis = self.detailed_quality_check(frame)
            self.alert_callback(frame, quality, detailed_analysis)
```

## Advanced Techniques

### 1. Stable Diffusion for Quality Enhancement
```python
class DiffusionQualityEnhancer:
    def __init__(self):
        self.sd_pipeline = self.load_stable_diffusion()
        self.quality_assessor = ComprehensiveIQA()
        
    def enhance_with_guidance(self, image_path):
        # First assess current quality
        initial_report = self.quality_assessor.analyze_image(image_path)
        
        # Generate enhancement prompt based on issues
        enhancement_prompt = self.create_enhancement_prompt(initial_report)
        
        # Use img2img with quality-focused prompt
        enhanced = self.sd_pipeline(
            prompt=enhancement_prompt,
            image=Image.open(image_path),
            strength=0.3,  # Preserve original content
            guidance_scale=7.5
        )
        
        # Verify improvement
        final_report = self.quality_assessor.analyze_image(enhanced)
        
        return {
            'original_quality': initial_report,
            'enhanced_image': enhanced,
            'enhanced_quality': final_report,
            'improvement': final_report.overall_score - initial_report.overall_score
        }
```

### 2. Multi-Resolution Quality Analysis
```python
class MultiResolutionIQA:
    def __init__(self):
        self.resolutions = [64, 128, 256, 512, 1024]
        
    def analyze_across_scales(self, image_path):
        image = Image.open(image_path)
        original_size = image.size
        
        scale_analyses = {}
        
        for resolution in self.resolutions:
            if resolution > max(original_size):
                continue
                
            # Resize image
            resized = image.resize(
                (resolution, resolution), 
                Image.Resampling.LANCZOS
            )
            
            # Analyze at this scale
            analysis = self.analyze_at_scale(resized, resolution)
            scale_analyses[resolution] = analysis
        
        # Aggregate findings
        return self.aggregate_multiscale_analysis(scale_analyses)
```

### 3. Adversarial Quality Assessment
```python
class AdversarialIQA:
    """Detect images designed to fool quality metrics"""
    
    def __init__(self):
        self.standard_assessor = StandardIQA()
        self.robust_assessor = RobustIQA()
        
    def detect_adversarial(self, image_path):
        # Get assessments from different models
        standard_score = self.standard_assessor.assess(image_path)
        robust_score = self.robust_assessor.assess(image_path)
        
        # Check for discrepancies
        discrepancy = abs(standard_score - robust_score)
        
        if discrepancy > 0.3:
            # Detailed analysis for potential adversarial
            detailed = self.detailed_adversarial_check(image_path)
            return {
                'is_adversarial': True,
                'confidence': discrepancy,
                'analysis': detailed
            }
        
        return {
            'is_adversarial': False,
            'quality_score': (standard_score + robust_score) / 2
        }
```

## Applications

### 1. Content Moderation
```python
class QualityBasedModeration:
    def __init__(self, quality_threshold=0.6):
        self.threshold = quality_threshold
        self.iqa_system = ComprehensiveIQA()
        
    def moderate_upload(self, image_path):
        report = self.iqa_system.analyze_image(image_path)
        
        if report.overall_score < self.threshold:
            return {
                'approved': False,
                'reason': 'Quality below threshold',
                'suggestions': report.improvement_suggestions,
                'detailed_scores': report.technical_scores
            }
        
        return {'approved': True, 'quality_report': report}
```

### 2. Automated Photo Curation
```python
class PhotoCurator:
    def __init__(self):
        self.quality_analyzer = ComprehensiveIQA()
        
    def curate_album(self, photo_paths, top_k=10):
        # Analyze all photos
        analyzed_photos = []
        
        for path in photo_paths:
            report = self.quality_analyzer.analyze_image(path)
            analyzed_photos.append({
                'path': path,
                'report': report,
                'score': report.overall_score
            })
        
        # Sort by quality
        analyzed_photos.sort(key=lambda x: x['score'], reverse=True)
        
        # Select top photos with diversity
        selected = self.select_diverse_top_photos(analyzed_photos, top_k)
        
        return selected
```

### 3. Real Estate Image Optimization
```python
class RealEstateImageOptimizer:
    def __init__(self):
        self.domain_iqa = DomainSpecificIQA(domain="real_estate")
        
    def optimize_listing_photos(self, property_photos):
        optimized_set = []
        
        for photo_path in property_photos:
            # Assess with real estate specific criteria
            assessment = self.domain_iqa.assess_domain_quality(photo_path)
            
            # Check specific requirements
            if assessment['interior_brightness'] < 0.6:
                enhanced = self.enhance_brightness(photo_path)
                optimized_set.append(enhanced)
            elif assessment['wide_angle_distortion'] > 0.7:
                corrected = self.correct_distortion(photo_path)
                optimized_set.append(corrected)
            else:
                optimized_set.append(photo_path)
                
        return self.order_by_room_type(optimized_set)
```

---

*Originally from umitkacar/IQA-with-LLMs repository*