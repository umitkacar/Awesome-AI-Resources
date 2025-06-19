# Food Generative AI: Stable Diffusion + LLMs for Culinary Innovation

## Overview

Food Generative AI combines the power of Stable Diffusion for food image generation with Large Language Models (LLMs) for recipe creation, ingredient analysis, and culinary insights. This comprehensive guide explores the intersection of AI and gastronomy, providing practical implementations for food-related generative AI applications.

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Food Image Generation with Stable Diffusion](#food-image-generation-with-stable-diffusion)
4. [Recipe Generation with LLMs](#recipe-generation-with-llms)
5. [Multimodal Food AI Pipeline](#multimodal-food-ai-pipeline)
6. [Training Custom Food Models](#training-custom-food-models)
7. [Best Practices](#best-practices)
8. [Advanced Techniques](#advanced-techniques)
9. [Production Deployment](#production-deployment)
10. [Resources and References](#resources-and-references)

## Introduction

Food Generative AI revolutionizes culinary creativity by:
- Generating photorealistic food images from text descriptions
- Creating novel recipes based on ingredients or dietary preferences
- Analyzing nutritional content from food images
- Suggesting ingredient substitutions and variations
- Generating plating and presentation ideas

## Architecture Overview

### System Components

```python
# Core architecture for Food Generative AI system
class FoodGenerativeAI:
    def __init__(self):
        self.image_generator = StableDiffusionFoodModel()
        self.recipe_generator = RecipeLLM()
        self.nutrition_analyzer = NutritionAI()
        self.ingredient_detector = IngredientVision()
        
    def generate_food_concept(self, prompt, dietary_restrictions=None):
        # Generate complete food concept with image and recipe
        image = self.image_generator.generate(prompt)
        recipe = self.recipe_generator.create_recipe(prompt, dietary_restrictions)
        nutrition = self.nutrition_analyzer.analyze(image)
        
        return {
            'image': image,
            'recipe': recipe,
            'nutrition': nutrition
        }
```

## Food Image Generation with Stable Diffusion

### Setting Up Stable Diffusion for Food

```python
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import numpy as np

class FoodImageGenerator:
    def __init__(self, model_id="stabilityai/stable-diffusion-2-1"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.pipe = self.pipe.to(self.device)
        
        # Optimize with DPM-Solver for faster generation
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
    def generate_food_image(self, prompt, style="professional food photography"):
        # Enhanced prompt engineering for food
        enhanced_prompt = f"{prompt}, {style}, highly detailed, 8k resolution, \
                          professional lighting, shallow depth of field, \
                          appetizing, fresh ingredients, vibrant colors"
        
        negative_prompt = "blurry, low quality, distorted, unappetizing, \
                          burnt, raw, spoiled, cartoon, illustration"
        
        image = self.pipe(
            enhanced_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            height=768,
            width=768
        ).images[0]
        
        return image

# Example usage
generator = FoodImageGenerator()
image = generator.generate_food_image("gourmet pasta carbonara with truffle shavings")
```

### Fine-tuning for Specific Cuisines

```python
from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn as nn

class CuisineSpecificModel(nn.Module):
    def __init__(self, base_model, cuisine_type):
        super().__init__()
        self.base_model = base_model
        self.cuisine_adapter = self._create_cuisine_adapter(cuisine_type)
        
    def _create_cuisine_adapter(self, cuisine_type):
        # Cuisine-specific style embeddings
        cuisine_styles = {
            'italian': {
                'colors': 'warm, rustic, tomato red, basil green',
                'presentation': 'elegant, traditional, family-style',
                'lighting': 'warm, natural, Mediterranean'
            },
            'japanese': {
                'colors': 'minimal, natural, subtle',
                'presentation': 'precise, artistic, zen-like',
                'lighting': 'soft, diffused, clean'
            },
            'french': {
                'colors': 'rich, sophisticated, butter yellow',
                'presentation': 'refined, classical, sauce-focused',
                'lighting': 'dramatic, professional, bistro-style'
            }
        }
        
        return cuisine_styles.get(cuisine_type, cuisine_styles['italian'])
    
    def generate_cuisine_specific(self, dish_description):
        style_prompt = f"{dish_description}, {self.cuisine_adapter['colors']}, \
                        {self.cuisine_adapter['presentation']}, \
                        {self.cuisine_adapter['lighting']}"
        
        return self.base_model.generate_food_image(style_prompt)
```

## Recipe Generation with LLMs

### Advanced Recipe Generation System

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

class RecipeGeneratorLLM:
    def __init__(self, model_name="meta-llama/Llama-2-13b-chat-hf"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
    def generate_recipe(self, ingredients, dietary_restrictions=None, 
                       cuisine_type=None, difficulty="medium"):
        prompt = self._create_recipe_prompt(
            ingredients, dietary_restrictions, cuisine_type, difficulty
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=1500,
                temperature=0.8,
                top_p=0.9,
                do_sample=True
            )
        
        recipe_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._parse_recipe(recipe_text)
    
    def _create_recipe_prompt(self, ingredients, dietary_restrictions, 
                            cuisine_type, difficulty):
        prompt = f"""Create a detailed recipe with the following requirements:
        
        Ingredients available: {', '.join(ingredients)}
        Dietary restrictions: {dietary_restrictions or 'None'}
        Cuisine type: {cuisine_type or 'Any'}
        Difficulty level: {difficulty}
        
        Please provide:
        1. Recipe name
        2. Serving size
        3. Prep time and cook time
        4. Complete ingredient list with measurements
        5. Step-by-step instructions
        6. Nutritional information per serving
        7. Chef's tips and variations
        
        Format the recipe in a clear, structured manner."""
        
        return prompt
    
    def _parse_recipe(self, recipe_text):
        # Parse the generated text into structured format
        recipe_data = {
            'name': self._extract_section(recipe_text, 'Recipe name:', '\n'),
            'servings': self._extract_section(recipe_text, 'Serving size:', '\n'),
            'prep_time': self._extract_section(recipe_text, 'Prep time:', '\n'),
            'cook_time': self._extract_section(recipe_text, 'Cook time:', '\n'),
            'ingredients': self._extract_list(recipe_text, 'Ingredients:', 'Instructions:'),
            'instructions': self._extract_list(recipe_text, 'Instructions:', 'Nutritional'),
            'nutrition': self._extract_section(recipe_text, 'Nutritional information:', 'Chef'),
            'tips': self._extract_section(recipe_text, "Chef's tips:", None)
        }
        
        return recipe_data
    
    def _extract_section(self, text, start_marker, end_marker):
        start = text.find(start_marker)
        if start == -1:
            return ""
        
        start += len(start_marker)
        if end_marker:
            end = text.find(end_marker, start)
            return text[start:end].strip() if end != -1 else text[start:].strip()
        
        return text[start:].strip()
    
    def _extract_list(self, text, start_marker, end_marker):
        section = self._extract_section(text, start_marker, end_marker)
        return [item.strip() for item in section.split('\n') if item.strip()]
```

### Ingredient-Based Recipe Suggestions

```python
class SmartRecipeSuggester:
    def __init__(self, recipe_llm, ingredient_db):
        self.recipe_llm = recipe_llm
        self.ingredient_db = ingredient_db
        
    def suggest_recipes_from_pantry(self, available_ingredients, preferences):
        # Analyze ingredient compatibility
        compatible_groups = self._find_compatible_ingredients(available_ingredients)
        
        # Generate multiple recipe options
        recipe_suggestions = []
        for group in compatible_groups:
            recipe = self.recipe_llm.generate_recipe(
                ingredients=group,
                dietary_restrictions=preferences.get('dietary_restrictions'),
                cuisine_type=preferences.get('preferred_cuisine')
            )
            
            # Calculate match score
            recipe['match_score'] = self._calculate_match_score(
                recipe, available_ingredients, preferences
            )
            
            recipe_suggestions.append(recipe)
        
        # Sort by match score
        return sorted(recipe_suggestions, key=lambda x: x['match_score'], reverse=True)
    
    def _find_compatible_ingredients(self, ingredients):
        # Group ingredients by flavor profiles and common pairings
        compatibility_matrix = self.ingredient_db.get_compatibility_matrix()
        
        groups = []
        for i in range(len(ingredients)):
            compatible_group = [ingredients[i]]
            for j in range(i+1, len(ingredients)):
                if compatibility_matrix.get((ingredients[i], ingredients[j]), 0) > 0.7:
                    compatible_group.append(ingredients[j])
            
            if len(compatible_group) >= 3:
                groups.append(compatible_group)
        
        return groups
    
    def _calculate_match_score(self, recipe, available_ingredients, preferences):
        score = 0.0
        
        # Ingredient utilization score
        used_ingredients = sum(1 for ing in available_ingredients 
                             if ing in ' '.join(recipe['ingredients']))
        score += (used_ingredients / len(available_ingredients)) * 0.4
        
        # Dietary preference match
        if preferences.get('dietary_restrictions'):
            if preferences['dietary_restrictions'] in recipe.get('tags', []):
                score += 0.3
        
        # Cuisine preference match
        if preferences.get('preferred_cuisine'):
            if preferences['preferred_cuisine'] in recipe.get('cuisine_type', ''):
                score += 0.3
        
        return score
```

## Multimodal Food AI Pipeline

### Integrated Vision-Language Model

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

class MultimodalFoodAnalyzer:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ).to("cuda")
        
        self.food_classifier = self._load_food_classifier()
        self.nutrition_estimator = self._load_nutrition_model()
        
    def analyze_food_image(self, image_path):
        image = Image.open(image_path)
        
        # Generate description
        inputs = self.processor(image, return_tensors="pt").to("cuda")
        out = self.model.generate(**inputs, max_length=50)
        description = self.processor.decode(out[0], skip_special_tokens=True)
        
        # Identify ingredients
        ingredients = self._detect_ingredients(image)
        
        # Estimate nutrition
        nutrition = self._estimate_nutrition(image, ingredients)
        
        # Generate recipe suggestions
        recipe_suggestions = self._suggest_recipes_from_image(ingredients, description)
        
        return {
            'description': description,
            'ingredients': ingredients,
            'nutrition': nutrition,
            'recipe_suggestions': recipe_suggestions
        }
    
    def _detect_ingredients(self, image):
        # Custom ingredient detection model
        features = self._extract_visual_features(image)
        
        # Multi-label classification for ingredients
        predictions = self.food_classifier(features)
        
        ingredients = []
        for ingredient, confidence in predictions:
            if confidence > 0.7:
                ingredients.append({
                    'name': ingredient,
                    'confidence': confidence,
                    'category': self._get_ingredient_category(ingredient)
                })
        
        return ingredients
    
    def _estimate_nutrition(self, image, ingredients):
        # Estimate portion size from image
        portion_size = self._estimate_portion_size(image)
        
        # Calculate nutrition based on ingredients and portions
        total_nutrition = {
            'calories': 0,
            'protein': 0,
            'carbs': 0,
            'fat': 0,
            'fiber': 0
        }
        
        for ingredient in ingredients:
            nutrition_data = self.nutrition_estimator.get_nutrition(
                ingredient['name'], 
                portion_size * ingredient['confidence']
            )
            
            for nutrient, value in nutrition_data.items():
                total_nutrition[nutrient] += value
        
        return total_nutrition
```

### Real-time Food Style Transfer

```python
import torch.nn as nn
import torchvision.transforms as transforms

class FoodStyleTransfer:
    def __init__(self):
        self.style_encoder = self._build_style_encoder()
        self.content_encoder = self._build_content_encoder()
        self.decoder = self._build_decoder()
        
    def transfer_food_style(self, content_image, style_reference, preservation_ratio=0.7):
        """
        Transfer cooking style while preserving food identity
        """
        # Extract content features (ingredients, composition)
        content_features = self.content_encoder(content_image)
        
        # Extract style features (plating, garnish, presentation)
        style_features = self.style_encoder(style_reference)
        
        # Adaptive instance normalization for style transfer
        stylized_features = self._adaptive_instance_norm(
            content_features, 
            style_features,
            preservation_ratio
        )
        
        # Generate stylized food image
        stylized_image = self.decoder(stylized_features)
        
        # Post-processing for food realism
        return self._enhance_food_realism(stylized_image)
    
    def _adaptive_instance_norm(self, content_feat, style_feat, alpha):
        """
        AdaIN specifically tuned for food images
        """
        size = content_feat.size()
        style_mean, style_std = self._calc_mean_std(style_feat)
        content_mean, content_std = self._calc_mean_std(content_feat)
        
        # Preserve food texture while applying style
        normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
        
        # Blend original and stylized features
        stylized = normalized_feat * style_std.expand(size) + style_mean.expand(size)
        return alpha * stylized + (1 - alpha) * content_feat
    
    def _enhance_food_realism(self, image):
        """
        Post-processing to ensure food looks appetizing
        """
        # Enhance colors for food appeal
        image = self._enhance_food_colors(image)
        
        # Add subtle highlights for freshness
        image = self._add_freshness_highlights(image)
        
        # Ensure proper food texture
        image = self._preserve_food_textures(image)
        
        return image
```

## Training Custom Food Models

### Dataset Preparation

```python
import os
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FoodDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.annotations = self._load_annotations(annotation_file)
        self.transform = transform or self._get_default_transform()
        
    def _get_default_transform(self):
        return A.Compose([
            A.Resize(512, 512),
            A.RandomCrop(480, 480),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1),
            A.GaussianBlur(p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def _load_annotations(self, annotation_file):
        """
        Load annotations with food-specific metadata
        """
        annotations = []
        with open(annotation_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                annotations.append({
                    'image_path': data['image_path'],
                    'caption': data['caption'],
                    'ingredients': data['ingredients'],
                    'cuisine': data['cuisine'],
                    'cooking_method': data['cooking_method'],
                    'dietary_tags': data['dietary_tags']
                })
        
        return annotations
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image = Image.open(os.path.join(self.image_dir, ann['image_path']))
        
        if self.transform:
            image = self.transform(image=np.array(image))['image']
        
        # Create comprehensive caption with metadata
        caption = self._create_detailed_caption(ann)
        
        return {
            'image': image,
            'caption': caption,
            'metadata': ann
        }
    
    def _create_detailed_caption(self, annotation):
        """
        Generate rich captions for better model training
        """
        base_caption = annotation['caption']
        
        # Add ingredient information
        if annotation['ingredients']:
            ingredients_str = ', '.join(annotation['ingredients'][:5])
            base_caption += f" Made with {ingredients_str}"
        
        # Add cuisine type
        if annotation['cuisine']:
            base_caption += f". {annotation['cuisine']} cuisine"
        
        # Add cooking method
        if annotation['cooking_method']:
            base_caption += f", {annotation['cooking_method']}"
        
        return base_caption
```

### Fine-tuning Strategy

```python
from accelerate import Accelerator
from diffusers import DDPMScheduler, UNet2DConditionModel
import torch.optim as optim

class FoodDiffusionTrainer:
    def __init__(self, model_name="runwayml/stable-diffusion-v1-5"):
        self.accelerator = Accelerator()
        self.model = self._load_model(model_name)
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            model_name, 
            subfolder="scheduler"
        )
        
    def train_food_model(self, train_dataset, epochs=100, learning_rate=1e-5):
        # Prepare dataloader
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=4, 
            shuffle=True
        )
        
        # Setup optimizer with different learning rates
        optimizer = optim.AdamW([
            {'params': self.model.unet.parameters(), 'lr': learning_rate},
            {'params': self.model.text_encoder.parameters(), 'lr': learning_rate * 0.1}
        ])
        
        # Learning rate scheduler
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10, 
            T_mult=2
        )
        
        # Prepare for distributed training
        self.model, optimizer, train_dataloader = self.accelerator.prepare(
            self.model, optimizer, train_dataloader
        )
        
        # Training loop
        for epoch in range(epochs):
            for batch in train_dataloader:
                with self.accelerator.accumulate(self.model):
                    # Add noise to images
                    noise = torch.randn_like(batch['image'])
                    timesteps = torch.randint(
                        0, self.noise_scheduler.num_train_timesteps, 
                        (batch['image'].shape[0],), 
                        device=batch['image'].device
                    )
                    
                    noisy_images = self.noise_scheduler.add_noise(
                        batch['image'], noise, timesteps
                    )
                    
                    # Predict noise
                    noise_pred = self.model(
                        noisy_images, 
                        timesteps, 
                        batch['caption']
                    ).sample
                    
                    # Calculate loss with food-specific weighting
                    loss = self._calculate_food_aware_loss(
                        noise_pred, noise, batch['metadata']
                    )
                    
                    self.accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
            
            # Validation and checkpointing
            if epoch % 10 == 0:
                self._validate_and_save(epoch)
    
    def _calculate_food_aware_loss(self, pred, target, metadata):
        """
        Custom loss function that emphasizes food-relevant features
        """
        base_loss = F.mse_loss(pred, target)
        
        # Additional loss components for food quality
        edge_loss = self._edge_preservation_loss(pred, target)
        color_loss = self._food_color_consistency_loss(pred, target)
        
        # Weight based on food type
        food_weight = self._get_food_type_weight(metadata)
        
        return base_loss + 0.1 * edge_loss + 0.05 * color_loss * food_weight
```

## Best Practices

### Prompt Engineering for Food Images

```python
class FoodPromptEngineer:
    def __init__(self):
        self.style_templates = {
            'professional': "professional food photography, shallow depth of field, \
                           studio lighting, high-end restaurant presentation",
            'rustic': "rustic style, natural lighting, wooden table, \
                      homemade, cozy atmosphere, farm-to-table",
            'modern': "modern plating, minimalist, geometric presentation, \
                      fine dining, molecular gastronomy",
            'street': "street food style, vibrant, casual, paper wrapping, \
                      authentic, local market"
        }
        
        self.enhancement_terms = {
            'texture': ['crispy', 'fluffy', 'creamy', 'crunchy', 'tender'],
            'freshness': ['fresh', 'just-cooked', 'steaming', 'glistening', 'vibrant'],
            'appeal': ['appetizing', 'mouth-watering', 'delicious-looking', 'tempting']
        }
    
    def create_optimized_prompt(self, dish_name, style='professional', 
                               enhancements=None):
        base_prompt = f"{dish_name}"
        
        # Add style
        style_addition = self.style_templates.get(style, self.style_templates['professional'])
        
        # Add enhancements
        if enhancements:
            enhancement_words = []
            for category, words in self.enhancement_terms.items():
                if category in enhancements:
                    enhancement_words.extend(words[:2])
            
            enhancement_str = ', '.join(enhancement_words)
            base_prompt = f"{base_prompt}, {enhancement_str}"
        
        # Combine all elements
        full_prompt = f"{base_prompt}, {style_addition}, 8k quality, \
                       award-winning food photography"
        
        return full_prompt
    
    def create_negative_prompt(self, avoid_list=None):
        base_negative = "blurry, low quality, distorted, unappetizing, \
                        overcooked, undercooked, messy plating, dirty plate"
        
        if avoid_list:
            additional_negatives = ', '.join(avoid_list)
            return f"{base_negative}, {additional_negatives}"
        
        return base_negative
```

### Quality Control and Validation

```python
class FoodImageQualityChecker:
    def __init__(self):
        self.quality_model = self._load_quality_assessment_model()
        self.food_detector = self._load_food_detection_model()
        
    def assess_food_image_quality(self, image):
        """
        Comprehensive quality assessment for generated food images
        """
        scores = {
            'technical_quality': self._assess_technical_quality(image),
            'food_realism': self._assess_food_realism(image),
            'composition': self._assess_composition(image),
            'color_appeal': self._assess_color_appeal(image),
            'presentation': self._assess_presentation(image)
        }
        
        # Calculate overall score
        weights = {
            'technical_quality': 0.2,
            'food_realism': 0.3,
            'composition': 0.2,
            'color_appeal': 0.15,
            'presentation': 0.15
        }
        
        overall_score = sum(scores[k] * weights[k] for k in scores)
        
        return {
            'overall_score': overall_score,
            'detailed_scores': scores,
            'recommendations': self._generate_improvement_recommendations(scores)
        }
    
    def _assess_food_realism(self, image):
        """
        Check if the generated food looks realistic
        """
        # Detect if food is present
        food_detection = self.food_detector(image)
        
        if not food_detection['food_detected']:
            return 0.0
        
        # Check for common unrealistic artifacts
        realism_checks = {
            'texture_quality': self._check_texture_realism(image),
            'lighting_consistency': self._check_lighting_consistency(image),
            'proportion_accuracy': self._check_proportions(image),
            'ingredient_visibility': self._check_ingredient_clarity(image)
        }
        
        return np.mean(list(realism_checks.values()))
    
    def _generate_improvement_recommendations(self, scores):
        recommendations = []
        
        if scores['technical_quality'] < 0.7:
            recommendations.append("Increase inference steps or adjust guidance scale")
        
        if scores['food_realism'] < 0.7:
            recommendations.append("Refine prompt with more specific food descriptors")
        
        if scores['composition'] < 0.7:
            recommendations.append("Add composition guidelines to prompt (rule of thirds, etc.)")
        
        if scores['color_appeal'] < 0.7:
            recommendations.append("Enhance color vibrancy in prompt or post-processing")
        
        return recommendations
```

## Advanced Techniques

### Multi-Stage Generation Pipeline

```python
class AdvancedFoodGenerator:
    def __init__(self):
        self.base_generator = FoodImageGenerator()
        self.inpainting_model = self._load_inpainting_model()
        self.super_resolution = self._load_super_resolution()
        self.style_mixer = FoodStyleTransfer()
        
    def generate_complex_dish(self, recipe_data, style_preferences):
        """
        Generate complex food images with multiple components
        """
        # Stage 1: Generate base plate/background
        base_prompt = f"empty {style_preferences['plate_style']} plate, \
                       {style_preferences['background']}"
        base_image = self.base_generator.generate_food_image(base_prompt)
        
        # Stage 2: Add main components
        for component in recipe_data['components']:
            component_mask = self._generate_placement_mask(
                base_image, 
                component['position']
            )
            
            component_prompt = self._create_component_prompt(component)
            component_image = self.inpainting_model(
                base_image,
                component_mask,
                component_prompt
            )
            
            base_image = component_image
        
        # Stage 3: Add garnishes and final touches
        garnish_image = self._add_garnishes(
            base_image, 
            recipe_data['garnishes']
        )
        
        # Stage 4: Enhance resolution and details
        final_image = self.super_resolution(garnish_image)
        
        # Stage 5: Apply style consistency
        if style_preferences.get('reference_image'):
            final_image = self.style_mixer.transfer_food_style(
                final_image,
                style_preferences['reference_image'],
                preservation_ratio=0.8
            )
        
        return final_image
    
    def _create_component_prompt(self, component):
        """
        Create detailed prompts for individual food components
        """
        base = component['name']
        
        # Add cooking method
        if component.get('cooking_method'):
            base += f", {component['cooking_method']}"
        
        # Add texture descriptors
        if component.get('texture'):
            base += f", {component['texture']} texture"
        
        # Add color and appearance
        if component.get('appearance'):
            base += f", {component['appearance']}"
        
        return base + ", photorealistic, high detail"
```

### Real-time Recipe Adaptation

```python
class AdaptiveRecipeGenerator:
    def __init__(self):
        self.recipe_llm = RecipeGeneratorLLM()
        self.substitution_engine = IngredientSubstitutionEngine()
        self.cultural_adapter = CuisineAdapter()
        
    def adapt_recipe_real_time(self, original_recipe, constraints):
        """
        Adapt recipes based on real-time constraints and preferences
        """
        adapted_recipe = original_recipe.copy()
        
        # Handle ingredient substitutions
        if constraints.get('missing_ingredients'):
            for missing in constraints['missing_ingredients']:
                substitute = self.substitution_engine.find_best_substitute(
                    missing,
                    available=constraints.get('available_ingredients'),
                    maintain_flavor=True
                )
                adapted_recipe = self._substitute_ingredient(
                    adapted_recipe, 
                    missing, 
                    substitute
                )
        
        # Adapt for dietary restrictions
        if constraints.get('dietary_restrictions'):
            adapted_recipe = self._adapt_for_diet(
                adapted_recipe,
                constraints['dietary_restrictions']
            )
        
        # Scale portions
        if constraints.get('servings'):
            adapted_recipe = self._scale_recipe(
                adapted_recipe,
                constraints['servings']
            )
        
        # Adjust cooking method
        if constraints.get('available_equipment'):
            adapted_recipe = self._adapt_cooking_method(
                adapted_recipe,
                constraints['available_equipment']
            )
        
        # Regenerate instructions
        adapted_recipe['instructions'] = self.recipe_llm.regenerate_instructions(
            adapted_recipe
        )
        
        return adapted_recipe
```

## Production Deployment

### API Service Implementation

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import asyncio
from typing import Optional, List
import redis
import hashlib

app = FastAPI(title="Food Generative AI API")

# Initialize models
food_generator = FoodImageGenerator()
recipe_generator = RecipeGeneratorLLM()
food_analyzer = MultimodalFoodAnalyzer()

# Redis for caching
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

class GenerateRequest(BaseModel):
    prompt: str
    style: Optional[str] = "professional"
    dietary_restrictions: Optional[List[str]] = None
    cuisine_type: Optional[str] = None
    include_recipe: bool = True
    include_nutrition: bool = True

class GenerateResponse(BaseModel):
    image_url: str
    recipe: Optional[dict] = None
    nutrition: Optional[dict] = None
    generation_time: float

@app.post("/generate", response_model=GenerateResponse)
async def generate_food_concept(request: GenerateRequest):
    start_time = time.time()
    
    # Check cache
    cache_key = hashlib.md5(
        f"{request.prompt}_{request.style}_{request.dietary_restrictions}".encode()
    ).hexdigest()
    
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return json.loads(cached_result)
    
    try:
        # Generate image
        image = await asyncio.to_thread(
            food_generator.generate_food_image,
            request.prompt,
            request.style
        )
        
        # Save image
        image_path = f"generated/{cache_key}.png"
        image.save(image_path)
        image_url = f"/static/{cache_key}.png"
        
        response_data = {
            "image_url": image_url,
            "generation_time": time.time() - start_time
        }
        
        # Generate recipe if requested
        if request.include_recipe:
            recipe = await asyncio.to_thread(
                recipe_generator.generate_recipe,
                ingredients=extract_ingredients_from_prompt(request.prompt),
                dietary_restrictions=request.dietary_restrictions,
                cuisine_type=request.cuisine_type
            )
            response_data["recipe"] = recipe
        
        # Analyze nutrition if requested
        if request.include_nutrition:
            nutrition = await asyncio.to_thread(
                food_analyzer._estimate_nutrition,
                image,
                response_data.get("recipe", {}).get("ingredients", [])
            )
            response_data["nutrition"] = nutrition
        
        # Cache result
        redis_client.setex(
            cache_key, 
            3600,  # 1 hour cache
            json.dumps(response_data)
        )
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_food_image(file: UploadFile = File(...)):
    """
    Analyze uploaded food image
    """
    # Save uploaded file
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Analyze image
    analysis = await asyncio.to_thread(
        food_analyzer.analyze_food_image,
        file_path
    )
    
    return analysis

@app.post("/adapt-recipe")
async def adapt_recipe(recipe_id: str, constraints: dict):
    """
    Adapt existing recipe based on constraints
    """
    # Retrieve original recipe
    original_recipe = get_recipe_from_db(recipe_id)
    
    # Adapt recipe
    adapted = await asyncio.to_thread(
        adaptive_generator.adapt_recipe_real_time,
        original_recipe,
        constraints
    )
    
    return adapted

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": True}
```

### Scaling and Optimization

```python
import torch.quantization
from torch.nn.parallel import DataParallel
import onnx
import onnxruntime as ort

class OptimizedFoodAI:
    def __init__(self):
        self.quantized_model = self._quantize_model()
        self.onnx_session = self._create_onnx_session()
        
    def _quantize_model(self):
        """
        Quantize model for faster inference
        """
        model = FoodImageGenerator().pipe.unet
        
        # Dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )
        
        return quantized_model
    
    def _create_onnx_session(self):
        """
        Convert to ONNX for deployment flexibility
        """
        # Export to ONNX
        dummy_input = torch.randn(1, 4, 64, 64)
        torch.onnx.export(
            self.quantized_model,
            dummy_input,
            "food_model.onnx",
            opset_version=11,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}}
        )
        
        # Create ONNX runtime session
        return ort.InferenceSession("food_model.onnx")
    
    def batch_generate(self, prompts, batch_size=4):
        """
        Efficient batch generation
        """
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            
            # Parallel generation
            with torch.cuda.amp.autocast():
                batch_results = self._generate_batch(batch)
            
            results.extend(batch_results)
        
        return results
```

## Resources and References

### Datasets
- Food-101: Large-scale food recognition dataset
- Recipe1M+: Recipe dataset with images
- Nutrition5k: Food images with nutritional information
- FoodKG: Food knowledge graph

### Pre-trained Models
- Stable Diffusion fine-tuned on food images
- CLIP models trained on food-text pairs
- Specialized food segmentation models
- Nutrition estimation models

### Tools and Libraries
- Diffusers: Hugging Face library for diffusion models
- Transformers: For LLMs and vision models
- Albumentations: Image augmentation for food photos
- ONNX Runtime: Production deployment

### Research Papers
- "FoodGAN: Food Image Generation with GANs" (2021)
- "Neural Chef: Deep Learning for Recipe Generation" (2022)
- "Multimodal Food Computing: A Survey" (2023)
- "Stable Diffusion for Culinary Arts" (2023)

### Community Resources
- Food AI Discord community
- GitHub repositories with food AI implementations
- Kaggle competitions on food recognition
- Food photography style guides for AI

*Originally from umitkacar/food-generative-ai repository*