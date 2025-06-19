# Awesome ComfyUI & Beyond

A comprehensive collection of ComfyUI workflows, custom nodes, extensions, and advanced techniques for Stable Diffusion and other generative AI models.

## 📚 Table of Contents
- [Introduction to ComfyUI](#introduction-to-comfyui)
- [Installation & Setup](#installation--setup)
- [Essential Custom Nodes](#essential-custom-nodes)
- [Advanced Workflows](#advanced-workflows)
- [Integration with Other Tools](#integration-with-other-tools)
- [Performance Optimization](#performance-optimization)
- [Workflow Gallery](#workflow-gallery)
- [Tips & Tricks](#tips--tricks)
- [Resources & Community](#resources--community)

## Introduction to ComfyUI

ComfyUI is a powerful node-based interface for Stable Diffusion that offers unprecedented control over the image generation process. Unlike traditional UIs, ComfyUI exposes the entire pipeline as modular nodes.

### Why ComfyUI?

1. **Complete Control**: Access to every step of the generation pipeline
2. **Workflow Automation**: Save and share complex workflows
3. **Memory Efficiency**: Better VRAM management than most UIs
4. **Extensibility**: Huge ecosystem of custom nodes
5. **Performance**: Optimized execution and caching

### Core Concepts

- **Nodes**: Individual processing units
- **Workflows**: Connected nodes forming a pipeline
- **Models**: Checkpoints, LoRAs, VAEs, etc.
- **Conditioning**: Prompt processing and manipulation
- **Samplers**: Different denoising algorithms

## Installation & Setup

### Basic Installation

```bash
# Clone repository
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Install dependencies
pip install -r requirements.txt

# Run ComfyUI
python main.py
```

### Advanced Setup

#### GPU Optimization
```bash
# For NVIDIA GPUs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For AMD GPUs (ROCm)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
```

#### Custom Node Manager
```bash
cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
```

### Directory Structure
```
ComfyUI/
├── models/
│   ├── checkpoints/     # SD models
│   ├── loras/          # LoRA models
│   ├── vae/            # VAE models
│   ├── embeddings/     # Textual inversions
│   └── controlnet/     # ControlNet models
├── custom_nodes/       # Extensions
├── input/             # Input images
└── output/            # Generated images
```

## Essential Custom Nodes

### 1. ComfyUI Manager
Essential for installing and managing other custom nodes
```
https://github.com/ltdrdata/ComfyUI-Manager
```

### 2. Efficiency Nodes
Streamlined workflows with better performance
```
https://github.com/LucianoCirino/efficiency-nodes-comfyui
```

### 3. ControlNet Preprocessors
All preprocessing nodes for ControlNet
```
https://github.com/Fannovel16/comfyui_controlnet_aux
```

### 4. IP-Adapter
Image prompt adapter for style transfer
```
https://github.com/cubiq/ComfyUI_IPAdapter_plus
```

### 5. AnimateDiff
Animation generation nodes
```
https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved
```

### 6. Face Restoration
GFPGAN, CodeFormer integration
```
https://github.com/mav-rik/facerestore_cf
```

### 7. Upscale Models
ESRGAN, Real-ESRGAN, and more
```
https://github.com/ssitu/ComfyUI_UltimateSDUpscale
```

### 8. Advanced Conditioning
Complex prompt manipulation
```
https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb
```

## Advanced Workflows

### 1. Multi-ControlNet Workflow
```json
{
  "workflow": "multi_controlnet",
  "nodes": [
    {
      "type": "LoadImage",
      "id": "depth_input",
      "outputs": ["depth_image"]
    },
    {
      "type": "LoadImage", 
      "id": "canny_input",
      "outputs": ["canny_image"]
    },
    {
      "type": "ControlNetApply",
      "inputs": {
        "image": "depth_image",
        "strength": 0.7,
        "control_net": "depth_controlnet"
      }
    },
    {
      "type": "ControlNetApply",
      "inputs": {
        "image": "canny_image",
        "strength": 0.5,
        "control_net": "canny_controlnet"
      }
    }
  ]
}
```

### 2. High-Resolution Fix Workflow
```python
# Latent upscale workflow for high-res images
workflow = {
    "initial_generation": {
        "width": 512,
        "height": 512,
        "steps": 20
    },
    "upscale_phase": {
        "scale_factor": 2,
        "denoising": 0.5,
        "steps": 15
    },
    "final_enhancement": {
        "face_restoration": True,
        "detail_enhance": True
    }
}
```

### 3. Style Transfer Pipeline
```yaml
nodes:
  - IPAdapterLoad:
      model: "ip-adapter_sd15_plus"
  - CLIPVisionEncode:
      input: "style_reference.jpg"
  - IPAdapterApply:
      weight: 0.8
      weight_type: "style transfer"
  - KSampler:
      cfg: 7.5
      steps: 30
```

### 4. Animation Workflow
```python
# AnimateDiff with prompt travel
animation_config = {
    "frame_count": 96,
    "fps": 24,
    "prompt_schedule": {
        0: "sunset landscape, golden hour",
        24: "twilight landscape, blue hour",
        48: "night landscape, starry sky",
        72: "dawn landscape, first light"
    },
    "motion_module": "mm_sd_v15_v2",
    "context_length": 16
}
```

### 5. Inpainting Advanced
```json
{
  "mask_preprocessing": {
    "blur": 15,
    "grow": 10,
    "feather": 5
  },
  "inpaint_settings": {
    "masked_only": true,
    "denoising": 1.0,
    "cfg_scale": 7.5
  },
  "post_processing": {
    "blend_mode": "multiply",
    "blend_factor": 0.2
  }
}
```

## Integration with Other Tools

### 1. Automatic1111 WebUI Bridge
```python
# Use A1111 models in ComfyUI
class A1111ModelLoader:
    def __init__(self, a1111_path):
        self.model_path = f"{a1111_path}/models/Stable-diffusion"
        self.lora_path = f"{a1111_path}/models/Lora"
    
    def load_checkpoint(self, name):
        return f"{self.model_path}/{name}"
```

### 2. Photoshop Plugin
```javascript
// ComfyUI Photoshop integration
const ComfyUIConnector = {
    endpoint: "http://localhost:8188",
    
    async sendToComfyUI(layer) {
        const imageData = await layer.toBase64();
        return fetch(`${this.endpoint}/api/image`, {
            method: 'POST',
            body: JSON.stringify({image: imageData})
        });
    }
};
```

### 3. Blender Integration
```python
# Blender to ComfyUI pipeline
import bpy
import requests

class ComfyUIBlenderBridge:
    def render_with_ai(self, render_result):
        # Send render to ComfyUI for enhancement
        workflow = self.load_workflow("enhance_render.json")
        workflow["nodes"]["LoadImage"]["image"] = render_result
        
        response = requests.post(
            "http://localhost:8188/api/prompt",
            json={"workflow": workflow}
        )
        return response.json()
```

### 4. Discord Bot Integration
```python
# Discord bot for ComfyUI
import discord
from comfyui_api import ComfyUIAPI

class ComfyBot(discord.Client):
    def __init__(self):
        super().__init__()
        self.comfy = ComfyUIAPI()
    
    async def on_message(self, message):
        if message.content.startswith('!generate'):
            prompt = message.content[10:]
            image = await self.comfy.generate(prompt)
            await message.channel.send(file=image)
```

## Performance Optimization

### 1. VRAM Management
```python
# Custom VRAM management node
class VRAMOptimizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["aggressive", "balanced", "quality"],),
                "max_vram_gb": ("FLOAT", {"default": 8.0})
            }
        }
    
    def optimize(self, mode, max_vram_gb):
        settings = {
            "aggressive": {
                "batch_size": 1,
                "tiled_vae": True,
                "cpu_offload": True
            },
            "balanced": {
                "batch_size": 2,
                "tiled_vae": False,
                "cpu_offload": False
            }
        }
        return settings[mode]
```

### 2. Batch Processing
```python
# Efficient batch processing workflow
def batch_process_images(image_folder, workflow):
    results = []
    
    # Load images in batches
    for batch in chunked(os.listdir(image_folder), 4):
        batch_tensors = [load_image(img) for img in batch]
        
        # Process batch
        with torch.inference_mode():
            outputs = workflow.process_batch(batch_tensors)
            results.extend(outputs)
    
    return results
```

### 3. Model Caching
```yaml
# Caching configuration
cache_config:
  models:
    max_size: 10  # Maximum cached models
    strategy: "lru"  # Least recently used
  latents:
    enable: true
    max_size_gb: 2
  conditioning:
    enable: true
    ttl_seconds: 3600
```

### 4. Execution Optimization
```python
# Optimize node execution order
class ExecutionOptimizer:
    def optimize_workflow(self, workflow):
        # Analyze dependencies
        dependency_graph = self.build_dependency_graph(workflow)
        
        # Find optimal execution order
        execution_order = self.topological_sort(dependency_graph)
        
        # Identify parallelizable nodes
        parallel_groups = self.find_parallel_nodes(execution_order)
        
        return {
            "order": execution_order,
            "parallel": parallel_groups
        }
```

## Workflow Gallery

### 1. Photorealistic Portrait
```yaml
name: "Photorealistic Portrait Workflow"
description: "High-quality portrait generation with face fix"
nodes:
  - checkpoint: "realistic_vision_v5"
  - vae: "vae-ft-mse-840000"
  - positive_prompt: |
      portrait photography, professional lighting,
      shallow depth of field, bokeh, 85mm lens
  - negative_prompt: |
      cartoon, anime, painted, artificial
  - sampler: "DPM++ 2M Karras"
  - steps: 30
  - cfg: 7
  - face_restore: "CodeFormer"
  - upscale: "4x-UltraSharp"
```

### 2. Artistic Style Mixing
```json
{
  "name": "Style Fusion",
  "nodes": [
    {
      "type": "CheckpointLoaderSimple",
      "checkpoint": "sd15_base"
    },
    {
      "type": "LoraLoader",
      "lora_1": "style_oil_painting",
      "strength_1": 0.6
    },
    {
      "type": "LoraLoader", 
      "lora_2": "style_watercolor",
      "strength_2": 0.4
    },
    {
      "type": "IPAdapter",
      "style_reference": "monet_painting.jpg",
      "weight": 0.3
    }
  ]
}
```

### 3. Product Photography
```python
product_workflow = {
    "lighting_setup": {
        "main_light": "soft box from right",
        "fill_light": "reflector left",
        "background": "pure white"
    },
    "camera_settings": {
        "focal_length": "100mm macro",
        "aperture": "f/8",
        "focus_stacking": True
    },
    "post_processing": {
        "background_removal": True,
        "shadow_generation": True,
        "color_correction": "product_lut"
    }
}
```

## Tips & Tricks

### 1. Prompt Engineering
```python
# Advanced prompt techniques
class PromptEnhancer:
    def __init__(self):
        self.quality_tags = [
            "masterpiece", "best quality", 
            "highly detailed", "professional"
        ]
        self.camera_tags = [
            "DSLR", "85mm lens", "shallow dof",
            "golden hour", "studio lighting"
        ]
    
    def enhance_prompt(self, base_prompt, style="photo"):
        if style == "photo":
            return f"{base_prompt}, {', '.join(self.quality_tags)}, {', '.join(self.camera_tags)}"
```

### 2. Seed Management
```python
# Seed variation techniques
def generate_seed_variants(base_seed, count=4):
    variants = []
    for i in range(count):
        # Subseed variation
        variant = {
            "seed": base_seed,
            "subseed": base_seed + i * 1000,
            "subseed_strength": 0.1
        }
        variants.append(variant)
    return variants
```

### 3. Custom Nodes Development
```python
# Template for custom node
class CustomImageProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "image/postprocessing"
    
    def process(self, image, strength):
        # Custom processing logic
        processed = image * strength
        return (processed,)
```

### 4. Workflow Debugging
```yaml
debugging_tips:
  - use_preview_nodes: "Add preview nodes after each step"
  - check_dimensions: "Ensure consistent dimensions"
  - monitor_vram: "Use system monitor nodes"
  - validate_models: "Check model compatibility"
  - test_incremental: "Build workflow step by step"
```

### 5. Advanced Conditioning
```python
# Regional conditioning
class RegionalConditioning:
    def create_regions(self, width, height):
        # Define regions with different prompts
        regions = [
            {
                "mask": self.create_circular_mask(width//2, height//2, 100),
                "prompt": "detailed face, sharp eyes",
                "weight": 1.5
            },
            {
                "mask": self.create_background_mask(),
                "prompt": "blurred background, bokeh",
                "weight": 0.8
            }
        ]
        return regions
```

## Resources & Community

### Official Resources
- **ComfyUI GitHub**: https://github.com/comfyanonymous/ComfyUI
- **Documentation**: https://docs.comfy.org
- **Examples**: https://comfyanonymous.github.io/ComfyUI_examples

### Community Nodes
- **ComfyUI Manager**: Node package manager
- **ComfyUI Custom Nodes**: https://github.com/topics/comfyui-nodes
- **Civitai**: Models and workflows

### Learning Resources
- **Video Tutorials**: 
  - Olivio Sarikas
  - Aitrepreneur
  - Scott Detweiler
- **Workflow Sharing**:
  - OpenArt
  - ComfyWorkflows
  - Reddit r/comfyui

### Advanced Topics
- **Node Development Guide**: Create custom nodes
- **API Documentation**: Integrate ComfyUI programmatically
- **Performance Tuning**: Optimize for your hardware
- **Workflow Automation**: CI/CD for AI art

### Community Links
- **Discord**: ComfyUI Official Discord
- **Reddit**: r/comfyui
- **Twitter**: #comfyui
- **Forums**: Stable Diffusion forums

---

*Originally from umitkacar/Awesome-ComfyUI-Beyond repository*