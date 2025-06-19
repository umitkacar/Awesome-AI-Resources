# Image Enhancement with AI: Advanced Techniques and Implementations

**Last Updated:** 2025-06-19

## Overview

Image Enhancement using AI represents a revolutionary approach to improving image quality, restoring damaged photos, and creating visually stunning results through deep learning techniques. This comprehensive guide covers state-of-the-art methods, practical implementations, and production-ready solutions for AI-powered image enhancement.

## Table of Contents

1. [Introduction](#introduction)
2. [Core Enhancement Techniques](#core-enhancement-techniques)
3. [Super Resolution](#super-resolution)
4. [Denoising and Restoration](#denoising-and-restoration)
5. [Color Enhancement and Correction](#color-enhancement-and-correction)
6. [HDR and Tone Mapping](#hdr-and-tone-mapping)
7. [Real-time Enhancement Pipeline](#real-time-enhancement-pipeline)
8. [Advanced Neural Networks](#advanced-neural-networks)
9. [Training Custom Models](#training-custom-models)
10. [Production Deployment](#production-deployment)
11. [Benchmarking and Evaluation](#benchmarking-and-evaluation)
12. [Best Practices](#best-practices)

## Introduction

AI-powered image enhancement leverages deep learning to:
- Increase image resolution (super-resolution)
- Remove noise and artifacts
- Enhance colors and contrast
- Restore old or damaged photos
- Improve low-light images
- Apply artistic enhancements

### Key Benefits

- **Automatic Processing**: No manual parameter tuning required
- **Context-Aware**: AI understands image content for optimal enhancement
- **Versatility**: Single model can handle multiple enhancement tasks
- **Quality**: Achieves results superior to traditional methods

## Core Enhancement Techniques

### Foundation Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image

class ImageEnhancementNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(ImageEnhancementNet, self).__init__()
        
        # Multi-scale feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Residual blocks for deep feature extraction
        self.res_blocks = nn.ModuleList([
            ResidualBlock(features) for _ in range(16)
        ])
        
        # Multi-scale processing branches
        self.branch1 = self._make_branch(features, features * 2)
        self.branch2 = self._make_branch(features, features * 2, scale=2)
        self.branch3 = self._make_branch(features, features * 2, scale=4)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(features * 6, features * 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 2, features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Output reconstruction
        self.output = nn.Sequential(
            nn.Conv2d(features, features // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features // 2, out_channels, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Extract initial features
        feat = self.initial(x)
        
        # Deep feature extraction with residual connections
        res_feat = feat
        for res_block in self.res_blocks:
            res_feat = res_block(res_feat)
        
        # Multi-scale processing
        b1 = self.branch1(res_feat)
        b2 = F.interpolate(self.branch2(
            F.interpolate(res_feat, scale_factor=0.5)
        ), size=res_feat.shape[2:])
        b3 = F.interpolate(self.branch3(
            F.interpolate(res_feat, scale_factor=0.25)
        ), size=res_feat.shape[2:])
        
        # Concatenate multi-scale features
        combined = torch.cat([b1, b2, b3], dim=1)
        
        # Fuse features
        fused = self.fusion(combined)
        
        # Generate enhanced output
        enhanced = self.output(fused + feat)  # Skip connection
        
        return enhanced
    
    def _make_branch(self, in_features, out_features, scale=1):
        layers = []
        
        if scale > 1:
            layers.append(nn.Conv2d(in_features, in_features, 3, 
                                  stride=scale, padding=1))
        
        layers.extend([
            nn.Conv2d(in_features, out_features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_features, out_features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
    def __init__(self, features):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)
        self.instance_norm1 = nn.InstanceNorm2d(features)
        self.instance_norm2 = nn.InstanceNorm2d(features)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.instance_norm1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.instance_norm2(out)
        
        return out + residual
```

### Attention Mechanisms

```python
class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionBlock, self).__init__()
        
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate query, key, value
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        proj_value = self.value(x).view(batch_size, -1, width * height)
        
        # Calculate attention
        attention = torch.bmm(proj_query, proj_key)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Apply learnable parameter and residual connection
        out = self.gamma * out + x
        
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Average pooling path
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        
        # Max pooling path
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine and apply sigmoid
        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1, 1)
        
        return x * out.expand_as(x)
```

## Super Resolution

### ESRGAN Implementation

```python
class ESRGANGenerator(nn.Module):
    def __init__(self, scale_factor=4, num_channels=3, num_filters=64, 
                 num_res_blocks=23):
        super(ESRGANGenerator, self).__init__()
        
        self.scale_factor = scale_factor
        
        # Initial convolution
        self.conv1 = nn.Conv2d(num_channels, num_filters, 3, padding=1)
        
        # Residual blocks with dense connections
        self.res_blocks = nn.ModuleList([
            ResidualDenseBlock(num_filters) for _ in range(num_res_blocks)
        ])
        
        # Feature fusion
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        
        # Upsampling layers
        self.upsampling = self._make_upsampling_layers(num_filters, scale_factor)
        
        # Final convolution
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters, num_channels, 3, padding=1)
        )
        
    def forward(self, x):
        # Initial feature extraction
        feat = self.conv1(x)
        trunk = feat
        
        # Pass through residual dense blocks
        for block in self.res_blocks:
            trunk = block(trunk)
        
        # Feature fusion
        trunk = self.conv2(trunk)
        feat = feat + trunk
        
        # Upsampling
        out = self.upsampling(feat)
        
        # Final reconstruction
        out = self.conv3(out)
        
        return out
    
    def _make_upsampling_layers(self, num_filters, scale_factor):
        layers = []
        
        if scale_factor == 2 or scale_factor == 4 or scale_factor == 8:
            for _ in range(int(np.log2(scale_factor))):
                layers.extend([
                    nn.Conv2d(num_filters, num_filters * 4, 3, padding=1),
                    nn.PixelShuffle(2),
                    nn.LeakyReLU(0.2, inplace=True)
                ])
        else:
            raise ValueError(f"Scale factor {scale_factor} not supported")
        
        return nn.Sequential(*layers)

class ResidualDenseBlock(nn.Module):
    def __init__(self, num_filters, growth_rate=32, num_layers=5):
        super(ResidualDenseBlock, self).__init__()
        
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(num_filters + i * growth_rate, growth_rate, 3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
        
        # Local feature fusion
        self.lff = nn.Conv2d(num_filters + num_layers * growth_rate, 
                            num_filters, 1)
        
    def forward(self, x):
        features = [x]
        
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        
        # Fuse all features
        out = self.lff(torch.cat(features, dim=1))
        
        # Residual learning
        return out * 0.2 + x
```

### Real-ESRGAN with Face Enhancement

```python
class RealESRGAN:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self._load_model(model_path)
        self.face_enhancer = FaceEnhancer()
        
    def enhance(self, image, face_enhance=True, tile_size=512, 
                tile_overlap=32):
        """
        Enhance image with optional face enhancement
        """
        # Convert to tensor
        img_tensor = self._preprocess(image)
        
        # Process in tiles for large images
        if max(image.size) > tile_size:
            enhanced = self._process_tiles(img_tensor, tile_size, tile_overlap)
        else:
            enhanced = self.model(img_tensor)
        
        # Convert back to image
        enhanced_img = self._postprocess(enhanced)
        
        # Apply face enhancement if requested
        if face_enhance:
            enhanced_img = self.face_enhancer.enhance_faces(enhanced_img)
        
        return enhanced_img
    
    def _process_tiles(self, img_tensor, tile_size, overlap):
        """
        Process large images in tiles to manage memory
        """
        b, c, h, w = img_tensor.shape
        
        # Calculate tile positions
        tiles_x = self._calculate_tile_positions(w, tile_size, overlap)
        tiles_y = self._calculate_tile_positions(h, tile_size, overlap)
        
        # Process each tile
        result = torch.zeros(b, c, h * self.model.scale_factor, 
                           w * self.model.scale_factor).to(self.device)
        
        for y_start, y_end in tiles_y:
            for x_start, x_end in tiles_x:
                # Extract tile
                tile = img_tensor[:, :, y_start:y_end, x_start:x_end]
                
                # Enhance tile
                enhanced_tile = self.model(tile)
                
                # Place in result with blending at overlaps
                self._blend_tile(result, enhanced_tile, 
                               x_start * self.model.scale_factor,
                               y_start * self.model.scale_factor,
                               overlap * self.model.scale_factor)
        
        return result
    
    def _blend_tile(self, result, tile, x_start, y_start, overlap):
        """
        Blend tile into result with smooth transitions
        """
        _, _, tile_h, tile_w = tile.shape
        
        # Create blend masks for smooth transitions
        blend_mask_x = self._create_blend_mask(tile_w, overlap, 'horizontal')
        blend_mask_y = self._create_blend_mask(tile_h, overlap, 'vertical')
        blend_mask = blend_mask_x * blend_mask_y
        
        # Apply blending
        y_end = y_start + tile_h
        x_end = x_start + tile_w
        
        result[:, :, y_start:y_end, x_start:x_end] = \
            result[:, :, y_start:y_end, x_start:x_end] * (1 - blend_mask) + \
            tile * blend_mask
```

## Denoising and Restoration

### Advanced Denoising Network

```python
class DenoisingNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(DenoisingNetwork, self).__init__()
        
        # Encoder path
        self.enc1 = self._encoder_block(in_channels, features)
        self.enc2 = self._encoder_block(features, features * 2)
        self.enc3 = self._encoder_block(features * 2, features * 4)
        self.enc4 = self._encoder_block(features * 4, features * 8)
        
        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 3, padding=1),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(inplace=True),
            SelfAttentionBlock(features * 8),
            nn.Conv2d(features * 8, features * 8, 3, padding=1),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(inplace=True)
        )
        
        # Decoder path with skip connections
        self.dec4 = self._decoder_block(features * 8, features * 4)
        self.dec3 = self._decoder_block(features * 4, features * 2)
        self.dec2 = self._decoder_block(features * 2, features)
        self.dec1 = self._decoder_block(features, features)
        
        # Output layer
        self.output = nn.Conv2d(features, out_channels, 1)
        
        # Noise level conditioning
        self.noise_embed = nn.Sequential(
            nn.Linear(1, features),
            nn.ReLU(inplace=True),
            nn.Linear(features, features)
        )
        
    def forward(self, x, noise_level=None):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))
        
        # Apply noise level conditioning if provided
        if noise_level is not None:
            noise_embed = self.noise_embed(noise_level.view(-1, 1))
            b = b + noise_embed.view(-1, noise_embed.size(1), 1, 1)
        
        # Decoder with skip connections
        d4 = self.dec4(F.interpolate(b, scale_factor=2))
        d4 = torch.cat([d4, e4], dim=1)
        
        d3 = self.dec3(F.interpolate(d4, scale_factor=2))
        d3 = torch.cat([d3, e3], dim=1)
        
        d2 = self.dec2(F.interpolate(d3, scale_factor=2))
        d2 = torch.cat([d2, e2], dim=1)
        
        d1 = self.dec1(F.interpolate(d2, scale_factor=2))
        d1 = torch.cat([d1, e1], dim=1)
        
        # Output
        out = self.output(d1)
        
        # Residual connection
        return out + x
    
    def _encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
```

### Old Photo Restoration

```python
class OldPhotoRestoration:
    def __init__(self):
        self.scratch_detector = ScratchDetectionNet()
        self.inpainting_net = InpaintingNetwork()
        self.color_restoration = ColorRestorationNet()
        self.denoising_net = DenoisingNetwork()
        self.super_resolution = ESRGANGenerator()
        
    def restore(self, image, enhance_faces=True):
        """
        Complete restoration pipeline for old photos
        """
        # Step 1: Detect and remove scratches/tears
        scratch_mask = self.scratch_detector.detect(image)
        if scratch_mask.sum() > 0:
            image = self.inpainting_net.inpaint(image, scratch_mask)
        
        # Step 2: Denoise
        image = self.denoising_net(image)
        
        # Step 3: Restore colors
        if self._is_grayscale(image) or self._has_faded_colors(image):
            image = self.color_restoration.restore(image)
        
        # Step 4: Enhance resolution
        image = self.super_resolution(image)
        
        # Step 5: Face enhancement if needed
        if enhance_faces:
            image = self._enhance_faces(image)
        
        # Step 6: Final adjustments
        image = self._apply_finishing_touches(image)
        
        return image
    
    def _enhance_faces(self, image):
        """
        Detect and enhance faces in the image
        """
        # Detect faces
        faces = self.face_detector.detect_faces(image)
        
        for face_bbox in faces:
            # Extract face region
            face_img = self._extract_region(image, face_bbox, padding=0.2)
            
            # Enhance face
            enhanced_face = self.face_enhancer.enhance(face_img)
            
            # Blend back into image
            image = self._blend_region(image, enhanced_face, face_bbox)
        
        return image
    
    def _apply_finishing_touches(self, image):
        """
        Apply final adjustments for optimal quality
        """
        # Adaptive histogram equalization
        image = self._adaptive_histogram_eq(image)
        
        # Smart sharpening
        image = self._smart_sharpen(image)
        
        # Color vibrance adjustment
        image = self._adjust_vibrance(image)
        
        return image
```

## Color Enhancement and Correction

### Advanced Color Correction

```python
class ColorCorrectionNet(nn.Module):
    def __init__(self):
        super(ColorCorrectionNet, self).__init__()
        
        # Color space transformation learning
        self.rgb_to_lab = nn.Conv2d(3, 3, 1, bias=False)
        self.lab_to_rgb = nn.Conv2d(3, 3, 1, bias=False)
        
        # Separate processing for luminance and chrominance
        self.luminance_net = self._build_luminance_net()
        self.chrominance_net = self._build_chrominance_net()
        
        # Global color adjustment
        self.global_adjust = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Convert to LAB color space
        lab = self.rgb_to_lab(x)
        
        # Separate channels
        l_channel = lab[:, 0:1, :, :]
        ab_channels = lab[:, 1:3, :, :]
        
        # Process luminance
        l_enhanced = self.luminance_net(l_channel)
        
        # Process chrominance
        ab_enhanced = self.chrominance_net(ab_channels)
        
        # Combine channels
        lab_enhanced = torch.cat([l_enhanced, ab_enhanced], dim=1)
        
        # Convert back to RGB
        rgb_enhanced = self.lab_to_rgb(lab_enhanced)
        
        # Apply global color adjustment
        global_adj = self.global_adjust(x)
        rgb_enhanced = rgb_enhanced + global_adj
        
        return torch.clamp(rgb_enhanced, -1, 1)
    
    def _build_luminance_net(self):
        return nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            ResidualBlock(32),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Tanh()
        )
    
    def _build_chrominance_net(self):
        return nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            ResidualBlock(64),
            ChannelAttention(64),
            nn.Conv2d(64, 2, 3, padding=1),
            nn.Tanh()
        )

class WhiteBalanceCorrection:
    def __init__(self):
        self.illuminant_estimator = IlluminantEstimationNet()
        
    def correct_white_balance(self, image):
        """
        Automatic white balance correction
        """
        # Estimate scene illuminant
        illuminant = self.illuminant_estimator.estimate(image)
        
        # Calculate correction matrix
        correction_matrix = self._calculate_correction_matrix(illuminant)
        
        # Apply correction
        corrected = self._apply_color_correction(image, correction_matrix)
        
        return corrected
    
    def _calculate_correction_matrix(self, illuminant):
        """
        Calculate von Kries transformation matrix
        """
        # Reference illuminant (D65)
        reference = torch.tensor([0.95047, 1.0, 1.08883])
        
        # Calculate scaling factors
        scale = reference / illuminant
        
        # Create diagonal matrix
        matrix = torch.diag(scale)
        
        return matrix
```

### Automatic Color Grading

```python
class AutoColorGrading:
    def __init__(self):
        self.style_analyzer = StyleAnalysisNet()
        self.color_transfer = ColorTransferNet()
        self.lut_generator = LUTGenerator()
        
    def apply_cinematic_grade(self, image, style='default'):
        """
        Apply cinematic color grading
        """
        # Analyze image characteristics
        image_features = self.style_analyzer.analyze(image)
        
        # Select appropriate grading style
        if style == 'auto':
            style = self._select_style(image_features)
        
        # Generate 3D LUT for the style
        lut_3d = self.lut_generator.generate(style, image_features)
        
        # Apply LUT with adaptive strength
        graded = self._apply_3d_lut(image, lut_3d)
        
        # Fine-tune specific aspects
        graded = self._fine_tune_grading(graded, style)
        
        return graded
    
    def _select_style(self, features):
        """
        Automatically select grading style based on image content
        """
        styles = {
            'portrait': features['face_count'] > 0,
            'landscape': features['sky_percentage'] > 0.3,
            'night': features['average_luminance'] < 0.3,
            'golden_hour': features['warm_tones'] > 0.6,
            'high_key': features['average_luminance'] > 0.7,
            'film_noir': features['contrast'] > 0.8 and features['saturation'] < 0.3
        }
        
        for style, condition in styles.items():
            if condition:
                return style
        
        return 'default'
    
    def _apply_3d_lut(self, image, lut_3d):
        """
        Apply 3D LUT transformation
        """
        # Flatten image
        h, w = image.shape[1:3]
        pixels = image.view(-1, 3)
        
        # Apply LUT
        transformed = torch.zeros_like(pixels)
        
        for i in range(pixels.shape[0]):
            r, g, b = pixels[i]
            
            # Trilinear interpolation in 3D LUT
            r_idx = r * (lut_3d.shape[0] - 1)
            g_idx = g * (lut_3d.shape[1] - 1)
            b_idx = b * (lut_3d.shape[2] - 1)
            
            transformed[i] = self._trilinear_interpolation(
                lut_3d, r_idx, g_idx, b_idx
            )
        
        # Reshape back
        return transformed.view(image.shape)
```

## HDR and Tone Mapping

### HDR Reconstruction

```python
class HDRReconstruction(nn.Module):
    def __init__(self):
        super(HDRReconstruction, self).__init__()
        
        # Multi-exposure fusion network
        self.exposure_fusion = ExposureFusionNet()
        
        # Tone mapping network
        self.tone_mapper = ToneMappingNet()
        
        # Detail enhancement
        self.detail_enhancer = DetailEnhancementNet()
        
    def forward(self, ldr_image):
        """
        Reconstruct HDR from single LDR image
        """
        # Generate multiple exposures
        exposures = self._generate_exposure_stack(ldr_image)
        
        # Fuse exposures into HDR
        hdr_linear = self.exposure_fusion(exposures)
        
        # Apply tone mapping
        tone_mapped = self.tone_mapper(hdr_linear)
        
        # Enhance details
        enhanced = self.detail_enhancer(tone_mapped, hdr_linear)
        
        return enhanced
    
    def _generate_exposure_stack(self, image, num_exposures=5):
        """
        Generate virtual exposure stack from single image
        """
        exposures = []
        
        # Base exposure values
        ev_range = torch.linspace(-2, 2, num_exposures)
        
        for ev in ev_range:
            # Apply exposure adjustment
            exposed = image * (2 ** ev)
            
            # Simulate camera response curve
            exposed = self._apply_camera_response(exposed)
            
            exposures.append(exposed)
        
        return torch.stack(exposures, dim=1)
    
    def _apply_camera_response(self, image):
        """
        Simulate realistic camera response curve
        """
        # Inverse gamma correction
        linear = torch.pow(image, 2.2)
        
        # Apply S-curve for highlights and shadows
        response = 3 * linear**2 - 2 * linear**3
        
        # Re-apply gamma
        return torch.pow(response, 1/2.2)

class ToneMappingNet(nn.Module):
    def __init__(self):
        super(ToneMappingNet, self).__init__()
        
        # Global tone mapping
        self.global_mapper = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Conv2d(64, 3, 3, padding=1)
        )
        
        # Local adaptation
        self.local_adapter = LocalAdaptationNet()
        
        # Highlight/shadow control
        self.highlight_control = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        self.shadow_control = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, hdr_input):
        # Apply global tone mapping
        global_mapped = self.global_mapper(hdr_input)
        
        # Local adaptation
        local_adapted = self.local_adapter(global_mapped, hdr_input)
        
        # Separate highlight and shadow control
        highlight_mask = self.highlight_control(hdr_input)
        shadow_mask = self.shadow_control(hdr_input)
        
        # Apply selective adjustments
        result = local_adapted
        result = result * (1 - highlight_mask) + \
                 self._compress_highlights(result) * highlight_mask
        result = result * (1 - shadow_mask) + \
                 self._lift_shadows(result) * shadow_mask
        
        return torch.clamp(result, 0, 1)
```

## Real-time Enhancement Pipeline

### GPU-Optimized Pipeline

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class RealtimeEnhancementPipeline:
    def __init__(self, model_path, target_fps=30):
        self.target_fps = target_fps
        self.trt_engine = self._build_trt_engine(model_path)
        self.cuda_stream = cuda.Stream()
        
        # Pre-allocate buffers
        self._allocate_buffers()
        
        # Frame buffer for temporal stability
        self.frame_buffer = deque(maxlen=3)
        
    def _build_trt_engine(self, model_path):
        """
        Build TensorRT engine for real-time inference
        """
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        
        # Parse ONNX model
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)
        
        with open(model_path, 'rb') as model:
            parser.parse(model.read())
        
        # Build engine
        engine = builder.build_engine(network, config)
        
        return engine
    
    def enhance_frame(self, frame):
        """
        Enhance single frame with temporal stability
        """
        start_time = time.time()
        
        # Add to frame buffer
        self.frame_buffer.append(frame)
        
        # Prepare input
        input_tensor = self._prepare_input(frame)
        
        # Copy to GPU
        cuda.memcpy_htod_async(
            self.d_input, input_tensor, self.cuda_stream
        )
        
        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.cuda_stream.handle
        )
        
        # Copy result back
        cuda.memcpy_dtoh_async(
            self.output, self.d_output, self.cuda_stream
        )
        
        # Synchronize
        self.cuda_stream.synchronize()
        
        # Apply temporal stability
        enhanced = self._apply_temporal_stability(self.output)
        
        # Ensure target FPS
        elapsed = time.time() - start_time
        if elapsed < 1.0 / self.target_fps:
            time.sleep(1.0 / self.target_fps - elapsed)
        
        return enhanced
    
    def _apply_temporal_stability(self, current_frame):
        """
        Reduce flickering between frames
        """
        if len(self.frame_buffer) < 2:
            return current_frame
        
        # Weighted average with previous frames
        weights = [0.6, 0.3, 0.1]  # Current, previous, 2 frames ago
        
        stabilized = np.zeros_like(current_frame)
        for i, (frame, weight) in enumerate(zip(
            reversed(self.frame_buffer), weights
        )):
            if i < len(self.frame_buffer):
                stabilized += frame * weight
        
        return stabilized

class StreamProcessor:
    def __init__(self, enhancement_pipeline):
        self.pipeline = enhancement_pipeline
        self.is_processing = False
        
    def process_video_stream(self, input_source, output_sink):
        """
        Process video stream in real-time
        """
        self.is_processing = True
        
        # Open video capture
        cap = cv2.VideoCapture(input_source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Setup output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_sink, fourcc, fps, 
                            (int(cap.get(3)), int(cap.get(4))))
        
        # Processing loop
        while self.is_processing and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Enhance frame
            enhanced = self.pipeline.enhance_frame(frame)
            
            # Write output
            out.write(enhanced)
            
            # Display preview (optional)
            cv2.imshow('Enhanced Stream', enhanced)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
```

## Advanced Neural Networks

### Transformer-based Enhancement

```python
class VisionTransformerEnhancer(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, 
                 embed_dim=768, depth=12, num_heads=12):
        super(VisionTransformerEnhancer, self).__init__()
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size,
            in_chans=in_chans, 
            embed_dim=embed_dim
        )
        
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, embed_dim)
        )
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) 
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Reconstruction head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, patch_size**2 * in_chans),
            nn.Tanh()
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Normalize
        x = self.norm(x)
        
        # Reconstruct patches
        x = self.head(x)
        
        # Reshape to image
        x = x.reshape(B, H // self.patch_embed.patch_size, 
                     W // self.patch_embed.patch_size,
                     self.patch_embed.patch_size, 
                     self.patch_embed.patch_size, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, C, H, W)
        
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super(TransformerBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
    def forward(self, x):
        # Self-attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x
```

### Diffusion Models for Enhancement

```python
class DiffusionEnhancer(nn.Module):
    def __init__(self, image_size=256, timesteps=1000):
        super(DiffusionEnhancer, self).__init__()
        
        self.image_size = image_size
        self.timesteps = timesteps
        
        # Noise schedule
        self.betas = self._linear_beta_schedule(timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        
        # U-Net denoiser
        self.denoise_net = UNetDenoiser(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            with_time_emb=True
        )
        
    def forward(self, degraded_image, steps=50):
        """
        Enhance image through reverse diffusion process
        """
        device = degraded_image.device
        batch_size = degraded_image.shape[0]
        
        # Start from degraded image with added noise
        img = degraded_image
        
        # Reverse diffusion process
        for i in reversed(range(0, self.timesteps, self.timesteps // steps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.denoise_net(img, t)
            
            # Remove noise
            img = self._denoise_step(img, predicted_noise, t)
            
            # Condition on degraded image
            img = self._condition_on_input(img, degraded_image, t)
        
        return img
    
    def _denoise_step(self, x_t, noise, t):
        """
        Single denoising step
        """
        alpha = self.alphas_cumprod[t][:, None, None, None]
        alpha_prev = self.alphas_cumprod[t-1][:, None, None, None] \
                     if t[0] > 0 else torch.ones_like(alpha)
        
        # Predict x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha) * noise) / torch.sqrt(alpha)
        x_0_pred = torch.clamp(x_0_pred, -1, 1)
        
        # Sample x_{t-1}
        variance = (1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)
        std = torch.sqrt(variance)
        
        eps = torch.randn_like(x_t) if t[0] > 0 else torch.zeros_like(x_t)
        
        x_prev = torch.sqrt(alpha_prev) * x_0_pred + \
                 torch.sqrt(1 - alpha_prev - variance) * noise + \
                 std * eps
        
        return x_prev
    
    def _condition_on_input(self, x_t, degraded, t):
        """
        Condition generation on input image
        """
        # Interpolate between generated and input based on timestep
        weight = (t.float() / self.timesteps)[:, None, None, None]
        return x_t * weight + degraded * (1 - weight)
```

## Training Custom Models

### Data Augmentation Pipeline

```python
class EnhancementDataAugmentation:
    def __init__(self):
        self.degradation_pipeline = self._build_degradation_pipeline()
        
    def _build_degradation_pipeline(self):
        return A.Compose([
            # Blur augmentations
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=(3, 7), p=1.0),
            ], p=0.5),
            
            # Noise augmentations
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            ], p=0.5),
            
            # Compression artifacts
            A.ImageCompression(quality_lower=40, quality_upper=95, p=0.3),
            
            # Color degradations
            A.OneOf([
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, 
                          b_shift_limit=20, p=1.0),
                A.HueSaturationValue(hue_shift_limit=20, 
                                   sat_shift_limit=30, 
                                   val_shift_limit=20, p=1.0),
                A.ChannelShuffle(p=1.0),
            ], p=0.3),
            
            # Brightness/contrast issues
            A.RandomBrightnessContrast(brightness_limit=0.3, 
                                      contrast_limit=0.3, p=0.5),
            
            # Downscaling (for super-resolution training)
            A.Downscale(scale_min=0.5, scale_max=0.9, p=0.5),
        ])
    
    def create_training_pair(self, clean_image):
        """
        Create degraded/clean image pair for training
        """
        # Apply degradations
        degraded = self.degradation_pipeline(image=clean_image)['image']
        
        # Ensure same size
        if degraded.shape != clean_image.shape:
            degraded = cv2.resize(degraded, 
                                (clean_image.shape[1], clean_image.shape[0]))
        
        return degraded, clean_image
```

### Training Loop

```python
class EnhancementTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.adversarial_loss = AdversarialLoss() if config.use_gan else None
        
        # Optimizers
        self.optimizer_g = torch.optim.Adam(
            model.parameters(), 
            lr=config.learning_rate,
            betas=(0.9, 0.999)
        )
        
        if self.adversarial_loss:
            self.optimizer_d = torch.optim.Adam(
                self.adversarial_loss.discriminator.parameters(),
                lr=config.learning_rate,
                betas=(0.9, 0.999)
            )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g, 
            T_max=config.epochs
        )
        
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (degraded, clean) in enumerate(dataloader):
            degraded = degraded.to(self.config.device)
            clean = clean.to(self.config.device)
            
            # Forward pass
            enhanced = self.model(degraded)
            
            # Calculate losses
            loss_l1 = self.l1_loss(enhanced, clean)
            loss_perceptual = self.perceptual_loss(enhanced, clean)
            
            loss_g = loss_l1 + self.config.perceptual_weight * loss_perceptual
            
            # Adversarial training
            if self.adversarial_loss and epoch > self.config.gan_start_epoch:
                # Train discriminator
                self.optimizer_d.zero_grad()
                loss_d = self.adversarial_loss.discriminator_loss(
                    clean, enhanced.detach()
                )
                loss_d.backward()
                self.optimizer_d.step()
                
                # Train generator
                loss_adv = self.adversarial_loss.generator_loss(enhanced)
                loss_g += self.config.adversarial_weight * loss_adv
            
            # Backward pass
            self.optimizer_g.zero_grad()
            loss_g.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.grad_clip
            )
            
            self.optimizer_g.step()
            
            total_loss += loss_g.item()
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                self._log_progress(epoch, batch_idx, len(dataloader), loss_g)
        
        return total_loss / len(dataloader)
    
    def validate(self, val_dataloader):
        self.model.eval()
        total_psnr = 0
        total_ssim = 0
        
        with torch.no_grad():
            for degraded, clean in val_dataloader:
                degraded = degraded.to(self.config.device)
                clean = clean.to(self.config.device)
                
                enhanced = self.model(degraded)
                
                # Calculate metrics
                psnr = calculate_psnr(enhanced, clean)
                ssim = calculate_ssim(enhanced, clean)
                
                total_psnr += psnr
                total_ssim += ssim
        
        return {
            'psnr': total_psnr / len(val_dataloader),
            'ssim': total_ssim / len(val_dataloader)
        }
```

## Production Deployment

### REST API Service

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import io
import torch
from PIL import Image

app = FastAPI(title="Image Enhancement API")

# Load models
models = {
    'super_resolution': load_model('models/esrgan.pth'),
    'denoising': load_model('models/denoising.pth'),
    'color_correction': load_model('models/color.pth'),
    'hdr': load_model('models/hdr.pth'),
    'all_in_one': load_model('models/unified.pth')
}

@app.post("/enhance")
async def enhance_image(
    file: UploadFile = File(...),
    enhancement_type: str = 'all_in_one',
    scale_factor: int = 2,
    denoise_strength: float = 0.5
):
    """
    Enhance uploaded image
    """
    # Validate input
    if enhancement_type not in models:
        raise HTTPException(status_code=400, 
                          detail=f"Unknown enhancement type: {enhancement_type}")
    
    # Read image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # Convert to tensor
    img_tensor = transforms.ToTensor()(image).unsqueeze(0)
    
    # Apply enhancement
    with torch.no_grad():
        if enhancement_type == 'super_resolution':
            enhanced = models[enhancement_type](img_tensor, scale_factor)
        elif enhancement_type == 'denoising':
            enhanced = models[enhancement_type](img_tensor, denoise_strength)
        else:
            enhanced = models[enhancement_type](img_tensor)
    
    # Convert back to image
    enhanced_img = transforms.ToPILImage()(enhanced.squeeze(0))
    
    # Return enhanced image
    img_byte_arr = io.BytesIO()
    enhanced_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return StreamingResponse(img_byte_arr, media_type="image/png")

@app.post("/batch_enhance")
async def batch_enhance(files: List[UploadFile] = File(...)):
    """
    Enhance multiple images
    """
    results = []
    
    for file in files:
        # Process each image
        enhanced = await enhance_image(file)
        results.append({
            'filename': file.filename,
            'enhanced': enhanced
        })
    
    return results

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "gpu_available": torch.cuda.is_available()
    }
```

### Docker Deployment

```dockerfile
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Download models
RUN python3 download_models.py

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: image-enhancement-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: image-enhancement
  template:
    metadata:
      labels:
        app: image-enhancement
    spec:
      containers:
      - name: api
        image: image-enhancement:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "4"
          requests:
            nvidia.com/gpu: 1
            memory: "4Gi"
            cpu: "2"
        ports:
        - containerPort: 8000
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: image-enhancement-service
spec:
  selector:
    app: image-enhancement
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

## Benchmarking and Evaluation

### Performance Metrics

```python
class EnhancementEvaluator:
    def __init__(self):
        self.metrics = {
            'psnr': self.calculate_psnr,
            'ssim': self.calculate_ssim,
            'lpips': self.calculate_lpips,
            'niqe': self.calculate_niqe,
            'brisque': self.calculate_brisque
        }
        
        # Load perceptual models
        self.lpips_model = lpips.LPIPS(net='alex')
        
    def evaluate_enhancement(self, enhanced, reference=None, degraded=None):
        """
        Comprehensive evaluation of enhancement quality
        """
        results = {}
        
        # Reference-based metrics
        if reference is not None:
            results['psnr'] = self.calculate_psnr(enhanced, reference)
            results['ssim'] = self.calculate_ssim(enhanced, reference)
            results['lpips'] = self.calculate_lpips(enhanced, reference)
        
        # No-reference metrics
        results['niqe'] = self.calculate_niqe(enhanced)
        results['brisque'] = self.calculate_brisque(enhanced)
        
        # Improvement metrics
        if degraded is not None and reference is not None:
            results['improvement'] = {
                'psnr_gain': self.calculate_psnr(enhanced, reference) - 
                            self.calculate_psnr(degraded, reference),
                'ssim_gain': self.calculate_ssim(enhanced, reference) - 
                            self.calculate_ssim(degraded, reference)
            }
        
        return results
    
    def calculate_psnr(self, img1, img2):
        """
        Peak Signal-to-Noise Ratio
        """
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    def calculate_ssim(self, img1, img2):
        """
        Structural Similarity Index
        """
        return structural_similarity(
            img1.cpu().numpy(), 
            img2.cpu().numpy(),
            multichannel=True,
            data_range=1.0
        )
    
    def calculate_lpips(self, img1, img2):
        """
        Learned Perceptual Image Patch Similarity
        """
        return self.lpips_model(img1, img2).item()
```

## Best Practices

### Model Selection Guide

```python
class ModelSelector:
    def __init__(self):
        self.image_analyzer = ImageAnalyzer()
        self.models = self._load_all_models()
        
    def select_best_model(self, image, user_preference=None):
        """
        Automatically select best enhancement model
        """
        # Analyze image characteristics
        analysis = self.image_analyzer.analyze(image)
        
        # Determine primary issues
        issues = self._identify_issues(analysis)
        
        # Select model based on issues and preferences
        if user_preference:
            return self.models[user_preference]
        
        if 'low_resolution' in issues:
            return self.models['super_resolution']
        elif 'noise' in issues:
            return self.models['denoising']
        elif 'poor_colors' in issues:
            return self.models['color_correction']
        elif 'low_dynamic_range' in issues:
            return self.models['hdr']
        else:
            return self.models['general_enhancement']
    
    def _identify_issues(self, analysis):
        issues = []
        
        if analysis['resolution'] < 720:
            issues.append('low_resolution')
        
        if analysis['noise_level'] > 0.1:
            issues.append('noise')
        
        if analysis['color_score'] < 0.5:
            issues.append('poor_colors')
        
        if analysis['dynamic_range'] < 0.3:
            issues.append('low_dynamic_range')
        
        return issues
```

### Quality Control

```python
class QualityController:
    def __init__(self, threshold_config):
        self.thresholds = threshold_config
        self.artifact_detector = ArtifactDetector()
        
    def validate_enhancement(self, original, enhanced):
        """
        Ensure enhancement meets quality standards
        """
        # Check for common artifacts
        artifacts = self.artifact_detector.detect(enhanced)
        
        if artifacts['over_sharpening'] > self.thresholds['sharpening']:
            enhanced = self._reduce_sharpening(enhanced)
        
        if artifacts['color_bleeding'] > self.thresholds['color']:
            enhanced = self._fix_color_bleeding(enhanced)
        
        if artifacts['blocking'] > self.thresholds['blocking']:
            enhanced = self._reduce_blocking(enhanced)
        
        # Ensure output is within valid range
        enhanced = torch.clamp(enhanced, 0, 1)
        
        # Final quality check
        if not self._passes_quality_check(original, enhanced):
            # Fall back to safer enhancement
            enhanced = self._safe_enhance(original)
        
        return enhanced
    
    def _passes_quality_check(self, original, enhanced):
        """
        Comprehensive quality validation
        """
        # Check if enhancement is actually better
        original_quality = self._calculate_quality_score(original)
        enhanced_quality = self._calculate_quality_score(enhanced)
        
        return enhanced_quality > original_quality * 1.1  # 10% improvement threshold
```

*Originally from umitkacar/image-enhancement repository*