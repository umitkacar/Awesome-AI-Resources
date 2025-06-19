# Awesome Video Inpainting: State-of-the-Art Techniques and Implementations

## Table of Contents
- [Overview](#overview)
- [Video Inpainting Challenges](#video-inpainting-challenges)
- [State-of-the-Art Methods](#state-of-the-art-methods)
- [Implementation Guide](#implementation-guide)
- [Temporal Consistency Techniques](#temporal-consistency-techniques)
- [Flow-Guided Inpainting](#flow-guided-inpainting)
- [Deep Learning Approaches](#deep-learning-approaches)
- [Evaluation Metrics](#evaluation-metrics)
- [Applications](#applications)
- [Future Directions](#future-directions)

## Overview

Video inpainting is the process of filling in missing or corrupted regions in video sequences with visually plausible content. Unlike image inpainting, video inpainting must maintain temporal coherence across frames while handling complex motion patterns, occlusions, and appearance changes.

### Key Requirements
- **Temporal Consistency**: Smooth transitions across frames
- **Motion Coherence**: Realistic motion in inpainted regions
- **Texture Preservation**: Maintaining visual quality
- **Semantic Correctness**: Plausible content generation
- **Efficiency**: Reasonable processing time for practical use

## Video Inpainting Challenges

### 1. Temporal Coherence
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TemporalCoherenceModule(nn.Module):
    def __init__(self, feature_dim=256, memory_size=5):
        super(TemporalCoherenceModule, self).__init__()
        
        # Temporal memory
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # Temporal consistency loss
        self.consistency_net = nn.Sequential(
            nn.Conv3d(feature_dim, 128, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 1, kernel_size=(3, 3, 3), padding=1)
        )
        
    def forward(self, features, temporal_memory=None):
        """
        Args:
            features: (B, T, C, H, W) current features
            temporal_memory: Previous frame features
        """
        B, T, C, H, W = features.shape
        
        # Reshape for attention
        features_flat = features.permute(1, 0, 3, 4, 2).reshape(T, B*H*W, C)
        
        if temporal_memory is not None:
            # Concatenate with memory
            memory_flat = temporal_memory.permute(1, 0, 3, 4, 2).reshape(-1, B*H*W, C)
            
            # Apply temporal attention
            attended_features, _ = self.temporal_attention(
                features_flat,
                memory_flat,
                memory_flat
            )
        else:
            # Self-attention within current features
            attended_features, _ = self.temporal_attention(
                features_flat,
                features_flat,
                features_flat
            )
            
        # Reshape back
        attended_features = attended_features.reshape(T, B, H, W, C).permute(1, 0, 4, 2, 3)
        
        # Compute temporal consistency score
        consistency_score = self.consistency_net(attended_features.permute(0, 2, 1, 3, 4))
        
        return attended_features, consistency_score
    
    def compute_temporal_loss(self, inpainted_frames, original_frames, masks):
        """Compute temporal consistency loss"""
        # Optical flow between consecutive frames
        flow_loss = self.compute_flow_loss(inpainted_frames, masks)
        
        # Feature similarity loss
        feature_loss = self.compute_feature_similarity_loss(inpainted_frames)
        
        # Motion smoothness loss
        motion_loss = self.compute_motion_smoothness_loss(inpainted_frames, masks)
        
        return flow_loss + 0.5 * feature_loss + 0.3 * motion_loss
    
    def compute_flow_loss(self, frames, masks):
        """Ensure consistent optical flow"""
        flow_loss = 0
        
        for t in range(len(frames) - 1):
            # Compute flow between consecutive frames
            flow = self.estimate_flow(frames[t], frames[t+1])
            
            # Warp frame t to t+1
            warped = self.warp_frame(frames[t], flow)
            
            # Compute difference in valid regions
            valid_mask = masks[t] * masks[t+1]
            diff = (warped - frames[t+1]) * valid_mask
            
            flow_loss += torch.mean(torch.abs(diff))
            
        return flow_loss / (len(frames) - 1)
```

### 2. Motion Handling
```python
class MotionAwareInpainting(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=64):
        super(MotionAwareInpainting, self).__init__()
        
        # Motion encoder
        self.motion_encoder = nn.Sequential(
            nn.Conv3d(input_channels * 2, hidden_dim, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, hidden_dim * 2, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU()
        )
        
        # Motion predictor
        self.motion_predictor = MotionPredictor(hidden_dim * 2)
        
        # Motion-guided inpainting
        self.inpainting_net = MotionGuidedGenerator(hidden_dim * 2)
        
    def forward(self, frames, masks, reference_frames=None):
        """
        Args:
            frames: (B, T, C, H, W) input frames with holes
            masks: (B, T, 1, H, W) binary masks (1 for holes)
            reference_frames: Optional reference frames for motion
        """
        B, T, C, H, W = frames.shape
        
        # Estimate motion patterns
        if reference_frames is not None:
            motion_input = torch.cat([frames, reference_frames], dim=2)
        else:
            # Use neighboring frames for motion estimation
            motion_input = self.create_motion_pairs(frames)
            
        # Encode motion
        motion_features = self.motion_encoder(motion_input.permute(0, 2, 1, 3, 4))
        
        # Predict motion in missing regions
        predicted_motion = self.motion_predictor(motion_features, masks)
        
        # Generate inpainted content
        inpainted = self.inpainting_net(frames, masks, predicted_motion)
        
        return inpainted, predicted_motion
    
    def create_motion_pairs(self, frames):
        """Create frame pairs for motion estimation"""
        T = frames.shape[1]
        pairs = []
        
        for t in range(T):
            if t > 0:
                prev_frame = frames[:, t-1]
            else:
                prev_frame = frames[:, t]
                
            if t < T-1:
                next_frame = frames[:, t+1]
            else:
                next_frame = frames[:, t]
                
            pair = torch.stack([prev_frame, next_frame], dim=2)
            pairs.append(pair)
            
        return torch.stack(pairs, dim=1)

class MotionPredictor(nn.Module):
    def __init__(self, feature_dim):
        super(MotionPredictor, self).__init__()
        
        # Flow estimation network
        self.flow_net = nn.Sequential(
            nn.Conv2d(feature_dim, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 3, padding=1)  # 2D flow field
        )
        
        # Motion refinement
        self.refine_net = nn.Sequential(
            nn.Conv2d(feature_dim + 2, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, 3, padding=1)
        )
        
    def forward(self, motion_features, masks):
        """Predict motion in masked regions"""
        B, C, T, H, W = motion_features.shape
        
        predicted_flows = []
        
        for t in range(T):
            # Extract features for frame t
            feat_t = motion_features[:, :, t]
            
            # Initial flow prediction
            flow = self.flow_net(feat_t)
            
            # Refine flow in masked regions
            mask_t = masks[:, t]
            flow_input = torch.cat([feat_t, flow * mask_t], dim=1)
            refined_flow = self.refine_net(flow_input)
            
            # Combine flows
            final_flow = flow * (1 - mask_t) + refined_flow * mask_t
            predicted_flows.append(final_flow)
            
        return torch.stack(predicted_flows, dim=1)
```

### 3. Complex Occlusion Handling
```python
class OcclusionAwareInpainting(nn.Module):
    def __init__(self):
        super(OcclusionAwareInpainting, self).__init__()
        
        # Occlusion detection
        self.occlusion_detector = OcclusionDetector()
        
        # Layered representation
        self.layer_decomposition = LayerDecomposition()
        
        # Layer-wise inpainting
        self.foreground_inpainter = InpaintingNetwork(name='foreground')
        self.background_inpainter = InpaintingNetwork(name='background')
        
        # Layer composition
        self.compositor = LayerCompositor()
        
    def forward(self, frames, masks):
        """Handle complex occlusions in video inpainting"""
        # Detect occlusion boundaries and types
        occlusion_maps = self.occlusion_detector(frames, masks)
        
        # Decompose into layers
        foreground, background, alpha_maps = self.layer_decomposition(
            frames, masks, occlusion_maps
        )
        
        # Inpaint each layer separately
        inpainted_fg = self.foreground_inpainter(foreground, masks * alpha_maps)
        inpainted_bg = self.background_inpainter(background, masks * (1 - alpha_maps))
        
        # Composite layers
        final_frames = self.compositor(
            inpainted_fg, inpainted_bg, alpha_maps, masks
        )
        
        return final_frames

class OcclusionDetector(nn.Module):
    def __init__(self):
        super(OcclusionDetector, self).__init__()
        
        self.edge_detector = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        self.motion_boundary_detector = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 16, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 1, kernel_size=(3, 3, 3), padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, frames, masks):
        """Detect occlusion boundaries"""
        B, T, C, H, W = frames.shape
        
        # Spatial edge detection
        edge_maps = []
        for t in range(T):
            edges = self.edge_detector(frames[:, t])
            edge_maps.append(edges)
        edge_maps = torch.stack(edge_maps, dim=1)
        
        # Motion boundaries
        flow_pairs = []
        for t in range(T-1):
            flow = self.compute_flow(frames[:, t], frames[:, t+1])
            flow_pairs.append(flow)
        
        if len(flow_pairs) > 0:
            flow_volume = torch.stack(flow_pairs, dim=2).permute(0, 1, 3, 4, 2)
            motion_boundaries = self.motion_boundary_detector(flow_volume)
        else:
            motion_boundaries = torch.zeros_like(edge_maps)
            
        # Combine cues
        occlusion_maps = torch.max(edge_maps, motion_boundaries)
        
        return occlusion_maps
```

## State-of-the-Art Methods

### 1. Deep Flow-Guided Video Inpainting
```python
class DeepFlowGuidedInpainting(nn.Module):
    def __init__(self):
        super(DeepFlowGuidedInpainting, self).__init__()
        
        # Flow completion network
        self.flow_completion = FlowCompletionNetwork()
        
        # Feature extraction
        self.encoder = EncoderNetwork()
        
        # Propagation network
        self.propagation = PropagationNetwork()
        
        # Content hallucination
        self.hallucination = ContentHallucination()
        
        # Decoder
        self.decoder = DecoderNetwork()
        
    def forward(self, frames, masks):
        """
        Complete video inpainting pipeline
        Args:
            frames: (B, T, C, H, W) input frames
            masks: (B, T, 1, H, W) inpainting masks
        """
        B, T, C, H, W = frames.shape
        
        # Step 1: Complete optical flow
        flow_forward, flow_backward = self.complete_flow(frames, masks)
        
        # Step 2: Extract features
        features = self.encoder(frames)
        
        # Step 3: Propagate features using completed flow
        propagated_features = self.propagation(
            features, flow_forward, flow_backward, masks
        )
        
        # Step 4: Hallucinate content for remaining holes
        hallucinated_features = self.hallucination(
            propagated_features, masks
        )
        
        # Step 5: Decode to frames
        inpainted_frames = self.decoder(hallucinated_features)
        
        # Step 6: Blend with original
        output = frames * (1 - masks) + inpainted_frames * masks
        
        return output
    
    def complete_flow(self, frames, masks):
        """Complete optical flow in masked regions"""
        flow_forward = []
        flow_backward = []
        
        # Forward flow
        for t in range(len(frames) - 1):
            flow = self.flow_completion(
                frames[:, t], frames[:, t+1], 
                masks[:, t], masks[:, t+1]
            )
            flow_forward.append(flow)
            
        # Backward flow
        for t in range(len(frames) - 1, 0, -1):
            flow = self.flow_completion(
                frames[:, t], frames[:, t-1],
                masks[:, t], masks[:, t-1]
            )
            flow_backward.append(flow)
            
        return flow_forward, flow_backward

class FlowCompletionNetwork(nn.Module):
    def __init__(self):
        super(FlowCompletionNetwork, self).__init__()
        
        # Initial flow estimation
        self.flow_estimator = nn.Sequential(
            nn.Conv2d(6, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Flow completion
        self.completion_net = nn.Sequential(
            nn.Conv2d(256 + 2, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 2, 4, stride=2, padding=1)
        )
        
    def forward(self, frame1, frame2, mask1, mask2):
        """Complete flow between two frames"""
        # Concatenate frames
        input_frames = torch.cat([frame1, frame2], dim=1)
        
        # Extract features
        flow_features = self.flow_estimator(input_frames)
        
        # Combine masks
        combined_mask = torch.max(mask1, mask2)
        mask_resized = F.interpolate(combined_mask, size=flow_features.shape[-2:])
        
        # Complete flow
        completion_input = torch.cat([flow_features, mask_resized], dim=1)
        completed_flow = self.completion_net(completion_input)
        
        return completed_flow

class PropagationNetwork(nn.Module):
    def __init__(self, feature_dim=256):
        super(PropagationNetwork, self).__init__()
        
        self.feature_dim = feature_dim
        
        # Bidirectional propagation
        self.forward_prop = nn.GRU(
            feature_dim, feature_dim // 2, 
            bidirectional=False, batch_first=True
        )
        self.backward_prop = nn.GRU(
            feature_dim, feature_dim // 2,
            bidirectional=False, batch_first=True
        )
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1)
        )
        
    def forward(self, features, flow_forward, flow_backward, masks):
        """Propagate features bidirectionally"""
        B, T, C, H, W = features.shape
        
        # Reshape for RNN
        features_flat = features.view(B, T, -1)
        
        # Forward propagation
        forward_out, _ = self.forward_prop(features_flat)
        
        # Backward propagation
        backward_input = features_flat.flip(dims=[1])
        backward_out, _ = self.backward_prop(backward_input)
        backward_out = backward_out.flip(dims=[1])
        
        # Combine bidirectional features
        combined = torch.cat([forward_out, backward_out], dim=-1)
        combined = combined.view(B, T, C, H, W)
        
        # Apply flow warping
        warped_features = self.flow_warp(combined, flow_forward, flow_backward)
        
        # Fusion
        fused = self.fusion(warped_features.view(B*T, C, H, W))
        fused = fused.view(B, T, C, H, W)
        
        return fused
    
    def flow_warp(self, features, flow_forward, flow_backward):
        """Warp features using optical flow"""
        warped = features.clone()
        
        # Forward warping
        for t in range(len(flow_forward)):
            if t < features.shape[1] - 1:
                warped[:, t+1] = self.warp_frame(
                    features[:, t], flow_forward[t]
                )
                
        # Backward warping
        for t in range(len(flow_backward)):
            if t < features.shape[1] - 1:
                warped[:, -(t+2)] = self.warp_frame(
                    features[:, -(t+1)], flow_backward[t]
                )
                
        return warped
```

### 2. Onion-Peel Network
```python
class OnionPeelNetwork(nn.Module):
    def __init__(self, num_layers=3):
        super(OnionPeelNetwork, self).__init__()
        
        self.num_layers = num_layers
        
        # Multi-scale encoder
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(4, 64 * (2**i), 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64 * (2**i), 64 * (2**i), 3, padding=1),
                nn.ReLU()
            ) for i in range(num_layers)
        ])
        
        # Recurrent propagation modules
        self.propagators = nn.ModuleList([
            RecurrentPropagation(64 * (2**i)) for i in range(num_layers)
        ])
        
        # Multi-scale decoder
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(64 * (2**i), 64 * (2**max(i-1, 0)), 
                                 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64 * (2**max(i-1, 0)), 3 if i == 0 else 64 * (2**max(i-1, 0)), 
                        3, padding=1)
            ) for i in range(num_layers)
        ])
        
    def forward(self, frames, masks):
        """
        Onion-peel inpainting from boundaries to center
        """
        B, T, C, H, W = frames.shape
        
        # Prepare input
        input_frames = torch.cat([frames, masks], dim=2)
        
        # Multi-scale encoding
        encoded_features = []
        current_input = input_frames
        
        for i, encoder in enumerate(self.encoders):
            # Encode at current scale
            features = []
            for t in range(T):
                feat = encoder(current_input[:, t])
                features.append(feat)
            features = torch.stack(features, dim=1)
            encoded_features.append(features)
            
            # Downsample for next scale
            if i < self.num_layers - 1:
                current_input = F.interpolate(
                    current_input.view(B*T, C+1, H, W),
                    scale_factor=0.5
                ).view(B, T, C+1, H//(2**(i+1)), W//(2**(i+1)))
                
        # Recurrent propagation at each scale
        propagated_features = []
        
        for i, (features, propagator) in enumerate(
            zip(encoded_features, self.propagators)
        ):
            # Create dilated masks for onion peeling
            scale_masks = self.create_dilated_masks(
                masks, 
                scale=2**i,
                num_iterations=5
            )
            
            # Propagate at current scale
            prop_feat = propagator(features, scale_masks)
            propagated_features.append(prop_feat)
            
        # Multi-scale decoding
        decoded = None
        
        for i in range(self.num_layers - 1, -1, -1):
            if decoded is None:
                decoded = propagated_features[i]
            else:
                # Upsample and combine
                decoded = F.interpolate(
                    decoded.view(B*T, -1, decoded.shape[-2], decoded.shape[-1]),
                    scale_factor=2
                ).view(B, T, -1, decoded.shape[-2]*2, decoded.shape[-1]*2)
                
                # Skip connection
                decoded = decoded + propagated_features[i]
                
            # Decode
            decoded_frames = []
            for t in range(T):
                frame = self.decoders[i](decoded[:, t])
                decoded_frames.append(frame)
            decoded = torch.stack(decoded_frames, dim=1)
            
        return decoded
    
    def create_dilated_masks(self, masks, scale, num_iterations):
        """Create dilated masks for onion peeling"""
        dilated_masks = []
        kernel_size = 3 * scale
        kernel = torch.ones(1, 1, kernel_size, kernel_size).to(masks.device)
        
        current_mask = masks
        for _ in range(num_iterations):
            # Dilate mask
            dilated = F.conv2d(
                current_mask.view(-1, 1, masks.shape[-2], masks.shape[-1]),
                kernel,
                padding=kernel_size//2
            )
            dilated = (dilated > 0).float()
            dilated = dilated.view(masks.shape)
            
            dilated_masks.append(dilated)
            current_mask = dilated
            
        return dilated_masks

class RecurrentPropagation(nn.Module):
    def __init__(self, feature_dim):
        super(RecurrentPropagation, self).__init__()
        
        self.feature_dim = feature_dim
        
        # Convolutional GRU
        self.conv_gru = ConvGRU(
            input_dim=feature_dim,
            hidden_dim=feature_dim,
            kernel_size=3
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, features, masks):
        """Recurrent propagation with attention"""
        B, T, C, H, W = features.shape
        
        # Initialize hidden state
        hidden = torch.zeros(B, C, H, W).to(features.device)
        
        outputs = []
        
        for t in range(T):
            # Current features and mask
            feat_t = features[:, t]
            mask_t = masks[t] if isinstance(masks, list) else masks[:, t]
            
            # Compute attention weights
            attn_input = torch.cat([feat_t, hidden], dim=1)
            attn_weights = self.attention(attn_input)
            
            # Apply attention
            attended_feat = feat_t * attn_weights
            
            # Update hidden state
            hidden = self.conv_gru(attended_feat, hidden, mask_t)
            
            outputs.append(hidden)
            
        return torch.stack(outputs, dim=1)
```

### 3. Copy-and-Paste Networks
```python
class CopyPasteNetwork(nn.Module):
    def __init__(self):
        super(CopyPasteNetwork, self).__init__()
        
        # Feature extractor
        self.encoder = ResNetEncoder()
        
        # Contextual attention
        self.contextual_attention = ContextualAttention(
            patch_size=3,
            propagate_size=3,
            stride=1
        )
        
        # Memory bank
        self.memory_bank = MemoryBank(capacity=100)
        
        # Refinement network
        self.refinement = RefinementNetwork()
        
    def forward(self, frames, masks):
        """Copy and paste inpainting"""
        B, T, C, H, W = frames.shape
        
        # Extract features
        features = self.encoder(frames.view(B*T, C, H, W))
        features = features.view(B, T, -1, features.shape[-2], features.shape[-1])
        
        inpainted_frames = []
        
        for t in range(T):
            # Get current frame and mask
            frame_t = frames[:, t]
            mask_t = masks[:, t]
            feat_t = features[:, t]
            
            # Search for best patches in memory
            if t > 0:
                # Use previous frames as reference
                ref_features = features[:, :t]
                ref_frames = frames[:, :t]
                
                # Contextual attention
                matched_patches, attention_map = self.contextual_attention(
                    feat_t, ref_features, mask_t
                )
                
                # Copy and paste
                pasted_frame = self.copy_paste(
                    frame_t, ref_frames, attention_map, mask_t
                )
            else:
                # For first frame, use spatial inpainting
                pasted_frame = self.spatial_inpaint(frame_t, mask_t)
                
            # Refine result
            refined_frame = self.refinement(pasted_frame, frame_t, mask_t)
            inpainted_frames.append(refined_frame)
            
            # Update memory bank
            self.memory_bank.update(refined_frame, feat_t)
            
        return torch.stack(inpainted_frames, dim=1)
    
    def copy_paste(self, target_frame, ref_frames, attention_map, mask):
        """Copy patches from reference frames"""
        B, T_ref, C, H, W = ref_frames.shape
        
        # Reshape attention map
        attn = attention_map.view(B, T_ref, H*W, H, W)
        
        # For each pixel in mask, find best matching patch
        pasted = target_frame.clone()
        
        for b in range(B):
            mask_indices = torch.nonzero(mask[b, 0] > 0)
            
            for idx in mask_indices:
                y, x = idx[0].item(), idx[1].item()
                
                # Find best matching position
                attn_scores = attn[b, :, :, y, x]
                best_t, best_pos = torch.unravel_index(
                    attn_scores.argmax(), 
                    attn_scores.shape
                )
                best_y, best_x = best_pos // W, best_pos % W
                
                # Copy patch
                patch_size = 3
                y1 = max(0, y - patch_size // 2)
                y2 = min(H, y + patch_size // 2 + 1)
                x1 = max(0, x - patch_size // 2)
                x2 = min(W, x + patch_size // 2 + 1)
                
                ref_y1 = max(0, best_y - patch_size // 2)
                ref_y2 = min(H, best_y + patch_size // 2 + 1)
                ref_x1 = max(0, best_x - patch_size // 2)
                ref_x2 = min(W, best_x + patch_size // 2 + 1)
                
                # Ensure same size
                h = min(y2-y1, ref_y2-ref_y1)
                w = min(x2-x1, ref_x2-ref_x1)
                
                pasted[b, :, y1:y1+h, x1:x1+w] = \
                    ref_frames[b, best_t, :, ref_y1:ref_y1+h, ref_x1:ref_x1+w]
                    
        return pasted

class ContextualAttention(nn.Module):
    def __init__(self, patch_size=3, propagate_size=3, stride=1):
        super(ContextualAttention, self).__init__()
        
        self.patch_size = patch_size
        self.propagate_size = propagate_size
        self.stride = stride
        
        # Feature transformation
        self.feat_transform = nn.Conv2d(256, 128, 1)
        
    def forward(self, query_features, ref_features, mask):
        """Compute attention between query and reference features"""
        B, C, H, W = query_features.shape
        B, T_ref, C, H_ref, W_ref = ref_features.shape
        
        # Transform features
        query_feat = self.feat_transform(query_features)
        ref_feat = self.feat_transform(ref_features.view(B*T_ref, C, H_ref, W_ref))
        ref_feat = ref_feat.view(B, T_ref, -1, H_ref, W_ref)
        
        # Extract patches
        query_patches = self.extract_patches(query_feat, self.patch_size)
        ref_patches = self.extract_patches(
            ref_feat.view(B*T_ref, -1, H_ref, W_ref), 
            self.patch_size
        )
        
        # Compute similarity
        attention_maps = []
        
        for t in range(T_ref):
            # Compute cosine similarity
            similarity = F.cosine_similarity(
                query_patches.unsqueeze(2),
                ref_patches[t].unsqueeze(1),
                dim=3
            )
            
            # Apply mask
            similarity = similarity * mask.view(B, -1, 1)
            
            # Softmax normalization
            attention = F.softmax(similarity, dim=2)
            attention_maps.append(attention)
            
        return ref_patches, torch.stack(attention_maps, dim=1)
```

## Implementation Guide

### Complete Video Inpainting Pipeline

```python
import cv2
import torch
import numpy as np
from collections import deque

class VideoInpaintingPipeline:
    def __init__(self, model_type='flow_guided', device='cuda'):
        self.device = torch.device(device)
        self.model = self.load_model(model_type)
        self.preprocessor = VideoPreprocessor()
        self.postprocessor = VideoPostprocessor()
        
    def load_model(self, model_type):
        """Load pre-trained inpainting model"""
        models = {
            'flow_guided': DeepFlowGuidedInpainting(),
            'onion_peel': OnionPeelNetwork(),
            'copy_paste': CopyPasteNetwork()
        }
        
        model = models[model_type]
        # Load pre-trained weights
        checkpoint = torch.load(f'weights/{model_type}_inpainting.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def inpaint_video(self, video_path, mask_path, output_path):
        """Inpaint entire video"""
        # Load video and masks
        frames, fps, frame_size = self.load_video(video_path)
        masks = self.load_masks(mask_path, len(frames))
        
        # Preprocess
        processed_frames, processed_masks = self.preprocessor(frames, masks)
        
        # Split into batches for memory efficiency
        batch_size = 16
        inpainted_frames = []
        
        for i in range(0, len(processed_frames), batch_size):
            batch_frames = processed_frames[i:i+batch_size]
            batch_masks = processed_masks[i:i+batch_size]
            
            # Convert to tensors
            frames_tensor = torch.from_numpy(batch_frames).to(self.device)
            masks_tensor = torch.from_numpy(batch_masks).to(self.device)
            
            # Inpaint
            with torch.no_grad():
                inpainted_batch = self.model(frames_tensor, masks_tensor)
                
            # Convert back to numpy
            inpainted_batch = inpainted_batch.cpu().numpy()
            inpainted_frames.extend(inpainted_batch)
            
        # Postprocess
        final_frames = self.postprocessor(
            np.array(inpainted_frames), 
            frames, 
            masks
        )
        
        # Save video
        self.save_video(final_frames, output_path, fps, frame_size)
        
    def load_video(self, video_path):
        """Load video frames"""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
        cap.release()
        
        return np.array(frames), fps, (width, height)
    
    def load_masks(self, mask_path, num_frames):
        """Load inpainting masks"""
        if os.path.isdir(mask_path):
            # Load frame-by-frame masks
            masks = []
            for i in range(num_frames):
                mask_file = os.path.join(mask_path, f'mask_{i:05d}.png')
                if os.path.exists(mask_file):
                    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                    mask = (mask > 127).astype(np.float32)
                else:
                    # Use previous mask or zeros
                    mask = masks[-1] if masks else np.zeros((height, width))
                masks.append(mask)
        else:
            # Single mask for all frames
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
            masks = [mask] * num_frames
            
        return np.array(masks)
    
    def save_video(self, frames, output_path, fps, frame_size):
        """Save inpainted video"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            
        out.release()

class VideoPreprocessor:
    def __init__(self, target_size=(432, 240)):
        self.target_size = target_size
        
    def __call__(self, frames, masks):
        """Preprocess video frames and masks"""
        # Resize
        processed_frames = []
        processed_masks = []
        
        for frame, mask in zip(frames, masks):
            # Resize frame
            frame_resized = cv2.resize(frame, self.target_size)
            processed_frames.append(frame_resized)
            
            # Resize mask
            mask_resized = cv2.resize(
                mask.astype(np.float32), 
                self.target_size
            )
            mask_resized = (mask_resized > 0.5).astype(np.float32)
            processed_masks.append(mask_resized)
            
        # Normalize frames
        processed_frames = np.array(processed_frames).astype(np.float32) / 255.0
        processed_masks = np.array(processed_masks)
        
        # Add channel dimension to masks
        processed_masks = np.expand_dims(processed_masks, axis=-1)
        
        # Convert to NCHW format
        processed_frames = processed_frames.transpose(0, 3, 1, 2)
        processed_masks = processed_masks.transpose(0, 3, 1, 2)
        
        return processed_frames, processed_masks

class VideoPostprocessor:
    def __init__(self):
        self.temporal_filter = TemporalFilter()
        self.color_corrector = ColorCorrector()
        
    def __call__(self, inpainted_frames, original_frames, masks):
        """Post-process inpainted frames"""
        # Convert back to HWC format
        inpainted_frames = inpainted_frames.transpose(0, 2, 3, 1)
        
        # Denormalize
        inpainted_frames = (inpainted_frames * 255).astype(np.uint8)
        
        # Resize to original size
        H, W = original_frames.shape[1:3]
        resized_frames = []
        for frame in inpainted_frames:
            resized = cv2.resize(frame, (W, H))
            resized_frames.append(resized)
        inpainted_frames = np.array(resized_frames)
        
        # Apply temporal filtering
        filtered_frames = self.temporal_filter(inpainted_frames, masks)
        
        # Color correction
        corrected_frames = self.color_corrector(
            filtered_frames, 
            original_frames, 
            masks
        )
        
        return corrected_frames

class TemporalFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        
    def __call__(self, frames, masks):
        """Apply temporal filtering for smoothness"""
        filtered = np.zeros_like(frames)
        T = len(frames)
        
        for t in range(T):
            # Get temporal window
            start = max(0, t - self.window_size // 2)
            end = min(T, t + self.window_size // 2 + 1)
            
            # Weighted average
            weights = np.exp(-np.abs(np.arange(start, end) - t) / 2)
            weights = weights / weights.sum()
            
            for i, w in enumerate(weights):
                filtered[t] += w * frames[start + i]
                
        return filtered.astype(np.uint8)
```

## Temporal Consistency Techniques

### 1. Multi-Frame Aggregation
```python
class MultiFrameAggregation(nn.Module):
    def __init__(self, num_frames=5):
        super(MultiFrameAggregation, self).__init__()
        
        self.num_frames = num_frames
        
        # Feature alignment
        self.alignment_net = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, 3, padding=1)
        )
        
        # Feature fusion
        self.fusion_net = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        )
        
    def forward(self, features, reference_frame_idx):
        """
        Aggregate features from multiple frames
        Args:
            features: (B, T, C, H, W) frame features
            reference_frame_idx: Index of reference frame
        """
        B, T, C, H, W = features.shape
        
        # Select frames around reference
        start_idx = max(0, reference_frame_idx - self.num_frames // 2)
        end_idx = min(T, reference_frame_idx + self.num_frames // 2 + 1)
        
        selected_features = features[:, start_idx:end_idx]
        ref_features = features[:, reference_frame_idx]
        
        # Align features to reference frame
        aligned_features = []
        
        for i in range(selected_features.shape[1]):
            if start_idx + i == reference_frame_idx:
                aligned_features.append(selected_features[:, i])
            else:
                # Compute alignment
                feat_pair = torch.cat([
                    selected_features[:, i], 
                    ref_features
                ], dim=1)
                
                alignment_flow = self.alignment_net(feat_pair)
                
                # Warp features
                aligned = self.warp_features(
                    selected_features[:, i], 
                    alignment_flow
                )
                aligned_features.append(aligned)
                
        # Stack aligned features
        aligned_features = torch.stack(aligned_features, dim=1)
        
        # Fuse features
        fused = self.fusion_net(aligned_features.permute(0, 2, 1, 3, 4))
        fused = fused.squeeze(2)
        
        return fused
```

### 2. Temporal Pyramids
```python
class TemporalPyramidInpainting(nn.Module):
    def __init__(self, pyramid_levels=3):
        super(TemporalPyramidInpainting, self).__init__()
        
        self.pyramid_levels = pyramid_levels
        
        # Inpainting at each temporal scale
        self.inpainters = nn.ModuleList([
            TemporalInpainter(scale=2**i) for i in range(pyramid_levels)
        ])
        
        # Fusion network
        self.fusion = PyramidFusion()
        
    def forward(self, frames, masks):
        """Multi-scale temporal inpainting"""
        # Build temporal pyramid
        pyramid_frames = [frames]
        pyramid_masks = [masks]
        
        for level in range(1, self.pyramid_levels):
            # Temporal downsampling
            downsampled_frames = frames[:, ::2**level]
            downsampled_masks = masks[:, ::2**level]
            
            pyramid_frames.append(downsampled_frames)
            pyramid_masks.append(downsampled_masks)
            
        # Inpaint at each level
        inpainted_pyramid = []
        
        for level in range(self.pyramid_levels):
            inpainted = self.inpainters[level](
                pyramid_frames[level], 
                pyramid_masks[level]
            )
            
            # Upsample to original temporal resolution
            if level > 0:
                inpainted = self.temporal_upsample(
                    inpainted, 
                    target_length=frames.shape[1]
                )
                
            inpainted_pyramid.append(inpainted)
            
        # Fuse pyramid levels
        final_inpainted = self.fusion(inpainted_pyramid)
        
        return final_inpainted
    
    def temporal_upsample(self, frames, target_length):
        """Upsample frames temporally"""
        B, T, C, H, W = frames.shape
        
        # Linear interpolation
        upsampled = F.interpolate(
            frames.permute(0, 2, 1, 3, 4).reshape(B*C, T, H*W),
            size=target_length,
            mode='linear',
            align_corners=True
        )
        
        upsampled = upsampled.reshape(B, C, target_length, H, W).permute(0, 2, 1, 3, 4)
        
        return upsampled
```

## Flow-Guided Inpainting

### Optical Flow Completion
```python
class OpticalFlowCompletion(nn.Module):
    def __init__(self):
        super(OpticalFlowCompletion, self).__init__()
        
        # Edge-aware flow completion
        self.edge_detector = EdgeDetector()
        
        # Flow diffusion
        self.flow_diffusion = FlowDiffusion()
        
        # Flow refinement
        self.flow_refiner = FlowRefinement()
        
    def forward(self, incomplete_flow, frame1, frame2, mask):
        """Complete optical flow in masked regions"""
        # Detect edges for guidance
        edges1 = self.edge_detector(frame1)
        edges2 = self.edge_detector(frame2)
        edge_map = torch.max(edges1, edges2)
        
        # Initial flow completion by diffusion
        diffused_flow = self.flow_diffusion(
            incomplete_flow, 
            mask, 
            edge_map
        )
        
        # Refine flow
        refined_flow = self.flow_refiner(
            diffused_flow, 
            frame1, 
            frame2, 
            mask
        )
        
        return refined_flow

class FlowDiffusion(nn.Module):
    def __init__(self, num_iterations=50):
        super(FlowDiffusion, self).__init__()
        
        self.num_iterations = num_iterations
        
        # Learnable diffusion parameters
        self.diffusion_weight = nn.Parameter(torch.tensor(0.25))
        self.edge_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, flow, mask, edge_map):
        """Diffuse flow from known to unknown regions"""
        # Create diffusion kernel
        kernel = torch.tensor([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        kernel = kernel.repeat(2, 1, 1, 1).to(flow.device)
        
        # Iterative diffusion
        current_flow = flow.clone()
        
        for _ in range(self.num_iterations):
            # Compute Laplacian
            padded_flow = F.pad(current_flow, (1, 1, 1, 1), mode='replicate')
            laplacian = F.conv2d(padded_flow, kernel, groups=2)
            
            # Edge-aware diffusion
            diffusion_rate = self.diffusion_weight * (1 - self.edge_weight * edge_map)
            
            # Update flow in masked regions
            update = laplacian * diffusion_rate * mask
            current_flow = current_flow + update
            
            # Keep known values fixed
            current_flow = flow * (1 - mask) + current_flow * mask
            
        return current_flow
```

## Deep Learning Approaches

### 1. 3D CNN Architecture
```python
class Video3DCNN(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super(Video3DCNN, self).__init__()
        
        # 3D Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 512, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU()
        )
        
        # Bottleneck with temporal attention
        self.bottleneck = TemporalAttentionBottleneck(512)
        
        # 3D Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, frames, masks):
        """
        3D CNN for video inpainting
        Args:
            frames: (B, T, C, H, W)
            masks: (B, T, 1, H, W)
        """
        # Concatenate frames and masks
        x = torch.cat([frames, masks], dim=2)
        
        # Permute to (B, C, T, H, W) for 3D convolutions
        x = x.permute(0, 2, 1, 3, 4)
        
        # Encode
        encoded = self.encoder(x)
        
        # Apply temporal attention
        attended = self.bottleneck(encoded)
        
        # Decode
        decoded = self.decoder(attended)
        
        # Permute back to (B, T, C, H, W)
        output = decoded.permute(0, 2, 1, 3, 4)
        
        return output

class TemporalAttentionBottleneck(nn.Module):
    def __init__(self, channels):
        super(TemporalAttentionBottleneck, self).__init__()
        
        # Temporal self-attention
        self.temporal_attn = nn.Sequential(
            nn.Conv3d(channels, channels // 8, kernel_size=1),
            nn.BatchNorm3d(channels // 8),
            nn.ReLU(),
            nn.Conv3d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(channels // 16, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Temporal attention
        t_attn = self.temporal_attn(x)
        x_t = x * t_attn
        
        # Channel attention
        c_attn = self.channel_attn(x_t)
        x_out = x_t * c_attn
        
        return x_out
```

### 2. Transformer-based Approach
```python
class VideoTransformerInpainting(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6):
        super(VideoTransformerInpainting, self).__init__()
        
        # Patch embedding
        self.patch_embed = VideoPatchEmbedding(
            patch_size=(2, 8, 8),
            in_channels=4,
            embed_dim=d_model
        )
        
        # Positional encoding
        self.pos_encoding = SpatioTemporalPositionalEncoding(d_model)
        
        # Transformer encoder-decoder
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=2048,
            dropout=0.1
        )
        
        # Mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Output projection
        self.output_proj = nn.Linear(d_model, 3 * 8 * 8 * 2)  # 3 channels * patch size
        
    def forward(self, frames, masks):
        """
        Transformer-based video inpainting
        """
        B, T, C, H, W = frames.shape
        
        # Patch embedding
        x = torch.cat([frames, masks], dim=2)
        patches, patch_info = self.patch_embed(x)
        
        # Replace masked patches with mask token
        mask_patches = F.interpolate(
            masks.permute(0, 2, 1, 3, 4),
            size=(T//2, H//8, W//8),
            mode='nearest'
        ).permute(0, 2, 3, 4, 1)
        
        mask_patches = mask_patches.reshape(B, -1, 1)
        patches = patches * (1 - mask_patches) + self.mask_token * mask_patches
        
        # Add positional encoding
        patches = self.pos_encoding(patches, patch_info)
        
        # Transformer processing
        memory = self.transformer.encoder(patches.transpose(0, 1))
        output = self.transformer.decoder(patches.transpose(0, 1), memory)
        output = output.transpose(0, 1)
        
        # Project to pixel space
        pixels = self.output_proj(output)
        pixels = pixels.reshape(B, T//2, H//8, W//8, 3, 2, 8, 8)
        pixels = pixels.permute(0, 1, 4, 5, 2, 6, 3, 7)
        pixels = pixels.reshape(B, T, 3, H, W)
        
        return pixels

class VideoPatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super(VideoPatchEmbedding, self).__init__()
        
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels, 
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        
        # Extract patches
        patches = self.proj(x)
        
        # Flatten patches
        patches = patches.flatten(2).transpose(1, 2)
        
        # Patch information for positional encoding
        patch_info = {
            'temporal_patches': patches.shape[0] // self.patch_size[0],
            'height_patches': H // self.patch_size[1],
            'width_patches': W // self.patch_size[2]
        }
        
        return patches, patch_info
```

## Evaluation Metrics

### Video Inpainting Metrics
```python
class VideoInpaintingMetrics:
    def __init__(self):
        self.lpips_model = self.load_lpips_model()
        self.fvd_model = self.load_fvd_model()
        
    def evaluate(self, inpainted_video, ground_truth_video, masks):
        """Comprehensive evaluation of video inpainting"""
        metrics = {}
        
        # Pixel-level metrics
        metrics['psnr'] = self.compute_psnr(inpainted_video, ground_truth_video, masks)
        metrics['ssim'] = self.compute_ssim(inpainted_video, ground_truth_video, masks)
        
        # Perceptual metrics
        metrics['lpips'] = self.compute_lpips(inpainted_video, ground_truth_video, masks)
        
        # Video-specific metrics
        metrics['temporal_consistency'] = self.compute_temporal_consistency(inpainted_video)
        metrics['flow_warping_error'] = self.compute_flow_warping_error(
            inpainted_video, ground_truth_video
        )
        
        # Distribution metrics
        metrics['fvd'] = self.compute_fvd(inpainted_video, ground_truth_video)
        
        return metrics
    
    def compute_psnr(self, pred, gt, masks):
        """Peak Signal-to-Noise Ratio"""
        mse = ((pred - gt) ** 2 * masks).sum() / masks.sum()
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        return psnr
    
    def compute_ssim(self, pred, gt, masks):
        """Structural Similarity Index"""
        from skimage.metrics import structural_similarity
        
        ssim_values = []
        for i in range(len(pred)):
            # Compute SSIM only in inpainted regions
            mask = masks[i].squeeze()
            if mask.sum() > 0:
                ssim = structural_similarity(
                    pred[i], gt[i],
                    multichannel=True,
                    data_range=1.0
                )
                ssim_values.append(ssim)
                
        return np.mean(ssim_values)
    
    def compute_lpips(self, pred, gt, masks):
        """Learned Perceptual Image Patch Similarity"""
        lpips_values = []
        
        for i in range(len(pred)):
            # Convert to tensor
            pred_tensor = torch.from_numpy(pred[i]).permute(2, 0, 1).unsqueeze(0)
            gt_tensor = torch.from_numpy(gt[i]).permute(2, 0, 1).unsqueeze(0)
            
            # Compute LPIPS
            with torch.no_grad():
                lpips = self.lpips_model(pred_tensor, gt_tensor)
                lpips_values.append(lpips.item())
                
        return np.mean(lpips_values)
    
    def compute_temporal_consistency(self, video):
        """Measure temporal consistency"""
        consistency_scores = []
        
        for t in range(1, len(video)):
            # Compute optical flow
            flow = cv2.calcOpticalFlowFarneback(
                cv2.cvtColor(video[t-1], cv2.COLOR_RGB2GRAY),
                cv2.cvtColor(video[t], cv2.COLOR_RGB2GRAY),
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Warp previous frame
            h, w = video[t].shape[:2]
            flow_map = np.meshgrid(np.arange(w), np.arange(h))
            flow_map = np.stack(flow_map, axis=-1).astype(np.float32)
            flow_map += flow
            
            warped = cv2.remap(
                video[t-1], 
                flow_map[..., 0], 
                flow_map[..., 1],
                cv2.INTER_LINEAR
            )
            
            # Compute consistency
            diff = np.abs(warped - video[t]).mean()
            consistency_scores.append(1 - diff)
            
        return np.mean(consistency_scores)
    
    def compute_flow_warping_error(self, pred, gt):
        """Flow warping error between consecutive frames"""
        errors = []
        
        for t in range(1, len(pred)):
            # Compute flow in ground truth
            flow_gt = cv2.calcOpticalFlowFarneback(
                cv2.cvtColor(gt[t-1], cv2.COLOR_RGB2GRAY),
                cv2.cvtColor(gt[t], cv2.COLOR_RGB2GRAY),
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Compute flow in prediction
            flow_pred = cv2.calcOpticalFlowFarneback(
                cv2.cvtColor(pred[t-1], cv2.COLOR_RGB2GRAY),
                cv2.cvtColor(pred[t], cv2.COLOR_RGB2GRAY),
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Flow error
            error = np.sqrt(((flow_pred - flow_gt) ** 2).sum(axis=-1)).mean()
            errors.append(error)
            
        return np.mean(errors)
```

## Applications

### 1. Object Removal in Videos
```python
class VideoObjectRemoval:
    def __init__(self):
        self.segmentation_model = self.load_segmentation_model()
        self.inpainting_model = self.load_inpainting_model()
        self.tracker = ObjectTracker()
        
    def remove_object(self, video_path, object_description=None, initial_mask=None):
        """Remove specified object from video"""
        # Load video
        frames = self.load_video(video_path)
        
        # Get object masks
        if initial_mask is not None:
            # Track object through video
            masks = self.tracker.track_from_mask(frames, initial_mask)
        else:
            # Segment object based on description
            masks = self.segment_object(frames, object_description)
            
        # Dilate masks for better removal
        dilated_masks = self.dilate_masks(masks, kernel_size=15)
        
        # Inpaint video
        inpainted_frames = self.inpainting_model(frames, dilated_masks)
        
        # Post-process for seamless integration
        final_frames = self.post_process(inpainted_frames, frames, dilated_masks)
        
        return final_frames
    
    def segment_object(self, frames, description):
        """Segment object based on text description"""
        masks = []
        
        for frame in frames:
            # Use CLIP or similar model for text-guided segmentation
            mask = self.segmentation_model.segment_by_text(frame, description)
            masks.append(mask)
            
        return np.array(masks)
    
    def dilate_masks(self, masks, kernel_size=15):
        """Dilate masks to ensure complete object removal"""
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (kernel_size, kernel_size)
        )
        
        dilated = []
        for mask in masks:
            dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
            dilated.append(dilated_mask)
            
        return np.array(dilated)
```

### 2. Video Restoration
```python
class VideoRestoration:
    def __init__(self):
        self.damage_detector = DamageDetector()
        self.inpainting_model = VideoInpaintingPipeline()
        
    def restore_damaged_video(self, video_path):
        """Restore damaged or corrupted video"""
        frames = self.load_video(video_path)
        
        # Detect damaged regions
        damage_masks = self.damage_detector.detect_damage(frames)
        
        # Classify damage types
        damage_types = self.classify_damage(frames, damage_masks)
        
        # Apply appropriate restoration
        restored_frames = []
        
        for i, (frame, mask, damage_type) in enumerate(
            zip(frames, damage_masks, damage_types)
        ):
            if damage_type == 'scratch':
                restored = self.restore_scratch(frame, mask)
            elif damage_type == 'stain':
                restored = self.restore_stain(frame, mask)
            elif damage_type == 'missing':
                # Use temporal information
                context_frames = self.get_temporal_context(frames, i)
                restored = self.inpainting_model.inpaint_with_context(
                    frame, mask, context_frames
                )
            else:
                restored = frame
                
            restored_frames.append(restored)
            
        # Apply temporal smoothing
        smoothed_frames = self.temporal_smoothing(restored_frames, damage_masks)
        
        return smoothed_frames
    
    def restore_scratch(self, frame, mask):
        """Restore film scratches"""
        # Use specialized scratch removal
        # Typically involves line detection and interpolation
        return self.scratch_removal_algorithm(frame, mask)
    
    def restore_stain(self, frame, mask):
        """Restore stains or spots"""
        # Color correction and blending
        surrounding_color = self.sample_surrounding_color(frame, mask)
        return self.blend_color(frame, mask, surrounding_color)
```

### 3. Privacy Protection
```python
class PrivacyProtectionInpainting:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.license_plate_detector = LicensePlateDetector()
        self.inpainting_model = VideoInpaintingPipeline()
        
    def anonymize_video(self, video_path, targets=['faces', 'license_plates']):
        """Remove privacy-sensitive information from video"""
        frames = self.load_video(video_path)
        
        # Detect sensitive regions
        all_masks = np.zeros((len(frames), 1, frames[0].shape[0], frames[0].shape[1]))
        
        if 'faces' in targets:
            face_masks = self.detect_faces(frames)
            all_masks = np.maximum(all_masks, face_masks)
            
        if 'license_plates' in targets:
            plate_masks = self.detect_license_plates(frames)
            all_masks = np.maximum(all_masks, plate_masks)
            
        # Apply consistent anonymization
        anonymized_frames = self.apply_anonymization(frames, all_masks)
        
        return anonymized_frames
    
    def detect_faces(self, frames):
        """Detect and track faces across frames"""
        face_masks = []
        face_tracks = {}
        
        for i, frame in enumerate(frames):
            # Detect faces
            faces = self.face_detector.detect(frame)
            
            # Create mask
            mask = np.zeros((frame.shape[0], frame.shape[1]))
            
            for face in faces:
                # Track face across frames for consistency
                face_id = self.match_face_to_track(face, face_tracks, i)
                
                # Add to mask with consistent ID
                x, y, w, h = face['bbox']
                mask[y:y+h, x:x+w] = 1
                
            face_masks.append(mask)
            
        return np.array(face_masks).reshape(-1, 1, *mask.shape)
    
    def apply_anonymization(self, frames, masks):
        """Apply temporally consistent anonymization"""
        # Use inpainting for natural-looking anonymization
        inpainted = self.inpainting_model.inpaint_video(frames, masks)
        
        # Ensure no residual information
        verified = self.verify_anonymization(inpainted, masks)
        
        return verified
```

## Future Directions

### 1. Neural Rendering Integration
```python
class NeuralRenderingInpainting:
    def __init__(self):
        self.nerf_model = NeuralRadianceField()
        self.inpainting_model = VideoInpaintingPipeline()
        
    def inpaint_with_3d_understanding(self, video, masks, camera_params):
        """Inpaint using 3D scene understanding"""
        # Reconstruct 3D scene from video
        scene_representation = self.nerf_model.reconstruct_scene(
            video, camera_params
        )
        
        # Identify missing 3D regions
        missing_3d_regions = self.project_masks_to_3d(
            masks, camera_params, scene_representation
        )
        
        # Complete 3D geometry
        completed_scene = self.complete_3d_geometry(
            scene_representation, missing_3d_regions
        )
        
        # Render completed views
        rendered_frames = []
        for i, cam_param in enumerate(camera_params):
            rendered = self.nerf_model.render_view(completed_scene, cam_param)
            rendered_frames.append(rendered)
            
        # Blend with original frames
        final_frames = self.blend_rendered_with_original(
            video, rendered_frames, masks
        )
        
        return final_frames
```

### 2. Interactive Video Inpainting
```python
class InteractiveVideoInpainting:
    def __init__(self):
        self.inpainting_model = VideoInpaintingPipeline()
        self.user_interface = UserInterface()
        
    def interactive_inpaint(self, video_path):
        """Interactive video inpainting with user guidance"""
        frames = self.load_video(video_path)
        
        # User draws initial mask
        initial_mask = self.user_interface.draw_mask(frames[0])
        
        # Propagate mask with user refinement
        masks = self.propagate_mask_interactive(frames, initial_mask)
        
        # Preview inpainting
        preview = self.quick_preview(frames[:30], masks[:30])
        
        # User provides guidance
        guidance = self.user_interface.get_inpainting_guidance(preview)
        
        # Apply guided inpainting
        if guidance['type'] == 'reference':
            # User provides reference frames
            result = self.inpainting_model.inpaint_with_reference(
                frames, masks, guidance['reference_frames']
            )
        elif guidance['type'] == 'style':
            # User specifies style
            result = self.style_guided_inpainting(
                frames, masks, guidance['style']
            )
        else:
            # Automatic inpainting
            result = self.inpainting_model.inpaint_video(frames, masks)
            
        return result
```

### 3. Real-time Video Inpainting
```python
class RealtimeVideoInpainting:
    def __init__(self):
        self.fast_model = self.load_optimized_model()
        self.frame_buffer = deque(maxlen=5)
        
    def stream_inpaint(self, video_stream, mask_stream):
        """Real-time video inpainting for streaming"""
        for frame, mask in zip(video_stream, mask_stream):
            # Add to buffer
            self.frame_buffer.append((frame, mask))
            
            if len(self.frame_buffer) >= 3:
                # Inpaint center frame using context
                center_idx = len(self.frame_buffer) // 2
                
                # Fast inpainting
                inpainted = self.fast_inpaint(
                    self.frame_buffer[center_idx][0],
                    self.frame_buffer[center_idx][1],
                    context_frames=[f[0] for f in self.frame_buffer if f != self.frame_buffer[center_idx]]
                )
                
                yield inpainted
            else:
                # Not enough context yet
                yield frame
                
    def fast_inpaint(self, frame, mask, context_frames):
        """Optimized inpainting for real-time performance"""
        # Reduce resolution if needed
        scale = 0.5 if frame.shape[0] > 720 else 1.0
        
        if scale < 1:
            small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
            small_mask = cv2.resize(mask, None, fx=scale, fy=scale)
            small_context = [cv2.resize(f, None, fx=scale, fy=scale) 
                           for f in context_frames]
        else:
            small_frame = frame
            small_mask = mask
            small_context = context_frames
            
        # Fast inference
        with torch.no_grad():
            inpainted = self.fast_model(
                torch.from_numpy(small_frame).unsqueeze(0),
                torch.from_numpy(small_mask).unsqueeze(0),
                torch.stack([torch.from_numpy(f) for f in small_context])
            )
            
        # Upscale if needed
        if scale < 1:
            inpainted = cv2.resize(
                inpainted.squeeze().numpy(),
                (frame.shape[1], frame.shape[0])
            )
        else:
            inpainted = inpainted.squeeze().numpy()
            
        return inpainted
```

## Conclusion

Video inpainting represents a complex challenge in computer vision, requiring sophisticated techniques to handle temporal consistency, motion coherence, and visual quality. The field has evolved from simple propagation methods to advanced deep learning approaches that leverage optical flow, 3D understanding, and transformer architectures.

Key takeaways:
- Temporal consistency is crucial for convincing video inpainting
- Flow-guided methods provide strong motion coherence
- Deep learning approaches enable complex pattern understanding
- Multi-scale and attention mechanisms improve quality
- Future directions include 3D understanding and real-time processing

As computational resources improve and new architectures emerge, video inpainting will continue to advance, enabling applications from content creation to video restoration and privacy protection.

*Originally from umitkacar/awesome-video-inpainting repository*