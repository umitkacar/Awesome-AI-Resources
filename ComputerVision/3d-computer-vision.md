# Awesome 3D Computer Vision: State-of-the-Art Methods and Implementations

**Last Updated:** 2025-06-19

## Table of Contents
- [Overview](#overview)
- [3D Reconstruction](#3d-reconstruction)
- [Depth Estimation](#depth-estimation)
- [Point Cloud Processing](#point-cloud-processing)
- [3D Object Detection](#3d-object-detection)
- [3D Scene Understanding](#3d-scene-understanding)
- [Neural Radiance Fields (NeRF)](#neural-radiance-fields-nerf)
- [3D Generation and Synthesis](#3d-generation-and-synthesis)
- [SLAM and Visual Odometry](#slam-and-visual-odometry)
- [Applications](#applications)
- [Future Directions](#future-directions)

## Overview

3D Computer Vision encompasses techniques for understanding and reconstructing the three-dimensional world from visual data. This field bridges the gap between 2D image analysis and full 3D scene understanding, enabling applications from autonomous driving to AR/VR and robotics.

### Key Challenges
- **Depth Ambiguity**: Recovering 3D from 2D projections
- **Occlusions**: Handling hidden surfaces and incomplete data
- **Scale Ambiguity**: Determining absolute scale from images
- **Computational Complexity**: Processing high-dimensional 3D data
- **Sensor Limitations**: Working with noisy and sparse measurements

## 3D Reconstruction

### 1. Multi-View Stereo (MVS)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DeepMVS(nn.Module):
    def __init__(self, num_depth_planes=128):
        super(DeepMVS, self).__init__()
        
        # Feature extraction network
        self.feature_extractor = self.build_feature_extractor()
        
        # Cost volume construction
        self.cost_regularization = CostVolumeRegularization()
        
        # Depth regression
        self.depth_regression = DepthRegression(num_depth_planes)
        
    def build_feature_extractor(self):
        """Build multi-scale feature extractor"""
        return nn.Sequential(
            # Initial convolutions
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Downsampling blocks
            self.conv_block(32, 64, stride=2),
            self.conv_block(64, 128, stride=2),
            self.conv_block(128, 256, stride=2),
            
            # Feature aggregation
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
    
    def conv_block(self, in_channels, out_channels, stride=1):
        """Convolutional block with residual connection"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, images, intrinsics, extrinsics, depth_values):
        """
        Multi-view stereo depth estimation
        Args:
            images: (B, N, 3, H, W) N views
            intrinsics: (B, N, 3, 3) camera intrinsics
            extrinsics: (B, N, 4, 4) camera extrinsics
            depth_values: (B, D) depth plane values
        """
        B, N, C, H, W = images.shape
        D = depth_values.shape[1]
        
        # Extract features from all views
        features = []
        for i in range(N):
            feat = self.feature_extractor(images[:, i])
            features.append(feat)
        features = torch.stack(features, dim=1)
        
        # Construct cost volume
        ref_feature = features[:, 0]  # Reference view
        cost_volume = self.build_cost_volume(
            features, intrinsics, extrinsics, depth_values
        )
        
        # Regularize cost volume
        regularized_cost = self.cost_regularization(cost_volume)
        
        # Predict depth
        depth_prob = self.depth_regression(regularized_cost)
        depth_map = self.compute_depth(depth_prob, depth_values)
        
        return depth_map, depth_prob
    
    def build_cost_volume(self, features, intrinsics, extrinsics, depth_values):
        """Build cost volume by warping features"""
        B, N, C, H, W = features.shape
        D = depth_values.shape[1]
        
        ref_feature = features[:, 0]
        ref_intrinsic = intrinsics[:, 0]
        ref_extrinsic = extrinsics[:, 0]
        
        cost_volume = []
        
        for d in range(D):
            depth = depth_values[:, d:d+1].view(B, 1, 1, 1)
            warped_features = []
            
            for i in range(1, N):
                # Compute homography for current depth
                src_intrinsic = intrinsics[:, i]
                src_extrinsic = extrinsics[:, i]
                
                homography = self.compute_homography(
                    ref_intrinsic, ref_extrinsic,
                    src_intrinsic, src_extrinsic,
                    depth
                )
                
                # Warp source feature to reference view
                warped = self.warp_feature(features[:, i], homography)
                warped_features.append(warped)
            
            # Compute matching cost
            warped_features = torch.stack(warped_features, dim=1)
            cost = self.compute_matching_cost(ref_feature, warped_features)
            cost_volume.append(cost)
            
        return torch.stack(cost_volume, dim=1)
    
    def compute_homography(self, K1, E1, K2, E2, depth):
        """Compute homography matrix for plane at given depth"""
        # Relative pose
        R = E2[:, :3, :3] @ E1[:, :3, :3].transpose(-1, -2)
        t = E2[:, :3, 3] - R @ E1[:, :3, 3]
        
        # Plane normal (assuming fronto-parallel)
        n = torch.tensor([0, 0, 1], device=R.device).float()
        
        # Homography
        H = K2 @ (R + t.unsqueeze(-1) @ n.unsqueeze(0) / depth) @ K1.inverse()
        
        return H
    
    def warp_feature(self, feature, homography):
        """Warp feature map using homography"""
        B, C, H, W = feature.shape
        
        # Create grid
        grid = self.create_grid(B, H, W, homography.device)
        
        # Apply homography
        warped_grid = self.apply_homography(grid, homography)
        
        # Sample features
        warped_feature = F.grid_sample(
            feature, warped_grid, mode='bilinear', padding_mode='zeros'
        )
        
        return warped_feature
    
    def compute_matching_cost(self, ref_feature, src_features):
        """Compute feature matching cost"""
        # Group-wise correlation
        B, N, C, H, W = src_features.shape
        
        # Compute correlation
        ref_feature = ref_feature.unsqueeze(1)
        correlation = (ref_feature * src_features).sum(dim=2) / (C ** 0.5)
        
        # Aggregate across views
        cost = correlation.mean(dim=1)
        
        return cost

class CostVolumeRegularization(nn.Module):
    def __init__(self):
        super(CostVolumeRegularization, self).__init__()
        
        # 3D CNN for cost volume filtering
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling
        self.deconv3d_1 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        
        self.deconv3d_2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        self.conv3d_out = nn.Conv3d(32, 1, kernel_size=3, padding=1)
        
    def forward(self, cost_volume):
        """Regularize cost volume using 3D convolutions"""
        # Add channel dimension
        x = cost_volume.unsqueeze(1)
        
        # Encoder
        conv1 = self.conv3d_1(x)
        conv2 = self.conv3d_2(conv1)
        conv3 = self.conv3d_3(conv2)
        
        # Decoder with skip connections
        deconv1 = self.deconv3d_1(conv3) + conv2
        deconv2 = self.deconv3d_2(deconv1) + conv1
        
        # Output
        out = self.conv3d_out(deconv2).squeeze(1)
        
        return out
```

### 2. Structure from Motion (SfM)
```python
class StructureFromMotion:
    def __init__(self):
        self.feature_detector = SIFTDetector()
        self.feature_matcher = FeatureMatcher()
        self.pose_estimator = PoseEstimator()
        self.triangulator = Triangulator()
        self.bundle_adjuster = BundleAdjuster()
        
    def reconstruct(self, images):
        """
        Complete SfM pipeline
        Args:
            images: List of images
        Returns:
            cameras: Estimated camera poses
            points3d: Reconstructed 3D points
        """
        # Step 1: Feature extraction and matching
        features = self.extract_features(images)
        matches = self.match_features(features)
        
        # Step 2: Initialize with two views
        initial_pair = self.select_initial_pair(matches)
        cameras, points3d = self.initialize_reconstruction(
            images[initial_pair[0]], 
            images[initial_pair[1]],
            features[initial_pair[0]], 
            features[initial_pair[1]],
            matches[initial_pair]
        )
        
        # Step 3: Incremental reconstruction
        registered = set(initial_pair)
        
        while len(registered) < len(images):
            # Find next best view
            next_view = self.find_next_view(
                registered, matches, points3d, features
            )
            
            if next_view is None:
                break
                
            # Register new view
            camera_pose = self.register_view(
                images[next_view], 
                features[next_view],
                points3d, 
                matches
            )
            
            cameras[next_view] = camera_pose
            
            # Triangulate new points
            new_points = self.triangulate_new_points(
                next_view, cameras, features, matches
            )
            
            points3d.update(new_points)
            registered.add(next_view)
            
            # Bundle adjustment
            if len(registered) % 5 == 0:
                cameras, points3d = self.bundle_adjuster.optimize(
                    cameras, points3d, features, matches
                )
                
        # Final bundle adjustment
        cameras, points3d = self.bundle_adjuster.optimize(
            cameras, points3d, features, matches
        )
        
        return cameras, points3d
    
    def initialize_reconstruction(self, img1, img2, feat1, feat2, matches):
        """Initialize reconstruction from two views"""
        # Compute essential matrix
        E, mask = cv2.findEssentialMat(
            feat1['keypoints'][matches[:, 0]], 
            feat2['keypoints'][matches[:, 1]],
            self.K  # Intrinsic matrix
        )
        
        # Recover pose
        _, R, t, mask = cv2.recoverPose(
            E, 
            feat1['keypoints'][matches[:, 0]], 
            feat2['keypoints'][matches[:, 1]],
            self.K
        )
        
        # Setup cameras
        cameras = {
            0: Camera(R=np.eye(3), t=np.zeros(3), K=self.K),
            1: Camera(R=R, t=t, K=self.K)
        }
        
        # Triangulate initial points
        points3d = self.triangulator.triangulate_points(
            cameras[0], cameras[1], 
            feat1['keypoints'][matches[:, 0]], 
            feat2['keypoints'][matches[:, 1]]
        )
        
        return cameras, points3d
    
    def register_view(self, image, features, points3d, matches):
        """Register new view using PnP"""
        # Find 2D-3D correspondences
        points_2d = []
        points_3d = []
        
        for point_id, point_3d in points3d.items():
            if point_id in features['point_ids']:
                idx = features['point_ids'].index(point_id)
                points_2d.append(features['keypoints'][idx])
                points_3d.append(point_3d)
                
        points_2d = np.array(points_2d)
        points_3d = np.array(points_3d)
        
        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            points_3d, points_2d, self.K, None
        )
        
        if success:
            R, _ = cv2.Rodrigues(rvec)
            return Camera(R=R, t=tvec.squeeze(), K=self.K)
        
        return None

class BundleAdjuster:
    def __init__(self):
        self.max_iterations = 100
        self.tolerance = 1e-6
        
    def optimize(self, cameras, points3d, observations):
        """
        Bundle adjustment using Levenberg-Marquardt
        """
        # Pack parameters
        camera_params, point_params = self.pack_parameters(cameras, points3d)
        x0 = np.concatenate([camera_params, point_params])
        
        # Setup sparse Jacobian structure
        jacobian_structure = self.compute_jacobian_structure(
            cameras, points3d, observations
        )
        
        # Optimize
        result = scipy.optimize.least_squares(
            self.residual_function,
            x0,
            jac_sparsity=jacobian_structure,
            method='lm',
            args=(observations, len(cameras)),
            max_nfev=self.max_iterations,
            ftol=self.tolerance
        )
        
        # Unpack results
        optimized_cameras, optimized_points = self.unpack_parameters(
            result.x, len(cameras)
        )
        
        return optimized_cameras, optimized_points
    
    def residual_function(self, params, observations, num_cameras):
        """Compute reprojection residuals"""
        cameras, points = self.unpack_parameters(params, num_cameras)
        residuals = []
        
        for obs in observations:
            camera = cameras[obs['camera_id']]
            point = points[obs['point_id']]
            
            # Project 3D point
            projected = camera.project(point)
            
            # Compute residual
            residual = projected - obs['pixel']
            residuals.extend(residual)
            
        return np.array(residuals)
```

### 3. Volumetric Reconstruction
```python
class VolumetricReconstruction:
    def __init__(self, voxel_size=0.01, truncation_distance=0.04):
        self.voxel_size = voxel_size
        self.truncation_distance = truncation_distance
        self.tsdf_volume = None
        
    def integrate_depth_map(self, depth_map, color_map, camera_pose, intrinsics):
        """
        Integrate depth map into TSDF volume
        """
        if self.tsdf_volume is None:
            self.initialize_volume(depth_map.shape, intrinsics)
            
        # Get voxel coordinates
        voxel_coords = self.get_voxel_coordinates()
        
        # Transform to camera coordinates
        world_coords = self.voxel_to_world(voxel_coords)
        camera_coords = self.world_to_camera(world_coords, camera_pose)
        
        # Project to image
        pixel_coords = self.project_to_image(camera_coords, intrinsics)
        
        # Sample depth values
        sampled_depths = self.sample_depth(depth_map, pixel_coords)
        
        # Compute TSDF values
        tsdf_values = self.compute_tsdf(camera_coords, sampled_depths)
        
        # Update volume
        self.update_tsdf_volume(tsdf_values, color_map, pixel_coords)
        
    def compute_tsdf(self, camera_coords, sampled_depths):
        """Compute truncated signed distance function"""
        # Distance along ray
        ray_distances = np.linalg.norm(camera_coords, axis=-1)
        
        # Signed distance
        signed_distances = sampled_depths - ray_distances
        
        # Truncate
        tsdf = np.clip(
            signed_distances / self.truncation_distance, 
            -1.0, 1.0
        )
        
        # Mask invalid depths
        valid_mask = sampled_depths > 0
        tsdf[~valid_mask] = 1.0
        
        return tsdf
    
    def extract_mesh(self):
        """Extract mesh using marching cubes"""
        # Get TSDF values
        tsdf_values = self.tsdf_volume['tsdf']
        
        # Apply marching cubes
        vertices, faces, normals, _ = marching_cubes(
            tsdf_values, level=0.0
        )
        
        # Transform vertices to world coordinates
        vertices = vertices * self.voxel_size + self.volume_origin
        
        # Get vertex colors
        colors = self.interpolate_colors(vertices)
        
        return {
            'vertices': vertices,
            'faces': faces,
            'normals': normals,
            'colors': colors
        }
```

## Depth Estimation

### 1. Monocular Depth Estimation
```python
class MonocularDepthEstimation(nn.Module):
    def __init__(self, encoder='efficientnet-b4', max_depth=10.0):
        super(MonocularDepthEstimation, self).__init__()
        
        self.max_depth = max_depth
        
        # Encoder
        self.encoder = self.build_encoder(encoder)
        
        # Decoder with skip connections
        self.decoder = self.build_decoder()
        
        # Multi-scale depth prediction
        self.depth_heads = nn.ModuleList([
            nn.Conv2d(ch, 1, 3, padding=1) 
            for ch in [256, 128, 64, 32]
        ])
        
    def build_encoder(self, encoder_name):
        """Build EfficientNet encoder"""
        encoder = timm.create_model(encoder_name, pretrained=True, features_only=True)
        return encoder
    
    def build_decoder(self):
        """Build decoder with adaptive bins"""
        return nn.ModuleList([
            # Decoder blocks with attention
            DecoderBlock(512, 256, use_attention=True),
            DecoderBlock(256, 128, use_attention=True),
            DecoderBlock(128, 64, use_attention=False),
            DecoderBlock(64, 32, use_attention=False)
        ])
    
    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        
        # Decoder with skip connections
        decoded_features = []
        x = features[-1]
        
        for i, decoder_block in enumerate(self.decoder):
            x = decoder_block(x)
            
            # Skip connection
            if i < len(features) - 1:
                skip = features[-(i+2)]
                x = x + F.interpolate(skip, size=x.shape[-2:], mode='bilinear')
                
            decoded_features.append(x)
            
        # Multi-scale depth prediction
        depth_maps = []
        
        for i, (feat, head) in enumerate(zip(decoded_features, self.depth_heads)):
            depth = head(feat)
            depth = torch.sigmoid(depth) * self.max_depth
            depth_maps.append(depth)
            
        # Upsample all to original resolution
        H, W = x.shape[-2:]
        depth_maps = [
            F.interpolate(d, size=(H*8, W*8), mode='bilinear', align_corners=False)
            for d in depth_maps
        ]
        
        return depth_maps

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=False):
        super(DecoderBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, 4, stride=2, padding=1
        )
        
        if use_attention:
            self.attention = SpatialAttention(out_channels)
        else:
            self.attention = None
            
    def forward(self, x):
        # Upsample
        x = self.upsample(x)
        
        # Convolutions
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Attention
        if self.attention is not None:
            out = self.attention(out)
            
        return F.relu(out + x)
```

### 2. Stereo Depth Estimation
```python
class StereoDepthEstimation(nn.Module):
    def __init__(self, max_disparity=192):
        super(StereoDepthEstimation, self).__init__()
        
        self.max_disparity = max_disparity
        
        # Feature extraction
        self.feature_extractor = self.build_feature_network()
        
        # Cost volume construction
        self.cost_aggregation = CostAggregation()
        
        # Disparity regression
        self.disparity_regression = DisparityRegression(max_disparity)
        
        # Refinement
        self.refinement = DisparityRefinement()
        
    def forward(self, left_image, right_image):
        # Extract features
        left_features = self.feature_extractor(left_image)
        right_features = self.feature_extractor(right_image)
        
        # Build cost volume
        cost_volume = self.build_cost_volume(left_features, right_features)
        
        # Aggregate cost
        aggregated_cost = self.cost_aggregation(cost_volume)
        
        # Predict disparity
        disparity = self.disparity_regression(aggregated_cost)
        
        # Refine
        refined_disparity = self.refinement(disparity, left_image)
        
        return refined_disparity
    
    def build_cost_volume(self, left_feat, right_feat):
        """Build cost volume using correlation"""
        B, C, H, W = left_feat.shape
        
        cost_volume = torch.zeros(B, self.max_disparity, H, W).to(left_feat.device)
        
        for d in range(self.max_disparity):
            if d == 0:
                cost = (left_feat * right_feat).mean(dim=1)
            else:
                # Shift right features
                shifted_right = F.pad(right_feat[:, :, :, d:], (0, d, 0, 0))
                cost = (left_feat * shifted_right).mean(dim=1)
                
            cost_volume[:, d] = cost
            
        return cost_volume

class CostAggregation(nn.Module):
    def __init__(self):
        super(CostAggregation, self).__init__()
        
        # 3D hourglass for cost aggregation
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        
        # Decoder
        self.deconv3d_1 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        
        self.deconv3d_2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        
        self.out_conv = nn.Conv3d(32, 1, 3, padding=1)
        
    def forward(self, cost_volume):
        # Add channel dimension
        x = cost_volume.unsqueeze(1)
        
        # Encoder
        conv1 = self.conv3d_1(x)
        conv2 = self.conv3d_2(conv1)
        conv3 = self.conv3d_3(conv2)
        
        # Decoder
        deconv1 = self.deconv3d_1(conv3)
        deconv2 = self.deconv3d_2(deconv1 + conv2)
        
        # Output
        out = self.out_conv(deconv2 + conv1).squeeze(1)
        
        return out
```

## Point Cloud Processing

### 1. PointNet++ Architecture
```python
class PointNetPlusPlus(nn.Module):
    def __init__(self, num_classes=40):
        super(PointNetPlusPlus, self).__init__()
        
        # Set abstraction layers
        self.sa1 = SetAbstraction(
            npoint=512, radius=0.2, nsample=32, 
            in_channel=3, mlp=[64, 64, 128]
        )
        self.sa2 = SetAbstraction(
            npoint=128, radius=0.4, nsample=64, 
            in_channel=128 + 3, mlp=[128, 128, 256]
        )
        self.sa3 = SetAbstraction(
            npoint=None, radius=None, nsample=None, 
            in_channel=256 + 3, mlp=[256, 512, 1024]
        )
        
        # Classification head
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, xyz):
        """
        Args:
            xyz: (B, N, 3) point cloud
        """
        B, N, _ = xyz.shape
        
        # Set abstraction
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Global feature
        x = l3_points.view(B, 1024)
        
        # Classification
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        
        return x

class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super(SetAbstraction, self).__init__()
        
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
            
    def forward(self, xyz, points):
        """
        Args:
            xyz: (B, N, 3) coordinates
            points: (B, N, C) features
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
            
        if self.npoint is not None:
            # Farthest point sampling
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz.permute(0, 2, 1), fps_idx).permute(0, 2, 1)
            
            # Ball query
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz.permute(0, 2, 1), idx)
            grouped_xyz -= new_xyz.permute(0, 2, 1).unsqueeze(2)
            
            if points is not None:
                grouped_points = index_points(points.permute(0, 2, 1), idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz
                
        else:
            # Global pooling
            grouped_points = xyz.permute(0, 2, 1).unsqueeze(2)
            if points is not None:
                grouped_points = torch.cat([points.permute(0, 2, 1).unsqueeze(2), grouped_points], dim=-1)
            new_xyz = None
            
        # PointNet layer
        grouped_points = grouped_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_points = F.relu(bn(conv(grouped_points)))
            
        # Max pooling
        new_points = torch.max(grouped_points, 2)[0]
        
        return new_xyz, new_points

def farthest_point_sample(xyz, npoint):
    """
    Farthest point sampling
    Args:
        xyz: (B, N, 3) point coordinates
        npoint: number of samples
    Returns:
        centroids: (B, npoint) sampled point indices
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
        
    return centroids
```

### 2. Point Cloud Segmentation
```python
class PointCloudSegmentation(nn.Module):
    def __init__(self, num_classes=13):
        super(PointCloudSegmentation, self).__init__()
        
        # Encoder
        self.sa1 = SetAbstraction(1024, 0.1, 32, 9, [32, 32, 64])
        self.sa2 = SetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128])
        self.sa3 = SetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256])
        self.sa4 = SetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512])
        
        # Decoder
        self.fp4 = FeaturePropagation(768, [256, 256])
        self.fp3 = FeaturePropagation(384, [256, 256])
        self.fp2 = FeaturePropagation(320, [256, 128])
        self.fp1 = FeaturePropagation(128, [128, 128, 128])
        
        # Segmentation head
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
        
    def forward(self, xyz, features):
        # Encoder
        l0_xyz = xyz
        l0_points = features
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        
        # Decoder
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        
        # Segmentation
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        
        return x

class FeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(FeaturePropagation, self).__init__()
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
            
    def forward(self, xyz1, xyz2, points1, points2):
        """
        Args:
            xyz1: (B, N, 3) target coordinates
            xyz2: (B, M, 3) source coordinates
            points1: (B, C1, N) target features
            points2: (B, C2, M) source features
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            
            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.view(B, N, 3, 1), 
                dim=2
            )
            
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
            
        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
            
        return new_points
```

## 3D Object Detection

### 1. VoxelNet
```python
class VoxelNet(nn.Module):
    def __init__(self, num_classes=3, voxel_size=[0.2, 0.2, 0.4], 
                 point_cloud_range=[0, -40, -3, 70.4, 40, 1]):
        super(VoxelNet, self).__init__()
        
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        
        # Voxel feature encoding
        self.vfe = VoxelFeatureExtractor()
        
        # Middle convolution layers
        self.middle_conv = MiddleConvolutions()
        
        # Region proposal network
        self.rpn = RegionProposalNetwork(num_classes)
        
    def forward(self, voxels, num_points, coordinates):
        """
        Args:
            voxels: (num_voxels, max_points, 7) voxel features
            num_points: (num_voxels,) number of points in each voxel
            coordinates: (num_voxels, 3) voxel coordinates
        """
        # Voxel feature extraction
        voxel_features = self.vfe(voxels, num_points)
        
        # Sparse to dense
        batch_size = coordinates[:, 0].max().item() + 1
        sparse_features = self.sparse_to_dense(
            voxel_features, coordinates, batch_size
        )
        
        # Middle layers
        middle_features = self.middle_conv(sparse_features)
        
        # RPN
        predictions = self.rpn(middle_features)
        
        return predictions
    
    def sparse_to_dense(self, voxel_features, coords, batch_size):
        """Convert sparse voxel features to dense feature map"""
        # Calculate output shape
        output_shape = [batch_size]
        output_shape.extend(self.grid_size[::-1])
        output_shape.append(voxel_features.shape[-1])
        
        # Create dense tensor
        dense_features = torch.zeros(output_shape).to(voxel_features.device)
        
        # Fill with voxel features
        indices = coords.long()
        dense_features[indices[:, 0], indices[:, 3], indices[:, 2], indices[:, 1]] = voxel_features
        
        # Permute to NCHW
        dense_features = dense_features.permute(0, 4, 1, 2, 3)
        
        return dense_features

class VoxelFeatureExtractor(nn.Module):
    def __init__(self, num_filters=[32, 128], voxel_size=35, use_norm=True):
        super(VoxelFeatureExtractor, self).__init__()
        
        self.use_norm = use_norm
        self.voxel_size = voxel_size
        
        # VFE layers
        self.vfe1 = VFELayer(7, num_filters[0], use_norm)
        self.vfe2 = VFELayer(num_filters[0], num_filters[1], use_norm)
        
    def forward(self, features, num_points):
        # Mask for valid points
        mask = self.get_paddings_indicator(num_points, features.shape[1])
        
        # VFE Layer 1
        x = self.vfe1(features, mask)
        
        # VFE Layer 2
        x = self.vfe2(x, mask)
        
        # Max pooling
        voxel_features = torch.max(x, dim=1)[0]
        
        return voxel_features
    
    def get_paddings_indicator(self, num_points, max_points):
        """Create mask for valid points in voxels"""
        batch_size = num_points.shape[0]
        mask = torch.zeros(batch_size, max_points).to(num_points.device)
        
        for i in range(batch_size):
            mask[i, :num_points[i]] = 1
            
        return mask

class VFELayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True):
        super(VFELayer, self).__init__()
        
        self.use_norm = use_norm
        self.fc = nn.Linear(in_channels, out_channels)
        if use_norm:
            self.bn = nn.BatchNorm1d(out_channels)
            
    def forward(self, x, mask):
        # Point-wise feature
        pwf = self.fc(x)
        
        if self.use_norm:
            # Reshape for batch norm
            pwf = pwf.transpose(1, 2).contiguous()
            pwf = self.bn(pwf)
            pwf = pwf.transpose(1, 2).contiguous()
            
        # Apply mask
        pwf = F.relu(pwf)
        pwf = pwf * mask.unsqueeze(-1)
        
        # Local aggregation
        laf = torch.max(pwf, dim=1, keepdim=True)[0]
        laf = laf.repeat(1, pwf.shape[1], 1)
        
        # Output feature
        output = torch.cat([pwf, laf], dim=-1)
        
        return output
```

### 2. PointPillars
```python
class PointPillars(nn.Module):
    def __init__(self, num_classes=3, voxel_size=[0.16, 0.16, 4], 
                 point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]):
        super(PointPillars, self).__init__()
        
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        
        # Pillar feature network
        self.pfn = PillarFeatureNet(
            num_filters=[64],
            use_norm=True,
            with_distance=False,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range
        )
        
        # Backbone
        self.backbone = PointPillarsBackbone()
        
        # Detection head
        self.head = SingleHead(num_classes)
        
    def forward(self, pillars, num_points_per_pillar, coors):
        # Pillar features
        pillar_features = self.pfn(pillars, num_points_per_pillar, coors)
        
        # Scatter to BEV
        spatial_features = self.scatter_to_bev(pillar_features, coors)
        
        # Backbone
        backbone_features = self.backbone(spatial_features)
        
        # Detection
        predictions = self.head(backbone_features)
        
        return predictions
    
    def scatter_to_bev(self, pillar_features, coords):
        """Scatter pillar features to BEV pseudo-image"""
        batch_size = coords[:, 0].max().item() + 1
        
        # Create pseudo-image
        nx = (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.voxel_size[0]
        ny = (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.voxel_size[1]
        
        canvas = torch.zeros(
            batch_size, pillar_features.shape[-1], int(ny), int(nx),
            dtype=pillar_features.dtype,
            device=pillar_features.device
        )
        
        # Scatter
        indices = coords.long()
        canvas[indices[:, 0], :, indices[:, 2], indices[:, 3]] = pillar_features
        
        return canvas

class PillarFeatureNet(nn.Module):
    def __init__(self, num_filters, use_norm, with_distance, 
                 voxel_size, point_cloud_range):
        super(PillarFeatureNet, self).__init__()
        
        self.use_norm = use_norm
        self.with_distance = with_distance
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        
        # Calculate input features
        num_input_features = 4  # x, y, z, r
        if with_distance:
            num_input_features += 1
            
        # Add pillar center features
        num_input_features += 5  # xc, yc, zc, xp, yp
        
        # PFN layers
        self.pfn_layers = nn.ModuleList()
        for i, num_filter in enumerate(num_filters):
            in_channels = num_input_features if i == 0 else num_filters[i-1]
            self.pfn_layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, num_filter),
                    nn.BatchNorm1d(num_filter) if use_norm else nn.Identity(),
                    nn.ReLU()
                )
            )
            
    def forward(self, features, num_points, coors):
        # Find pillar center
        points_mean = features.sum(dim=1, keepdim=True) / num_points.unsqueeze(-1).unsqueeze(-1)
        
        # Calculate offset from pillar center
        f_cluster = features - points_mean
        
        # Find distance to pillar center
        if self.with_distance:
            f_distance = torch.norm(f_cluster[..., :2], dim=-1, keepdim=True)
            features = torch.cat([features, f_cluster, f_distance], dim=-1)
        else:
            features = torch.cat([features, f_cluster], dim=-1)
            
        # Add pillar center
        pillar_center = self.get_pillar_center(coors)
        pillar_center = pillar_center.unsqueeze(1).repeat(1, features.shape[1], 1)
        features = torch.cat([features, pillar_center], dim=-1)
        
        # PFN
        for pfn in self.pfn_layers:
            features = pfn(features)
            
        # Max pooling
        features = torch.max(features, dim=1)[0]
        
        return features
```

## 3D Scene Understanding

### Scene Graph Generation
```python
class SceneGraphGeneration3D(nn.Module):
    def __init__(self, num_obj_classes=160, num_rel_classes=26):
        super(SceneGraphGeneration3D, self).__init__()
        
        # Object detection
        self.object_detector = VoteNet(num_obj_classes)
        
        # Object feature extraction
        self.obj_feature_extractor = PointNet2MSG()
        
        # Relationship prediction
        self.rel_predictor = RelationshipPredictor(num_rel_classes)
        
        # Graph refinement
        self.graph_refiner = GraphRefinementNetwork()
        
    def forward(self, point_cloud):
        # Detect objects
        obj_proposals = self.object_detector(point_cloud)
        
        # Extract object features
        obj_features = []
        for prop in obj_proposals:
            obj_points = self.crop_object_points(point_cloud, prop['bbox'])
            obj_feat = self.obj_feature_extractor(obj_points)
            obj_features.append(obj_feat)
            
        obj_features = torch.stack(obj_features)
        
        # Predict relationships
        rel_predictions = self.rel_predictor(obj_features, obj_proposals)
        
        # Build scene graph
        scene_graph = self.build_scene_graph(obj_proposals, rel_predictions)
        
        # Refine graph
        refined_graph = self.graph_refiner(scene_graph)
        
        return refined_graph
    
    def build_scene_graph(self, objects, relationships):
        """Build scene graph from objects and relationships"""
        nodes = []
        edges = []
        
        # Create nodes
        for i, obj in enumerate(objects):
            node = {
                'id': i,
                'class': obj['class'],
                'bbox': obj['bbox'],
                'features': obj['features']
            }
            nodes.append(node)
            
        # Create edges
        for rel in relationships:
            if rel['score'] > 0.5:  # Threshold
                edge = {
                    'subject': rel['subject_id'],
                    'object': rel['object_id'],
                    'predicate': rel['predicate'],
                    'score': rel['score']
                }
                edges.append(edge)
                
        return {'nodes': nodes, 'edges': edges}

class RelationshipPredictor(nn.Module):
    def __init__(self, num_rel_classes):
        super(RelationshipPredictor, self).__init__()
        
        # Pairwise feature extraction
        self.pair_feature_extractor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Spatial feature extraction
        self.spatial_encoder = nn.Sequential(
            nn.Linear(9, 64),  # 3D bbox difference
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Relationship classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_rel_classes)
        )
        
    def forward(self, obj_features, obj_proposals):
        num_objs = len(obj_features)
        relationships = []
        
        for i in range(num_objs):
            for j in range(num_objs):
                if i == j:
                    continue
                    
                # Pairwise features
                pair_feat = torch.cat([obj_features[i], obj_features[j]])
                pair_feat = self.pair_feature_extractor(pair_feat)
                
                # Spatial features
                spatial_feat = self.compute_spatial_features(
                    obj_proposals[i]['bbox'], 
                    obj_proposals[j]['bbox']
                )
                spatial_feat = self.spatial_encoder(spatial_feat)
                
                # Combine features
                combined_feat = torch.cat([pair_feat, spatial_feat])
                
                # Predict relationship
                rel_scores = self.classifier(combined_feat)
                
                relationships.append({
                    'subject_id': i,
                    'object_id': j,
                    'scores': rel_scores
                })
                
        return relationships
```

## Neural Radiance Fields (NeRF)

### 1. Original NeRF Implementation
```python
class NeRF(nn.Module):
    def __init__(self, pos_dim=3, view_dim=3, feat_dim=256):
        super(NeRF, self).__init__()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(pos_dim, L=10)
        self.view_encoder = PositionalEncoding(view_dim, L=4)
        
        # MLP
        self.mlp_base = nn.Sequential(
            nn.Linear(self.pos_encoder.out_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
        )
        
        self.density_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 1)
        )
        
        self.feature_head = nn.Linear(feat_dim, feat_dim)
        
        self.rgb_head = nn.Sequential(
            nn.Linear(feat_dim + self.view_encoder.out_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, 3),
            nn.Sigmoid()
        )
        
    def forward(self, positions, view_dirs):
        """
        Args:
            positions: (N, 3) 3D positions
            view_dirs: (N, 3) viewing directions
        """
        # Encode positions
        pos_enc = self.pos_encoder(positions)
        
        # Base MLP
        features = self.mlp_base(pos_enc)
        
        # Density prediction
        density = self.density_head(features)
        
        # RGB prediction
        features = self.feature_head(features)
        view_enc = self.view_encoder(view_dirs)
        rgb_input = torch.cat([features, view_enc], dim=-1)
        rgb = self.rgb_head(rgb_input)
        
        return rgb, density

class PositionalEncoding:
    def __init__(self, input_dim, L):
        self.L = L
        self.input_dim = input_dim
        self.out_dim = input_dim * (2 * L + 1)
        
    def __call__(self, x):
        """Apply positional encoding"""
        encodings = [x]
        
        for l in range(self.L):
            encodings.append(torch.sin(2**l * np.pi * x))
            encodings.append(torch.cos(2**l * np.pi * x))
            
        return torch.cat(encodings, dim=-1)

class NeRFRenderer:
    def __init__(self, near=2.0, far=6.0, n_samples=64, n_importance=128):
        self.near = near
        self.far = far
        self.n_samples = n_samples
        self.n_importance = n_importance
        
    def render_rays(self, rays_o, rays_d, nerf_coarse, nerf_fine=None):
        """
        Volume rendering
        Args:
            rays_o: (N, 3) ray origins
            rays_d: (N, 3) ray directions
            nerf_coarse: coarse NeRF model
            nerf_fine: fine NeRF model (optional)
        """
        # Sample points along rays
        z_vals = self.sample_along_rays(rays_o, rays_d, self.n_samples)
        points = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)
        
        # Query NeRF
        view_dirs = rays_d.unsqueeze(1).expand_as(points)
        rgb_coarse, density_coarse = nerf_coarse(
            points.reshape(-1, 3), 
            view_dirs.reshape(-1, 3)
        )
        rgb_coarse = rgb_coarse.reshape(points.shape)
        density_coarse = density_coarse.reshape(points.shape[:-1])
        
        # Volume rendering
        rgb_map_coarse, weights = self.volume_rendering(
            rgb_coarse, density_coarse, z_vals, rays_d
        )
        
        # Hierarchical sampling
        if nerf_fine is not None:
            z_vals_fine = self.sample_pdf(z_vals, weights, self.n_importance)
            z_vals_combined, _ = torch.sort(torch.cat([z_vals, z_vals_fine], dim=-1))
            
            points_fine = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals_combined.unsqueeze(-1)
            view_dirs_fine = rays_d.unsqueeze(1).expand_as(points_fine)
            
            rgb_fine, density_fine = nerf_fine(
                points_fine.reshape(-1, 3),
                view_dirs_fine.reshape(-1, 3)
            )
            rgb_fine = rgb_fine.reshape(points_fine.shape)
            density_fine = density_fine.reshape(points_fine.shape[:-1])
            
            rgb_map_fine, _ = self.volume_rendering(
                rgb_fine, density_fine, z_vals_combined, rays_d
            )
            
            return rgb_map_coarse, rgb_map_fine
            
        return rgb_map_coarse, None
    
    def volume_rendering(self, rgb, density, z_vals, rays_d):
        """Classical volume rendering"""
        # Compute distances
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
        dists = dists * torch.norm(rays_d, dim=-1, keepdim=True)
        
        # Compute alpha
        alpha = 1.0 - torch.exp(-F.relu(density) * dists)
        
        # Compute weights
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
            dim=-1
        )[..., :-1]
        
        # Composite
        rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)
        
        return rgb_map, weights
```

### 2. Instant-NGP
```python
class InstantNGP(nn.Module):
    def __init__(self, encoding_config, network_config):
        super(InstantNGP, self).__init__()
        
        # Multi-resolution hash encoding
        self.encoding = MultiResolutionHashEncoding(
            n_levels=encoding_config['n_levels'],
            n_features_per_level=encoding_config['n_features_per_level'],
            log2_hashmap_size=encoding_config['log2_hashmap_size'],
            base_resolution=encoding_config['base_resolution'],
            finest_resolution=encoding_config['finest_resolution']
        )
        
        # Small MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.encoding.n_output_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        
        self.density_head = nn.Linear(16, 1)
        self.rgb_head = nn.Sequential(
            nn.Linear(16 + self.encoding.n_output_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Sigmoid()
        )
        
    def forward(self, positions, view_dirs=None):
        # Hash encoding
        encoded_pos = self.encoding(positions)
        
        # MLP
        features = self.mlp(encoded_pos)
        
        # Density
        density = self.density_head(features)
        
        # RGB
        if view_dirs is not None:
            encoded_dirs = self.encoding(view_dirs)
            rgb_input = torch.cat([features, encoded_dirs], dim=-1)
            rgb = self.rgb_head(rgb_input)
        else:
            rgb = None
            
        return rgb, density

class MultiResolutionHashEncoding(nn.Module):
    def __init__(self, n_levels=16, n_features_per_level=2, 
                 log2_hashmap_size=19, base_resolution=16, 
                 finest_resolution=512):
        super(MultiResolutionHashEncoding, self).__init__()
        
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        
        # Compute per-level resolutions
        self.resolutions = self.compute_resolutions()
        
        # Initialize hash tables
        self.hash_tables = nn.ModuleList([
            nn.Embedding(2**log2_hashmap_size, n_features_per_level)
            for _ in range(n_levels)
        ])
        
        # Initialize embeddings
        for table in self.hash_tables:
            nn.init.uniform_(table.weight, -1e-4, 1e-4)
            
        self.n_output_dims = n_levels * n_features_per_level
        
    def compute_resolutions(self):
        """Compute resolution for each level"""
        growth_factor = np.exp(
            (np.log(self.finest_resolution) - np.log(self.base_resolution)) / (self.n_levels - 1)
        )
        
        resolutions = [
            int(self.base_resolution * growth_factor**i) 
            for i in range(self.n_levels)
        ]
        
        return resolutions
    
    def hash_function(self, coords, resolution):
        """Spatial hash function"""
        primes = [1, 2654435761, 805459861]
        coords = coords * resolution
        coords = coords.long()
        
        hash_value = torch.zeros_like(coords[:, 0])
        for i in range(3):
            hash_value ^= coords[:, i] * primes[i]
            
        return hash_value % (2**self.log2_hashmap_size)
    
    def forward(self, positions):
        """Multi-resolution hash encoding"""
        encodings = []
        
        for level, resolution in enumerate(self.resolutions):
            # Get integer coordinates
            coords = positions * resolution
            coords_floor = torch.floor(coords).long()
            
            # Trilinear interpolation
            local_features = []
            for dx in [0, 1]:
                for dy in [0, 1]:
                    for dz in [0, 1]:
                        corner_coords = coords_floor + torch.tensor([dx, dy, dz]).to(positions.device)
                        hash_idx = self.hash_function(corner_coords, resolution)
                        corner_features = self.hash_tables[level](hash_idx)
                        
                        # Compute weights
                        weights = 1.0 - torch.abs(coords - corner_coords.float())
                        weight = torch.prod(weights, dim=-1, keepdim=True)
                        
                        local_features.append(weight * corner_features)
                        
            # Sum weighted features
            level_encoding = torch.sum(torch.stack(local_features), dim=0)
            encodings.append(level_encoding)
            
        return torch.cat(encodings, dim=-1)
```

## 3D Generation and Synthesis

### 1. 3D-GAN
```python
class Generator3D(nn.Module):
    def __init__(self, z_dim=128, voxel_size=64):
        super(Generator3D, self).__init__()
        
        self.z_dim = z_dim
        self.voxel_size = voxel_size
        
        # Initial projection
        self.fc = nn.Linear(z_dim, 256 * 4 * 4 * 4)
        
        # 3D deconvolution layers
        self.deconv1 = nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(128)
        
        self.deconv2 = nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        
        self.deconv3 = nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(32)
        
        self.deconv4 = nn.ConvTranspose3d(32, 1, 4, stride=2, padding=1)
        
    def forward(self, z):
        # Project and reshape
        x = self.fc(z)
        x = x.view(-1, 256, 4, 4, 4)
        
        # Deconvolutions
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = torch.sigmoid(self.deconv4(x))
        
        return x

class Discriminator3D(nn.Module):
    def __init__(self, voxel_size=64):
        super(Discriminator3D, self).__init__()
        
        # 3D convolution layers
        self.conv1 = nn.Conv3d(1, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(32, 64, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 128, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.conv4 = nn.Conv3d(128, 256, 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        
        # Output layer
        self.fc = nn.Linear(256 * 4 * 4 * 4, 1)
        
    def forward(self, x):
        # Convolutions
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        
        # Flatten and output
        x = x.view(-1, 256 * 4 * 4 * 4)
        x = self.fc(x)
        
        return x

class GAN3DTrainer:
    def __init__(self, generator, discriminator, device='cuda'):
        self.G = generator.to(device)
        self.D = discriminator.to(device)
        self.device = device
        
        # Optimizers
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss
        self.criterion = nn.BCEWithLogitsLoss()
        
    def train_step(self, real_voxels):
        batch_size = real_voxels.size(0)
        
        # Labels
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        
        # Train Discriminator
        self.opt_D.zero_grad()
        
        # Real voxels
        real_output = self.D(real_voxels)
        d_loss_real = self.criterion(real_output, real_labels)
        
        # Fake voxels
        z = torch.randn(batch_size, self.G.z_dim).to(self.device)
        fake_voxels = self.G(z)
        fake_output = self.D(fake_voxels.detach())
        d_loss_fake = self.criterion(fake_output, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.opt_D.step()
        
        # Train Generator
        self.opt_G.zero_grad()
        
        fake_output = self.D(fake_voxels)
        g_loss = self.criterion(fake_output, real_labels)
        
        g_loss.backward()
        self.opt_G.step()
        
        return d_loss.item(), g_loss.item()
```

### 2. Point Cloud Generation
```python
class PointCloudVAE(nn.Module):
    def __init__(self, num_points=2048, latent_dim=128):
        super(PointCloudVAE, self).__init__()
        
        self.num_points = num_points
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = PointNetEncoder(latent_dim * 2)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_points * 3),
            nn.Tanh()
        )
        
    def encode(self, x):
        """Encode point cloud to latent distribution"""
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=1)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent code to point cloud"""
        output = self.decoder(z)
        return output.reshape(-1, self.num_points, 3)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var
    
    def sample(self, num_samples=1):
        """Sample new point clouds"""
        z = torch.randn(num_samples, self.latent_dim).to(next(self.parameters()).device)
        samples = self.decode(z)
        return samples

class PointNetEncoder(nn.Module):
    def __init__(self, output_dim):
        super(PointNetEncoder, self).__init__()
        
        # Point-wise MLPs
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        
        # Global feature
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, output_dim)
        
        # Batch norms
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        # x: (B, N, 3) -> (B, 3, N)
        x = x.transpose(1, 2)
        
        # Point-wise features
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global pooling
        x = torch.max(x, dim=2)[0]
        
        # FC layers
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.fc2(x)
        
        return x
```

## SLAM and Visual Odometry

### 1. Visual SLAM System
```python
class VisualSLAM:
    def __init__(self):
        self.map = Map()
        self.tracker = Tracker()
        self.mapper = LocalMapper()
        self.loop_closer = LoopCloser()
        self.optimizer = GraphOptimizer()
        
    def process_frame(self, frame, timestamp):
        """Process new frame"""
        # Feature extraction
        keypoints, descriptors = self.extract_features(frame)
        
        # Tracking
        if self.map.initialized:
            pose = self.tracker.track(frame, keypoints, descriptors, self.map)
        else:
            # Initialize map with first frame
            self.initialize_map(frame, keypoints, descriptors)
            pose = np.eye(4)
            
        # Create frame object
        current_frame = Frame(
            image=frame,
            timestamp=timestamp,
            keypoints=keypoints,
            descriptors=descriptors,
            pose=pose
        )
        
        # Check if keyframe
        if self.is_keyframe(current_frame):
            # Add keyframe
            self.map.add_keyframe(current_frame)
            
            # Local mapping
            self.mapper.process_keyframe(current_frame, self.map)
            
            # Loop closure detection
            loop_frame = self.loop_closer.detect_loop(current_frame, self.map)
            if loop_frame is not None:
                self.close_loop(current_frame, loop_frame)
                
        return pose
    
    def initialize_map(self, frame, keypoints, descriptors):
        """Initialize map with first frame"""
        # Create first keyframe
        first_keyframe = KeyFrame(
            image=frame,
            keypoints=keypoints,
            descriptors=descriptors,
            pose=np.eye(4)
        )
        
        self.map.add_keyframe(first_keyframe)
        
        # Create initial map points from depth or stereo
        if self.has_depth:
            points_3d = self.triangulate_from_depth(keypoints, self.depth_map)
        else:
            # Wait for second frame for triangulation
            self.initialization_pending = True
            
        # Add map points
        for i, point_3d in enumerate(points_3d):
            map_point = MapPoint(
                position=point_3d,
                descriptor=descriptors[i],
                keyframe=first_keyframe
            )
            self.map.add_point(map_point)
            
        self.map.initialized = True

class Tracker:
    def __init__(self):
        self.matcher = FeatureMatcher()
        self.pnp_solver = PnPSolver()
        
    def track(self, frame, keypoints, descriptors, map):
        """Track camera pose"""
        # Get visible map points
        visible_points = self.get_visible_points(map)
        
        # Match features
        matches = self.matcher.match(
            descriptors, 
            [p.descriptor for p in visible_points]
        )
        
        # Get 2D-3D correspondences
        points_2d = []
        points_3d = []
        
        for match in matches:
            points_2d.append(keypoints[match.queryIdx].pt)
            points_3d.append(visible_points[match.trainIdx].position)
            
        points_2d = np.array(points_2d)
        points_3d = np.array(points_3d)
        
        # Solve PnP
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d, points_2d, self.K, None
        )
        
        if success:
            # Convert to pose matrix
            R, _ = cv2.Rodrigues(rvec)
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = tvec.squeeze()
            
            return pose
        else:
            return None
```

### 2. Visual-Inertial Odometry
```python
class VisualInertialOdometry:
    def __init__(self):
        self.state = VIOState()
        self.feature_tracker = FeatureTracker()
        self.imu_integrator = IMUIntegrator()
        self.estimator = EKFEstimator()
        
    def process_imu(self, accel, gyro, timestamp):
        """Process IMU measurement"""
        # Add to buffer
        self.imu_buffer.append({
            'accel': accel,
            'gyro': gyro,
            'timestamp': timestamp
        })
        
        # Propagate state
        if self.state.initialized:
            dt = timestamp - self.state.timestamp
            self.state = self.imu_integrator.propagate(
                self.state, accel, gyro, dt
            )
            
    def process_image(self, image, timestamp):
        """Process image measurement"""
        # Track features
        tracked_features = self.feature_tracker.track(image)
        
        # Get IMU measurements between frames
        imu_measurements = self.get_imu_between_frames(
            self.last_image_time, timestamp
        )
        
        # Initialize if needed
        if not self.state.initialized:
            self.initialize(tracked_features, imu_measurements)
        else:
            # Update state estimate
            self.estimator.update(
                self.state, 
                tracked_features, 
                imu_measurements
            )
            
        self.last_image_time = timestamp
        
        return self.state.pose

class IMUIntegrator:
    def __init__(self):
        self.gravity = np.array([0, 0, -9.81])
        
    def propagate(self, state, accel, gyro, dt):
        """Propagate state using IMU measurements"""
        # Extract state
        p = state.position
        v = state.velocity
        R = state.rotation
        ba = state.bias_accel
        bg = state.bias_gyro
        
        # Remove biases
        accel_unbiased = accel - ba
        gyro_unbiased = gyro - bg
        
        # Update rotation
        dR = self.exp_so3(gyro_unbiased * dt)
        R_new = R @ dR
        
        # Update velocity
        v_new = v + (R @ accel_unbiased + self.gravity) * dt
        
        # Update position
        p_new = p + v * dt + 0.5 * (R @ accel_unbiased + self.gravity) * dt**2
        
        # Create new state
        new_state = VIOState(
            position=p_new,
            velocity=v_new,
            rotation=R_new,
            bias_accel=ba,
            bias_gyro=bg
        )
        
        return new_state
    
    def exp_so3(self, w):
        """Exponential map for SO(3)"""
        theta = np.linalg.norm(w)
        
        if theta < 1e-6:
            return np.eye(3) + self.skew(w)
        else:
            k = w / theta
            K = self.skew(k)
            return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
            
    def skew(self, v):
        """Skew-symmetric matrix"""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
```

## Applications

### 1. Autonomous Driving
```python
class AutonomousDriving3D:
    def __init__(self):
        self.lidar_processor = LiDARProcessor()
        self.camera_processor = CameraProcessor()
        self.fusion_module = SensorFusion()
        self.object_tracker = MultiObjectTracker()
        self.map_builder = HDMapBuilder()
        
    def process_sensor_data(self, lidar_data, camera_data, vehicle_pose):
        """Process multi-sensor data for autonomous driving"""
        # Process LiDAR
        lidar_objects = self.lidar_processor.detect_objects(lidar_data)
        
        # Process camera
        camera_objects = self.camera_processor.detect_objects(camera_data)
        
        # Sensor fusion
        fused_objects = self.fusion_module.fuse(
            lidar_objects, 
            camera_objects,
            self.calibration
        )
        
        # Track objects
        tracked_objects = self.object_tracker.update(fused_objects)
        
        # Update HD map
        self.map_builder.update(tracked_objects, vehicle_pose)
        
        # Plan trajectory
        trajectory = self.plan_trajectory(
            tracked_objects, 
            self.map_builder.get_local_map(),
            vehicle_pose
        )
        
        return {
            'objects': tracked_objects,
            'trajectory': trajectory,
            'map': self.map_builder.get_local_map()
        }
    
    def plan_trajectory(self, objects, local_map, vehicle_pose):
        """Plan safe trajectory"""
        # Get drivable area
        drivable_area = local_map.get_drivable_area()
        
        # Predict object trajectories
        predicted_trajectories = self.predict_object_motion(objects)
        
        # Generate candidate trajectories
        candidates = self.generate_trajectory_candidates(
            vehicle_pose, 
            drivable_area
        )
        
        # Evaluate candidates
        best_trajectory = None
        best_cost = float('inf')
        
        for trajectory in candidates:
            # Check collision
            if self.check_collision(trajectory, predicted_trajectories):
                continue
                
            # Compute cost
            cost = self.compute_trajectory_cost(
                trajectory, 
                local_map, 
                objects
            )
            
            if cost < best_cost:
                best_cost = cost
                best_trajectory = trajectory
                
        return best_trajectory
```

### 2. AR/VR Applications
```python
class ARVRSystem:
    def __init__(self):
        self.slam = VisualSLAM()
        self.mesh_reconstructor = MeshReconstructor()
        self.renderer = ARRenderer()
        self.hand_tracker = HandTracker()
        
    def process_frame(self, rgb_frame, depth_frame):
        """Process frame for AR/VR"""
        # SLAM for camera tracking
        camera_pose = self.slam.process_frame(rgb_frame, depth_frame)
        
        # Reconstruct 3D mesh
        self.mesh_reconstructor.integrate_frame(
            rgb_frame, 
            depth_frame, 
            camera_pose
        )
        
        # Hand tracking for interaction
        hand_pose = self.hand_tracker.track(rgb_frame, depth_frame)
        
        # Render AR content
        ar_frame = self.render_ar_content(
            rgb_frame,
            camera_pose,
            hand_pose,
            self.mesh_reconstructor.get_mesh()
        )
        
        return ar_frame
    
    def render_ar_content(self, frame, camera_pose, hand_pose, scene_mesh):
        """Render AR content"""
        # Place virtual objects
        virtual_objects = self.place_virtual_objects(scene_mesh)
        
        # Handle interactions
        if hand_pose is not None:
            self.handle_hand_interaction(hand_pose, virtual_objects)
            
        # Render
        rendered_frame = self.renderer.render(
            frame,
            camera_pose,
            virtual_objects,
            scene_mesh
        )
        
        return rendered_frame

class HandTracker:
    def __init__(self):
        self.hand_model = self.load_hand_model()
        self.pose_estimator = HandPoseEstimator()
        
    def track(self, rgb_frame, depth_frame):
        """Track hand pose in 3D"""
        # Detect hand
        hand_bbox = self.detect_hand(rgb_frame)
        
        if hand_bbox is None:
            return None
            
        # Crop hand region
        hand_rgb = self.crop_region(rgb_frame, hand_bbox)
        hand_depth = self.crop_region(depth_frame, hand_bbox)
        
        # Estimate 2D keypoints
        keypoints_2d = self.hand_model(hand_rgb)
        
        # Lift to 3D using depth
        keypoints_3d = self.lift_to_3d(keypoints_2d, hand_depth)
        
        # Estimate pose
        hand_pose = self.pose_estimator.estimate(keypoints_3d)
        
        return hand_pose
```

## Future Directions

### 1. Neural Implicit Representations
```python
class NeuralImplicitSurface(nn.Module):
    def __init__(self):
        super(NeuralImplicitSurface, self).__init__()
        
        # SDF network
        self.sdf_net = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            *[ResidualBlock(256) for _ in range(8)],
            nn.Linear(256, 257)  # SDF + features
        )
        
        # Color network
        self.color_net = nn.Sequential(
            nn.Linear(256 + 3, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )
        
    def forward(self, points, view_dirs=None):
        """
        Predict SDF and color for 3D points
        """
        # SDF and features
        sdf_output = self.sdf_net(points)
        sdf = sdf_output[:, 0:1]
        features = sdf_output[:, 1:]
        
        # Color
        if view_dirs is not None:
            color_input = torch.cat([features, view_dirs], dim=-1)
            colors = self.color_net(color_input)
        else:
            colors = None
            
        return sdf, colors
    
    def extract_mesh(self, resolution=256, threshold=0.0):
        """Extract mesh using marching cubes"""
        # Create grid
        x = np.linspace(-1, 1, resolution)
        y = np.linspace(-1, 1, resolution)
        z = np.linspace(-1, 1, resolution)
        xx, yy, zz = np.meshgrid(x, y, z)
        
        points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        points = torch.tensor(points, dtype=torch.float32)
        
        # Evaluate SDF
        with torch.no_grad():
            sdf_values = []
            batch_size = 10000
            
            for i in range(0, len(points), batch_size):
                batch_points = points[i:i+batch_size]
                sdf, _ = self.forward(batch_points)
                sdf_values.append(sdf.numpy())
                
        sdf_values = np.concatenate(sdf_values).reshape(resolution, resolution, resolution)
        
        # Marching cubes
        vertices, faces = mcubes.marching_cubes(sdf_values, threshold)
        vertices = vertices / resolution * 2 - 1
        
        return vertices, faces
```

### 2. Differentiable Rendering
```python
class DifferentiableRenderer:
    def __init__(self, image_size=512):
        self.image_size = image_size
        self.rasterizer = self.setup_rasterizer()
        
    def setup_rasterizer(self):
        """Setup PyTorch3D rasterizer"""
        from pytorch3d.renderer import (
            RasterizationSettings,
            MeshRasterizer,
            MeshRenderer,
            SoftPhongShader,
            TexturesVertex
        )
        
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        
        return MeshRasterizer(raster_settings=raster_settings)
    
    def render(self, vertices, faces, vertex_colors, camera):
        """
        Differentiable mesh rendering
        """
        # Create mesh
        mesh = Meshes(
            verts=[vertices],
            faces=[faces],
            textures=TexturesVertex(verts_features=[vertex_colors])
        )
        
        # Render
        fragments = self.rasterizer(mesh, cameras=camera)
        images = self.shader(fragments, mesh)
        
        return images
```

### 3. Self-Supervised 3D Learning
```python
class SelfSupervised3D:
    def __init__(self):
        self.encoder = PointCloudEncoder()
        self.decoder = PointCloudDecoder()
        
    def train_with_augmentation(self, point_cloud):
        """Self-supervised training with augmentation"""
        # Apply augmentations
        aug1 = self.augment(point_cloud)
        aug2 = self.augment(point_cloud)
        
        # Encode
        z1 = self.encoder(aug1)
        z2 = self.encoder(aug2)
        
        # Contrastive loss
        loss = self.contrastive_loss(z1, z2)
        
        return loss
    
    def augment(self, point_cloud):
        """Apply 3D augmentations"""
        # Random rotation
        angle = np.random.uniform(0, 2*np.pi)
        rotation = self.rotation_matrix(angle)
        augmented = point_cloud @ rotation
        
        # Random scaling
        scale = np.random.uniform(0.8, 1.2)
        augmented = augmented * scale
        
        # Random jittering
        noise = np.random.normal(0, 0.02, size=point_cloud.shape)
        augmented = augmented + noise
        
        return augmented
```

## Conclusion

3D Computer Vision continues to evolve rapidly with advances in deep learning, neural rendering, and geometric understanding. The integration of classical geometric methods with modern deep learning approaches has led to robust solutions for 3D reconstruction, understanding, and synthesis.

Key developments include:
- Multi-view geometry combined with deep learning
- Efficient 3D representations (NeRF, neural implicit surfaces)
- Real-time 3D perception for robotics and AR/VR
- Self-supervised learning for 3D understanding
- Differentiable rendering for inverse graphics

As hardware capabilities improve and new algorithms emerge, 3D computer vision will enable even more sophisticated applications in autonomous systems, digital content creation, and human-computer interaction.

*Originally from umitkacar/awesome-3D-Computer-Vision repository*