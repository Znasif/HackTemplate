#!/usr/bin/env python3
"""
Test script for loading and using the SceneScript model
"""
import torch
import argparse
import os
import sys
import numpy as np
import cv2
from PIL import Image

def inspect_checkpoint(checkpoint_path):
    """
    Inspect a checkpoint file and print its structure
    """
    print(f"Inspecting checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"Successfully loaded checkpoint")
        
        # Check if it's a dictionary
        if isinstance(checkpoint, dict):
            print(f"Checkpoint is a dictionary with keys: {list(checkpoint.keys())}")
            
            # Check for known keys
            if 'state_dict' in checkpoint:
                print(f"Found 'state_dict' with {len(checkpoint['state_dict'])} entries")
                # Print a few sample entries
                print("Sample entries:")
                for i, (k, v) in enumerate(checkpoint['state_dict'].items()):
                    if i < 5:  # Print just a few for brevity
                        print(f"  {k}: Tensor of shape {v.shape} and dtype {v.dtype}")
                    else:
                        break
            
            # Check for other common keys
            for key in ['optimizer_states', 'lr_schedulers', 'epoch', 'global_step']:
                if key in checkpoint:
                    print(f"Found '{key}': {checkpoint[key]}")
        else:
            print(f"Checkpoint is not a dictionary but a {type(checkpoint)}")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

def try_load_model(checkpoint_path):
    """
    Attempt to load model from checkpoint in various ways
    """
    print(f"Attempting to load model from: {checkpoint_path}")
    
    # First, try standard PyTorch loading
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("Successfully loaded checkpoint using torch.load")
        
        # If it's a state_dict, try creating a simple model
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            print("Checkpoint contains a state_dict, attempting to create model...")
            
            # Try PyTorch Lightning if available
            try:
                import pytorch_lightning as pl
                
                class GenericModel(pl.LightningModule):
                    def __init__(self):
                        super().__init__()
                        # Create a placeholder model
                        self.backbone = torch.nn.Sequential(
                            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
                        )
                    
                    def forward(self, x):
                        return self.backbone(x)
                
                # Try to load entire model
                print("Attempting to load as Lightning model...")
                model = GenericModel.load_from_checkpoint(checkpoint_path)
                print("Success! Loaded complete Lightning model")
                return model
                
            except Exception as e:
                print(f"Error loading as Lightning model: {e}")
                
                # Try loading just the state_dict
                try:
                    print("Attempting to create generic model and load state_dict...")
                    
                    # Create a simple placeholder model - adjust as needed
                    model = torch.nn.Sequential(
                        torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
                    )
                    
                    # Try to load state_dict
                    state_dict = {k.replace('backbone.', ''): v for k, v in checkpoint['state_dict'].items() 
                                 if k.startswith('backbone.')}
                    
                    if len(state_dict) > 0:
                        # Only try loading if we have relevant state dict entries
                        model.load_state_dict(state_dict, strict=False)
                        print("Loaded state_dict into generic model (non-strict)")
                        return model
                    else:
                        print("No compatible layers found in state_dict")
                        
                except Exception as e:
                    print(f"Error loading state_dict into generic model: {e}")
        
        # If it's not a state_dict, it might be the model itself
        elif hasattr(checkpoint, 'forward') and callable(getattr(checkpoint, 'forward')):
            print("Checkpoint appears to be a model object with forward method")
            return checkpoint
        
        else:
            print("Checkpoint format not recognized as a model or state_dict")
        
    except Exception as e:
        print(f"Error during model loading: {e}")
    
    return None

def test_inference(model, image_path=None):
    """
    Test inference with the loaded model
    """
    if model is None:
        print("No model provided for inference test")
        return
    
    print("Preparing test input...")
    
    # Create test input (either from image or random tensor)
    if image_path and os.path.exists(image_path):
        print(f"Loading image from {image_path}")
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((256, 256))
        
        # Convert to tensor and normalize
        img_tensor = torch.FloatTensor(np.array(img)).permute(2, 0, 1)
        img_tensor = img_tensor / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
    else:
        print("Creating random input tensor")
        # Create a random input tensor
        img_tensor = torch.rand(3, 256, 256)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    # Set model to eval mode
    if hasattr(model, 'eval'):
        model.eval()
    
    # Try inference
    try:
        print("Running inference...")
        with torch.no_grad():
            output = model(img_tensor)
        
        # Print output information
        if isinstance(output, dict):
            print(f"Model output is a dictionary with keys: {list(output.keys())}")
            for k, v in output.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: Tensor of shape {v.shape}")
                else:
                    print(f"  {k}: {type(v)}")
        elif isinstance(output, (list, tuple)):
            print(f"Model output is a {type(output).__name__} with {len(output)} elements")
            for i, item in enumerate(output):
                if isinstance(item, torch.Tensor):
                    print(f"  {i}: Tensor of shape {item.shape}")
                else:
                    print(f"  {i}: {type(item)}")
        elif isinstance(output, torch.Tensor):
            print(f"Model output is a tensor of shape {output.shape}")
        else:
            print(f"Model output is of type {type(output)}")
        
        print("Inference successful!")
        
    except Exception as e:
        print(f"Error during inference: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test script for SceneScript model loading")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint file (.ckpt)')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to test image (optional)')
    parser.add_argument('--inspect-only', action='store_true',
                        help='Only inspect checkpoint without loading model')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        return
    
    # Inspect checkpoint
    inspect_checkpoint(args.checkpoint)
    
    # Exit if inspect-only mode
    if args.inspect_only:
        return
    
    # Try to load model
    model = try_load_model(args.checkpoint)
    
    # Test inference if model was loaded
    if model is not None:
        test_inference(model, args.image)
    else:
        print("Could not load a usable model")

if __name__ == "__main__":
    main()
