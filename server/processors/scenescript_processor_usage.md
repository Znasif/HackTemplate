# SceneScript Processor

This document explains how to use the SceneScript processor for 3D scene understanding and reconstruction in the vidServer project.

## Overview

The SceneScript processor integrates the SceneScript model for 3D scene understanding from 2D images. It enables real-time 3D reconstruction, object detection, and scene graph generation from camera input.

## Features

- 3D scene understanding from single RGB images
- Object detection with 3D position, rotation, and scale
- Room layout estimation
- Scene graph generation for relationship understanding
- Multiple visualization modes: mesh, wireframe, bounding box
- Performance metrics tracking

## Installation

1. Install the base vidServer requirements
2. Install additional SceneScript dependencies:
```bash
pip install torch torchvision opencv-python pillow matplotlib
```

3. Download the SceneScript model:
```bash
# Replace with the actual download command for your model
wget https://example.com/scenescript_model_non_manhattan_class_agnostic_model.ckpt -O /home/znasif/scenescript_model_non_manhattan_class_agnostic_model.ckpt
```

## Usage in Python Code

```python
from processors.scenescript_processor import SceneScriptProcessor

# Initialize the processor
processor = SceneScriptProcessor(
    model_path="/home/znasif/scenescript_model_non_manhattan_class_agnostic_model.ckpt",
    use_gpu=True,
    confidence_threshold=0.5,
    display_mode="mesh"
)

# Process a frame
processed_frame, scene_description = processor.process_frame(frame)
```

## Test Script

A test script is provided to verify the processor is working correctly:

```bash
python server/test_scenescript.py --model-path /home/znasif/scenescript_model_non_manhattan_class_agnostic_model.ckpt --display-mode mesh
```

## Parameters

The SceneScript processor accepts the following parameters:

- `model_path`: Path to the SceneScript model checkpoint
- `use_gpu`: Whether to use GPU acceleration (default: True)
- `confidence_threshold`: Threshold for object detection confidence (default: 0.5)
- `display_mode`: Visualization mode - "mesh", "wireframe", "bbox", or "all"

## Display Modes

SceneScript supports multiple visualization modes:

1. **mesh**: Full 3D mesh rendering with textures (if available)
2. **wireframe**: Simplified 3D wireframe representation
3. **bbox**: Only 2D/3D bounding boxes of detected objects
4. **all**: Combines all visualization modes

In the test script, you can cycle through display modes by pressing 'm'.

## Output Format

The processor returns two items:
1. **processed_frame**: Frame with visualizations
2. **scene_description**: Text description of the scene, including:
   - Detected objects with counts
   - Room dimensions and type
   - Spatial relationships between objects
   - Inference time metrics

## Performance Considerations

- Using a GPU is strongly recommended for real-time performance
- The full mesh visualization mode is more computationally intensive
- First inference is slower due to model initialization and compilation

## Integration with Stream Processing

To integrate with the vidServer framework:

1. Import the processor in your stream processing code
2. Initialize the processor with appropriate parameters
3. Call `process_frame()` on each video frame
4. Use the returned processed frame and scene description for display or further processing

## Extending SceneScript

The SceneScript processor can be extended in several ways:

- Adding support for multi-frame tracking and temporal consistency
- Integrating with AR/VR applications for immersive visualization
- Creating interactive 3D scene editors based on the reconstructions
- Combining with other processors for multimodal understanding
