# FastVLM Processor

This document explains how to use the FastVLM processor for vision-language tasks in the vidServer project.

## Overview

The FastVLM processor integrates [FastVLM](https://github.com/apple/ml-fastvlm), an efficient vision-language model developed by Apple's ML Research team. FastVLM uses a hybrid vision encoder called FastViTHD designed for faster inference with high-resolution images, resulting in significantly reduced Time-to-First-Token (TTFT).

## Features

- Processes images with the FastVLM model to generate descriptive text
- Supports multiple model sizes (0.5B, 1.5B, 7B parameters)
- Measures and displays inference time metrics
- Overlays generated text on processed frames
- Customizable prompts for different vision-language tasks

## Installation

1. Install the base vidServer requirements
2. Install additional FastVLM dependencies:
```bash
pip install -r server/processors/fastvlm_requirements.txt
```

3. Download FastVLM model weights:
```bash
cd /home/znasif/ml-fastvlm
bash get_models.sh
```

## Usage in Python Code

```python
from processors.fastvlm_processor import FastVLMProcessor

# Initialize the processor
processor = FastVLMProcessor(
    model_path="/home/znasif/ml-fastvlm/checkpoints/llava-fastvithd_1.5b_stage3",
    prompt="Describe what you see in the image.",
    use_gpu=True,
    model_size="0.5b"
)

# Process a frame
processed_frame, text_result = processor.process_frame(frame)
```

## Test Script

A test script is provided to verify the processor is working correctly:

```bash
python server/test_fastvlm.py --model-path /home/znasif/ml-fastvlm/checkpoints/llava-fastvithd_0.5b_stage3 --prompt "Describe what you see in the image."
```

## Parameters

The FastVLM processor accepts the following parameters:

- `model_path`: Path to the FastVLM model checkpoint directory
- `prompt`: Default prompt to use for the model (e.g., "Describe what you see in the image.")
- `use_gpu`: Whether to use GPU acceleration (default: True)
- `model_size`: Model size variant ('0.5b', '1.5b', or '7b')

## Model Variants

FastVLM comes in three sizes:

1. **FastVLM-0.5B**: Small and fast - great for mobile devices where speed matters
2. **FastVLM-1.5B**: Well balanced - great for larger devices where speed and accuracy matters
3. **FastVLM-7B**: Fast and accurate - ideal for situations where accuracy matters over speed

## Example Prompts

- General description: "Describe what you see in the image."
- Detailed analysis: "Analyze this image in detail."
- Object counting: "How many people are in this image?"
- OCR-like tasks: "Read and transcribe any text visible in this image."
- Scene understanding: "What is happening in this scene?"

## Performance Considerations

- The 0.5B model is recommended for real-time applications
- Using a GPU is strongly recommended for optimal performance
- First inference is slower due to model initialization

## Integration with Stream Processing

To integrate with the vidServer framework:

1. Import the processor in your stream processing code
2. Initialize the processor with appropriate parameters
3. Call `process_frame()` on each video frame
4. Use the returned processed frame and text for display or further processing
