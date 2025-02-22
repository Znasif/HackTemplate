from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
from transformers.image_utils import load_image
import torch

# Load model and processor once outside the loop
model_id = "google/paligemma2-3b-mix-448"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto").eval()
processor = PaliGemmaProcessor.from_pretrained(model_id)

def process_image(filename):
    try:
        image = load_image(filename)
        prompt = "describe en"
        model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(torch.bfloat16).to(model.device)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]
            decoded = processor.decode(generation, skip_special_tokens=True)
            return decoded
    except Exception as e:
        return f"Error processing {filename}: {str(e)}"

def main():
    print("Model loaded and ready. Enter image filenames (or 'quit' to exit):")
    while True:
        filename = input("\nEnter image filename: ").strip()
        if filename.lower() == 'quit':
            break
        
        if filename:
            result = process_image(filename)
            print(f"\nResult: {result}")

if __name__ == "__main__":
    main()