import gradio as gr
import torch
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel 

# Setup device, model, tokenizer, and feature extractor
device = 'mps'


model_checkpoint1 = "Stoneman/IG-caption-generator-vit-gpt2-last-block"
feature_extractor1 = ViTImageProcessor.from_pretrained(model_checkpoint1)
tokenizer1 = GPT2TokenizerFast.from_pretrained(model_checkpoint1)
model1 = VisionEncoderDecoderModel.from_pretrained(model_checkpoint1).to(device)

model_checkpoint2 = "Stoneman/IG-caption-generator-vit-gpt2-all"
model2 = VisionEncoderDecoderModel.from_pretrained(model_checkpoint2).to(device)

model_checkpoint3 = "Stoneman/IG-caption-generator-nlpconnect-last-block"
model3 = VisionEncoderDecoderModel.from_pretrained(model_checkpoint3).to(device)

model_checkpoint4 = "Stoneman/IG-caption-generator-nlpconnect-all"
model4 = VisionEncoderDecoderModel.from_pretrained(model_checkpoint4).to(device)

models = {
    1: model1,
    2: model2,
    3: model3,
    4: model4
}

# Prediction function
def predict(image, max_length=128):
    captions = {}

    image = image.convert('RGB')
    pixel_values = feature_extractor1(images=image, return_tensors="pt").pixel_values.to(device)
    for i in range(1,5):
        caption_ids = models[i].generate(pixel_values, max_length=max_length)[0]
        caption_text = tokenizer1.decode(caption_ids, skip_special_tokens=True)
        captions[i] = caption_text
    # Return a single string with all captions
    return '\n\n'.join(f'Model {i}: {caption}' for i, caption in captions.items())


# Define input and output components
input_component = gr.components.Image(label="Upload any Image", type="pil")
output_component = gr.components.Textbox(label="Captions")

# Example images
examples = [f"/Users/stoneman/Desktop/test_images/example{i}.JPG" for i in range(1, 10)]

# Interface
title = "IG-caption-generator"
description = "Made by: Jiayu Shi"
interface = gr.Interface(
    fn=predict,
    description=description,
    inputs=input_component,
    theme="huggingface",
    outputs=output_component,
    examples=examples,
    title=title,
)

# Launch interface
interface.launch(debug=True)
