# Project: Self-Generated Instagram Captions

## Overview
According to Forbes, the number of social media users worldwide has swelled to 4.9 billion globally for the year of 2023 (Forbes, 2023). Across platforms, approximately 50 million are self-proclaimed “influencers”, those who utilize platforms to make some form of income. Influencers constantly compete for attention in the world of social media, and victory heavily relies on the quality of their content, especially the quality of their captions.

In this project, I will be using the [Instagram Images with Captions](https://www.kaggle.com/datasets/prithvijaunjale/instagram-images-with-captions) dataset from Kaggle to fine tune a pretrained model on Hugging Face that can generate high quality IG-like captions for photos automatically. 


## Model Selection
1. encoder-decoder model which has not been fine-tuned for image captioning.  
    - For image_encoder_model = **"google/vit-base-patch16-224-in21k"** 
    - For text_decoder_model = **"gpt2"**
2. encoder-decoder model which has been fine-tuned for image captioning. 
    - For image_encoder_model = **"nlpconnect/vit-gpt2-image-captioning"** 
    - For text_decoder_model = **"nlpconnect/vit-gpt2-image-captioning"**.

The fine-tuning will be done twice for each selected pre-trained model. The first fine-tuning will be done only on the last GPT2 block in the selected model, which means I need to freeze all the prior layers. The second fine-tuning will be done on all the parameters in the selected model.

## Model Evaluation
![example5](https://github.com/Stonemannn/Transformers/blob/124f2c7ade694a7d302463f5385c0a786d90ae92/Final-Project-Automatic-IG-Caption-Generator/test_images/example5.JPG)

- Model 1: I'm so excited to finally be able to share my story with you guys. I'm so excited to finally be able to share my story with you guys. 

- Model 2: I'm so excited to finally share my first collection with you guys! I've been working on this for so long that I'm almost ready to reveal it 😂 I'm so excited to finally share my first collection with you guys! I've been working on this for so long that I'm almost ready to reveal it 😂 I'm so excited to finally share my first collection with you guys! I've been working on this for so long that I'm almost ready to reveal it 😂 I'm so excited to finally share my first collection with you guys! I've been working on this for so long that I'm almost ready

- Model 3: I'm not sure what this is but I can tell you that it's a big house. 

- Model 4: I'm sorry I couldn't get to the bottom of this post. I'm sorry I couldn't get to the bottom of this post. 

![example1](https://github.com/Stonemannn/Transformers/blob/124f2c7ade694a7d302463f5385c0a786d90ae92/Final-Project-Automatic-IG-Caption-Generator/test_images/example1.JPG)

- Model 1: I'm so excited to finally be able to share my story with you guys. I'm so excited to finally be able to share my story with you guys. 

- Model 2: I'm so happy I'm not a vampire 

- Model 3: I can't wait to see what the next lake looks like. 

- Model 4: I'm going to be back in a few days but this is the best I've had so far. 

![example8](https://github.com/Stonemannn/Transformers/blob/124f2c7ade694a7d302463f5385c0a786d90ae92/Final-Project-Automatic-IG-Caption-Generator/test_images/example8.JPG)

- Model 1: I'm so excited to finally be able to share my new collection with you guys! 

- Model 2: I'm so happy I got to spend my birthday with my bestie 💗 

- Model 3: I love this little dog 💙💙 

- Model 4: 💜 

![example7](https://github.com/Stonemannn/Transformers/blob/124f2c7ade694a7d302463f5385c0a786d90ae92/Final-Project-Automatic-IG-Caption-Generator/test_images/example7.JPG)

- Model 1: I'm so excited to be launching my first app! 

- Model 2: I'm so excited to finally be apart of this family. I love you guys so much. 

- Model 3: I'm not sure if this is a good place to start my journey. 

- Model 4: I'm not sure what kind of city this is in but it's definitely not me.

### Critical Analysis
- **Future Applications**: FlashAttention-2 makes it possible for video generation which require long sequences.

- **Advantages**: FlashAttention-2 works well for longer sequences.

- **Disadvantages**: FlashAttention-2 is not as efficient as FlashAttention for shorter sequences.

## Fine-Tuning Dataset
- https://www.kaggle.com/datasets/prithvijaunjale/instagram-images-with-captions

## HuggingFace Spaces
- https://huggingface.co/spaces/Stoneman/IG-caption-generator

## HuggingFace Model
- https://huggingface.co/Stoneman/IG-caption-generator-vit-gpt2-last-block/tree/main
- https://huggingface.co/Stoneman/IG-caption-generator-vit-gpt2-all/tree/main
- https://huggingface.co/Stoneman/IG-caption-generator-nlpconnect-last-block/tree/main
- https://huggingface.co/Stoneman/IG-caption-generator-nlpconnect-all/tree/main

## References

- https://ankur3107.github.io/blogs/the-illustrated-image-captioning-using-transformers/
