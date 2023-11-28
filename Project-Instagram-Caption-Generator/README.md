# Project: Instagram Caption Generator

## Overview
According to Forbes, the global count of social media users has reached about 5 billion in 2023. Among these users, about 50 million identify as "influencers" — individuals who use these platforms to earn income. In the competitive world of social media, influencers vie for attention, and the success of their content is crucial. However, the captions accompanying their posts are equally important. Captions provide context, allowing influencers to showcase their creativity and personality.

![post_sample](https://github.com/Stonemannn/Transformers/blob/989ec6686e74a4327d1de097bda3c46e98669a15/Final-Project-Automatic-IG-Caption-Generator/post_sample.png)

In this project, I'm exploring the use of [Instagram Images with Captions](https://www.kaggle.com/datasets/prithvijaunjale/instagram-images-with-captions) dataset from Kaggle titled Instagram Images with Captions. This dataset features 20.5K Instagram posts, complete with images in JPG format and their respective captions in a CSV file. Due to the limitations of my laptop’s computing power, I’ll be focusing on a subset of this dataset, which includes 5,000 images and captions.

## Fine-Tuning Dataset
- https://www.kaggle.com/datasets/prithvijaunjale/instagram-images-with-captions

## Data Preprocessing
1. **Conversion to Dataset Format:** First, I'll transform both the images and captions into a format called Dataset which is easier for later training.
2. **Feature Extraction:** Next, each image in the dataset will be processed through feature extraction using a chosen image encoder model.
3. **Caption Tokenization:** The captions will be processed using a text decoder model to tokenize them.
4. **Dataset Splitting:** Finally, I'll divide the dataset into two parts: 80% for training and 20% for testing. (Note: Ultimately, I chose to test the model using some of my own photos, since the original dataset contained too much noise.)


## Model Selection
1. encoder-decoder model which has not been fine-tuned for image captioning.  
    - For image_encoder_model = **"google/vit-base-patch16-224-in21k"** 
    - For text_decoder_model = **"gpt2"**
2. encoder-decoder model which has been fine-tuned for image captioning. 
    - For image_encoder_model = **"nlpconnect/vit-gpt2-image-captioning"** 
    - For text_decoder_model = **"nlpconnect/vit-gpt2-image-captioning"**.

The fine-tuning will be done twice for each selected pre-trained model. The first fine-tuning will be done only on the last GPT2 block in the selected model, which means I need to freeze all the prior layers. The second fine-tuning will be done on all the parameters in the selected model.

### Quick introduction to ViT (Vision Transformer)
Vision transformers are an implementation of transformer architecture for images. The Vision Transformer (ViT) transforms images into fixed-number patches, treating them as tokens (Dosovitskiy, 2020). These patches are linearly embedded and combined with positional encodings. The transformer architecture processes these embeddings, capturing both local and global dependencies through self-attention mechanisms. Unlike traditional convolutional neural networks, ViT operates on global information simultaneously. 


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

**Conclusion** In my opinion, Model 3, which is only fine-tuned on the last GPT2 block of pre-fine-tuned image-captioning model: nlpconnect/vit-gpt2-image-captioning, stands out as the best choice. It impressively learned to use emojis in captions, and at the same time, identifies the subjects of the photos.

### Critical Analysis
- **Future Applications**: This model can be extended to generate captions for other social media platforms such as Twitter, Facebook, and TikTok. It can also be extended to generate captions for other types of media such as videos.

- **Advantages**: Saves time and effort for influencers who are constantly competing for attention in the world of social media.

- **Disadvantages**: Only fine-tuned on a subset of the dataset due to limited computing power. The model is not able to generate really attractive captions that can help influencers gain more followers.

## HuggingFace Spaces
- https://huggingface.co/spaces/Stoneman/IG-caption-generator

## HuggingFace Model
- https://huggingface.co/Stoneman/IG-caption-generator-vit-gpt2-last-block/tree/main
- https://huggingface.co/Stoneman/IG-caption-generator-vit-gpt2-all/tree/main
- https://huggingface.co/Stoneman/IG-caption-generator-nlpconnect-last-block/tree/main
- https://huggingface.co/Stoneman/IG-caption-generator-nlpconnect-all/tree/main

## References

- https://ankur3107.github.io/blogs/the-illustrated-image-captioning-using-transformers/
