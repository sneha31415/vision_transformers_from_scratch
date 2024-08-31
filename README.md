<p>
<h1 align = "center" > <strong>🚀 Vision Transformers From Scratch</strong> <br></h1>

<h2 align = "center">

</p>

[SRA](https://www.sravjti.in/) Eklavya 2024 ✨<br></h2>

<!-- TABLE OF CONTENTS -->
# ✏️ Table of contents

- [About the project](#-about-the-project)
  - [Aim](#aim)
  - [Description](#description)
  - [Tech Stack](#Tech-Stack)
  - [Dataset](#Dataset)
  - [File Structure](#-file-structure)
- [Getting Started](#-getting-started)
  - [Installation](#installation)
- [Theory and Approach](#-theory-and-approach)
  - [CNN + LSTM Model](#cnn--lstm-model)
  - [ViT Model](#vision-transformers-model-vit)
- [Results](#-results)
  - [CNN + LSTM model](#cnn--lstm-model-1)
  - [ViT model](#vit-model)
- [Applications](#-applications)
- [Contributors](#contributors)
- [Acknowledgements](#acknowledgements-and-resources)
 

<!-- ABOUT PROJECT -->
# ⭐ About the project
## Aim
The aim of this project is to generate descriptive captions for images by combining the power of Transformers and computer vision.

## Description
This project focuses on image captioning using Vision Transformers (ViT), implemented from scratch. Initially, a basic CNN + LSTM approach was employed to establish a baseline. We then transitioned to a more advanced Vision Transformer (ViT) model to leverage its capability in capturing long-range dependencies in image data.

## Tech Stack
### Programming Language
 ![Static Badge](https://img.shields.io/badge/Python-white?style=for-the-badge&logo=python&labelColor=black&color=%5C)

### Deeplearning Frameworks
![Static Badge](https://img.shields.io/badge/Pytorch-orange?style=for-the-badge&logo=pytorch&labelColor=black)

![Static Badge](https://img.shields.io/badge/Tensorflow-orange?style=for-the-badge&logo=Tensorflow&labelColor=black)

![Static Badge](https://img.shields.io/badge/Keras-orange?style=for-the-badge&logo=Keras&labelColor=black)

### Data handling
![Static Badge](https://img.shields.io/badge/Numpy-blue?style=for-the-badge&logo=Numpy&labelColor=black)

![Static Badge](https://img.shields.io/badge/Pandas-blue?style=for-the-badge&logo=Pandas&labelColor=black)

![Static Badge](https://img.shields.io/badge/OpenCV-blue?style=for-the-badge&logo=OpenCV&labelColor=black)

### Natural Language Processsing
![Static Badge](https://img.shields.io/badge/NLTK-silver?style=for-the-badge&logo=python&logoColor=pink&labelColor=black)
<br><br>

## Dataset
The project uses the [COCO 2017 dataset](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) , a comprehensive dataset comprising of 5 descriptive captions for each image.<br>
<br>

## 📁 File structure
![Static Badge](https://img.shields.io/badge/coming%20soon-red?style=flat-square)
<br>
<br>



<!-- GETTING STARTED -->
# 🛠 Getting started
## Installation
1) Clone the repo<br>
`git clone https://github.com/sneha31415/vision_transformers_from_scratch.git`

2) Navigate to the project directory<br>
`cd vision_transformers_from_scratch` 
<br><br>

<!-- THEORY AND APPROACH -->
# 📝 Theory and Approach
## CNN + LSTM Model
This is the complete architecture of the CNN + LSTM image captioning model. The CNN encoder basically finds patterns in images and encodes it into a vector that is passed to the LSTM decoder that outputs a word at each time step to best describe the image. Upon reaching the {endseq} token or the maximum length of the sentence, the entire caption is generated and that is our output for that particular image.
![cnn+lstm model](assets/image_capt_cnn+lstm.png)

#### 1) Encoder: 
A pretrained CNN model (ResNet50) is used for feature extraction, transforming input images into fixed-length feature vectors.
#### 2) Decoder: 
An LSTM network is utilized to generate captions by taking image features and previous word embeddings as input to predict the next word.


## Vision Transformers model (ViT) 
### What are transformers?
Before heading into vision transformers, lets understand transformers.<br> Since the introduction of transformers in the 2017 paper [Attention is all you need ](https://arxiv.org/abs/1706.03762)by Google Brain, it steered an interest in its capability in NLP
#### Transformer Architecture
![Transformer](assets/transformer_encoder_decoder.png)
<!-- 
The transformer is an architecture that relies on the concept of attention, a technique used to provide weights to different parts of an input sequence so that a better understanding of its underlying context is achieved. <br>
In addition, transformers process inputs in parallel making them more efficient and scalable in comparison to traditional sequential models such as RNN and LSTM. -->

**In the Transformer model:**

- **Encoder**: Converts input tokens into continuous representations using self-attention to capture relationships between all tokens simultaneously.
- **Decoder**: Generates output tokens by attending to both the encoder’s output and previously generated tokens, using masked self-attention and cross-attention.

### So... What are Vision Transformers?
Vision Transformers were introduced in the 2020 paper [An Image is worth 16x16 words](https://arxiv.org/abs/2010.11929).<br >
The Vision Transformer, or ViT, is a model that employs a Transformer-like architecture over patches of the image. An image is split into fixed-size patches, each of them are then linearly embedded, position embeddings are added, and the resulting sequence of vectors is fed to a standard Transformer encoder.<br>

### 🌟 Image Captioning using ViT from scratch 💥

![vit caption](assets/image_capt_ViT.png)
#### 1) ViT Encoder
As depicted in the figure, instead of using a pretrained CNN or
Faster R-CNN model to extract spatial features or bottom-up
features like the previous methods, we divide the original image into a sequence of image patches to adapt to the input
form of Transformer. We used a Conv2D layer(due to performance gain reasons) with a stride and kernel size equal to patch size. Alternatively you can also use linear layer here. and then reshape the 4D tensor to 3D to flatten it. Then add the learnable position embeddings.<br>

The encoder of CPTR consists of `Nx`
 stacked identical
layers, each of which consists of a multi-head self-attention
(MHA) sublayer followed by a positional feed-forward sublayer. MHA contains H parallel heads and each head hi corresponds to an independent scaled dot-product attention function which allows the model to jointly attend to different subspaces.

![alt text](assets/attention_formula.png)

#### 2) Transformer Decoder
In the decoder side, we add positional embedding to
the word embedding features and take the addition of encoder output and decoder 1st layer results as the input.<br> 

The decoder consists of `Nd` stacked identical layers with each layer containing a masked multi-head self-attention sublayer followed
by a multi-head cross attention sublayer and a feedforward sublayer sequentially.
The output feature of the last decoder layer is utilized to
predict next word via a linear layer whose output dimension
equals to the vocabulary size

## 🤖 Results
### CNN + LSTM Model

#### 1) Bleu score
- The CNN + LSTM model achieved a BLEU-1 score of <u>0.553085</u> and BLEU-2 score of <u>0.333717</u> 

#### 2) Predicted Captions
![result1](assets/result1.png)

![result2](assets/result2.png)

![result3](assets/result3.png)

![result4](assets/result4.png)

#### some fails
![fail1](assets/fail1.png)
![fail2](assets/fail2.png)

### ViT Model
![Static Badge](https://img.shields.io/badge/coming%20soon-red?style=flat-square)


## 🌎 Applications:
- **Enhanced Image Understanding: 
Generates more accurate and context-aware captions by capturing complex 
relationships within images.**

- **Accessibility: 
Improves accessibility for visually impaired users by converting visual information into descriptive text.**

- **Image Search and Organization: Enhances image search engines by providing detailed descriptions, aiding in better indexing and retrieval.**

- **E-commerce: Provides detailed product descriptions in online catalogs, improving user experience and product discovery.**


## Contributors

- [Sneha Singh](https://github.com/sneha31415) - sneha.singh.31415@gmail.com

- [Prithvi Tambewagh](https://github.com/rkt-1597) - patambewagh_b23@et.vjti.ac.in

- [Akash Kawle](https://github.com/shinymack) - ackawle_b23@et.vjti.ac.in

## Acknowledgements and Resources

- Special thanks to [SRA VJTI](https://www.sravjti.in/) for Eklavya 2024

- A heartful gratitude to our mentors [Aryan Nanda](https://github.com/AryanNanda17) and [Abhinav Ananthu](https://github.com/Herculoxz) for guiding us throughout this project

- [Deep learning specialization](https://www.coursera.org/specializations/deep-learning) for their course on Neural Networks and Deep Learning.

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) for understanding Vision Transformers(ViTs)

- [Attention is all you need](https://arxiv.org/abs/1706.03762) for understanding the transformer architecture

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) for understanding residual functions





  <br/>


