<p>
<h1 align = "center" > <strong>Vision Transformers From Scratch</strong> <br></h1>

<h2 align = "center">

</p>

[SRA](https://www.sravjti.in/) Eklavya 2024 ✨<br></h2>

<!-- ABOUT PROJECT -->
# 🚀  About the project
## ⭐ Aim
The aim of this project is to generate descriptive captions for images by combining the power of Transformers and computer vision.

## ✏️ Description
This project focuses on image captioning using Vision Transformers (ViT), implemented from scratch. Initially, a basic CNN + LSTM approach was employed to establish a baseline. We then transitioned to a more advanced Vision Transformer (ViT) model to leverage its capability in capturing long-range dependencies in image data.

## 🤖 Tech Stack
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


## Dataset
The project uses the [COCO 2017 dataset](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) , a comprehensive dataset comprising of 5 descriptive captions for each image.<br>


## File structure
![Static Badge](https://img.shields.io/badge/coming%20soon-red?style=flat-square)



<!-- GETTING STARTED -->
# Getting started
## 🛠 Installation
1) Clone the repo<br>
`git clone https://github.com/sneha31415/vision_transformers_from_scratch.git`

2) Navigate to the project directory<br>
`cd vision_transformers_from_scratch` 

<!-- THEORY AND APPROACH -->
# Theory and Approach
## CNN + LSTM Model
This is the complete architecture of the CNN + LSTM image captioning model. The CNN encoder basically finds patterns in images and encodes it into a vector that is passed to the LSTM decoder that outputs a word at each time step to best describe the image. Upon reaching the <end> token or the maximum length of the sentence, the entire caption is generated and that is our output for that particular image.
![cnn+lstm model](assets/image_capt_cnn+lstm.png)

#### 1) Encoder: 
A pretrained CNN model (ResNet50) is used for feature extraction, transforming input images into fixed-length feature vectors.
#### 2) Decoder: 
An LSTM network is utilized to generate captions by taking image features and previous word embeddings as input to predict the next word.


## Vision Transformer (ViT) 
### What are transformers?
Before heading into the vision transformer, lets understand transformers.<br> Since the introduction of transformers in 2017 in the paper [Attention is all you need ](https://arxiv.org/abs/1706.03762)by Google Brain, it steered an interest in its capability in NLP
#### Transformer Architecture
![Transformer](assets/transformer_encoder_decoder.png)
<!-- 
The transformer is an architecture that relies on the concept of attention, a technique used to provide weights to different parts of an input sequence so that a better understanding of its underlying context is achieved. <br>
In addition, transformers process inputs in parallel making them more efficient and scalable in comparison to traditional sequential models such as RNN and LSTM. -->

**In the Transformer model:**

- Encoder: Converts input tokens into continuous representations using self-attention to capture relationships between all tokens simultaneously.
- Decoder: Generates output tokens by attending to both the encoder’s output and previously generated tokens, using masked self-attention and cross-attention.

### What are Vision Transformers?
Vision Transformers are models that apply the Transformer architecture to image data by treating image patches as sequences of tokens, enabling the capture of global context and complex dependencies.<br>
So, the task of image feature extraction that

![ViT](assets/ViT.png)










## Contributors

- [Sneha Singh](https://github.com/sneha31415) - sneha.singh.31415@gmail.com

- [Prithvi Tambewagh](https://github.com/rkt-1597) - patambewagh_b23@et.vjti.ac.in

- [Akash Kawle](https://github.com/shinymack) - ackawle_b23@et.vjti.ac.in

## Acknowledgements 
- [SRA VJTI](https://www.sravjti.in/) Eklavya 2024
  
A heartful gratitude to the mentors of this project:
- [Aryan Nanda](https://github.com/AryanNanda17)
- [Abhinav Ananthu](https://github.com/Herculoxz)
  <br/>


