<p>
<h1 align = "center" > <strong>Vision Transformers From Scratch</strong> <br></h1>

<h2 align = "center">

</p>

[SRA](https://www.sravjti.in/) Eklavya 2024 ✨<br></h2>

<!-- ABOUT PROJECT -->
# About the project
## ⭐ Aim
The aim of this project is to implement Vision Transformers (ViT) from scratch for image captioning, demonstrating their superiority over traditional CNN + LSTM models in generating more accurate and descriptive captions. 

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



<!-- ### 🚀 Project Description 
Transformers are state-of-the-art (SOTA) model architectures, with models like GPT, BERT, T5, XLNet, and RoBERTa excelling in natural language processing tasks such as understanding, generation, and translation.

---

#### An Image is Worth 16x16 Words 

Despite their success in NLP, the use of Transformers in computer vision (CV) is still emerging. This project aims to explore the Transformer architecture for CV applications:

1. **Introduction to Deep Learning Models** :
   - Basics of naive deep-learning models.

2. **Sequential Data Processing** :
   - Using RNNs and LSTMs.

3. **Vision Transformers** :
   - Understanding and implementation.

4. **Image Captioning Model** :
   - Generating descriptive captions for images.
 -->

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


