# course 1- Neural Networks and Deep Learning

## overview:

week 1: Introduction to deep learning

week 2: Neural network basis

week 3: shallow(one hidden layer) neural networks

week 4: Deep neural networks

## **Week 1: Introduction to Deep Learning:**

**What is a neural network**

simple neural network

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled.png)

bit complex neural network

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%201.png)

we need to give only x, and y and all the hidden layer things will be guessed by it 

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%202.png)

**Supervised learning with neural networks**

In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.

*Examples of supervised learning applications*:

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%203.png)

image → use CNN

audio and language→ sequence data(1 D )→ we use RNN

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%204.png)

*#### Structured vs unstructured data*

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%205.png)

- Structured data refers to things that has a defined meaning such as price, age
- Unstructured data refers to thing like pixel, raw audio, text.

**Why is deep learning taking off**

Deep learning is taking off now due to a large amount of data available through the digitization of the society, faster computation and innovation in the development of neural network algorithm.

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%206.png)

*Two things have to be considered to get to the high level of performance*:

1. Being able to train a big enough neural network
2. Huge amount of labeled data

### need of faster computation:

Algorithms- make neural networks run faster

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%207.png)

one recent example is the sigmoid to relu function

In the sigmoid function, where the gradient approaches zero, the parameters change very slowly and learning gets slow. So relu makes gradient descent work much faster. Thus fast computation makes bigger neural network learn faster

in this course , m → no. of training examples

## Week 2

**Binary Classification**

here a Cat vs. Non-Cat classification, which would take an image as an input and output a label to propagation whether this image is a cat (label 1) or not (label 0).

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%208.png)

An image is store in the computer in three separate matrices corresponding to the Red, Green, and Blue color channels of the image. The three matrices have the same size as the image, for example, the resolution of the cat image is 64 pixels x 64 pixels, the three matrices (RGB) are 64 by 64 each. To create a feature vector, x, the pixel intensity values will be “unroll” or “reshape” for each color. The dimension of the input feature vector x is .

$n_x$ = 64 * 64 * 3

64 * 64 is the total pixels in an image. The input feature x contains all the 64 pixels of all the three images one below other