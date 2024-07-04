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

### Notation

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%209.png)

**Logistic Regression Cost Function**

Given a cat picture X, we want to know if its a cat picture or not (i.e Y =1 or 0) 

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2010.png)

What are the parameters of logistic regression?

**W, an 𝑛𝑥 dimensional vector, and b, a real number.**

In Logistic regression, we want to train the parameters `w` and `b`, we need to define a cost function.

The loss function measures the discrepancy between the prediction (𝑦̂(𝑖)) and the desired output (𝑦(𝑖)). In other words, the loss function computes the error for a single training example.

Justification for loss function:

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2011.png)

when y = 1, for our loss func to be small, negative of log (y hat) must be small, thus log (y hat) must be large , thus y hat must be large, but y hat comes after applying sigmoid function so y hat must be 1 for the loss to be minimum

Similarly, for y = 0, y hat must be small, and the minimum it can become is zero

- **loss function is applied to a single training example**

The cost function is the average of the loss function of the **entire training set**. We have to find the parameters 𝑤 𝑎𝑛𝑑 𝑏 that minimize the overall cost function.

The loss function measures how well the model is doing on the single training example, whereas the cost function measures how well the parameters w and b are doing on the entire training set.

**Gradient Descent**

The goal of the training model is to minimize the loss function, usually with randomly initialized parameters, and using a gradient descent method .

Derivative = slope of function at that point(dJ / dW)

In the right side of graph, dJ/dw is positive so W = W - (some positive), so W decreases and moves to the global minimum eventually

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2012.png)

Notation: if J is a function of one variable only, then derivative is shown using small d
If J is a function of more than one variable we use partial derivative(dell )

- In real J is function of both w, b

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2013.png)

**Derivatives**

Derivatives are slopes

Slope can be different at diff points of the curve

**Computation Graph**

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2014.png)

**Derivatives with a Computation Graph**

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2015.png)

**Logistic Regression Gradient Descent**

Logistic regression gradient descent computation using the computation 

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2016.png)