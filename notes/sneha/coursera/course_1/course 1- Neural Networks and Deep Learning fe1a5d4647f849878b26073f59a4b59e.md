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

### log is to the base e and not 10

when y = 1, for our loss func to be small, negative of log (y hat) must be small, thus log (y hat) must be large , thus y hat must be large, but y hat comes after applying sigmoid function so y hat must be 1 for the loss to be minimum

Similarly, for y = 0, y hat must be small, and the minimum it can become is zero

- **loss function is applied to a single training example**

The cost function is the average of the loss function of the **entire training set**. We are going to find the parameters 𝑤 𝑎𝑛𝑑 𝑏 that minimize the overall cost function.

The loss function measures how well the model is doing on the single training example, whereas the cost function measures how well the parameters w and b are doing on the entire training set.

**Gradient Descent**

The goal of the training model is to minimize the loss function, usually with randomly initialized parameters, and using a gradient descent method .

1. Start calculating the cost and gradient for the given training set of (x,y) with the parameters w and b.
2. update parameters w and b. Repeat these steps until you reach the minimal values of cost function.

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

If one wants to understand derivatives in a computational graph, the key is to understand derivatives on the edges. If a directly affects c, then we want to know how it affects c. If a changes a little bit, how does c change? We call this the partial derivative of c with respect to a.

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2015.png)

**Logistic Regression Gradient Descent**

Logistic regression gradient descent computation using the computation 

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2016.png)

The bottom-rightmost equation means that if on increasing W1, loss(L) increases(this total is denoted by {delL/delW1} or dW1) then dw1 will be positive so as per W1 = W1 - alpha dW1, we should decrease W1 because that would decrease loss 

**Gradient Descent on m Examples**

The cost function is computed as an average of the `m` individual loss values, the gradient with respect to each parameter should also be calculated as the mean of the `m` gradient values on each example.

This is the calculation for m examples, refer with the above calculation of one example to understand nicely

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2017.png)

```
j = j / m
dw = dw / m
db = db / m
since j, dw, db were over m examples, we need to average it
```

But this is less efficient due to for loops needed for each feature 

We will thus use vectorization to get rid of for loops

## vectorization:

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2018.png)

how vectorization speeds up code:

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2019.png)

np.random.rand(n) generates a list of n numbers each between 0 to 1
np.dot(a,b) computes the dot product of a and b

GPUs are faster due to parallelization but than can be done is CPUs using python inbuilt functions like numpy

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2020.png)

### More examples

example1:

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2021.png)

example2:

always try to see if you can use numpy instead of for loop

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2022.png)

learnings applied our logistic regression gradient descent implementation

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2023.png)

instead of taking dW1 , dW2 and so on all the examples seperately, we put them in an array and find the dot product.
Here we removed one out of the two for loops

### more vectorization

Z contains all the smallcase z

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2024.png)

similarly:

[https://www.notion.so](https://www.notion.so)

wT is a 1*m matrix, X is n*m(where n is the number of input variables) 

so their dot product is an 1*m matrix which contains all the small z

Summary: Instead of looping for each training example to compute lowercase a and z one at a time, we computed A, Z which computed all a and z at the same time

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2025.png)

the left one is the one with 1 for loop and the right one is with numpy(got rid of for loop )

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2026.png)

one for loop eliminated

## Broadcasting in python

### Broadcasting can happen only and only if:

1. they are equal, or
2. one of them is 1.

eg. 

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2027.png)

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2028.png)

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2029.png)

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2030.png)

axis = 0 means sum verticaly

axis = 0 means sum horizontally
reshape commands shapes the matrix to 1*4. Here this command is redundant. But when we aren’t sure of the size, we call reshape on the matrix  to make it to the size we need it to be. It is a cheap command so can be called to be sure with the size

matrix + some number → python expands the number into a matrix of the same size

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2031.png)

The above is also true for [1 2 3 4] i.e a column matrix

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2032.png)

In the above example, [100 200 300] is duplicated m-1 times

similarly

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2033.png)

### note:: broadcasting also takes place for multiplication and division

a*b is multiplication not dot product

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2034.png)

ans = 3*3

- Note on python/ Numpy Vectors

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2035.png)

a = np.random.randn(5)

creates a rank 1 array with shape (5, )

In neural network problems do not use a rank 1 array 

![Screenshot from 2024-07-06 18-41-34.png](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Screenshot_from_2024-07-06_18-41-34.png)

np.random.randn(5, 1) creates a definite column matrix 

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2036.png)

if you end up with a rank 1 array then do a = a.reshape(5,1) or (1,5) as per need

use assert function to check the dimension it is a cheap function

**Explanation of Logistic Regression Cost Function**

[https://www.notion.so](https://www.notion.so)

### some doubts cleared:

Query : what is dz and why is db the sum of it

In the context of neural networks and backpropagation, dZtypically represents the gradient of the loss function with respect to the linear component Z(which is Z=WX+bZ = WX + bZ=WX+b). This gradient, dZ, is computed during the backpropagation process.

Here's a more detailed breakdown:

1. **Forward Propagation**:
    - Z=WX+b
    - A=σ(Z)A = \σ(Z) (where σ is the activation function)
        
        σ\sigma
        
2. **Loss Computation**:
    - L=$L$(A,Y)  (where $L$ is the loss function and Y is the true label
3. **Backpropagation**:
    - To update the parameters W and b, you need to compute the gradients of the loss with respect to these parameters.
    - dZ=∂L/∂Z
    - dW=∂L/∂W
    - db=∂L/∂b

The gradient db is the derivative of the loss with respect to the bias term b. During backpropagation, the gradient with respect to the bias term b is computed as the sum of all elements in dZ divided by the number of examples m. This is because the bias term b affects all the outputs in the same way, and hence the gradient needs to accumulate the contribution from all training examples.

In mathematical terms, the derivative of the loss with respect to b is given by:

db=∑dZi/m for i = 1 to m

where m is the number of examples.

### Example

Here's a simple example to illustrate this:

```python
pythonCopy code
import numpy as np

# Suppose dZ is computed from the backpropagation step
dZ = np.array([[1.0, -2.0], [3.0, -4.0]])  # Example gradient array
m = dZ.shape[1]  # Number of examples, in this case, 2

# Compute db
db = np.sum(dZ) / m

print(db)  # Output will be 0.5 in this case

```

In this example, `dZ` contains the gradients of the loss with respect to the linear component `Z` for each training example. Summing `dZ` and dividing by `m` gives the gradient with respect to the bias term `b`, which is necessary for updating `b` during gradient descent.

REMEMBER

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2037.png)

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2038.png)

### ASSIGNMENT QUESTIONS:

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2039.png)

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2040.png)

### Note : * indicates element wise multiplication

assignment2

Flattening and standarding:

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2041.png)

# WEEK : 3

**() brackets will refer to the training example**

**square [ ] brackets will refer to the layer of the neural network**

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2042.png)

like logistic regression only but multiple times z and a calculation

**Neural Network Representation**

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2043.png)

**Computing a Neural Network's Output**

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2044.png)

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2045.png)

for hidden layer, b is a 4*3 matrix and for output layer, 1*4 matrix. Same for W

Also note that W = wT where wT is the transpose of w

So, for given input x of the neural network we can compute the output of the neural network with the given 4 lines of code

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2046.png)

**Vectorizing Across Multiple Examples**

**$a^{[2](i)}$** here 2 refers to the layer number and i refers to the $i^{th}$ training example

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2047.png)

X denotes  for 1 training  example  
$X^{(1)}$ denotes the first training example, and so on

### vectorization for the for loop

The uppercase matrices are made by stacking the lower case letters horizontally

horizontally we are going across different training examples
and vertical indexing corresponds to the different nodes in the neural network

So, The value on the left topmost corresponds to the activation of the first hidden unit on the first training example

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2048.png)

The right side shows the vectorized implementation

For 3 training examples:

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2049.png)

$Z^{[i]} = W^{[i]} + b^{[i]}$ is the correct vectorization 

RECAP: for loop  to  vectorization

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2050.png)

## **Activation Functions**

The purpose of the activation function is to introduce ***non-linearity into the network***

They are denoted by g

we can use different activation funcs for different layers, so we use superscripts to denote g1 g2 etc

cons of tanh(z) and sigmoid(z):

Both’s slope ends up close to being zero when the values of z are large. This slows down gradient descent. So relu was introduced

- How to choose which activation function to use?

1) If our o/p is 0 or 1 (binary classification) we can use sigmoid, mostly don’t use this
2) tanh(z) (superior to sigmoid but less used)
3) RELU for other- widely used
cons of RELU: when Z is negative, activation is zero, but this is fine in practical world. Neural network learns much faster in RELU
4) Leaky RELU(Z = negative then activation is non zero, less used tho)

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2051.png)

### why do you need non-linear activation functions?

if we use linear activation functions, then the neural network will just outputing a linear function of the input which is not enough to form a universal function approximator. Such a network can just be represented as a matrix multiplication, and you would not be able to obtain very interesting behaviors from such a network.

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2052.png)

so, if we use a linear activation func or if we dont have a linear activation function, then no matter how many layers are there, all its doing is just computing a linear activation function. This is equivalent to having no hidden layers

**THE COMPOSITION OF TWO LINEAR FUNCS IS ALSO A LINEAR FUNC , so hidden layers become useless**

One place where we can use linear activation functions is house price prediction. But the hidden layers should still use RELU , sigmoid etc

**Derivatives of Activation Functions**

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2053.png)

If z is large (say 10), then g(z) = 1, so slope = g(z) *(g(z) - 1) ⇒ 1 * (1-1) = 0*

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2054.png)

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2055.png)

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2056.png)

## **Random Initialization**

If we initialize all weights to zero, all the hidden layers get symmetric. This symmetry leads to no learning because all neurons in a layer are effectively identical and learn the same features. The network cannot break this symmetry and learn diverse features

Solution to this random initialization:

for activation functions such as sigmoid, tanh we need to initialize W with small quantities

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2057.png)

*Why small numbers?*

This is for sigmoid or tanh activation function. If weight parameters are initially large, we are more likely to get large values of `z` calculated by `z=wx+b`. If we check this in the graph of sigmoid(tanh) function, we can see the slope in large `z` is very close to zero, which would slow down the learning process since parameters are updated by only a very small amount each time.

# **Week 4: Deep Neural Networks**

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2058.png)

Notation:

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2059.png)

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2060.png)

There is no vectorization for l=4 to total number of layers
we have to use for loop for all layers

**Getting your Matrix Dimensions Right**

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2061.png)

for m examples, dimensions are: 

Z[1] = (n[1], m) 
W[1] = (n[1], n[0])
X = (n[0], m)
b[1] = (n[1], 1) so bias remains same for m examples also, but due to python broadcasting it becomes (n[1], m)

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2062.png)

**Why Deep Representations?**

- deep neural network over big neural network
- Deep neural network with multiple hidden layers might be able to have the earlier layers learn lower level simple features and then have the later deeper layers then put together the simpler things it's detected in order to detect more complex things like recognize specific words or even phrases or sentences.
- If there aren't enough hidden layers, then we might require exponentially more hidden units to compute in shallower networks.

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2063.png)

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2064.png)

**Building Blocks of Deep Neural Networks**

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2065.png)

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2066.png)

da[0] is not useful to calculate as it is derivative wrt input parameters

**Forward and Backward Propagation**

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2067.png)

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2068.png)

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2069.png)

**Parameters vs Hyperparameters**

*Parameters*:

- weight matrices `W` of each layer
- bias terms `b` of each layer

*Hyper parameters*:

- number of hidden units `n[l]`
- learning rate
- number of iteration
- number of layers `L`
- choice of activation functions

see the values of hyperparameters which fit the best

### What does this have to do with the brain?

![Untitled](course%201-%20Neural%20Networks%20and%20Deep%20Learning%20fe1a5d4647f849878b26073f59a4b59e/Untitled%2070.png)