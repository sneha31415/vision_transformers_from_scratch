# Deep Learning

# Course 2: Improving Deep Neural Network: Hyperparameter Tuning, Regularization and Optimization

## Train, Validation (Dev) and Test sets:

First the model trained using the training set and then we can implement different types of algorithms/model and test them on the dev test, and then do the final evaluation using test set.

Previously, there was 70/30 train/test data split, But now as there is access to more data (>1 million training examples) so the test set is now only 1-2 %.

Note: dev and test sets should be from the same distribution.

It is also okay to not have test set at all, we can just do both evaluate different models on the dev set itself.

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled.png)

---

## Bias / Variance

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%201.png)

We can identify if we have a high bias or a high variance issue from the train set and dev set errors. 

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%202.png)

For high bias and high variance the model is underfitting and overfitting the data at the same time.

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%203.png)

In the past, there used to be a **tradeoff** between reducing bias and reducing variance.

In the era of deep learning and big data, increasing the network size and obtaining more data can reduce bias without significantly increasing variance, as long as appropriate **regularization** is applied.

---

## Regularization

We can use regularization to prevent overfittng.

This is how we implement L2 Regularization for Logistic Regression:

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%204.png)

L2 is preferred over L1

Here we use lambda which is the regularization parameter, which by adding to the cost function penalizes for large parameter values.

For Neural Networks:

the regularization term is added to the gradient descent update equation for the weight parameters. Regularization helps to prevent overfitting by shrinking the weights and making the model less complex.

It is also called weight decay.

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%205.png)

---

### How does regularization prevent overfitting?

By changing the values for lambda we take the values of some weights to zero (not completely but reducing their impact sounds more good)  making it less over fitting and takes it towards underfitting but there is a state in between where it fits just right.

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%206.png)

We take the example for the tanh activation function: by reducing the weights we push the activation towards the linear part of tanh, so the network becomes more linear which makes it less overfitting.

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%207.png)

---

## Dropout Regularization

We have probabilities for eliminating nodes for each layer, these probabilities are also changed for each example so we are training a different diminished network for each example.

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%208.png)

### Inverted dropout (Implementation for dropout)

The implementation of dropout involves setting a probability for eliminating nodes in each layer of the network. We use a random matrix to determine which nodes to keep and which to eliminate. 

**Note:** In this we also divide the activations by the keep probability to maintain the expected value of the activations. 

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%209.png)

During Test time, we don’t use dropout.

**Intuiton** behind dropout :

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2010.png)

For layers with less neurons we don’t care about overfitting so we give them a higher keep probability.

### Other Regularization methods

- We can flip out images horizontally to get double the size of out dataset.
- We can also randomly zooming in our images
- We can impose random rotations or distortions.

**Early Stopping:** 

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2011.png)

---

## Normalizing Inputs

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2012.png)

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2013.png)

---

## Vanishing / Exploding Gradients

These problems occur when the derivatives or slopes of the network's parameters become very large or very small, making training difficult.

If the weight matrices are slightly larger than the identity matrix, the activations can explode exponentially. On the other hand, if the weight matrices are slightly smaller than the identity matrix, the activations decrease exponentially.

---

### Weight Initialization for Deep Networks

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2014.png)

---

## Gradient Checking

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2015.png)

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2016.png)

This is only done for debugging.

Doesn’t work with dropout.

---

# Optimization Algorithms

## Mini-batch Gradient descent

Rather than applying gradient descent for the whole training set, we can divide it into mini batches which is more efficient, this technique also make the training way faster (i.e. faster convergence to minima of cost function).

So we run a for loop for gradient descent having (m / mini_batch_size) iterations.

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2017.png)

For a large training set, mini batch GD will always be faster than batch GD.

We can even multiple iterations of this mini batch GD for better results.

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2018.png)

We do not get a smooth curve for this method because there are variations within every batch, like some may be easy or hard.

We set a hyperparameter - **mini_ batch_size**

- If mini_batch_size = m, it is just Batch Gradient Descent
- If mini_batch_size = 1, It is called Stochastic G.D which was taught previously, In which we apply gradient descent for every training example. ( No vectorization is used as only one example is used at a time).

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2019.png)

Choosing the mini-batch size:

- If small training set ( m < 2000 ): Use Batch g.d
- Typical mini-batch size: 64, 128, 256, 512, Mini batch should fit in the CPU/GPU memory only then it would be efficient.

---

## Exponentially weighted averages

Moving average:  Vt = βVt-1 + (1-β)θt, where β is a parameter that determines the weight given to the previous value.

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2020.png)

β = 0.9 is red, β = 0.98 is green, β = 0.5 is yellow

Beta determines the number of days over which the average is computed.

so if beta = 0.9,  this is approximately averaging over 10 days of temp. data. 

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2021.png)

For computing we use the same formula in the for loop and replace the values every time so it is time and memory efficient.

---

### Bias Correction in EWA

When implementing exponentially weighted averages, there is a bias in the initial estimates. The initial estimates tend to be lower than the actual values. Bias correction helps to make the estimates more accurate, especially during the initial phase.

To correct this, the estimate is divided by 1 -  beta^t, where t is the current day. 

for going from purple line to the green line initially.

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2022.png)

---

## Gradient Descent with Momentum

It smooths out the steps of gradient descent, reducing oscillations in the vertical direction and allowing for faster learning in the horizontal direction.

It will always work better than normal gd algorithm.

We use the EWA of the gradient terms.

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2023.png)

Beta = 0.9 works pretty well and considered common choice.

Bias correction is not usually necessary in practice.

In some literature the 1-beta term is omitted.

---

## RMSprop

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2024.png)

---

## Adam Optimization Algorithm

Momentum + RMSprop

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2025.png)

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2026.png)

---

## Rate Decay

The concept of learning rate decay involves gradually reducing the learning rate over time to achieve better convergence and avoid getting stuck in local optima.

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2027.png)

Another type of decay is exponential decay:

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2028.png)

---

## The Problem of Local Optima

We learn that in high-dimensional spaces, most points of zero gradient in a cost function are actually saddle points rather than local optima

Plateaus can also slow down learning, and algorithms like momentum, RMSprop, and Adam can help overcome this challenge.

---

## Tuning Process

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2029.png)

Alpha is most important, then beta, no of hidden units and mini_batch_size.

The Adam parameters works good with their default values,

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2030.png)

We can try random values for two hyperparameters simultaneously using random points as coordinates, Using values off of a grid is not good because we get to use less values for a parameter as one stays the same while the other changes along a line.

This can be done for multiple hyperparameters using a multidimensional vector.

### Using an Appropriate Scale to pick Hyperparameters

If we want a random number between 0.0001 and 1 then we will bet 90% of the numbers from 0.1 to 1, so to get uniformity among the random selections we can use log scale rather than a linear scale.

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2031.png)

So here we are randomly sampling r using -4*np.random.rand() which gives values form [-4,0] then using alpha = 10^-r to get random values of alpha uniformly.

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2032.png)

---

### Panda vs Caviar Approach

There are two major approaches to searching for hyperparameters: the "**panda approach**" and the "**caviar strategy**." 

The panda approach involves **babysitting one model** and gradually adjusting hyperparameters based on its performance. This approach is suitable when **computational resources are limited.** 

The caviar strategy involves **training multiple models in parallel** with different hyperparameter settings and selecting the best-performing one.

---

## Normalizing Activations in a Network

We previously used normalization on weights and biases to efficiently train them.

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2033.png)

**Batch normalization** is an algorithm that helps in the training of neural networks by normalizing the mean and variance of hidden unit values.

### Implementing Batch Norm

The algorithm computes the mean and variance of the hidden unit values and then normalizes them. It also introduces learnable parameters, gamma and beta, which allow for control over the mean and variance of the hidden unit values. 

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2034.png)

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2035.png)

This Beta is different from the one used in Adam, RMSprop
In most deep learning frameworks batch norm can be done in one line.

Since we subtract the mean in normalization step, the effect of b (bias) get subtracted as it is a constant, so we can omit the use of bias while usig batch norm.

so here beta is actually doing the job of the bias b.

**We also apply gradient descent to beta and gamma.**

---

### Why does Batch Norm work?

It works by reducing the amount that the distribution of hidden unit values shifts around, making the later layers of the neural network more stable. 

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2036.png)

Here if we train our model on the images of black cats and test it on these images of coloured cats it might not perform well even when the function is same but the distribution of X has changed, this is called covariate shift.

It reduces the problem of covariate shift, where the distribution of input data changes, and allows each layer to learn more independently. As a result, batch normalization speeds up the learning process and improves the overall performance of the network.

The mean and variance is a little bit noisy because it's estimated with just a relatively small sample of data (each mini-batch). So similar to dropout, it adds some noise to each hidden layer's activations.

Batch norm also has a slight regularization effect:

### Batch norm at test time

During testing, a separate estimate of the mean and variance is needed. This is typically done using an exponentially weighted average across mini-batches. The estimated mean and variance are then used to scale the data during testing.

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2037.png)

---

## Softmax Regression

generalization of logistic regression for multiple classes. 

The Softmax layer computes the linear part of the final layer and applies the Softmax activation function to generate the output probabilities for each class.

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2038.png)

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2039.png)

---

![Untitled](Deep%20Learning%20168f15615dfe42babc39173c0ab4c401/Untitled%2040.png)