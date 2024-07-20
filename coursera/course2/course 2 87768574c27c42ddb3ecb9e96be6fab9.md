# course 2

## Setting up your machine learning application

**Train / Dev / Test sets**

Setting up the training, development (dev, also called validate set) and test sets has a huge impact on productivity. It is important to choose the dev and test sets from the same distribution, The training set can come from a different distribution and it must be taken randomly from all the data.

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled.png)

train different models on test and check those models on dev set

**Bias and variance-**

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%201.png)

Bias refers to errors due to overly simplistic assumptions in the learning algorithm, while variance refers to errors due to excessive complexity in the model

by being a linear classifier, the data on the left is not getting fitted

high train set error = high bias

high dev set error = high variance (i.e overfitting has taken place)

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%202.png)

the above is valid only when the optimal(Bayes) error is quite small and the train and dev sets are from the same distribution

if the images are blurry, then even humas cant tell whether it is a cat or a dog pic so optimal (bayes) error is high. Thus at such cases even 15 % train or dev set error is considered okay and not high bias or variance

Both high bias and variance := 

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%203.png)

**Basic Recipe for Machine Learning**

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%204.png)

hight variance : Collect more training data to help the model generalize better and reduce overfitting

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%205.png)

## Regularizing your neural network

**Regularization**

L2 regularization(used more oftenly)

lambda here is called the regularization parameter. It is a hyperparameter that you might have to tune 

Note: lambda is a reserved keyword in python so we will be using lambd in our excercises

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%206.png)

`b` is just one parameter over a very large number of parameters, so no need to include it in the regularization. While W is a very large dimensional matrix

L1 regularization ends up making W to be sparse i.e W  vector will have a lot of zeros into it

**Regularization for a Neural Network**:

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%207.png)

In case of neural networks, after we have calculated the derivative of the cost func wrt Weights, we add the regularization term. **This regularization term penalizes large values of W**

The alternative term for L2 regularization is weight decay

For the matrix `w`, this norm is called the Frobenius norm. Its definition looks like `L2` norm but is not called the `L2` norm:

**Why Regularization Reduces Overfitting?**

frobenius norm penalizes the large vales of W

Simpler → means less layer neural network → more generalised 

use each of the hiiden layer but each will have low impact thus the neural network becomes less complex and generalises the data

If we make regularization lambda to be very big, then weight matrices will be set to be reasonably close to zero, effectively zeroing out a lot of the impact of the hidden units. Then the simplified neural network becomes a much smaller neural network, eventually almost like a logistic regression. We'll end up with a much smaller network that is therefore less prone to overfitting.

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%208.png)

If lambda is large, W lessens and thus Z also lessens. If Z is small, then every layer will be roughly linear and if each layer is roughly linear, the whole neural network is linear. Thus overfitting is avoided

So, complex and highly non - linear functions become simple func which avoids overfitting

### Comparing L1 and L2 Regularization

- **Sparsity:** L1 regularization tends to produce sparse models with many zero coefficients, which can aid in feature selection. L2 regularization generally results in smaller but non-zero coefficients.
- **Effect on Coefficients:** L1 regularization can completely eliminate less important features by setting their coefficients to zero. L2 regularization reduces the size of all coefficients but rarely sets them exactly to zero.
- **Use Cases:** L1 regularization is useful when there are many irrelevant features, promoting a simpler model. L2 regularization is effective when all features are potentially relevant but their weights need regularization.

**Dropout Regularization**

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%209.png)

keep-probs ⇒ indicates how many units of a layer are to be kept. So, 1 - (keep-probs) part of units will be removed(i.e zero out those hidden units)

On each iteration of the gradient descent, the units that will be zeroed will be different

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2010.png)

- Dropout is another powerful regularization technique.
- With dropout, what we're going to do is go through each of the layers of the network and set some probability of eliminating a node in neural network. It's as if on every iteration you're working with a smaller neural network, which has a regularizing effect.
- Inverted dropout technique, `a3 = a3 / keep_prob(scaling parameter)`, ensures that the expected value of `a3` remains the same, which makes test time easier because you have less of a scaling problem.

dividing by keep-prob   

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2011.png)

Keep key-prob lower when there is a higher chance of overfitting
key it high when there is a lower chance of overfitting, like in the layer where there are less parameters/ units, the keep-prob is kept high 

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2012.png)

**Understanding Dropout:**

- **Purpose**: Dropout is a regularization technique used in neural networks to prevent overfitting. It works by randomly ignoring (dropping out) a fraction of neurons during training. This forces the network to not rely too heavily on any one neuron, spreading out the learning and making the network more robust.
- **Implementation**: In Keras or similar frameworks, you typically specify a dropout rate (e.g., 0.2 means dropping out 20% of neurons) for each layer where dropout is applied.
- **Effect**: Dropout forces the network to learn redundant representations, reducing co-adaptation between neurons and improving generalization. It's particularly useful in large networks with many parameters, such as those used in computer vision tasks with high-dimensional input data (like images).

### Challenges and Considerations

- **Hyperparameter Tuning**: Setting the dropout rate involves choosing a balance between reducing overfitting and maintaining model performance. This rate needs to be tuned using techniques like cross-validation to find an optimal value.
- **Training Behavior**: Dropout can affect the behavior of the training process. Since neurons are randomly dropped during each iteration, the cost function (J) may not always decrease monotonically. It's essential to monitor the training progress and ensure that the loss function generally trends downward over time.
- **Debugging**: Dropout can sometimes make it harder to debug the model because the cost function is no longer well-defined in the traditional sense (since different neurons are dropped randomly). It's recommended to first train the model without dropout to ensure the basic training process is working correctly before implementing dropout.

### **what does dropout do?**

- **Before Dropout**: Neurons in the network are fully connected, and there's a risk of overfitting if certain neurons dominate in learning.
- **After Dropout**: Neurons are randomly dropped during training, forcing the network to learn more robust and generalized features.

*Note*:

- A **common mistake** when using dropout is to use it both in training and testing. You should use dropout (randomly eliminate nodes) only in training.

**Other Regularization Methods:**

- **Data augmentation**: getting more training data can be expensive and somtimes can't get more data, so flipping horizontally, random cropping, random distortion and translation of image can make additional fake training examples.
- **Early stopping**: stopping halfway to get a mid-size `w`.
    - *Disadvantage*: early stopping couples two tasks of machine learning, optimizing the cost function `J` and not overfitting, which are supposed to be completely separate tasks, to make things more complicated.
    - *Advantage*: running the gradient descent process just once, you get to try out values of small `w`, mid-size `w`, and large `w`, without needing to try a lot of values of the L2 regularization hyperparameter lambda.

## Setting up your optimization problem

### **Normalizing Inputs**

**When to normalize inputs?**
⇒ when the input features are in very different ranges like x1 = {0 to 1000} and x = {0 to 1}

subtract mean from x and then divide x by variance

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2013.png)

mean gets set to 0 and variance to 1

Due to  this, all the features get to a similar scale

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2014.png)

**Vanishing / Exploding gradients**

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2015.png)

- In a very deep network derivatives or slopes can sometimes get either very big or very small, maybe even exponentially, and this makes training difficult.
- **The weights W, if they're all just a little bit bigger than one or just a little bit bigger than the identity matrix, then with a very deep network the activations can explode. And if W is just a little bit less than identity, the activations will decrease exponentially.**

**Weight Initialization for Deep Networks**

role of weight initialization :

- **Proper Initialization**: By starting with weights that have a reasonable variance (e.g., `1/n` or `2/n`), the network is more likely to converge to a lower training (and generalization) error. Proper initialization can help in finding a good solution in the loss landscape.
- **Improper Initialization**: If the weights are poorly initialized, gradient descent might struggle to find a good minimum, potentially getting stuck in poor local minima or taking an excessively long time to converge.
- Preventing vanishing and exploding gradients.
- Ensuring proper signal propagation.
- Speeding up the convergence of gradient descent.
- Increasing the odds of finding a good minimum in the loss landscape.

how to train a deep neural network without the weights exploding to large values or depleting to zero

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2016.png)

A partial solution to the problems of vanishing and exploding gradients is better or more careful choice of the random initialization for neural network.

For a single neuron, suppose we have `n` features for the input layer, then we want `Z = W1X1 + W2X2 + ... + WnXn` not blow up and not become too small, so the larger `n` is, the smaller we want `Wi` to be.

- It's reasonable to set variance of `Wi` to be equal to `1/n`
- It helps reduce the vanishing and exploding gradients problem, because it's trying to set each of the weight matrices `W` not too much bigger than `1` and not too much less than `1`.
- Generally for layer `l`, set `W[l]=np.random.randn(shape) * np.sqrt(1/n[l-1])`.
    - For `relu` activation, set `Var(W)=2/n` by `W[l]=np.random.randn(shape) * np.sqrt(2/n[l-1])`. (aka He initialization by [Kaiming He](http://kaiminghe.com/))
    - For `tanh` activation, `W[l]=np.random.randn(shape) * np.sqrt(1/n[l-1])`. (Xavier initialization)
    - `W[l]=np.random.randn(shape) * np.sqrt(2/(n[l-1]+n[l]))` (Yoshua Bengio)
- `1` or `2` in variance `Var(W)=1/n or 2/n` can be a hyperparameter, but not as important as other hyperparameters.

**Numerical approximation of gradients**

Numerically verify implementation of derivative of a function is correct and hence to check if there is a bug in the backpropagation implementation.

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2017.png)

two sided difference formula is much more accurate for gradient checking

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2018.png)

**Gradient Checking**

**Gradient checking is a valuable technique to ensure that the backpropagation implementation in our neural network is correct. By numerically approximating the gradients and comparing them with the gradients computed by backpropagation, we can verify the accuracy of our implementation.**

**concatenate** into a giant vector theta

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2019.png)

for each example i , check if `d(theta) approx` is equal to `d(theta)`

![Screenshot from 2024-07-18 23-38-27.png](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Screenshot_from_2024-07-18_23-38-27.png)

1. `diff_ratio ≈ 10^-7`, great, backprop is very likely correct.
2. `diff_ratio ≈ 10^-5`, maybe OK, better check no component of this difference is particularly large.
3. `diff_ratio ≈ 10^-3`, worry, check if there is a bug.

**Gradient Checking Implementation Notes**

d(theta)approx takes a lot of time to compute so compute only in debugging time and not training time

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2020.png)

- for dropout, we can first check grad, then turn on dropout

**Quiz:**

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2021.png)

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2022.png)

## Week 2

**Mini-batch Gradient Descent**

for each little step of gradient descent, we need to process the entire training set
If m is large, this will take a lot of time
so we split the training set into mini batches(baby training sets) 

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2023.png)

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2024.png)

now, we will implement one step of gradient descent using X^{t}, Y^{t}

### Process Overview(no need to wait for entire training set to get processed)

1. **Initialization**: Initialize model parameters.
2. **Epoch Loop**: For each epoch:
    - **Shuffle Data**: Optionally shuffle the training data to ensure each epoch has a different order of training examples.
    - **Minibatch Loop**: For each minibatch:
        - **Select Minibatch**: Choose the next minibatch of data.
        - **Compute Gradient**: Calculate the gradient of the loss function with respect to the model parameters using only the minibatch data.
        - **Update Parameters**: Adjust the model parameters using the computed gradients.
    - **Repeat**: Continue until the entire dataset has been processed.

# Understanding Mini-batch Gradient Descent

the batch gradient descent must always and always decrease
But in case of mini-batch gradient descent, it is possible that X{2}, Y{2} is a harder batch than the X{1}, Y{1} batch (in ways like the X{2}, Y{2} mini batch might have more mislabeled things that increase the cost). Thus, the graph can go a bit noisy

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2025.png)

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2026.png)

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2027.png)

mini-batch size is a hyperparameter;

batch size better in [64, 128, 256, 512], a power of 2;

make sure that mini-batch fits in CPU/GPU memory.

**exponentially Weighted Averages**

DEFINITION:

Exponentially Weighted Averages (EWA), also known as Exponential Moving Averages (EMA), are used to smooth time series data or other sequences to identify trends and patterns by giving more weight to recent observations. This method is particularly useful in various fields such as finance, signal processing, and machine learning, especially in optimization algorithms like Adam.

### Key Concepts

1. **Weighting Factor (β)**:
    - The weighting factor β (between 0 and 1) determines the rate at which the influence of older observations decays. A higher β means slower decay (more weight to older values), while a lower β means faster decay (more weight to recent values).
2. **Recursive Calculation**:
    - EWA is calculated recursively using the current observation and the previous average. This allows for efficient computation without needing to store the entire history of observations.

### 

yellow ⇒ beta = 0.5 → that means  averaging over 2 days(so its more fluctuating)
green ⇒ beta = 0.98 → that means averaging over 50 days
red  ⇒ beta = 0.9 → averaging over 10 days

*** theta is the actual temp of $ith$ day

v is the exponentially weighted parameter

more is beta, more will it be affected by the temp of the previous day as per the formula Vt = beta(Vt-1) + (1-beta)(theta t)

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2028.png)

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2029.png)

**Understanding Exponentially Weighted Averages**

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2030.png)

just one line of code

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2031.png)

**Bias Correction in Exponentially Weighted Averages**

aim : making the computation of the averages more accurate

We need bias correction in the initial phase of learning

The expected average line is green but what we get is purple one. This is because initial average is zero, so v1 and v2 are very low. So, we use bias correction term = Vt/(1 - Bt)

The bias correction term is very less significant for higher values of beta so need not worry about that

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2032.png)

**Gradient Descent with Momentum** 

This smoothes out the steps of gradient descent. The oscillations in the vertical direction average out to something close to zero

Theory:

Because mini-batch gradient descent makes a parameter update after seeing just a subset of examples, the direction of the update has some variance, and so the path taken by mini-batch gradient descent will "oscillate" toward convergence. Using momentum can reduce these oscillations.

- gradient descent with momentum, which computes an EWA of gradients to update weights almost always works faster than the standard gradient descent algorithm.
- algorithm has two hyperparameters of `alpha`, the learning rate, and `beta` which controls your exponentially weighted average. common value for `beta` is `0.9`.

dw =  acceleration for going down the bowl
V_(dw) = velocity for going down the bowl

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2033.png)

V_dw and V_db is initialised as zero vector with dimensions same as W and b

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2034.png)

left formula for V_dw is more preffered

**RMSprop:**

RMSprop(root mean square), similar to momentum, has the effects of damping out the oscillations in gradient descent and mini-batch gradient descent and allowing you to maybe use a larger learning rate alpha.

We know that we have to slow down the learning rate in the vertical direction and fasten in the horizontal direction

so, The learning rate in the vertical direction must get low. So, Sdb must be large because then b will get updated slowly

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2035.png)

The algorithm computes the exponentially weighted averages of the squared gradients and updates weights by the square root of the EWA.

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2036.png)

**Adam Optimization Algorithm:**

- Adam (Adaptive Moment Estimation) optimization algorithm is basically putting momentum and RMSprop together and combines the effect of gradient descent with momentum together with gradient descent with RMSprop.

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2037.png)

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2038.png)

**Learning Rate Decay**

The learning algorithm might just end up wandering around, and never really converge, because you're using some fixed value for alpha. Learning rate decay methods can help by making learning rate smaller when optimum is near.

formula for learning rate decay 𝛼*α* is:

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2039.png)

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2040.png)

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2041.png)

**The Problem of Local Optima**

plateau ⇒ region where derivative is close to zero for a long time

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2042.png)

- First, you're actually pretty unlikely to get stuck in bad local optima, but much more likely to run into a saddle point, so long as you're training a reasonably large neural network, save a lot of parameters, and the cost function J is defined over a **relatively high dimensional space**.
- second, since the gradient is low at the plateau, the learning gets slow. And this is where algorithms like **momentum** or **RMSProp** or **Adam** can really help your learning algorithm.

![Untitled](course%202%2087768574c27c42ddb3ecb9e96be6fab9/Untitled%2043.png)

we can see in the image that the learning is quite slow due to plateau