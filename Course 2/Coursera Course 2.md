# Coursera Course 2

# Tuning Process

![Untitled](Coursera%20Course%202%20e115727e39824e80a5f8c8505605c92e/Untitled.png)

There are several types of hyperparameters used in deep nets as shown above. We can organise them priority wise.

One way to find the best values, for instance out of 2 hyperparameters, is to plot some random points in a 2-D space, where each axis denotes each of those 2 hyperparameters. Choosing uniform points leads to inaccuracy due to redundancy of any of the hyperparameters. Hence we choose random points. Same idea can be extended for multiple hyperparameters. Also, after we find a best point in this grid, optionally, we can choose to repeat this process for a small region around it. This is known as going from coarse to fine.

# Scale of Hyperparameter tuning

For scaling integral hyperparameters we can scale randomly e.g. no. of hidden units/layers. But for hyperparameters like learning rate, which can lie between 0.0001 and 0.1, most of the points would occur between 0.001 and 0.1; so in such case we use logarithmic scales.

![Untitled](Coursera%20Course%202%20e115727e39824e80a5f8c8505605c92e/Untitled%201.png)

![Untitled](Coursera%20Course%202%20e115727e39824e80a5f8c8505605c92e/Untitled%202.png)

Even if scaling goes less accurately, still it can be improved by coarse to fine method.

# Panva vs Caviar Approach

The hyperparameters must be periodically checked for their effectiveness.

![Untitled](Coursera%20Course%202%20e115727e39824e80a5f8c8505605c92e/Untitled%203.png)

Panda approach -

- When less computational resources
- Vary parameters by checking past progress
- If found poor progress can change model and retrain, but only 1 model at a time.

Caviar Approach -

- When large number of computational resources available, train a model with a set of hyperparameters on each
- Map their overall progress (of data error/cost function w.r.t. time) and choose best set of hyperparametrs.
- This approach is preferred

# Normalising Activations in a network

![Untitled](Coursera%20Course%202%20e115727e39824e80a5f8c8505605c92e/Untitled%204.png)

We tend to have zero mean and unit standard variance for input layer in logistic regression for better optimization of W and b. Likewise, we implement it on the value z of each layer of DNN (this is also done for activations rather than z, but more commonly for z). Known as ‘Batch Norm’.We can apply the computations for zero mean and unit standard variance for z[l] and then modify the equation using the hyperparameters gamma and beta. This is because we may necessarily want zero mean and unit standard variance due to which activation of that layer, for e.g. sigmoid function, will lie in the central linear range.