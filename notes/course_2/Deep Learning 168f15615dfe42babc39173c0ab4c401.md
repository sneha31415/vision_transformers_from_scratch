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