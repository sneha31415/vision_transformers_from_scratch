# Coursera  Course

# Backpropagation :-

We now aim to find the derivative of final output variable (in this case, the cost ‘J’) w.r.t various intermediate variables. Hence we are going backwards in the computation graph, due to which this process is termed as ‘Backpropagation’. (going from right to left). 

In this process, we also use the Chain Rule of Derivatives.

For forward computations, we go fro  left to right, and, as contrary to backpropagation, attain the value of the final output variable.

# Logistic Regression Gradient Descent :-

For the method of Logistic Regression, we consider, at p[resent, 2 features, X1 and X2; so, matrix w will be [w1 w2]. Hence, we get the output of neural network as :

y_hat = sigmoid( X1w1 + X2w2+b ); 

Here, we consider everything w.r.t. only 1 training example, so fuinal output variable is L (Loss) instead of J (cost).

We employ backpropagation in the computational graph of this example of logistic regression, to get various derivatives, of the final output variable, L(a, y) (where a=X1w1 + X2w2+b) w.r.t. various intermediate variables.

![Screenshot from 2024-07-05 14-34-02.png](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-05_14-34-02.png)

Here also we follow the convention that derivative of a function f w.r.t. variable x is represented as dx, where f is the final output variable.

Using formulae of calculus, in the above example, values of da and dz are found out, as shown in the image above. Here also, we employ the chain rule as is shown in the formulation of dz, in above image. The derivatives of concern here, are dw1, dw2 and db, as they will be used in the further calculation of improvising the values of w1, w2 and b, like as shown in bottom right corner of above image.

# Gradient Descent on ‘m’ examples :-

![Screenshot from 2024-07-05 17-52-05.png](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-05_17-52-05.png)

![Screenshot from 2024-07-05 17-49-56.png](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-05_17-49-56.png)

We know that the cost function is the average of the loss function over all training examples. Hence, it turns out that the derivative of J w.r.t. any value of w (w1, w2, w3, … wm) or value of b, is the average, of the derivative of the loss function w.r.t. the same intermediate variable, over all training examples. The algorithm for the same is depicted in Fig. 

Algorithm explanation : 

Very firstly, we declare a variable for the loss function (here denoted as ‘J’), each value of dw (for each feature) and value of db.We initialize a for loop, for accessing training examples one by one. We then calculate the values of z(i) and a(i) as per their definitions. This new value of y_hat(i) i.e. a, is used, along with the correspnding value of y(i) i.e. expected output, in the calculation of the loss function and this value of loss function is added to its value from previous iterations. A for loop is also used for changing value of dw of each feature  acording to their respective formulae derived in section . After passing through this for loop, the values of J, dw (of each feature) and db are divided by m to get their average values and they are then used to vary the values of the elements of matrix w and value of b.

# Basics of Vcetorization :-

![Screenshot from 2024-07-05 18-52-27.png](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-05_18-52-27.png)

Vectorization are features (in-built) available in languages like Python. They help us to increase computation time significantly while providing the same result. This is because using vectorization, like the ‘dot’ function in numpy library, we are able to take advantage of SIMD (Single Instruction Multiple Data) so that processes run parallely and computation speed increases, as opposed to working of for loops, which kind of run sequentially, for each iteration, thus not taking complete advantage of SIMD and parallel computing.

For instance, we initialize 2 random arrays of 1 million dimensions - a and b, and store their dot product in c, using both methods - vectorization and for loops. It is observed that vectorization is clearly 50x (or even more) faster than using for loops.

```python
import numpy as np
import time as t

a=np.random.rand(1000000)
b=np.random.rand(1000000)
tic=t.time()
c=np.dot(a,b)
tac=t.time()
print(c)
print("Time by vectrization : " + str(1000*(tac-tic)) + " ms\n")

c=0;
tic=t.time()
for i in range(1000000):
  c+=a[i]*b[i]
tac=t.time()
print(c)
print("Time by for loop : " + str(1000*(tac-tic)) + " ms\n")
```

![Screenshot from 2024-07-05 18-58-26.png](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-05_18-58-26.png)

# More examples on vectorization :-/

![Screenshot from 2024-07-05 19-17-28.png](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-05_19-17-28.png)

![Screenshot from 2024-07-05 19-16-48.png](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-05_19-16-48.png)

![Screenshot from 2024-07-05 19-19-44.png](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-05_19-19-44.png)

There are several use cases of vectorization. As shown in Fig., we would need a for loop for multiplying corresponding elements of a matrix A and a vector v; however, the task could be sped up using vectorization, by first creating an array of zeroes, then applying vectorization and storing results in this new array.

Likewise, we can apply several mathematical functions to any vector/array using vectorizatin in a faster way, rather than using for loops. (Fig.)

Fig. shows the algorithm that is going to be used for gradient descent of our Lgistic Regression model. Instead of explicitly declaring variables for each element of matrix w, initialising it to zero and modifying them, we can create an array for all such variables and apply vectorization on it.

# Vectorising Logistic Regression :-

![Screenshot from 2024-07-05 20-13-11.png](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-05_20-13-11.png)

We now aim to apply vectorization to our Logistic Regression model. For this, we simply create vectors for values of z as well as activations i.e. a. then, we just implement the following lines of code : 

```python
Z=np.dot(w.T, X)+b
```

Here w.T means transpose of w (as it is a column vector), X is the matrix of features and b is a real number - the bias. As all operations in this line are matrix operations, hence, Python Interpreter automatically converts the number b to a 1*m dimensional matrix for the calculation. This is termed as ‘Broadcasting’.

# Vectorizing Logistic Regression’s Gradient Output :-

![Screenshot from 2024-07-05 20-54-19.png](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-05_20-54-19.png)

![Screenshot from 2024-07-05 20-54-43.png](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-05_20-54-43.png)

As shown in Fig., we initialize arrays for various values of z and y (expected output) to implement vectorization for dz. Then, according to the formula fr dz, using vectorization, we compute values of dw vector for all features and in the end, divide it by m. We observe that the computation of db is basically the average of the sum of elements of dz. We simplify it accordingly. As shown in Fig., we modify initial algorithm, implementing vectorization. However, we need to repeat this new algorithm for multiple iterations, which can be accommplished only using for loop.

# Broadcasting in Python :-

![Screenshot from 2024-07-06 13-41-38.png](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-06_13-41-38.png)

![Screenshot from 2024-07-06 13-42-09.png](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-06_13-42-09.png)

![Screenshot from 2024-07-06 13-43-48.png](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-06_13-43-48.png)

As shown in the general principle of brodcasting, If we give a (m, n) dimensional matrix and do any arithmetic operation (addition. subtraction, multiplication. division), then Python Interpreter automatically converts the other array/vector : if row vector of dimension m, then that row vector is stacked horizontally n times; if column vector of dimension n, then that row vector is stacked vertically m times. Then, the operation is done m times. Several examples are considered for this; like summing all columns of a matrix of nutritional components - calories from carbs, protein and fat, per 100g of 3 food items, and then, dividing original matrix by this 1*4 column vector to get percentage of the calories provided by each nuitritional component in each food item.

![Screenshot from 2024-07-06 13-48-54.png](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-06_13-48-54.png)

# Note on Python/Numpy Vectors :-

![Screenshot from 2024-07-06 13-59-46.png](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-06_13-59-46.png)

We should refrain from using rank 1 arrays due to thier non-intuitive behaviour. We can always reshape rank 1 arrays or any vector as vectors/matrices of required size. We can use assert statements to ensure matrices/vectors of correct size.

# Week 3

# Neural Networks Overview :-

![Screenshot from 2024-07-07 16-48-33.png](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-07_16-48-33.png)

Now, we just repeat what calculations we did for logistic regression mdel over several neurons. Such a group of stacked neurons is termed as ‘Layer’ and variable g of a layer 1 can be represented as g superscript [1]. g superscript (1) refers training example 1. We do calculations of z, a=y_hat in each layer, in each neuron and pass on the activations of all these neurons to those of the next layer. Derivatives, which are used in backpropagation, can also be represented in a fashion similar to that of variables as shown in red in above image.

# Neural Networks Representation :-

![Screenshot from 2024-07-07 16-58-46.png](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-07_16-58-46.png)

The input layer is indexed as 0, and subsequent indexing is applied to layers. Neurons/Nodes are numberd as 1, 2, 3, …, starting from the top. While describing a neural network, the input layer isn’t taken into consideration; hence, above neural network can be said to be a ‘2 layer Neural Network’. The dataset that we feed in the neural network, contains labels for input layer and output layer, but not hidden layers. Likewise, we have specific matrices w and b for each layer (as we are implementing logistic regression). The activation matrix can be formed for each layer as a vector/Python Matrix as shown for the layer 1 in above image.

# Computing a Neural Network’s output :-

![Screenshot from 2024-07-07 17-29-54.png](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-07_17-29-54.png)

Every neuron of the netwrok for our logistic regression model, performs these 2 operations - calculation of z and calculation of a (activation). If we consider each neuron individually (for layer 1 at present), we can form equations of ‘z’ and ‘a’ similar to those of our previous neural network’s model, by giving proper superscript and subscript according to layer index and neuron index respectively. Hence, we get matrices for w and b for first layer.

![Screenshot from 2024-07-07 17-30-09.png](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-07_17-30-09.png)

We can group the values of w, b and z, as well as those of a and sigmoid(z) together to form corresponding matrices. Hence, by using vectorization, we get : 

z[1]=(W[1].T)(X)+b …(each term is a vector/matrix; W[1]=w[1].T)

Extending the same idea to layer 2, we get final 4 equations for this Neural Network :

![Screenshot from 2024-07-07 17-30-18.png](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-07_17-30-18.png)

We can also represent X as ‘a’ superscript [0]

# Vectorizing across multiple examples :-

![Screenshot from 2024-07-07 20-11-38.png](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-07_20-11-38.png)

We now take int consideration m examples in the X vector, as well as the neural network. Note, that capital letters denote vectorized version. We can dentoe any variable related to ith training data with superscript of round brackets with i. Non-vectorized version of this prblem suggests use of for loop from 1 to m. 

![Screenshot from 2024-07-07 20-14-40.png](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-07_20-14-40.png)

Now, in vectorizd versin we will just take into consideration that each matrix will now have m times as much columns as it previously had. Hence, ultimately we can get activation of each of the neuron, for each training example. Each matrix has as much rows as there are hidden units in previous layer and as much columns as there are total number of training examples.

# Activation Functions :-

![Screenshot from 2024-07-07 21-56-21.png](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-07_21-56-21.png)

We have several options for activation functions. One of them is a=tanh(z). It can be thought of to be a version of the sigmoid function, but centred around zero. This characteristic makes it preferrable as we try to have a zero mean for our data. However, as z becomes very large or very small, the derivative of tanh(z) becomes very small, slowing down the gradient descent process. Sigmoid function is still preferrable for output layer, as it gives output between 0 and 1, which can be usefuk in the case of binary classification. Another example is ReLU - Rectified Linear Unit. It has the advantage that it has slope 1 for all positive values of z and slope 0 for all negative values. The slope at z=0 is not defined, however, in actual computations, the odds of attaining the point z=0.00000000 is very small, so thats not a thing of concern. ReLU is mostly used in practice, as the slope doesn’t cause the gradient process to slow down as z→0. There is another version of ReLU i.e. Leaky ReLU, which as a slight slope for ngative values of z. The parameter of 0.01 in Leaky ReLU can be changed in the process of ML.

# Need of non-linear activation functions :-

Linear / Identity activation function has the form : g(z)=z.

Without activation fn the neural network is just computing linear actiavtion function based on the input, irrespective of the no. of hidden layers, making extra hidden layers redundant.Hence, if we use nin-linear activation function just at the output, then it is equivalent to a standard logistic regression model without any hidden layer. However, in case of regression (linear regression) it can be useful to hve linear activation functions, as the output y_hat can have any real value and not just 0 or 1 like for binary classification.

![Screenshot from 2024-07-08 12-13-13.png](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-08_12-13-13.png)

# Derivatives of various activation functions :-

![Screenshot from 2024-07-08 12-10-14.png](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-08_12-10-14.png)

![Screenshot from 2024-07-08 12-10-55.png](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-08_12-10-55.png)

![Screenshot from 2024-07-08 12-11-19.png](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-08_12-11-19.png)

For ReLU and Leaky ReLU we can choose any of the 2 values of slope, of z>0 and z<0, at z=0.

# Gradient Descent for Neural Networks :-

We must initialize all parameters to random numbers rather than to zero.

![Algorithm (same as previous neural network)](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-08_13-15-55.png)

Algorithm (same as previous neural network)

![Screenshot from 2024-07-08 13-16-25.png](Coursera%20Course%20ad96fa4029d842568f1526cf6c66dc59/Screenshot_from_2024-07-08_13-16-25.png)

# Random initialization of parameters :-

In Logistic regression, initialization of parameters w and b to 0 won’t be a problem; but not so in Neural Network, where it will still be alright if b is initialized to zero but not w. This is because, by principle of induction, we can state that after several iterations as well, the matrix W[1] will have identical rows.