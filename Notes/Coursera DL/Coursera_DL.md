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