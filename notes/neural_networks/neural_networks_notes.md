# Deep Learning

## Chapter 1: what is a neural network?

A **neural network** is a computational model which consists of **interconnected layers of nodes** (neurons) that can learn to perform tasks by adjusting some **parameters (weights and biases in this case)** connecting them together.

e.g. A basic form of this is a **Linear Regression** model which aims to find the line of best fit for the data which it is given, so the parameters here are the **slope and the y-intercept** of the [line.IN](http://line.IN) this model there is only one input layer and one output layer.

Convolutional Neural Network(CNN) → Good of image recognition

Long short-term memory network(LSTM) → Good for speech recognition

**The type of Neural Network used in this playlist is a Multilayer Perceptron.**

### Neuron:

It is something that holds a number which is called its “**activation**”, it can pass those activation to the next layers by the connection which it has with them.

In the video, 3b1b uses images of handwritten digits from the MNIST database and the **first layer** of the N.N, takes the pixel values of the **28x28** image as **784** neurons.(0.00 → black, 1.00→white).

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled.png)

The **output layer** is just **10** neurons for **0-9,** and the activation of those tells us what digit the neural network thinks the input is.

3b1b takes **2 hidden layers with 16 neurons each**, which is solely based upon our choice.

The activation’s of one layer bring about the activation in the next layer.

### Why a layered structure?

- Lower layers learn simple features, such as edges or textures in an image.
- Higher layers combine these simple features to recognise more complex patterns, such as shapes or objects.
- **Example:**
    - In image recognition, the first layer might detect edges, the second layer might detect corners or textures, and the final layers might recognise parts of objects or even entire objects.
    
    ![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%201.png)
    
    ![Untitled.png](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%202.png)
    
    ![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%203.png)
    
    ### Activation of a neuron:
    
    The activation of each neuron in the next layer is linked to previous layer neurons by some weights.
    
    ![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%204.png)
    
    ![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%205.png)
    
    We take the **weighted sum of the activations of the previous layers** to find the activation of a neuron.
    
    We also add a **bias** to the weighted sum if we want a threshold value to be passed before **activation function** is applied.
    
    So for this model, for the first layer after the input layer there are **784 weights per neuron, and 1 bias per neuron.**   
    
    We use some functions known as activation functions to squish the activation value to some determined range here that is 0 to 1.(This is done to introduce **non-linearity** into the model).
    
    **This can be shown as Matrix Multiplication:**
    
    ![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%206.png)
    
    ### Activation functions:
    
    1. **Sigmoid function**
        
        ![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%207.png)
        
    2. **ReLU (Rectified Linear Unit)**
        
        Most modern networks use ReLU because it provides better learning that sigmoid.
        
        ![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%208.png)
        
    3. **tanh**
    
    ---
    
    ## Chapter 2: Gradient descent, how neural networks learn
    
    It is used to minimize the cost function in neural networks.
    
    Initially all the **weights and biases are set at random**.
    
    Its objective is to adjust the parameters (weights and biases) of the network to reduce the error between predicted and actual outputs.
    
    ### Cost Function:
    
    As we have a labeled dataset, i.e. knowing the correct activations for the last layer we can calculate the loss for each neuron in the output layer.
    
    So the cost is actually the **Mean Squared Error** for all the neurons in the output layer.
    
    It can be said that it is a function which takes all the weights and biases and input and gives out the cost, with its parameters being all the training examples.
    
    ### Gradient Descent:
    
    Gradient of a function gives a vector pointing in the direction of fastest ascent, so negative gradient in the direction of fastest descent i.e to find the local minimum.
    
    Each step in changing the weights and biases in proportional to the gradient of cost function, so we reach the minimum without overshooting.
    
    ![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%209.png)
    
    The algorithm used to make these changes to weights and biases is called ***Backpropagation.***
    

---

## Chapter 3: What is backpropagation really doing?

Initially when the weights and biases are all random the output for a training example is going to look like this:

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%2010.png)

Now we know what adjustments should be made,

For example we consider this training example in which the “2” neuron which has 0.2 activation which should be 1.0.

So we pay attention to its connections to the previous layer,

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%2011.png)

Here the weights of the connection with higher activation neurons in the previous layer has a greater impact than neurons wit lower activation.

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%2012.png)

Now we get what changes should be made to weights corresponding to each connection for the last second layer.

Once we have list of those changes we can apply the same to third to last later. This can be applied recursively to all layers before them too.

This process is repeated to for all the training examples.

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%2013.png)

This average of nudges is somewhat proportional to the negative gradient of the cost function.

### Stochastic Gradient Descent:

In SGD, instead of using the entire dataset for each iteration, only a single random training example (or a **small batch**) is selected to calculate the gradient and update the model parameters. This random selection introduces randomness into the optimization process, hence the term “stochastic” in stochastic Gradient Descent.

This way of dividing into mini batches does not take as much computation power as we would need to train from the whole dataset at once.

---

## Chapter 4: Backpropagation Calculus

For example we consider a network with only one neuron in each layer.

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%2014.png)

and lets focus on only last 2 layers of the network.

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%2015.png)

So here y being the desired output the cost is the squared error, with the notations for the neurons as shown in the image.

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%2016.png)

This tree diagram shows relation between everything.

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%2017.png)

Applying the chain rule we get something line this,

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%2018.png)

we also find the sensitivity of the cost function to the biases.

Still this is one component of the gradient vector, as it takes for all weights and biases.

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%2019.png)

Now we can apply the same chain rule idea to layers before that.