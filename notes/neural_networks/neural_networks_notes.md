# Deep Learning

## Chapter 1: what is a neural network?

A **neural network** is a computational model which consists of **interconnected layers of nodes** (neurons) that can learn to perform tasks by adjusting some **parameters (weights and biases in this case)** connecting them together.

e.g. A basic form of this is a **Linear Regression** model which aims to find the line of best fit for the data which it is given, so the parameters here are the **slope and the y-intercept** of the line. In this model there is only one input layer and one output layer.

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