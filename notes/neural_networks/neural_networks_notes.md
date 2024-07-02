# Deep Learning

## Chapter 1: what is a neural network?

A **neural network** is a computational model which consists of **interconnected layers of nodes** (neurons) that can learn to perform tasks by adjusting some **parameters (weights and biases in this case)** connecting them together.

e.g. A basic form of this is a **Linear Regression** model which aims to find the line of best fit for the data which it is given, so the parameters here are the **slope and the y-intercept** of the line, In this model there is only one input layer and one output layer.

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

---

## Chapter 5: Intro to Transformers

Specific kind of neural network mostly used for voice to text, text to voice, text to image and language translation.

GPT - Generative Pre-Trained Transformer

It takes some text, image, voice and predicts what comes next.

It first takes some snippet of text then appends the word predicted and passes on that snippet again.

It predicts the next work as a form of probability distribution.

So this is what a Vision Transformer would do,

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%2020.png)

Here the tunable parameters are simply the weights and biases.

### Tokenization:

The first step in a transformer is to break the input/prompt into small pieces called tokens.

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%2021.png)

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%2022.png)

Then each token is assigned a vector (matrix) of a very high dimension which encodes the meaning the word without context.

In this video, we take example of the GPT-3 model,

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%2023.png)

These are all the types of matrices used in this model or just a standard transformer model, 

### Embedding Matrix:

It is a matrix of all pre-defined words in the language with each being assigned its particular vector.

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%2024.png)

Each column is for a word, and the values are generated at random at first but then tuned using data.

Each token in our input is assigned a vector based on this embedding matrix, it can be considered that each direction in this embedded vector refers to some context like age, number,location, gender etc.

If we take a 3d slice of the high dimensional vector we can represent the words like this:

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%2025.png)

Words with meanings close to each other are closer to each other,

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%2026.png)

And this signifies the sense of direction in this huge matrix somewhat,

Here to know how close the words are we can simply do the dot product.

This vectors assigned to the tokens also contain their positional embeddings.

Initially when the tokens are assigned the vectors from the embedding matrx it only consists of their meaning individually without any context of its surroundings.

The model makes multiple repetitions of Attention and MLP blocks which will be explained later.

Basically they add some more context in the vectors about each other.

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%2027.png)

In the final Matrix after all these repetitions the last column( the last token) get updated for the particular context.

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%2028.png)

A dot product of this last column in taken with an Unembedding matrix with all the 50k words, so this gives us how close is a word to some other word.

So to convert the last output matrix into a probability distribution we use ***softmax.***

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%2029.png)

### Softmax with Temperature:

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%2030.png)

The input to this softmax is also called **Logits.**

---

## Chapter 6: Attention in transformers

We consider tokens to be just full words for learning purposes but in reality they can also be parts of words, depending on the model.

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%2031.png)

The Attention Mechanism adds context in all the vectors for the word mole. Initially the token embedding has no context.

This attention process can be considered like asking questions like are there any adjectives beside a word, so for that a **Query** matrix in generated for a token,

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%2032.png)

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%2033.png)

For now we suppose that this query matrix also transforms the 12288 dimensional embedding space to 128-dimensional query/key space.

There is also another matrix similar to the query, a **Key** matrix, we can consider it supposedly like answering the query matrix. we also multiply this matrix to out embedded token.

Now we arrange the outputs of this, in a table for the Keys and Querys and their corresponding dot products.

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%2034.png)

This dot products give us the score by which some word enhances the meaning of the other word,

We also apply softmax to normalize the values.

But also the later words should not influence earlier words since the model simultaneously also predict what would come after each word so that one training example can act as many.

So before applying softmax to the columns, we change the values below the diagonal to negative infinity, after 

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%2035.png)

This is called Masking. 

Now fox example we want the word “fluffy” to make changes to “creature”, we use a third matrix called the Value Matrix,

we multiply this values matrix to the word fluffy and add the output to creature.

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%2036.png)

Now in place of the Key matrix we multiply the embeddings to the Value matrix, and then multiply the value vector to each column.

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%2037.png)

now after adding the rescaled values we get the change in embedding for “creature” vector.

This is also done for all the other words not just for “creature”.

This step is called **Self-Attention Head.**

**Cross Attention Head:** Key and Query act on two different data, for example, language translation, query would in english but key would be in some other language.

Full attention block consists of multiple heads of attention where all have their own Key, Query and Value maps, These all head get some other type of context and change the embeddings in that way.

![Untitled](Deep%20Learning%20a528f30c4a7e4a59b61042c8642b13d1/Untitled%2038.png)

the embedded token goes through multiple Attention blocks and MLP block to finally get all the context updated in the matrices