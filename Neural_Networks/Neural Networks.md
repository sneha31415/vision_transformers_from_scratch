# Neural Networks

*from 3Blue1Brown*

- ***Note :-*** Throughout the explanation, we consider a neural network taking a square image of 28*28 = 784 pixels and giving one of the 10 outputs - 0 to 9. This Neural Network consists of 2 hidden layers, each having 16 Neurons. The input and output layers have 784 neurons and 10 neurons respectively.
- Machine Learning - We use data to determine how a Model behaves.

# 1. Terminology, Sturctural Aspects and Basic Working :-

## a. *Neuron -*

 A thing that holds a number ranging from 0 to 1 (termed as ‘Activation’ of the neuron).

## b. *Layers -*

A set of neurons, which are connected to several other sets of Neurons. The layers of neurons present between the input layer and the output layer, are termed as ‘*Hidden Layers*’. 

- In case of a neural network identifying handwritten digits, the neurons in the hidden layers can be thought of to be activated when certain characteristic parts e.g. a straight line or a loop in upper side of image, is detected.
- The activation of neurons of input layer cause activation of some specific neurons in subsequent layers and ultimately, some neurons in last layer activate to varying extents. The neuron which is the mst activated, in the output layer, provides the neural network’s best guess of output for the input provided.

![Curves in digit structures](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-06-28_15-16-56.png)

Curves in digit structures

- It can also be thought that neurons from previous hidden layer detect parts of these; loops or lines. Hence, detection of small structures in image in various layers cause activation of certain neurons in subsequent layers and thus, can detect the image.

![Structural components of constituent structures of digits](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-06-28_15-23-12.png)

Structural components of constituent structures of digits

## c. *Weights -*

Weights are real numbers, either positive or negative, assigned to the connection between 2 partcular neurns, each being from adjacent layers. Weights are multiplied with activation of neuron from previous layer and then, carrying out the same for all neurons of previous layer, sum of all these numbers, i.e. ‘*weighted sum*’ is passed to the connected neuron of next layer. Here, ‘w’ are weights and ‘a’ represents activation of neurons of previous layer, which has ‘n’ neurons.

![Expression of weighted sum](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-06-28_15-30-18.png)

Expression of weighted sum

## d. *Bias -*

Bias is any number, which indicates threshold value of the activation of a neuron due to the neurons frm previous layer. It is subtracted from the weighted sum. 

For example, here, the bias is ‘10’.

![Expression of weighted sum, incorporating bias](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-06-28_16-26-57.png)

Expression of weighted sum, incorporating bias

## e. *Squishification function -*

The value of weighted sum, along with correction of bias, can turn out to be any real number; s as to simplify calculations, we try to scale down the values of weighted sum to a value ranging from 0 to 1 using a squishificatin function. Here, sigmoid function is used.

E.g. Sigmoid, ReLU, softmax, etc.

![Sigmoid Function](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-06-28_16-38-54.png)

Sigmoid Function

![Rectified Linear Unit (ReLU)](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-06-28_16-40-23.png)

Rectified Linear Unit (ReLU)

- Hence, the functioning of Neural Network depends solely on the values of weights and biases. Basically, in the process of *‘**Learning**’* the Neural Network adjusts its weights and biases after each input and thus, imroves itself.
- General Representation of above equations :

![Matrix representation of expression of activation of a neuron](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-06-28_16-36-05.png)

Matrix representation of expression of activation of a neuron

# 2. Operation Mechanism of Neural Network :-

- A labelled dataset is provided as input to the neural network. Depending on the differences between the output f neural netwprk and the expected output, the values of required weights and biases are adjusted so that the neural network will perform better on subsequest data. This is the process by which a Neural Network learns.
- Thus, the main goal is to generalize the action of Neural Network. Hence, after training it on ‘Training Data’, which is labelled, it is also checked with ‘Testing Data’, whichj cntains unlabelled data elements, whu=ich are completely new for the Neural Network and its accuracy is analysed.

## a. *Cost Function -*

- Initially, we randomly assign weights and biases of the Neural Network. In this scenario, the Neural network performs very badly, with (maybe) some exceptions of good guesses.
- Now, we take the square of the difference of the activations of each of the neurons of the output layer and add them together. This value is termed as ‘***Cost***’ of the corresponding training example (training input). Higher the cost, worse is the performance of the Neural Network.
- The cost of all training examples are noted; they form the data points of the ‘***Cost Function**’ .*

## b. *Gradient Descent and Introduction to Backpropagation -*

- We aim to minimize the cost, i.e. we are concerned about the minima of the cost function. Hence, we use the result from Multivariable Differential Calculus, that, the gradient of a function gives the direction of maximum slope. We then tend to the minima by traversing path along the function, in direction opposite to that of the Gradient. Then, we can trace the values of parameters of that function; for cost function, they would be the weights and biases in the Neural Network, and change them accordingly.
- One drawback of this method is that it turns out to be complicated for complicated cost functions, which have a larger number of minima.
- Hence, we choose any arbritry point on the curve of cost function, and measure its Gradient around that point; we tend to traverse the function in the direction opposite to that of the Gradient. Continuing this repeatedly, one can approach the local minimum of the cost function.
- The negative Gradient of the cost function for all weights and biases, for each step, gives the changes that need to made in the weights and biases so that the value of the cost function decreases. Each of its values also gives information as to which input of the cost function needs to be modified to how much extent, and to either be increased or decreased.
- The algorithm which performs the above changes to weights and biases, is termed as ‘***Backpropagation***’.
- Smoother variations in cost functions are needed for finding the corresponding local minimum accurately. Hence, the activations of neurons of last layer (as cost function depends on them) have values which vary smoothly; rather, all neurons have continuoulsy variable activations.
- ***Gradient Descent*** - the process of repeatedly varying the input of a function by using a multiple of negative gradient of the function.
- It is observed that for this particular Neural Network under consideration, it is not like the neurons of the hidden layers identify specific patterns in the input image, but its like of a more random form of arrangements of pixels. ALs, this is why even when we provide the Neural Network with a random image, it gives a confident guess out of the 10 digits, though it is wrong.
- It is observed that Neural Networks learn properly labelled data faster than randomly labelled data.

# 3. Backpropagation :-

## *a. Backpropagation with traditional Gradient Descent -*

- Consider an input image is provided to the Neural Network and it produces certain activations in the output layer. We compare these activations with the activations that should be produced in the output layer. This gives an idea of how the activations of each of the neurons of the previous layer need to be modified - either increased or decreased, and to what extent - to a large extent or a small extent.
- For this example, we consider we are giving an image of digit ‘2’ as input to the neural network and hence, in the ideal case, expect all neurons in output layer, except the one corresponding to digit ‘2’ should be zero, as shown below.
- This means that, if the neuron corresponding to digit ‘8’ in output layer, has a value close to zero, and in the ideal scenario, it is zero, then its value needs to be decreased to a small extent. Likewise, calculations are done for all output layer neurons.

![Fig. 1](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-06-28_19-44-09.png)

Fig. 1

![Fig. 2](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-06-28_20-51-47.png)

Fig. 2

- The activations of output layer neurons depends on the weights of their connections, the bias of that neuron and the activations of the preceding layer.
- Let us focus on the effect of penultimate layer, on the output layer neuron corresponding to digit ‘2’, whose activation is required to be maximum possible. (Fig. 1)
1. ***Effect of weights of penultimate layer neurons on activtion of output layer neuron :***

The weights, corresponding to the neurons of preceding layer, whose activatins are high, should be increased.

1. ***Effect of activation of penultimate layer neurons on activtion of output layer neuron :***

The neurons of preceding layer having high weights should be more active and vice versa. However, this is not possible, as we can change only weights and biases but not activations of any neurons in the Neural Network.

1. ***Effect of biases of penultimate layer neurons on activtion of output layer neuron :***

The biases shuld be increased.

- Likewise, we consider changes in weights and biases  of all neurons in penultimate layer and sum them up to get final change that needs to be done, in each neuron of penultimate layer.(Fig. 2)
- In the same manner, the weights and biases of preceding layers are adjusted for each training example.
- Now, each weight has a series of adjustments that needs to be made, obtained after training it on each training example. The average of all these adjustments is finally added to each weight and bias and the Neural Network is tuned.
- Until now, everything was discussed in relation with Gradient Descent, as dsicussed above.

### 

## *b. Stochastic Gradient Descent -*

- It is computationally expensive to check for slope and descent step by taking thw whole training data set as input. Hence, generally, the method of ‘Stochastic Gradient Descent’ is used.
- In this method, the whole training data is divided into mini-batches, consisting of a relatively smaller number of data elements. The descent at each mini-batch is calculated and process is repeated over all mini-batches.
- This method is faster than the traditional Gradient Descent, though it can be a little less accurate. however, it also has computational advantage to traditional Gradient Descent.

# 4. Calculus behind Backpropagation :-

- Firstly, we consider a simple neural network, with 4 neurons and 4 layers :

![Simple Neural Network](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-06-29_17-14-00.png)

Simple Neural Network

- For calculating change in weight(s), we need to find the change in cost function w.r.t. change in the corresponding weight(s). It is convinient to use Chain Rule of Derivatives for this purpose.
- We note that : (w.r.t Fig. 3 & Fig. 4)
1. Cost of a particular training example depends on the activation of last layer neuron and the expected activation of that neuronm amd is denoted by Co.
2. Activation of that last layer neuron depends on the corresponding weight(s), activation(s) of the neuron(s) of the preceding layer and the corresponding bias; this weighted sum with bias correction quantity is denoted by z(l). 
3. The activation of last layer neuron will be obtained when z(l) will be operated by some squishification function.
4. These various quantities are interdependent as shown in Fig. 3.

![Fig. 3](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-06-29_16-59-58.png)

Fig. 3

![Fig. 4](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-06-29_16-55-09.png)

Fig. 4

![Fig. 6](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-06-29_16-57-10.png)

Fig. 6

![Fig. 5](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-06-29_16-55-53.png)

Fig. 5

![Fig. 7](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-06-29_17-05-38.png)

Fig. 7

- With reference to Fig. 3, we can derive the required formula for chain rule of derivatives, as shown in Fig. 5, and find the respective derivatives from the quantities as given in Fig. 4.
- This derivative is just for 1 weight of the neural network, fior 1 training example. The required derivative for this weight would be the average of this derivative over all training data. (Fig. 6)
- This value forms just one component of the column matrix, which represents the gradient of cost function. This has to be done for all the weights and biases. (Fig. 7)

![Fig. 8](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-06-29_17-08-03.png)

Fig. 8

- For the derivative of cost function w.r.t. bias, we only replace all the terms of w(l) with b(l) i Fig. 5 and 6 and the first derivative, of z(l) w.r.t. b(l) in Fig. 8 only would be having a different value: 1, which needs to be replaced.
- This same procedure can be repeated for weights of preceding layers for obtaining respective derivatives.
- Now, let us consider a network with layers having several neurons (Fig. 9)
- Subscripts ‘j’ ank ‘k’ respectively refer to neurons from last and penultimate layer and both begin from 0, as 0, 1, 2, … (Fig. 9)
- The Cost can be defined according to its traditional definition, as shown in Fig. 10.
- We also define zj(l) for jth neuron of last layer, incorporating activations and weights from all neurons from preceding layer.

![Fig. 9](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-06-29_17-17-47.png)

Fig. 9

![Fig. 10](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-06-29_17-18-25.png)

Fig. 10

![Fig.11](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-06-29_17-20-44.png)

Fig.11

![Fig. 12](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-06-29_17-22-36.png)

Fig. 12

![Fig. 13](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-06-29_17-24-40.png)

Fig. 13

# 5. A Basic Idea of transformers :-

## *a. Overview :*

- ***‘Transformer’*** is a special kind of Neural Network, which can be used for various purposes :
1. Voice-to-text
2. Text-to-voice
3. Text-to-image
4. Image-to-text
- Working of ChatGPT [(Text predictive) Generative Pre-trained Transformer] :-

An initial text snippet is provided to the model; it works on it and it gives a sampler distribution as output; depending on this output, it predicts next word. Then this whole snippet is taken as input and words are predicted further.

- Basically, in Transformers,  the data provided as input is converted into an array/matrix of numbers, also termed as ‘Tensors’ (general term).
- Vectors are drawn from the input matrix. There is another matrix, which contains tunable parametrs, which, are often termed as ‘Weights’. The matrix of weights is multiplied with each vectors drawn from input matrix (shown in yellow in Fig. 14) to eventually form output matrx, after several repetitions of this procedure.
- This way, input data is transformed into output data.

![Fig. 14](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-06-29_19-44-47.png)

Fig. 14

## *b. Terminology, Structural Aspects and Mechanism of Operation -*

### 1. Tokens -

- The sentence which is to be completed, is broken down into parts termed as ‘Tokens’. Tokens can be defined to be pieces of words or punctuation. Just for sake of simplicity, for explanation here, we consider tokens as words of sentence.

![Tokens](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-06-29_19-53-14.png)

Tokens

### 2. Embedding Matrix -

- Embedding matrix is a matrix, the columns contains a column vector for each word of its vocabulary. The vocabulary of GPT-3 is 50,257 words.
- This Embedding Matrix has 12,288 rows. Initially it has random numbers, but as the training progresses, all its elements get finely adjusted
- The values of Embedding Matrix are changed and tuned as the model is traind on Data.
- These column vectors are termed as ‘Embeddings’; thus the embeddings/word vectors can be thought of to be vectors in a very high dimensional space.
- As  the model is trained, several patterns are observed :
1. the embeddings get values such that embeddings with similar meaning tend to lie near each other.
2. The difference between the embeddings of ‘man’ and ‘woman are found to be similar to that of ‘king’ and ‘queen’. Thus, to find the feminine form of a word, one may  add this difference embedding to the embedding of masculine form of the entity.
3. One can also roughly infer that certain directions in the high dimensional space of the embeddings encode certain meanings, e.g. one direction encodes gender information, other, familial relation, and so on. 
4. Interestingly, if we take a vector ‘plur’ as the difference of the embeddings of the word ‘cats’ and that of ‘cat’, and calculate the dot product of ‘plur’ with the singular and plural form of various nouns, we find that the plural forms align more with ‘plur’ than singular forms, as they have poistive dot product while the latter have negative dot products.
5. Likewise, the dot product of ‘plur’ with various words denoting numbers, like ‘one’, ‘two’, ‘three’ etc. are found to be increasing order. 

![Fig. 15](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-06-29_21-47-07.png)

Fig. 15

![Fig. 16](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-06-29_21-47-15.png)

Fig. 16

### 3. Context Size and Context -

- The ‘Context Size’ of a model is the number of words that the model can process at a time. For GPT-3, it is 2048.
- As the context of a particular embedding is analysed by the model, operations are performed on the embeddings, and during this process, they are modified to numerically represent the required context more accurately.

### 4. Output Side -

- Desired output is a probability distribution of the tokens that should continue the given text.
- There is another matrix, termed the ‘Unembedding Matrix’, which has 50,257 rows (as much as vocabulary size of the model) and 12,288 columns (as much as the no. of rows in the embeddings).
- The Unembedding Matrix is multiplied with the last embedding after the corresponding embeddings w.r.t. given text have passed the blocks of the transformer, which maps the modified version of the laste embedding to 50,257 values (vocabulary size of transformer). All the elements of this matrix are termed as ‘Logits’.
- This matrix of 50,257 elements won’t necessarily follow to form a Probability Distribution. Hence, to convert it into a probability distribution, these values are given as input to a special function called ‘softmax’ (Fig. 17)
- Elements of output matrix, after softmax, are termed ‘Probabilities’.

![Fig. 17](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-06-29_22-16-46.png)

Fig. 17

- Another term ‘temperature’ may also be incorporated into softmax to modify it. Higher the temperature, lower values in probability distribution also rise. Lower the temperature, only the higher values dominate the prbability distribution.
- It is called ‘temperature’ as in  analogous thermodynamical equations, the role of that term, is played by temperature.

![Fig. 18](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-06-29_22-22-09.png)

Fig. 18

![Fig. 19](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-06-29_22-17-31.png)

Fig. 19

![Fig. 20](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-06-29_22-26-04.png)

Fig. 20

# 6. Attention in Transformers :-

## a. *Overview :*

- The aim of a transformer is to change the embedding so that it ultimately encodes the contextual meaning of words in a sentence rater than just its literal meaning.
- Consider the word ‘,ole’ in the 3 sentences below. The contextual meaning of the word ‘mole’ is diefferent in all 3 of them, but the initial embedding of the word ‘mole’ is the same for all 3 sentences.

 

![Fig. 21](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-07-01_11-35-31.png)

Fig. 21

- An ‘Attention’ block (where the process of ‘Attention’ occurs) is the place where the embeddings of the tokens in a sentence interact with each other and change themselves to reflect their contextual meaning.

![Fig. 22](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-07-01_11-35-55.png)

Fig. 22

## b. *Single Head of Attention (Self Attention):-*

- The embedding, which is in the form of a vector, tells what the token is and what is its position in the given sentence.
- When the tokens are ‘attended’, the embeddings are changed by opeartions like matrix multiplication, so that the embeddings gain contextual meaning, just like relating a noun to adjectives preceding it.
- There is a vector, with relatively smaller dimensions (here, 128) which is the ‘Query’ vector. This is obtained, for each embedding, by multiplying a ‘Query Matrix’, with 128 rows and 12,288 columns, with each of the embeddings. The Query matrix initially has random elements but they are refined with training.
- This ca be thought of as though the query matrix maps the high dimensional embedding to a low dimensional Query/Key space.

![Fig. 23](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-07-01_12-21-21.png)

Fig. 23

- There is another matrix - ‘Key’ Matrix, with 128 rows and 12,288 columns, which is also multiplied with each embedding to obtain ‘Key’ Vectors. Initially they have random elements but they get refined with training. Like the query matrix, it also maps the high dimensional embedding to low dimensiona query/key space.
- Basicallly, the query vectors is like questioning the embeddings about the contextual meaning and the key vectors are like answers to those questions.

![Fig. 24](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-07-01_12-32-24.png)

Fig. 24

- For instance, in the above sentence ‘**A fluffy blue creature roamed the verdant forest**’, the key vectors of tokens ‘fluffy’ and ‘blue’ align to a great extent with the query of the token ‘creature’, indicating contextal relation among those words. This similarity can be understood mathematically as a high value of the dot product of query vector with key vector. These values are filled in a grid termed as ‘Attention Pattern’, which signifies how relevant tokens at the left are, w.r.t. to the tokens at the top.

![Fig. 25](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-07-01_12-36-38.png)

Fig. 25

![Fig. 26](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-07-01_12-37-45.png)

Fig. 26

- The larger dots correspond to a higher dot prduct of query vector and key vector. As per the above discussion, in the language of machine learning, we can state that the embeddings ‘fluffy’ and ‘blue’ attend to the embedding ‘creature’. (Fig. 26)
- These values can lie anywhere from negative infinity to positive infinity. For the next step, where we need to normalize thes values as that of a probability distribution. So, we apply softmax to each column.
- This operation can be represented as an expression as shown below. The meaning of various terms is same as discussed above. Q and K dentoe the array of query and key vectors for al tokens. Also, it has been found out that it is better for numerical stability to divide each term in the attention pattern by the squate root of the dimension of the key query space, before applying softmax.

![Fig. 27](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-07-01_12-46-35.png)

Fig. 27

![Fig. 28](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-07-01_12-47-00.png)

Fig. 28

- It turns out to be more efficient to let the transformer predict the next token for all the tokens in the starting sentece given to the transformer as well; this is because due to this, a single example trains it as much as a bunch of examples.

![Fig. 29](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-07-01_16-04-26.png)

Fig. 29

- Also, the later tokens shouldn’t affect the contextual meaning of previous tokens, so we need the following entries in the attention pattern, marked in red, to be zero, in the probability distribution. Hence, we set them to negative infinity. This process is termed as ‘Masking’ and was applied in the training phase of GPT-3.

![Fig. 30](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-07-01_16-20-32.png)

Fig. 30

![Fig. 31](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-07-01_16-23-15.png)

Fig. 31

- The size of attention pattern is the square of context size.
- Now, the tokens, which are related to other tokens, are multiplied with another matrix termed ‘Value Matrix’, with 12,288 rows and 12,288 columns, which produces a ‘Value Vector’. The value vector is added to the embedding of the other vector (both are of smae dimensions) to get the change in embedding, reflecting its contextual meaning. Likewise, in this example, the embedding of token ‘fluffy’ is multiplied with value matrix to get its value vwctor, which is then added to the embedding of creature to get the embedding for ‘fluffy creature’. However, this is just a part of a large process.

![Fig. 32](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-07-01_16-28-45.png)

Fig. 32

- Like we had in attention head, we now replace all query vectors with their corresponding embeddings and the key vectors with corresponding value vectors. We multiplythe value vectors with the probability distribution under the column of each embedding and take a column-wise sum. This sum gives the change in each embedding that should be made, to reflect its current contextual menaing. Alling this change to each embedding, we get new embeddings, and this is the change in embeding which was being referred to, since beginning of the topic.
    
    ![Change in embedding for attaining the meaning of a ‘fluffy blue creature’](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-07-01_16-43-45.png)
    
    Change in embedding for attaining the meaning of a ‘fluffy blue creature’
    

![Fig. 33](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-07-01_16-44-01.png)

Fig. 33

- It is efficient, especially for running multiple attention heads in parallel, that the value of parameters in value matrix and the sum of the n umber of parameters in the key and query matrices is the same. Hence, the value matrix can be factored as a product of a 12,288*128 matrix (which maps the embedding to a smaller dimensional space) and a 128*12,288 matrix (which re-maps the embedding to a larger dimensional space). Here, we are using a ‘Low-Rank’ transformation here.

## c. *Cross-Attention :-*

- Cross-Attention involves perocessing 2 different types of data, like translation from one language to another, or transcripting speech (audio to text)

![Screenshot from 2024-07-01 17-06-13.png](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-07-01_17-06-13.png)

- Here, in attention pattern, the main difference is that Query vectors come from one type of data say one language, and key vectors, from other language. Also, there is no need of masking in this case, as words further in a sentence can also affect meaning of sentence, in context of previous words, in dfferent languages.

## d. *Multi-Head Attention and 1 Attention block :-*

- An attention block consists of 96 Attention heads, running in parallel and there are 96 attention blocks in GPT-3.
- Each attention head has its own query, key and value matrices and each head produces some change in embedding.

![Screenshot from 2024-07-01 17-10-44.png](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-07-01_17-10-44.png)

- At the end of an attention block, the changes in each embeddings produced by each head of attention are added to the original embedding to produce a new embedding and then, that new embedding is passed onto next attention block.

![Screenshot from 2024-07-01 17-10-52.png](Neural%20Networks%20182a99e3c172458bac92a1430a6cff2e/Screenshot_from_2024-07-01_17-10-52.png)

## e. *Output Matrix :-*

- In the actual practice, to ease calculations, in each of the heads of attention, only the value matrix which has dimensions of 128*12,288 (which we refer to as ‘Value Down Matrix’) is used.
- For each embedding, we get a 28 dimensional column vector. At the end of each attention block, all of these (for each embedding individually)they are combined to form a matrix having dimensions 96*128.
- This 96*128 dimensional vector is then multiplied with the output matrix, which is formed by combining all the components of value matrix which have dimensions 12,288*128. Hence, when referring to a sinlge head, what is being referred is the ‘Value  Down Matrix’.

## f. *Multilayer Perceptron :-*

- After passing through an attention block, the embeddings go through a multilayer perceptron, and then this process is repeated.
- Thus, eventually, the embeddings go through a series of attention blocks and then multilayer perceptrons.
- The mre the embeddings get refined, the more the transformer is able to gauge finer details like the sentiment and tone in the gven sentence.