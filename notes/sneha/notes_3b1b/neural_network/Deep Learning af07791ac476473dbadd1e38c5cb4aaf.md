# Deep Learning

## chapter 1: But what is a neural network

aim = put together a neural network that learns to recognise digits

 neuron = thing that holds a number (between 0 and 1), the number inside it is called its activation

![Untitled](Deep%20Learning%20af07791ac476473dbadd1e38c5cb4aaf/Untitled.png)

these 784 neurons make the first layer of our neural network

![Untitled](Deep%20Learning%20af07791ac476473dbadd1e38c5cb4aaf/Untitled%201.png)

### edge detection(work of 1st hidden layer) -

![Untitled](Deep%20Learning%20af07791ac476473dbadd1e38c5cb4aaf/Untitled%202.png)

the weighted sum might come as any number, so we need function to squish things between 0 to 1

sigmoid (logistic curve) does this

![Untitled](Deep%20Learning%20af07791ac476473dbadd1e38c5cb4aaf/Untitled%203.png)

bias is used for inactivity below a threshold

therefore bias is an indication of the fact whether the neuron is active or inactive 

bias = added to the weighted sum before squishing function

![Untitled](Deep%20Learning%20af07791ac476473dbadd1e38c5cb4aaf/Untitled%204.png)

13002 weights and biases (knobs that can be tweaked and turned to make the network behave in a certain way)

![Untitled](Deep%20Learning%20af07791ac476473dbadd1e38c5cb4aaf/Untitled%205.png)

condensed equation:  $a^1 = sigma(Wa^0 + b)$

the network or neuron can be considered as a function which takes in a input number(784 grayscaled pixel values) and spits out a  number (activation)

## chapter 2 : **Gradient descent, how neural networks learn**

cost function -

![Untitled](Deep%20Learning%20af07791ac476473dbadd1e38c5cb4aaf/Untitled%206.png)

cost function is large when the network does not know what its doing

avg cost is the avg of all costs over all the examples

Gradient gives the direction in which we should step to increase the function 
so, taking the negative of that gradient will give the direction to decrease the function most quickly

![Screenshot from 2024-07-01 13-59-01.png](Deep%20Learning%20af07791ac476473dbadd1e38c5cb4aaf/Screenshot_from_2024-07-01_13-59-01.png)

the above is for 2 input (x, y) and z is the cost function graph
the same will be for a 13002 number of inputs 

![Untitled](Deep%20Learning%20af07791ac476473dbadd1e38c5cb4aaf/Untitled%207.png)

more is -delta(c(W)) more effect on the cost function it will have.
Like more magnitude ones cause a more rapid decrease is cost function i.e change of weight of more slope one will have a larger impact(in short we are exploring which changes to which weights matter the most)

![Untitled](Deep%20Learning%20af07791ac476473dbadd1e38c5cb4aaf/Untitled%208.png)

second layer (i.e 1st hidden layer)

![Untitled](Deep%20Learning%20af07791ac476473dbadd1e38c5cb4aaf/Untitled%209.png)

## Chapter 3: What **is back-propagation really doing?**

the magnitude of each component in the negative gradient tells us how sensitive each component is to the cost function

![Untitled](Deep%20Learning%20af07791ac476473dbadd1e38c5cb4aaf/Untitled%2010.png)

wiggle to the first weight(whose negative gradient is 3.20) will cause a 32 times more change in cost function than the wiggle of 2nd weight would

How can output be made to be a ‘2’ ?

![Untitled](Deep%20Learning%20af07791ac476473dbadd1e38c5cb4aaf/Untitled%2011.png)

- increase w for those activation is high so that activation of neuron corresponding to digit 2 increases
- increase a for those whose weights are positive and decrease a for  those weights are negative.

now, this is the required changes only for the digit 2
every digits has its own desired changes 
so, the desires of 2 are added with the desires of other digits

the figure below shows the summing of desires 

![Untitled](Deep%20Learning%20af07791ac476473dbadd1e38c5cb4aaf/Untitled%2012.png)

BACKPROPAGATION( is how a single training example would like to  change the weights and biases)

see what changes in weights and bias each digit wants and average out the changes

fig-1

![Untitled](Deep%20Learning%20af07791ac476473dbadd1e38c5cb4aaf/Untitled%2013.png)

This is nothing but the negative gradient of the cost function

![Untitled](Deep%20Learning%20af07791ac476473dbadd1e38c5cb4aaf/Untitled%2014.png)

### stochastic gradient descent-

averaging all the training examples as shown in fig.1 is computationally slow

so we randomly subdivide the data set into mini batches and compute each step’s gradient using the mini batch

![Untitled](Deep%20Learning%20af07791ac476473dbadd1e38c5cb4aaf/Untitled%2015.png)

In the above image, left one computes minima slow and carefully computing gradient for every training example
While the right one computes minima faster and using mini batches 

![Untitled](Deep%20Learning%20af07791ac476473dbadd1e38c5cb4aaf/Untitled%2016.png)

This is faster than carefully finding the local minimum i.e 

computing on each training example

## Chapter 4 : backpropagation Calculus

terminologies and basic formulae

$a^L$  is the networks output and y is the desired output

![Untitled](Deep%20Learning%20af07791ac476473dbadd1e38c5cb4aaf/Untitled%2017.png)

each of $w^L$, $Z^L , a^L$ , C₀ are some numbers, so we can represent them on a number line  

if, $W^L$ is changes, $Z^L$ also changes, due to which $a^L$ also changes, which inback changes $Co$

so by chain rule,delta Co/delta $W^L$ can we given as -

![Untitled](Deep%20Learning%20af07791ac476473dbadd1e38c5cb4aaf/Untitled%2018.png)

![Untitled](Deep%20Learning%20af07791ac476473dbadd1e38c5cb4aaf/Untitled%2019.png)

![Untitled](Deep%20Learning%20af07791ac476473dbadd1e38c5cb4aaf/Untitled%2020.png)

$del\ C_o / del\ b^L$ (how sensitive the cost function is to the bias of the current layer)=

![Untitled](Deep%20Learning%20af07791ac476473dbadd1e38c5cb4aaf/Untitled%2021.png)

$del\ C_o /del\ a^{L-1}$(how sensitive the cost function is to the activation of the previous layer neuron)

![Untitled](Deep%20Learning%20af07791ac476473dbadd1e38c5cb4aaf/Untitled%2022.png)

now, when we give multiple neurons in each layer, now we will have a subscript k for each neuron indicating which number th of neuron of the Lth layer it is

![Untitled](Deep%20Learning%20af07791ac476473dbadd1e38c5cb4aaf/Untitled%2023.png)

![Untitled](Deep%20Learning%20af07791ac476473dbadd1e38c5cb4aaf/Untitled%2024.png)

what changes slightly is  the derivative of the cost function wrt activation, cuz here $a_k^{L-1} has \ influence \ on\ a_o^1\ and\ also\ on\ a_1^L$

![Untitled](Deep%20Learning%20af07791ac476473dbadd1e38c5cb4aaf/Untitled%2025.png)

## chapter 5: **But what is a GPT? Visual intro to transformers**

transformer = specific kind of neural network( An ML model)