# course 5

**Why Sequence Models?**

sequence models are applicable to all of these settings
The length of input and output may be different
The input and output both necessarily need not be sequence 

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled.png)

If $T_x$  is the length of a sequence then $T_x^{(i)}$ is the length of the sequence for training example i

How to represent Individual words in a sentence

come up with a vocabulary / dictionary. One way to build this dictionary is to look up into the training sets and find the top 10,000 occuring words

Now, we can use one hot representations to represent each of these words

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%201.png)

For a word not in your vocabulary, we need create a new token or a new fake word called unknown word denoted by `<UNK>`.

**Recurrent Neural Network Model**

we can have a standard network that outputs y which tells if the word is a part of a name or not. but this does not seem to work much. If we build a neural network to learn the mapping from x to y using the one-hot representation for each word as input, it might not work well.

 There are two main problems:

- Inputs and outputs can be different lengths in **different examples**. not every example has the same input length T or the same output length T. Even with a maximum length, zero-padding every input up to the maximum length doesn't seem like a good representation.
- For a naive neural network architecture, it doesn't share features learned across different positions of texts. eg. Harry appearing in a position tells that it is a part of a name, but if harry appears in other position also then that necessarily doesn’t mean that that is also a part of a name
- Also, we will end with a lot of parameters since each of $x^{<i>}$ are one hot encoded vectors with large dimensions , so eventually we will have a lot of parameters

   **unrolled representation**

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%202.png)

      **rolled representation**

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%203.png)

- A recurrent neural network does not have either of these disadvantages.

What is RNN?
Unlike traditional feedforward neural networks, RNNs have connections that form directed cycles. This allows them to maintain a "memory" of previous inputs by using their internal state, making them particularly useful for tasks that involve sequential data or time series, such as:

The $W_{aa} , W_{ax}, W_{ya} \ are \ the \ parameters$ 

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%204.png)

**In the image above, we feed  the output of the prv layer to the curr layer**

- At each time step, the recurrent neural network that passes on the activation to the next time step for it to use.
- The recurrent neural network scans through the data from **left to right.** The parameters it uses for each time step are shared.
- One limitation of unidirectional neural network architecture is that the prediction at a certain time uses inputs or uses information from the inputs earlier in the sequence but not information later in the sequence.
    - `He said, "Teddy Roosevelt was a great president."`
    - `He said, "Teddy bears are on sale!"`
    
    since RNN uses the info only from the previous steps, thus it is not appropriate to tell if Teddy is a name or not . We can tell Teddy is a name only when we see the next word to it
    
    Forward prop
    
    ![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%205.png)
    

-Instead of carrying around two parameter matrices Waa and Wax, we can simplifying the notation by compressing them into just one parameter matrix Wa.

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%206.png)

**Backpropagation Through Time**