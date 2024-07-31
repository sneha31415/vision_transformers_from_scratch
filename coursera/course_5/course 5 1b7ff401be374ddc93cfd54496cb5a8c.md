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

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%207.png)

**Different types of RNNs**

There are different types of RNN:

- One to One
- One to Many (music generation)
- Many to One (movie rating from the review given)
- Many to Many (eg. name recognizer, machine translation)

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%208.png)

in machine translation, Tx is not equal to Ty

Also, in music generation, the output is given as an input

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%209.png)

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2010.png)

**Language model and sequence generation**

So what a language model does is to tell you what is the probability of a particular sentence.

For example, we have two sentences from speech recognition application:

[https://www.notion.so](https://www.notion.so)

For language model it will be useful to represent a sentence as output `y` rather than inputs `x`. So what the language model does is to estimate the probability of a particular sequence of words `𝑃(y<1>, y<2>, ..., y<T_y>)`

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2011.png)

### *How to build a language model*?

`Cats average 15 hours of sleep a day <EOS>` Totally 9 words in this sentence.

- The first thing you would do is to tokenize this sentence.
- Map each of these words to one-hot vectors or indices in vocabulary.
    - Maybe need to add extra token for end of sentence as `<EOS>` or unknown words as `<UNK>`.
    - Omit the period. if you want to treat the period or other punctuation as explicit token, then you can add the period to you vocabulary as well.
- **Set the inputs x <t>= y<t-1>**
this means that when predicting the 2nd word, we will also **give it the correct 1st word.** And while predicting the 3rd word, we will give it the **correct 1st and 2nd word**
- What `a1` does is it will make a softmax prediction to try to figure out what is the probability of the first words $y^{<1>}$. That is what is the probability of any word in the dictionary. Such as, what's the chance that the first word is *Aaron*?

**$y hat^{<2>}$  is the probability of 2nd word being “average”  in the event that the first word is “cats”**

- Until the end, it will predict the chance of `<EOS>`.
- To train the NN, Define the cost function. **$yhat^{<t>}$ is the predicted word by the RNN and $y^{<t>}$ is the correct word.** The overall loss is just the sum over all time steps of the loss associated with the individual predictions.
    
    Training this RNN over a large dataset, then it will be able to predict the next word in the sequence 
    

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2012.png)

After you train a sequence model, one way you can informally get a sense of what is learned is to have it sample novel sequences.

.Sampling is essential in sequence generation tasks because it leverages the probabilistic nature of the model's predictions to generate diverse, realistic, and natural text. It allows the model to explore various possibilities and produce outputs that reflect the variability and richness of human languages

**Sampling Novel Sequences:**

### Why Not Just Use Argmax?

- **Repetitiveness**: Always choosing the word with the highest probability (argmax) often leads to repetitive and less natural text. It misses out on the natural variability of language.
- **Lack of Creativity**: Argmax can make the model's output very predictable and dull, as it won't explore less probable, but still valid, continuations.

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2013.png)

*Character level language model*:

If you build a character level language model rather than a word level language model, then your sequence y1, y2, y3,(eg cat will be as → y1 for c, y2 for a, y3 for t) would be the individual characters in your training data, rather than the individual words in your training data. Using a character level language model has some pros and cons. As computers gets faster there are more and more applications where people are, at least in some special cases, starting to look at more character level models.

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2014.png)

- Advantages:
    - You don't have to worry about `<UNK>`.
- Disadvantages:
    - The main disadvantage of the character level language model is that you end up with much longer sequences.
    - And so character language models are not as good as word level language models at capturing long range dependencies between how the the earlier parts of the sentence also affect the later part of the sentence.
    - More computationally expensive to train.

If the model was trained on news articles then the left one shows the sampling novel sequences

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2015.png)

RNNs are not good at capturing long range dependencies due to the vanishing gradient problem. The output is influenced only by the closer words and not by the words that are far away
eg. the **cat** which already ate ………. **was** full
eg. the **cats** which already ate ………. **were** full
If, the …… was very long then was and were will not be able to learn from cat and cats

The basic RNN we've seen so far is not very good at capturing very long-term dependencies. It's difficult for the output to be strongly influenced by an input that was very early in the sequence. This is bcuz of the vanishing gradient problem as the gradients vanish and the earlier layers fail to learn

- When doing backprop, the gradients can not just decrease exponentially, they may also increase exponentially with the number of layers going through. 
Exploding gradients are easier to spot because the parameters just blow up and you might often see NaNs, or not a numbers, meaning results of a numerical overflow in your neural network computation.
One solution to that is apply *gradient clipping*: it is bigger than some threshold, re-scale some of your gradient vector so that is not too big.
Vanishing gradients is much harder to solve and it will be the subject of GRU or LSTM.

RNN unit :

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2016.png)

The output activation is passed through the softmax unit and the prediction i.e y hat is obtained

**Gated Recurrent Unit (GRU)**

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2017.png)

gamma is such that it is either close to 0 or close to 1

When gate update = 0, then it means that don’t update this value i.e remember the word “cats” so that we use “were” and not “was”

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2018.png)

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2019.png)

gamma(f) ⇒ forget gate , so instead  so (1 - gamma_update) we have gamma(f) here

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2020.png)

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2021.png)

lstm ⇒ more complicated, but  more powerful due to 3 gates

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2022.png)

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2023.png)

**Bidirectional RNN** 

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2024.png)

• Blocks can be not just the standard RNN block but they can also be GRU blocks or LSTM blocks. In fact, BRNN with LSTM units is commonly used in NLP problems.

The disadvantage of the bidirectional RNN is that you do need the entire sequence of data before you can make predictions anywhere. So, for example, if you're building a speech recognition system, then the BRNN will let you take into account the entire speech utterance but if you use this straightforward implementation, you need to wait for the person to stop talking to get the entire utterance before you can actually process it and make a speech recognition prediction.

**Deep RNNs**

deep rnn with 3 hidden layer→

- For learning very complex functions sometimes is useful to stack multiple layers of RNNs together to build even deeper versions of these models.
- The blocks don't just have to be standard RNN, the simple RNN model. They can also be GRU blocks LSTM blocks.
- And you can also build deep versions of the bidirectional RNN.

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2025.png)

QUIZ :

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2026.png)