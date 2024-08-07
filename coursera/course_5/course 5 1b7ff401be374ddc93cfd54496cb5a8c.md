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

### REFERENCE : [https://chatgpt.com/share/7ceb9e5c-a74e-469b-a222-835b083d54f1](https://chatgpt.com/share/7ceb9e5c-a74e-469b-a222-835b083d54f1)

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

## WEEK 2- **Natural Language Processing & Word Embeddings**

**Word Representation**

O stands for one hot encoded vector 

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2027.png)

One of the weaknesses of one-hot representation is that it treats each word as a thing unto itself, and it doesn't allow an algorithm to easily generalize across words.

Using one hot encoded vector for our model, orange and apple are two distinct things . so even if we train it on “**orange juice**” , it will not be able to guess apple juice given **apple ______ ?**

This is because each one hot encoded vector has 1 only at the place where there is that word in the vocab and thus inner product between any two one hot encoded vectors is zero. So, orange and apple cannot be related. 
Thus, It doesn't know that somehow apple and orange are much more similar than king and orange or queen and orange.

Note : 5391 is the place where that word (man) occurs in the vocab

**Featurised representation over one hot encoding**

gender = say -1 for man and 1 for woman
age= kings and queen are mostly elderly so feature is ranged 0.7, 0.69. man and woman cannot be generalised here so they are neural 

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2028.png)

so, now we see that apple and orange have somewhat similar feature vectors so if we have orange juice then predicting apple juice from apple ____ will be somewhat easier cuz apple and orange will be much more related than man and apple

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2029.png)

**Using Word Embeddings(transfer learning)**

word embeddings ⇒ dense and low dimensional
one hot ⇒ sparse and high dimensional

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2030.png)

no need of fine tuning if data set is small

Word embeddings tend to make the biggest difference when the task you're trying to carry out has a relatively smaller training set.

- Useful for NLP standard tasks.
    - Named entity recognition
    - Text summarization
    - Co-reference
    - Parsing
- Less useful for:
    - Language modeling
    - Machine translation

**Properties of Word Embeddings**

Anologies 

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2031.png)

sim = similarity e_w is such that it is similar to e_king - e_man + e_woman
where e_w = embedding of word w

We can use cosine similarity to calculate this similarity.

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2032.png)

**What t-SNE does is, it takes 300-D data, and it maps it in a very non-linear way to a 2D space. And so the mapping that t-SNE learns, this is a very complicated and very non-linear mapping. So after the t-SNE mapping, you should not expect these types of parallelogram relationships, like the one we saw on the left, to hold true. And many of the parallelogram analogy relationships will be broken by t-SNE.**

cosine similarity 

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2033.png)

**Embedding Matrix:**

**300 ⇒ feature size**
10000 ⇒ vocab size

O_6257 ⇒ one hot vector of 

### how is the embedding of a word formed?

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2034.png)

e is our embedding vector. It is 300 dimensional. e is formed by multiplying E i.e embedding matrix and one hot encoded vector

- Our goal will be to learn an embedding matrix E by initializing E randomly and then learning all the parameters of this (300, 10000) dimensional matrix.
- E times the one-hot vector gives you the embedding vector.

**Learning Word Embeddings**

A relatively complex neural language model  to generate  good word embeddings

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2035.png)

since we have 6 input words and each word embedding is a 300 dimensional matrix, so total params = 6 * 300 = 1800

- If we have a fixed historical window of 4 words (4 is a hyperparameter), then we take the four embedding vectors and stack them together, and feed them into a neural network, and then feed this neural network output to a softmax, and the softmax classifies among the 10,000 possible outputs in the vocab for the final word we're trying to predict. These two layers have their own parameters W1,b1 and W2, b2.
- This is one of the earlier and pretty successful algorithms for learning word embeddings.

A more generalized algorithm **to learn word embeddings**-

- We have a longer sentence: `I want a glass of orange juice to go along with my cereal`. The task is to predict the word `juice` in the middle.
- If it goes to build a language model then it is natural for the context to be a few words right before the target word. But if your goal isn't to learn the language model but to just get good embeddings, then you can choose other contexts.
- Contexts:
    - Last 4 words: descibed previously.
    - 4 words on left & right: `a glass of orange ___ to go along with`
    - Last 1 word: `orange`, much more simpler context.
    - Nearby 1 word: `glass`. This is the idea of a **Skip-Gram** model, which works surprisingly well.
- If your main goal is really to learn a word embedding, then you can use all of these other contexts and they will result in very meaningful work embeddings as well.

refer for understanding "neural langauge model to learn good embedding” -

[https://chatgpt.com/share/b68276b0-cc0a-42cb-8cac-67c24fbdab06](https://chatgpt.com/share/b68276b0-cc0a-42cb-8cac-67c24fbdab06)

A more simpler and computationally effective way to learn good embeddings is Word2Vec 

SUMMARY :

**If similar words have similar embeddings, then learning how to use 1 word can automatically train the network to use other words with similar embeddings. 
This similar embedding can be done using training a NN. The word embeddings are initially randomly initialised but as the training proceeds, the embeddings are learnt(just like weights)**

 ****

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2036.png)

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2037.png)

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2038.png)

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2039.png)

SEE HOW TROLL2 AND GYMKATA ARE CLOSE TO EACH OTHER SINCE THEY ARE SIMILAR IN MEANING

NOTE: Simply using one number to represent single word wont work. This is because each word may have different different meanings like a positive and a negative meaning. So to represent one word with multiple numbers, we use that many activations(here since we have used 2 activations, so we will get two embeddings per word)

Simply using the next word predictor NN does not give a lot of context to understand each one. So learn word2Vec which is a popular method for creating word embeddings but used to include more context

**Word2Vec** 

Two methods : 

1) Continuous bag of words

2) skip gram

**Continuous bag of words** increases the context by using the surrounding words to predict what occurs in the middle.

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2040.png)

**Skip gram** uses the word in the middle to predict the surrounding words

**speeding up training with negative sampling:(much more efficient than skip gram)**

In practice, we don’t create just two embeddings per word but rather like 100 per word. so we need to use 100 activations.
also, word2Vec uses the entire wikipedia for training and not just 2 sentences.
Therefore, word2Vec has a vocabulary of about 3,000,000 words and phrases

Therefore, the total number of weights that we need to optimize is:
3,000,000 phrases * 100 activations * 2 (*2 for the weights after activation) = 600,000,000 weights. So training can be slow. this is solved via negative sampling. This is done by removing the words we don’t want to predict for optimization. 
Suppose the **aardvark** wants to predict the word **A.
S**o, in the input only aardvark has 1  in it. So we can ignore all the other weights coming from other words. Because the other words multiply their weights by 0

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2041.png)

So, we are now left with 300,000,000 weights

Now, after this word2Vec randomly selects 2 to 20 words that we don’t want to predict. Suppose here we choose “**abandon”** as that undesired prediction.
We know that  “A” is the desired prediction 

Now we can ignore all the other weights and optimize with **A and abandon**

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2042.png)

### Skip-Gram Model

- **Definition**: The Skip-Gram model is a type of word embedding model that uses a word to predict its surrounding words within a fixed-size window. This model is trained to maximize the likelihood of observing the context words given a central word.
- **Example**:
    - For the sentence "I want a glass of orange juice to go along with my cereal," if "juice" is the central word, the Skip-Gram model will try to predict "glass," "orange," etc., as context words.

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2043.png)

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2044.png)

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2045.png)

*How to sample context `c`*:

- One thing you could do is just sample uniformly, at random, from your training corpus.
    - When we do that, you find that there are some words like `the, of, a, and, to` and so on that appear extremely frequently.
    - In your context to target mapping pairs just get these these types of words extremely frequently, whereas there are other words like `orange`, `apple`, and also `durian` that don't appear that often.
- In practice the distribution of words `p(c)` isn't taken just entirely uniformly at random for the training set purpose, but instead there are different heuristics that you could use in order to balance out something from the common words together with the less common words.

This ensures that $e_c$ of not only the most frequent words is updated but also of the less frequent words

**Negative Sampling**

first one is the positive example i.e with the correct target
The subsequent rows are the context with the wrong word i.e with random word from the dict (NEGATIVE EXAMPLES)

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2046.png)

on every iteration, we are going to train only ( k + 1 ) of them and not updating a 10000 way softmax classifier

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2047.png)

How to select The negative examples

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2048.png)

where f is the frequency of the word

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2049.png)

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2050.png)