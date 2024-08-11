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

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2046.png)

*How to sample context `c`*:

- One thing you could do is just sample uniformly, at random, from your training corpus.
    - When we do that, you find that there are some words like `the, of, a, and, to` and so on that appear extremely frequently.
    - In your context to target mapping pairs just get these these types of words extremely frequently, whereas there are other words like `orange`, `apple`, and also `durian` that don't appear that often.
- In practice the distribution of words `p(c)` isn't taken just entirely uniformly at random for the training set purpose, but instead there are different heuristics that you could use in order to balance out something from the common words together with the less common words.

This ensures that $e_c$ of not only the most frequent words is updated but also of the less frequent words

**Negative Sampling**

first one is the positive example i.e with the correct target
The subsequent rows are the context with the wrong word i.e with random word from the dict (NEGATIVE EXAMPLES)

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2047.png)

on every iteration, we are going to train only ( k + 1 ) of them and not updating a 10000 way softmax classifier

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2048.png)

How to select The negative examples

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2049.png)

where f is the frequency of the word

GloVe (global vector for word representation)

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2050.png)

minimizing this square cost function allows you to learn meaningful word embeddings

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/518d809e-e533-476a-b985-1876364e8458.png)

**Sentiment Classification**

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2051.png)

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2052.png)

Here we are summing everything into a big vector.
But since the sequence of words also matters and not just the embeddings, (like here there is a lot of repetition of the word “good” but still it is a negative review), so we need a rnn model here

many to one: 

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2053.png)

Therefore, Instead of just summing all of your word embeddings, you can instead use a RNN for sentiment classification.

**Debiasing Word Embeddings**

Word embeddings maybe have the bias problem such as gender bias, ethnicity bias and so on. As word embeddings can learn analogies like man is to woman like king to queen. The paper shows that a learned word embedding might output:

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2054.png)

**Reducing bias in word embeddings**

- *Identify bias direction*
    - The first thing we're going to do is to identify the direction corresponding to a particular bias we want to reduce or eliminate.
    - And take a few of these differences and basically average them. And this will allow you to figure out in this case that what looks like this direction is the gender direction, or the bias direction. Suppose we have a 50-dimensional word embedding.
        
        ![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/9c06af32-b9ec-48ca-9b6b-34beafd3632b.png)
        
    - Then we have
        - `cosine_similarity(sophie, g)) = 0.318687898594`
        - `cosine_similarity(john, g)) = -0.23163356146`
        - to see male names tend to have positive similarity with gender vector whereas female names tend to have a negative similarity. This is acceptable.
    - But we also have
        - `cosine_similarity(computer, g)) = -0.103303588739`
        - `cosine_similarity(singer, g)) = 0.185005181365`
        - It is astonishing how these results reflect certain unhealthy gender stereotypes.
    - The bias direction can be higher than 1-dimensional. Rather than taking an average, SVD (singular value decomposition) and PCA might help.
- *Neutralize*
    - For every word that is not definitional, project to get rid of bias.
    - That means put them in the non-bias direction(eg babysitter, doctor are gender neutral)
    
    ![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2055.png)
    
- *Equalize pairs*
    - In the final equalization step, what we'd like to do is to make sure that words like grandmother and grandfather are both exactly the same similarity, or exactly the same distance, from words that should be gender neutral, such as babysitter or such as doctor.
    - The key idea behind equalization is to make sure that a particular pair of words are equi-distant from the 49-dimensional g⊥.
    

Quiz:

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2056.png)

## Week 3 sequence models and attention mechanism

**Basic Models**

For converting to english translation :
Use an encoder network that inputs a french sentence that will generate the and then a decoder to decode the vector and output the english sentence

**Image captioning**

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2057.png)

input an image into a convolutional neural network(maybe a pretrained alexNet) and have that learn the features of the input image. And if we get rid of the final softmax unit, the pretrained alexnet will give us a 4096 dimensional feature vector which represents the input image
so, this pretrained network will be the input vector of the encoder

Then we can feed this to an rnn to output one word at a time 

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2058.png)

machine translation is quite similar to language model

The decoder part of the machine translation is same a language model

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2059.png)

- You can use a language model to estimate the probability of a sentence.
- The decoder network of the machine translation model looks pretty much identical to the language model, except that instead of always starting along with the vector of all zeros, it has an encoder network that figures out some representation for the input sentence.
- Instead of modeling the probability of any sentence, it is now modeling the probability of the output English translation conditioned on some input French sentence. In other words, you're trying to estimate the probability of an English translation.

Its not optimal to pick one word at a time as we see that “going” is a much better word considering x but the whole translation does not end up being the very best

![Untitled](course%205%201b7ff401be374ddc93cfd54496cb5a8c/Untitled%2060.png)

It is better to use an approximate search algorithm. Here, it will try to pick the sentence y that maximizes the conditional probability

**Beam search ⇒ approximate search algorithm**

given an input sentence, we do not want to output a random translation but rather the best and a more likely translation. 
Beam search is an algo to this .

If B = 3, then we need 3 copies of the network.
Now, we pick the 3 words with the most probabilty of being the first word. So we have 3 choices of first word

![image.png](course%205%201b7ff401be374ddc93cfd54496cb5a8c/image.png)

Now since for each 1st word out of the 3 choices, the 2nd word can be any of the 10,000 words from the vocab. so, we have total 10,000 * 3 = 30,000 choices. Each first word is hardcoded into one of the neural network copies

now out of these 30,000 choices we will select the ones that have the most probability of being the first 2 words
eg . “in september”, “jane is”, “jane visits” will be selected as they have the highest probability

Now for the 3 word we hardwire the 3 possibilities of the first 2 words in each of the NN

Now  again, we will pick the top 3 possibilies of the first 3 words of the sentence 

![image.png](course%205%201b7ff401be374ddc93cfd54496cb5a8c/image%201.png)

So, beam search is maximising the probability of the next word given the previous words and the input sentence(french sentence)

B = 1 means greedy search 

**Refinements to Beam Search**

we were maximizing the product of probabilities till now. 
But multiplying a lot of numbers less than 1 will result in a very tiny number, which can result in numerical underflow.

So now we will maximise the sum of the log of probabilities

![image.png](course%205%201b7ff401be374ddc93cfd54496cb5a8c/image%202.png)

- If you have a very long sentence, the probability of that sentence is going to be low, because you're multiplying many terms less than 1. And so the objective function (the original version as well as the log version) has an undesirable effect, that maybe it unnaturally tends to prefer very short translations. It tends to prefer very short outputs. B**ecause shorter sentences have higher probability due to less words**

### **Solution: Normalized Log-Likelihood Objective**

- **Normalized Log-Likelihood:** To counter this bias, a normalized log-likelihood objective is sometimes used. The idea is to adjust the log-likelihood by normalizing it, for example, by the length of the sentence. This normalization reduces the tendency of the model to favor shorter sentences, making it more likely to produce translations of appropriate length, even when they are longer.
- **Why Normalize:** By normalizing the log-likelihood, you prevent the model from being unfairly biased towards shorter outputs. It helps the model focus on the actual quality of the translation, rather than just the length.

**A normalized log-likelihood objective:**

![image.png](course%205%201b7ff401be374ddc93cfd54496cb5a8c/image%203.png)

Ty = vocab size

alpha is also an hyperparameter. If alpha = 0 then we do not normalise by length

**size of B ⇒**

large B = considers more possibilities(so better result) but slower computation
small B = less possibilities but faster computation

![image.png](course%205%201b7ff401be374ddc93cfd54496cb5a8c/image%204.png)

**Error Analysis in Beam Search**

- Beam search is an approximate search algorithm, also called a heuristic search algorithm. And so it doesn't always output the most likely sentence.
- In order to know whether it is the beam search algorithm that's causing problems and worth spending time on, or whether it might be the RNN model that's causing problems and worth spending time on, we need to do error analysis with beam search.
- Getting more training data or increasing the beam width might not get you to the level of performance you want.
- You should break the problem down and figure out what's actually a good use of your time.
- *The error analysis process:*
    - Problem:
        - To translate: `Jane visite l’Afrique en septembre.` (x)
        - Human: `Jane visits Africa in September.` (y)
            - 
        - Algorithm: `Jane visited Africa last September.` (ŷ) which has some error.
        
        y* ⇒ human translation
        
        y hat ⇒ by algorithm
        

![image.png](course%205%201b7ff401be374ddc93cfd54496cb5a8c/image%205.png)

If probability of the human translation is greater but still the less prob sentence is picked then the fault is of beam search bcuz its his responsibility to pick the sentence with the max probability
Job of beam search ⇒ maximise the prob of the output sentence

But if P of the human translation is less than its not the fault of the beam search

![image.png](course%205%201b7ff401be374ddc93cfd54496cb5a8c/image%206.png)

**Bleu Score (Optional)**

used when the human references are given

Blue ⇒ bilingual evaluation understudy 

![image.png](course%205%201b7ff401be374ddc93cfd54496cb5a8c/image%207.png)

we will also look over pairs of words

![image.png](course%205%201b7ff401be374ddc93cfd54496cb5a8c/image%208.png)

clip count ⇒ max number of times the bigram appears in either reference 1 or reference 2
count ⇒ the max times the bigram appears the candidate

Now, for bleu score, P_n  = (sum of clip count) / (sum of count)

![image.png](course%205%201b7ff401be374ddc93cfd54496cb5a8c/image%209.png)

blue score calculation ⇒

![image.png](course%205%201b7ff401be374ddc93cfd54496cb5a8c/image%2010.png)

**Attention Model Intuition**

We‘ve been using an Encoder-Decoder architecture for machine translation. Where one RNN reads in a sentence and then different one outputs a sentence. its difficult for the NN to memorize a super long sentence. so, for long sentence the encoder decoder model may not work well as it first encodes the whole sentence. 

There's a modification to this called the Attention Model that makes all this work much better.

The French sentence:

> Jane s'est rendue en Afrique en septembre dernier, a apprécié la culture et a rencontré beaucoup de gens merveilleux; elle est revenue en parlant comment son voyage était merveilleux, et elle me tente d'y aller aussi.
> 

The english translation :

> Jane went to Africa last September, and enjoyed the culture and met many wonderful people; she came back raving about how wonderful her trip was, and is tempting me to go too.
> 

The way a human translator would translate this sentence is not to first read the whole French sentence and then memorize the whole thing and then regurgitate an English sentence from scratch. Instead, what the human translator would do is read the first part of it, maybe generate part of the translation, look at the second part, generate a few more words, look at a few more words, generate a few more words and so on.

![image.png](course%205%201b7ff401be374ddc93cfd54496cb5a8c/image%2011.png)

The Encoder-Decoder architecture above is that it works quite well for short sentences, so we might achieve a relatively high Bleu score, but for very long sentences, maybe longer than 30 or 40 words, the performance comes down. (The blue line)

green ⇒ through attention model
blue ⇒  by NN memorising the whole sentence 

![image.png](course%205%201b7ff401be374ddc93cfd54496cb5a8c/image%2012.png)

First step of RNN

![image.png](course%205%201b7ff401be374ddc93cfd54496cb5a8c/image%2013.png)

alpha<1,1> denotes How much attention should we be paying to first word of the french input sentence for computing the 1st word of the translation.

So, to generalise, alpha<t, t’> indicates how much attention should you give to the $t^{'th}$ word of the input french sentence to generate the $t^{th}$ word of the english sentence

The context C depends on the various alpha coming 
S denotes State

![image.png](course%205%201b7ff401be374ddc93cfd54496cb5a8c/image%2014.png)

Assume you have an input sentence and you use a bidirectional RNN, or bidirectional GRU, or bidirectional LSTM to compute features on every word. In practice, GRUs and LSTMs are often used for this, maybe LSTMs be more common. The notation for the Attention model is shown below.

![image.png](course%205%201b7ff401be374ddc93cfd54496cb5a8c/image%2015.png)

how to calculate attention weights?

Compute e<t,t'> using a small neural network:

![image.png](course%205%201b7ff401be374ddc93cfd54496cb5a8c/image%2016.png)

• One downside to this algorithm is that it does take quadratic time or quadratic cost to run this algorithm. If you have Tx words in the input and Ty words in the output then the total number of these attention parameters are going to be Tx * Ty.

Visualize the attention weights 𝛼<t,t'>:

![image.png](course%205%201b7ff401be374ddc93cfd54496cb5a8c/image%2017.png)

**Speech Recognition**

**Speech recognition**

- What is the speech recognition problem? You're given an audio clip, x, and your job is to automatically find a text transcript, y.
- So, one of the most exciting trends in speech recognition is that, once upon a time, speech recognition systems used to be built using *phonemes* and this were, I want to say, hand-engineered basic units of cells.
    - Linguists use to hypothesize that writing down audio in terms of these basic units of sound called phonemes would be the best way to do speech recognition.
- But with end-to-end deep learning, we're finding that phonemes representations are no longer necessary. But instead, you can built systems that input an audio clip and directly output a transcript without needing to use hand-engineered representations like these.
    - One of the things that made this possible was going to much larger data sets.
    - Academic data sets on speech recognition might be as a 300 hours, and in academia, 3000 hour data sets of transcribed audio would be considered reasonable size.
    - But, the best commercial systems are now trains on over 10,000 hours and sometimes over a 100,000 hours of audio.

![image.png](course%205%201b7ff401be374ddc93cfd54496cb5a8c/image%2018.png)

*How to build a speech recognition?*

- **Attention model for speech recognition**: one thing you could do is actually do that, where on the horizontal axis, you take in different time frames of the audio input, and then you have an attention model try to output the transcript like, "the quick brown fox".

![image.png](course%205%201b7ff401be374ddc93cfd54496cb5a8c/image%2019.png)

**CTC cost for speech recognition**

![image.png](course%205%201b7ff401be374ddc93cfd54496cb5a8c/image%2020.png)

- For simplicity, this is a simple of what uni-directional for the RNN, but in practice, this will usually be a bidirectional LSTM and bidirectional GRU and usually, a deeper model. But notice that the number of time steps here is very large and in speech recognition, usually the number of input time steps is much bigger than the number of output time steps.
    - For example, if you have 10 seconds of audio and your features come at a 100 hertz so 100 samples per second, then a 10 second audio clip would end up with a thousand inputs. But your output might not have a thousand alphabets or characters.
- The CTC cost function allows the RNN to generate an output like `ttt_h_eee___[]___qqq__`, here `_` is for "blank", `[]` for "space".
- The basic rule for the CTC cost function is to collapse repeated characters not separated by "blank".

![image.png](course%205%201b7ff401be374ddc93cfd54496cb5a8c/image%2021.png)

With a RNN what we really do, is to take an audio clip, maybe compute spectrogram features, and that generates audio features x<1>, x<2>, x<3>, that you pass through an RNN. So, all that remains to be done, is to define the target labels y.

- In the training set, you can set the target labels to be zero for everything before that point, and right after that, to set the target label of one. Then, if a little bit later on, the trigger word was said again at this point, then you can again set the target label to be one.

![image.png](course%205%201b7ff401be374ddc93cfd54496cb5a8c/image%2022.png)

quiz:

![image.png](course%205%201b7ff401be374ddc93cfd54496cb5a8c/image%2023.png)