# Course 5: Sequence Models

## Week 1:

## Why Sequence Models?

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled.png)

---

## Notation

- The input sequence is a sentence, and the goal is to determine if each word is part of a person's name.
- The notation used includes X to represent the input sequence, Y to represent the target output, and T_x and T_y to denote the lengths of the input and output sequences.
- Individual words in the sentence are represented using a vocabulary or dictionary, with each word assigned a unique index.
- One-hot representations are used to represent each word, with a vector of zeros and a single one indicating the position of the word in the dictionary.
- If a word is not in the vocabulary, a special token called "Unknown Word" is used to represent it.
- The goal is to learn a mapping from the input sequence to the target output using a sequence model, such as a recurrent neural network.

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%201.png)

## Recurrent Neural Network Model

- The inputs and outputs can have different lengths in different examples.
- A standard neural network architecture does not share features learned across different positions of texts.
- Recurrent Neural Networks (RNNs) address these limitations.
- RNNs use a hidden layer that takes input from the previous time step to make predictions.

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%202.png)

## Backpropagration Through Time

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%203.png)

## Different Types of RNNs

- RNN architectures can have different numbers of inputs (Tx) and outputs (Ty).
- Different types of RNN architectures include:
    - Many-to-Many: Multiple inputs and multiple outputs, with the input and output sequences having the same length.
    - Many-to-One: The RNN reads a sequence of inputs and outputs a single output at the last time-step.
    - One-to-Many: Used for tasks like music generation.
    - One-to-One: A standard neural network architecture with one input and one output.
    - Many-to-One: Used for tasks like sentiment classification, where the RNN reads a sequence of inputs and outputs a single output.
    - Many-to-One: Used for tasks like machine translation, where the input and output sequences can have different lengths.

---

## Language Model and Sequence Generation

- A language model estimates the probability of a given sentence.
- To build a language model using an RNN, you need a training set of text.
- Tokenization is the process of mapping words to vectors or indices.
- An RNN predicts the probability of the next word in a sequence.

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%204.png)

---

- To sample from the model, you first randomly choose the first word based on the softmax distribution.
- Then, you pass the sampled word as input to the next time step and sample the next word.
- This process continues until you reach the end of the sequence or a specified number of time steps.
- The content also discusses the difference between word-level and character-level language models.
- Character-level models can handle unknown words but have longer sequences and are more computationally expensive.
- Word-level models are more commonly used but character-level models have specialized applications.

---

## Vanishing Gradients with RNNs

- The vanishing gradient problem occurs because the error associated with later timesteps has a hard time propagating back to affect the computations of earlier timesteps.
- The basic RNN model has many local influences, meaning that the output at a certain timestep is mainly influenced by values close to that timestep.
- Exploding gradients can also occur in RNNs, where the gradients increase exponentially and can cause the parameters of the neural network to become unstable.
- **Gradient Clipping:** rescaling the gradient vectors if they exceed a certain threshold.

---

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%205.png)

## Gated Recurrent Unit (GRU)

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%206.png)

- The GRU has a memory cell (C) that provides memory for long-range connections and helps with the vanishing gradient problem.
- The GRU unit computes the output activation (a) and can be used to make predictions (y hat).
- The GRU unit has two gates: the update gate (Gamma_u) and the relevance gate (Gamma_r).
- The update gate determines when to update the memory cell value, while the relevance gate determines the relevance of the previous memory cell value.
- The equations governing the computations of a GRU unit are explained, including the candidate value for the memory cell (c tilde) and the update equation for the memory cell (C).

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%207.png)

---

## LSTM

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%208.png)

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%209.png)

- LSTM units are more powerful than GRU units and can learn long-range connections in a sequence.
- LSTM units have three gates: the update gate, the forget gate, and the output gate.
- The update gate controls whether or not to update the memory cell with a candidate value.
- The forget gate controls which information to forget from the memory cell.
- The output gate determines the output of the LSTM unit.
- LSTM units are capable of capturing long-term dependencies in sequences.
- GRU units are simpler and faster than LSTM units, but LSTM units are more powerful and flexible.
- Historically, LSTM units have been the default choice, but GRUs have gained momentum in recent years.

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%2010.png)

---

## Bidirectional RNN

- The forward recurrent units influence the current input, while the backward recurrent units influence the previous inputs.
- The network computes the forward activations first and then the backward activations.
- The bidirectional RNN can make predictions anywhere in the sequence by taking into account information from the past, present, and future.

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%2011.png)

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%2012.png)

---

## Week 2:

## Word Representation

Previously we used one-hot vectors for each word with a size of the vocabulary of words we had which was 10k words so it didnt help to create and relation/context between any two words.

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%2013.png)

Rather than having one hot vectors of the size of vocabulary we can have them in the size of how many features we can get, which also helps to find similarities between words.

## Using Word Embeddings

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%2014.png)

We also embed/encode images for face recognition systems.

---

## Properties of Words Embeddings

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%2015.png)

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%2016.png)

**t-SNE(t-distributed stochastic neighbor embedding)**   **is** **a statistical method that visualizes high-dimensional data by mapping it to a two or three-dimensional space**. 

Also many of the parallelogram analogies will be broken using t-SNE.

### Finding Similarity:

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%2017.png)

---

## Embedding Matrix

We have an embedding matrix of order (embedding dimensions, no of words) and we multiply it by the one hot vector of the word whose embedding vector we want.

---

## Learning Word Embeddings

First we get the embedding vectors of all previous words.

then a FC layer and then a softmax to predict the next words.

We can also have a hyperparameter to decide how many previous words to be used for predicting the next word.

This method is used to learn word embeddings.

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%2018.png)

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%2019.png)

---

## Word2Vec

Simpler and computationally efficient algorithm.

1. Skip-grams : The model learns the mapping from a context word to a target word within a certain window.
2. The model uses a softmax unit to output the probabilities of different target words given the context word.
3. The loss function for softmax is -y*log(y_hat).

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%2020.png)

We also use a hierarchical softmax classifier to reduce computation time for large vocabulary sizes.

The skip-gram model is one version of the Word2Vec algorithm, with another version called CBOW (continuous bag-of-words).

The key problem with the skip-gram model is the computational cost of the softmax step.

---

## Negative Sampling

- Negative sampling involves creating a supervised learning problem where the goal is to predict whether a pair of words is a context-target pair or not.
- Positive examples are generated by sampling a context word and a target word that appear together, while negative examples are generated by randomly selecting words from the dictionary.

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%2021.png)

Here, **k is a parameter which tells how many negative examples to use for each positive one.**

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%2022.png)

This turns our problem from 10k softmax to 10k binary classification problems.  

In one iteration we only train the positive and negative examples we took rather than all vocab words.

The choice of negative examples can be based on the empirical frequency of words in the corpus, with a heuristic value of the word **frequency raised to the power of 3/4** often used.

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%2023.png)

---

## GloVe Word Vectors

- The algorithm optimizes the difference between theta i transpose e j and log of X ij squared.
- X ij represents the number of times word i appears in the context of word j.
- The algorithm uses gradient descent to minimize the sum of the difference for all word pairs.
- The algorithm handles cases where X ij is equal to zero by adding an extra weighting term.
- The weighting factor ensures that frequent and infrequent words are given appropriate weight.

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%2024.png)

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%2025.png)

- The individual components of word embeddings learned using GloVe may not be easily interpretable.
- Despite this, the parallelogram map for figure analogies still works with GloVe embeddings.

---

## Sentiment Classification

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%2026.png)

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%2027.png)

As this simple model does not take in word order into consideration, So in the 4th review the word “good” is appearing many times and the model may predict this review with more stars, So thats why we will RNN for sentiment classification.

### RNN for sentiment classification

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%2028.png)

---

## Debiasing Word Embeddings

The problem of bias is mainly related to gender, ethinicity, age, sexual orientation and other biases of text used to train the model.

1. Man : Woman as King : Queen
2. Man: Programmer as Woman: Homemaker
3. Father: Doctor as Mother: Nurse
- One approach to reducing bias in word embeddings is to identify the bias direction and the non-bias direction.
- The next steps involve neutralizing words that are not definitional and equalizing pairs of words to eliminate bias.

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%2029.png)

Handpicking the pairs we want to neutralize is also feasible.

---

## Week 3:

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%2030.png)

- Encoder: The encoder network takes an input sequence, such as a sentence in a source language, and processes it to create a fixed-length representation called the "context vector" or "thought vector". The encoder network captures the important information from the input sequence and encodes it into a dense vector representation.
- Decoder: The decoder network takes the context vector generated by the encoder and uses it to generate an output sequence, such as a translated sentence in a target language. The decoder network uses the context vector as an initial input and generates one word at a time, conditioning each word on the previously generated words.

### Picking the Most likely sentence

• Greedy search, where words are chosen one at a time based on their likelihood, is not always optimal for finding the best translation.
• Approximate search is a common algorithm used to find the most likely translation by maximizing the conditional probability.

## Beam Search

• Beam search considers multiple alternatives instead of just one, using a parameter called the **beam width.**

• The algorithm keeps track of the top choices at each step and narrows down the possibilities.

• Beam search can be more accurate than greedy search, which only considers the most likely word at each step.

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%2031.png)

![Untitled](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/Untitled%2032.png)

---

### Refinements:

### Length Normalization

Since these probabilities are all numbers less than 1, multiplying a lot of numbers less than 1 will result in a tiny, tiny, tiny number, which can result in numerical underflow. So in practice, instead of maximizing this product, we will take logs and log of a product becomes sum of a log which is the 2nd formula. This provides us with a numerically stable algorithm that is less prone to rounding errors. Because the log function is strictly monotonically increasing function, we know that maximizing log P(y) given x should give the same result as maximizing P(y) given x.

![image.png](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/image.png)

![image.png](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/image%201.png)

---

## Error Analysis in Beam Search

- The process involves computing the probabilities of the correct translation (y*) and the generated translation (y-hat) using the RNN model.
- If P(y* given x) is greater than P(y-hat given x), it suggests that beam search failed to find the best translation and is at fault.
- If P(y* given x) is less than or equal to P(y-hat given x), it suggests that the RNN model is at fault for generating a less likely translation.

---

## Bleu Score

Bilingual Evaluation Understudy

- The BLEU score is a metric used to evaluate the quality of machine translation output.
- It measures the degree to which the machine translation output overlaps with the reference translations.
- The BLEU score can be computed on different n-grams, such as unigrams, bigrams, trigrams, and higher values of n.
- The modified precision is used to calculate the BLEU score for each n-gram.
- To compute the final BLEU score, the modified precision scores for each n-gram are averaged.

![image.png](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/image%202.png)

![image.png](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/image%203.png)

![image.png](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/image%204.png)

---

## Attention Model Intuition

The model uses attention weights to determine how much attention should be given to each word in the input sentence.

![image.png](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/image%205.png)

By focusing on a local window of the input sentence, the model can generate translations more accurately.

---

## Attention Model

The context vector is a weighted sum of the features from different time steps, based on the attention weights.

![image.png](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/image%206.png)

The attention model can be computationally expensive, as it requires computing attention weights for each word in the input sentence. This results in a **quadratic cost** if the input and output sentences are long. However, there are research efforts to reduce the computational cost of the attention model.

![image.png](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/image%207.png)

![image.png](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/image%208.png)

---

## Speech Recognition

End-to-end deep learning models have made it possible to directly input an audio clip and output a transcript, without the need for hand-engineered representations like phonemes.

Phonemes: smallest distinct units of sound that can change the meaning of a word.  the word "cat" is made up of three phonemes: /k/ /æ/ /t/.

![image.png](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/image%209.png)

 In speech recognition, the number of **input time steps (audio frames)** is often **larger than** the number of **output time steps (transcript characters).** This misalignment makes it challenging to directly map the input to the output. The **CTC cost function allows for flexible alignment** between the input and output sequences.

Blank Character: The CTC cost function introduces a **special blank character** to represent **gaps or repeated characters in the output sequence**. This blank character **helps collapse repeated characters that are not separated by other characters,** simplifying the output sequence.

---

## Trigger Word Detection

![image.png](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/image%2010.png)

The target labels for training a trigger word detection system can be set to 0 for everything before the trigger word and 1 right after it.

![image.png](Course%205%20Sequence%20Models%20f0c078ae211d4963a2bf701e92125782/image%2011.png)