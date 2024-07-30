# Course 5: Sequence Models

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