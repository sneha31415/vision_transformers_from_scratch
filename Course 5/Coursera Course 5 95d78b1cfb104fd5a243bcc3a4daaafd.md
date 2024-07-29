# Coursera Course 5

# Week 1

## Why Sequence Models?

![Untitled](Coursera%20Course%205%2095d78b1cfb104fd5a243bcc3a4daaafd/Untitled.png)

## Notation

![Untitled](Coursera%20Course%205%2095d78b1cfb104fd5a243bcc3a4daaafd/Untitled%201.png)

![Untitled](Coursera%20Course%205%2095d78b1cfb104fd5a243bcc3a4daaafd/Untitled%202.png)

## Recurrent Neural Network Model

a<0> can be a vector of zeros or of random values.

One Weakness of RNN is that it uses previous data and not lter data for predictions.

tanh is generally used activation function, or even ReLU. Can use sigmoid for binary classification, or softmax also.

![Untitled](Coursera%20Course%205%2095d78b1cfb104fd5a243bcc3a4daaafd/Untitled%203.png)

![Untitled](Coursera%20Course%205%2095d78b1cfb104fd5a243bcc3a4daaafd/Untitled%204.png)

![Untitled](Coursera%20Course%205%2095d78b1cfb104fd5a243bcc3a4daaafd/Untitled%205.png)

![Untitled](Coursera%20Course%205%2095d78b1cfb104fd5a243bcc3a4daaafd/Untitled%206.png)

## Backpropagation through time

![Untitled](Coursera%20Course%205%2095d78b1cfb104fd5a243bcc3a4daaafd/Untitled%207.png)

## Different types of RNN

![Untitled](Coursera%20Course%205%2095d78b1cfb104fd5a243bcc3a4daaafd/Untitled%208.png)

![Untitled](Coursera%20Course%205%2095d78b1cfb104fd5a243bcc3a4daaafd/Untitled%209.png)

![Untitled](Coursera%20Course%205%2095d78b1cfb104fd5a243bcc3a4daaafd/Untitled%2010.png)

## Language Model and sequence generation

![Untitled](Coursera%20Course%205%2095d78b1cfb104fd5a243bcc3a4daaafd/Untitled%2011.png)

![Untitled](Coursera%20Course%205%2095d78b1cfb104fd5a243bcc3a4daaafd/Untitled%2012.png)

![Untitled](Coursera%20Course%205%2095d78b1cfb104fd5a243bcc3a4daaafd/Untitled%2013.png)

## Sampling Novel Sqeuences

![Untitled](Coursera%20Course%205%2095d78b1cfb104fd5a243bcc3a4daaafd/Untitled%2014.png)

![Untitled](Coursera%20Course%205%2095d78b1cfb104fd5a243bcc3a4daaafd/Untitled%2015.png)

![Untitled](Coursera%20Course%205%2095d78b1cfb104fd5a243bcc3a4daaafd/Untitled%2016.png)

## Vanishing Gradients with RNNs

For very long sentences we might not be able to keep track of previous changes to affect later changes effectively.

Exploding Gradients - Gradient clipping

![Untitled](Coursera%20Course%205%2095d78b1cfb104fd5a243bcc3a4daaafd/Untitled%2017.png)

## GRU

![Untitled](Coursera%20Course%205%2095d78b1cfb104fd5a243bcc3a4daaafd/Untitled%2018.png)

![Untitled](Coursera%20Course%205%2095d78b1cfb104fd5a243bcc3a4daaafd/Untitled%2019.png)

![Untitled](Coursera%20Course%205%2095d78b1cfb104fd5a243bcc3a4daaafd/Untitled%2020.png)

## Long Short Term Memory (LSTM)

A modern version of LSTMs ca have c and a nterchangale. 

Only corresponding elements of the gates can affect each other.

GRU is simpler and faster, LSTM is flexible and powerful.

![Untitled](Coursera%20Course%205%2095d78b1cfb104fd5a243bcc3a4daaafd/Untitled%2021.png)

![Untitled](Coursera%20Course%205%2095d78b1cfb104fd5a243bcc3a4daaafd/Untitled%2022.png)

## BRNN

BRNN with LSTM is generally preferred.

Disadvantage - need entire sequence of data before making prediction.

![Untitled](Coursera%20Course%205%2095d78b1cfb104fd5a243bcc3a4daaafd/Untitled%2023.png)

![Untitled](Coursera%20Course%205%2095d78b1cfb104fd5a243bcc3a4daaafd/Untitled%2024.png)

## Deep RNNs

![Untitled](Coursera%20Course%205%2095d78b1cfb104fd5a243bcc3a4daaafd/Untitled%2025.png)