# Deep Learning

# Course 4: Convolutional Neural Networks

## Computer Vision

Problems:

- Image Classification
- Object Detection
- Neural Style Transfer

One of the challenges in computer vision is dealing with large input sizes, but convolutional neural networks provide a solution by implementing the convolution operation.

### Edge Detection

The process involves convolving a filter matrix with an input image to produce an output matrix. The filter/kernel matrix is a small matrix that is moved across the image, computing element-wise products and summing them to obtain the output matrix. The resulting matrix can be interpreted as an image that highlights the detected edges.

![Untitled](Deep%20Learning%208dcaec6fa9734afb8acadc409ee154b6/Untitled.png)

![Untitled](Deep%20Learning%208dcaec6fa9734afb8acadc409ee154b6/Untitled%201.png)

Instead of hand-coding edge detectors, we can use deep learning algorithms to learn the parameters of the filters automatically. This allows for more robust and flexible edge detection. 

![Untitled](Deep%20Learning%208dcaec6fa9734afb8acadc409ee154b6/Untitled%202.png)

## Padding

Padding is a modification to the basic convolutional operation that helps address **two downsides**: the **shrinking output size** and the l**oss of information near the edges** of the image. By padding the image with additional pixels, we can **preserve the original input size** and ensure that all pixels in the image contribute to the output. 

The convention is to pad by **zeros.**

The padding amount can be specified using the formula **n + 2p - f + 1,** where n is the input size, p is the padding amount, and f is the filter size

- Valid convolutions: no padding
- Same convolutions: output size is the same as the input size
- f is usually odd

![Untitled](Deep%20Learning%208dcaec6fa9734afb8acadc409ee154b6/Untitled%203.png)

## Strided Convolutions

Stride is the no of steps we move the kernel by after each convolution operation

**$output size = (input size + 2 * padding - filter size) / stride + 1$.** We also learn that if the result is not an integer, it is rounded down. 

![Untitled](Deep%20Learning%208dcaec6fa9734afb8acadc409ee154b6/Untitled%204.png)

In convolution we are needed to flip the filter matrix along the horizontal and vertical axis, but in most deep learning literatures we do this without the flipping and still call it convolution but it is cross correlation.

## Convolutions Over Volume

![Untitled](Deep%20Learning%208dcaec6fa9734afb8acadc409ee154b6/Untitled%205.png)

n x n x nc * f x f x nc = (n-f+1)x(n-f+1)xnc’

---

## One Layer of a Convolutional Network

![Untitled](Deep%20Learning%208dcaec6fa9734afb8acadc409ee154b6/Untitled%206.png)

![Untitled](Deep%20Learning%208dcaec6fa9734afb8acadc409ee154b6/Untitled%207.png)

---

## Simple Convolutional Network

![Untitled](Deep%20Learning%208dcaec6fa9734afb8acadc409ee154b6/Untitled%208.png)

### Pooling Layers

Convolutional Neural Networks (ConvNets) often use pooling layers to **reduce the size of the representation**, speed up computation, and make features more robust.

• **Max pooling** involves breaking the input into regions and taking the maximum value from each region to create the output.
• **Average pooling** is another type of pooling that takes the average value from each region.
• Max pooling is used more frequently than average pooling in ConvNets.
• Pooling has hyperparameters such as **filter size (f) and stride (s), which are not learnable.**

---

## CNN Example

![Untitled](Deep%20Learning%208dcaec6fa9734afb8acadc409ee154b6/Untitled%209.png)

![Untitled](Deep%20Learning%208dcaec6fa9734afb8acadc409ee154b6/Untitled%2010.png)

### Why convolutions?

- **Parameter Sharing:** allows feature detectors to be used in multiple positions in an image, reducing the number of parameters needed.
- **Sparsity of connections** means that each output unit is only connected to a subset of input features, further reducing the number of parameters.

---

## Classic Networks

![Untitled](Deep%20Learning%208dcaec6fa9734afb8acadc409ee154b6/Untitled%2011.png)

## LeNet-5

![Untitled](Deep%20Learning%208dcaec6fa9734afb8acadc409ee154b6/Untitled%2012.png)

---

![Untitled](Deep%20Learning%208dcaec6fa9734afb8acadc409ee154b6/Untitled%2013.png)

---

VGG-16 is a relatively large network with approximately 138 million parameters.

Its simplicity and effectiveness have made it a popular choice for various computer vision tasks.

![Untitled](Deep%20Learning%208dcaec6fa9734afb8acadc409ee154b6/Untitled%2014.png)

---

## ResNets

**Skip connections** allow the activation from one layer to be fed to another layer deeper in the network, which helps to overcome the **vanishing and exploding gradient problems.** The main concept introduced is the residual block, which consists of a shortcut connection that allows information to bypass certain layers and flow directly to deeper layers. 

ResNet has been shown to be effective in training very deep networks, even with over 100 layers. The performance of the training error tends to improve as the number of layers increases, unlike in plain networks without skip connections.   

![Untitled](Deep%20Learning%208dcaec6fa9734afb8acadc409ee154b6/Untitled%2015.png)

---

## 

In plain networks, their performance on the train set is hurt if they are too much deeper.

ResNets' success lies in their ability to learn the identity function(if it cannot learn that feature it just passes the same input further in the network), which allows them to easily copy the activation from one layer to another. By adding skip connections, ResNets can learn the identity function and maintain or improve performance even with the addition of extra layers. 

The use of same convolutions helps preserve dimensions and enables the short-circuit connections. ResNets are commonly used in image recognition tasks.

![Untitled](Deep%20Learning%208dcaec6fa9734afb8acadc409ee154b6/Untitled%2016.png)

---

## Networks in Networks and 1x1 convolutions

• One-by-one convolutions are more useful when applied to images with multiple channels.

• It looks at each position in the image, takes the element-wise product between the numbers in the filter and the image, and applies a ReLU nonlinearity.

## Inception Network Motivation

- Instead of choosing a specific filter size or pooling layer, the Inception network uses a combination of them all.
- This makes the network architecture more complex but also improves its performance.
- The Inception network reduces computational cost by using 1x1 convolutions as bottleneck layers.
- These bottleneck layers shrink the representation size before increasing it again.
- Shrinking the representation size significantly reduces the computational cost without significantly impacting performance.

![Untitled](Deep%20Learning%208dcaec6fa9734afb8acadc409ee154b6/Untitled%2017.png)

---

## Inception Network: Case Studies

![Untitled](Deep%20Learning%208dcaec6fa9734afb8acadc409ee154b6/Untitled%2018.png)

In order to really concatenate all of these outputs at the end we are going to use the same type of padding for pooling.

![Untitled](Deep%20Learning%208dcaec6fa9734afb8acadc409ee154b6/Untitled%2019.png)

---

## MobileNet

![Untitled](Deep%20Learning%208dcaec6fa9734afb8acadc409ee154b6/Untitled%2020.png)

For the pointwise convolution we have 5 1x1x3 filters.

in further pictures, the 3 layered rgb filter will be used for denoting depthwise convolution.

we can get an idea for the computational cost by :
#filter_params x #filter_positions x #of_filters\

![Untitled](Deep%20Learning%208dcaec6fa9734afb8acadc409ee154b6/Untitled%2021.png)

## MobileNet architecture

![Untitled](Deep%20Learning%208dcaec6fa9734afb8acadc409ee154b6/Untitled%2022.png)

The bottleneck block it is increasing the size of the representation (by Expansion) to learn in a better way then reducing it using pointwise/projection conv.. to convert it into smaller block so it takes less memory to pass to the next block.

![Untitled](Deep%20Learning%208dcaec6fa9734afb8acadc409ee154b6/Untitled%2023.png)

---

## EfficientNet

It allows you to scale up or down the size of the network based on the computational resources of a device.

It provides a way to choose the **optimal trade-off** between i**mage resolution, network depth, and layer width** to achieve the best performance within a given computational budget.

![Untitled](Deep%20Learning%208dcaec6fa9734afb8acadc409ee154b6/Untitled%2024.png)

---

## Using Open-Source Implementation

• Replicating neural network architectures can be challenging due to the tuning of hyperparameters.

• Accessing the author's implementation can help you get started faster and avoid the need to reimplement from scratch.

---

## Transfer Learning

- Using pre-trained models and weights to initialize your own neural network.
- The computer vision research community has posted many datasets online, such as ImageNet, MS COCO, and Pascal, which researchers have trained their algorithms on.
- Depending on the size of your training set, you can freeze certain layers of the network and only train the parameters associated with your specific task.
- You can pre-compute the activations from the frozen layers and save them to disk.
- If you have a larger training set, you can freeze fewer layers and train more layers on top.
- If you have a lot of data, you can use the entire pre-trained network as initialization and train the whole network

---

## Data Augmentation

- Data augmentation is a technique used to improve the performance of computer vision systems by increasing the amount of training data.
- Common data augmentation methods in computer vision include mirroring, random cropping, rotation, shearing, and color shifting.