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