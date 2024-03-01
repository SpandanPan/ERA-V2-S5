# ERA_BackProp

## Part1

### Steps to update gradient and parameters

 *  Block1: Calculates forward propagation. The error for E1 and E2 are added
#### Block2: Calculates the gradient of Loss Function with respect to w5(parameter)
#### Block3: The calculation involves chain rule from E1 (as E2 is not affected by W5) to a_o1 to o1 To w5
#### Block4: The calulation is done to calculate gradients for weights of innitial layers (w1)
#### Block 5: Gradient for all weights in innitial layers are calulated

#### Updating loss
Once the gradients are calculated, they are updated after multiplying with learning rate
This gets us the new weights

## Part2

#### The notebook for running MNIST is attached

## Training
### Input image size is 28*28
### CNN Layers Used

### Layer1 - Conv with 8 channels, kernel 3*3, Padding=0, Stride=0
### BN used
### DropOut Used (p=0.1)
### Layer2 - Conv with 16 channels, kernel 3*3, Padding=0, Stride=0
### BN Used
### DropOut Used (p=0.1)

### Transitional Layer - 
### 1. MaxPooling (2*2)
### 2. 1*1 Conv - * channels
### BN used
###  Layer3 - Conv with 8 channels, kernel 3*3, Padding=0, Stride=0
### BN Used
### Layer4 - Conv with 16 channels, kernel 3*3, Padding=0, Stride=0
### BN
### Transitional Layer: Max Pooling

### Layer 5 - Conv with 32 channels, kernel 3*3, Padding=0, Stride=0

### FC Layer 1 - Dim(2*2*32,30)
### FC Layer 2 -Dim (30,10)
### Output - Log Softmax

### Other hyperparams
### LR=0.01

