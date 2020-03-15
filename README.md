# CNN-for-piston

The network topology was inspired by VGG. We compared the performance about four types of VGGs with different number of layers to select the most appropriate network that is not only time-saving but of high accuracy. We also split the network into five branches with same loss functions for each.

We trained five CNNs with the same structure for identifying piston error of each submirror, except the error of the first submirror. 
One branch has twelve convolutional layers with ReLU activations, five pooling layers and a fully connected layer at the end. This is used to predict the piston step values in the range . The branch is in charge of performing a regression task to predict a continuous value. The cost function in this branch is the mean squared error between the output scores of the last fully connected layer and the ground truth labels or, equivalently, the L2 norm of the difference between predicted scores and labels. This prediction can be carried out for the information contained in one single wavelength. The CNN is trained in a fully supervised manner, so input images and labels must be supplied in the process.

We use TensorFlow to build the framework of the CNN. The network parameters are updated with a mini-batch gradient descent algorithm and the Adam update rule. The size of the batch is 64, the number of the learning rate is 0.0001, and the maximum number of iterations is 5000.

