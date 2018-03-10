# Manual-TensorFlow-Perceptron
A perceptron programmed from scratch by simulating tensorflow API for simple linear classification.


# Code Main Contents
There are 9 classes, and 1 main function :

+ **class Operation()** This Operation class will be inherited by other classes that actually compute the specific
    operation, such as adding or matrix multiplication.
+ **class add(Operation)** Extended class from Operation class to do Addition.
+ **class multily(Operation)** Extended class from Operation class to do Multiplication.
+ **class matmul(Operation)** Extended class from Operation class to do Matrix Multiplication.
+ **class Sigmoid(Operation)** Extended class from Operation class to do binary classification (0 / 1).

+ **class Placeholder()** A placeholder is a node that needs to be provided a value for computing the output in the Graph.
+ **class Variable()** This variable is a changeable parameter of the Graph.
+ **class Graph()** Which make the connections between Placeholders, Variables and Operations.

+ **def traverse_postorder(operation)** A function which is basically makes sure computations are done in the correct order.

+ **class Session()** To Execute the whole Computational Graph operations.

The Dataset used: **make_blobs**, from scikit learn.

# Note
Based on Course: "Complete Guide to TensorFlow for Deep Learning with Python".


