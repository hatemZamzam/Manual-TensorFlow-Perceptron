# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 12:19:23 2018

@author: HatemZam
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


## ---------------------------- OPERATION
class Operation():
    
    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.output_nodes = []
        
        for node in input_nodes:
            node.output_nodes.append(self)
        
        _default_graph.operations.append(self)
    
    def compute(self):
        pass


class add(Operation):
    
    def __init__(self, x, y):
        super().__init__([x, y])
    
    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var + y_var


class multiply(Operation):
    
    def __init__(self, x, y):
        super().__init__([x, y])
    
    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var * y_var


class matmul(Operation):
    
    def __init__(self, x, y):
        super().__init__([x, y])
    
    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var.dot(y_var)  ## Assume it's numpy array


class Sigmoid(Operation):
    
    def __init__(self, x):
        super().__init__([x])
    
    def compute(self, x_val):
        return 1/(1 + np.exp(-x_val))


## ------------------------ Placeholders, Variables and Graph

class Placeholder():
    
    def __init__(self):
        self.output_nodes = []
        _default_graph.placeholders.append(self)


class Variable():
    
    def __init__(self, initial_value=None):
        self.value = initial_value
        self.output_nodes = []
        
        _default_graph.variables.append(self)


class Graph():
    
    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []
    
    def set_as_default(self):
        global _default_graph
        _default_graph = self


## ------------------- Traversal Postorder Tree >> to keep computations of nodes in order

def traverse_postorder(operation):
    
    nodes_postorder = []
    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder


## --------------------- Session class to execute the operations on 

class Session():
    
    def run(self, operation, feed_dict = {}):
        nodes_postorder = traverse_postorder(operation)
        
        for node in nodes_postorder:
            
            if type(node) == Placeholder:
                node.output = feed_dict[node]
            
            elif type(node) == Variable:
                node.output = node.value
            
            else:
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)
            
            if type(node.output) == list:
                node.output = np.array(node.output)
        
        return operation.output


## -------------------------- testing ....

def ex1():
    g = Graph()
    g.set_as_default()
    
    A = Variable(10)
    b = Variable(1)
    x = Placeholder()
    
    y = multiply(A, x)
    z = add(y, b)
    
    sess = Session()
    result = sess.run(operation = z, feed_dict = {x:10})
    print(result)


def ex2():
    g2 = Graph()
    g2.set_as_default()
    
    A2 = Variable([[10, 20],
                   [30, 40]])
    b2 = Variable([1, 1])
    x2 = Placeholder()
    
    y2 = matmul(A2, x2)
    z2 = add(y2, b2)
    
    sess2 = Session()
    result2 = sess2.run(operation = z2, feed_dict = {x2:10})
    print(result2)


ex1()
ex2()


# -------------------------------------------- Classification
# ------- Activation Function :

"""
def sigmoid(x):
    return 1/(1 + np.exp(-x))

sample_x = np.linspace(-10, 10, 100)
sample_y = sigmoid(sample_x)

plt.plot(sample_x, sample_y)
"""


data = make_blobs(n_samples=50, n_features=2, centers=2, random_state=75)

#print(data)

features = data[0]
labels = data[1]

def plot_class_data():
    
    plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='coolwarm')

    x = np.linspace(0, 11, 10)
    y = -x * 2 + 8
    plt.plot(x, y)
    
    
    np.array([2, 1]).dot([[8],[10]]) - 8
    np.array([2, 1]).dot([[2],[-10]]) - 8

plot_class_data()

## ------------------- Now use the Graph for Simple Linear Classifier

def linearClassifierGraph():
    g = Graph()
    g.set_as_default()
    
    w = Variable([2, 1])
    b = Variable(-8)
    x = Placeholder()
    
    y = matmul(w, x)
    z = add(y, b)
    a = Sigmoid(z)
    
    sess = Session()
    result = sess.run(operation = a, feed_dict = {x:[8, 10]})
    result2 = sess.run(operation = a, feed_dict = {x:[2, -10]})
    print(result)
    print(result2)

linearClassifierGraph()
