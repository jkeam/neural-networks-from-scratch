import numpy as np
print("--------------------------------------------------------------------------------------")
print("Concept 1: Every neuron has it's own weights and bias")
print("\tWeight - 1 per input, so if 3 input nodes, then this is a list of len 3.")
print("\tBias - always just 1")
print("\nSo if there are 4 inputs, but 3 neurons:")
print("\tWeights: [ [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3] ]")
print("\tBiases: [1, 2, 3]")
print("\tEach neuron has a weight for each input, which is why there are 4 inputs per list.")
print("\tBut there are only 3 lists of weights and 3 values in baises bc there are only 3 neurons.")
print("""
This is a 3x2 array.
It is also a matrix because it is retangular (2 dimensions) and homologous (same length and width).
array = [
    [1, 2], # Row 1
    [3, 4], # Row 2
    [5, 6]  # Row 3
]""")
print("")
inputs:list[float] = []

print("Single neuron")
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0
output = (inputs[0]*weights[0] +
 inputs[1]*weights[1] +
 inputs[2]*weights[2] +
 inputs[3]*weights[3] + bias)
print(output)

# using numpy
print("Single neuron using numpy")
outputs = np.dot(weights, inputs) + bias
print(outputs)

print("Three neurons in a single layer using just Python")
inputs = [1, 2, 3, 2.5]
weights1 = [0.2, 0.8, -0.5, 1]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]
bias1 = 2
bias2 = 3
bias3 = 0.5
outputs = [
 # Neuron 1:
 inputs[0]*weights1[0] +
 inputs[1]*weights1[1] +
 inputs[2]*weights1[2] +
 inputs[3]*weights1[3] + bias1,
 # Neuron 2:
 inputs[0]*weights2[0] +
 inputs[1]*weights2[1] +
 inputs[2]*weights2[2] +
 inputs[3]*weights2[3] + bias2,
 # Neuron 3:
 inputs[0]*weights3[0] +
 inputs[1]*weights3[1] +
 inputs[2]*weights3[2] +
 inputs[3]*weights3[3] + bias3]
print(outputs)


print("Three neurons in a single layer using just numpy")
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
 [0.5, -0.91, 0.26, -0.5],
 [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]
layer_outputs = np.dot(weights, inputs) + biases
print(layer_outputs)


print("--------------------------------------------------------------------------------------")
