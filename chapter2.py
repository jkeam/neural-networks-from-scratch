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
This is a 3x2 array (tall by wide).
It is also a matrix because it is retangular (2 dimensions) and homologous (same length and width).
array = [
    [1, 2], # Row 1
    [3, 4], # Row 2
    [5, 6]  # Row 3
]""")
print("""
Feature set instance or observation or sample is best done in batches to avoid overfitting and encourage generalization.
      However, this is often an art to figure out the right batch size.""")
print("")
print("""
Matrix product:
    Given 2 matrices, multiplies them.  Shape matters and must match, example:
    (5x4) * (4x5) -> results in a matrix of shape (5x5)
        The "inner" 4 must match and the result are the "outer" numbers.
    This is near because if we have a row vector of shape (1x5) and a column vector of (5x1)
        we get a result that is (1x1) which is a single number!

Transposition:
    This is a fundamental concept because the input batch matrix and the weight matrix will typically be the same shape
        for example, (3x4) and (3x4).
        In order to be able to perform the matrix product, we can transpose the second matrix to get the desired:
            (3x4) and (4x3), which then creates the right shapes we need.
        This works because we care about the relationships between the inputs against the weights.
        The matrix product does this as it multiplies all input and weight combinations against each other,
            so transposition has no effect on the final result.
        We also transpose the weights, so that the output matrix will have the same relationship as the input matrix,
            where each row represents a feature set/observation/sample,
            and the column then represents the neurons (bc that is what we transposed).
            So each row is a feature set and each column is one neuron.
            Our bias (single vector) is then applied to the result as an addition.
                so the first element is added to the first column (neuron), which makes sense, each neuron has 1 bias.
                The second element in the bias vector is added to all elements in the second column of the resultant matrix,
                so on and so forth.

    Rows becomes columns. So this (2x3):
    [
        [1, 2, 3],
        [4, 5, 6]
    ]
    becomes (3x2):
    [
        [1, 4],
        [2, 5],
        [3, 6]
    ]
""")
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

print("Transposition is a neat trick to create a column vector from a regular python list")
a = [1, 2, 3]
b = [2, 3, 4]
# notice that we have to turn both python lists into mathmatical arrays or matrices
#   which are basically 2 dimensional lists
a = np.array([a])
b = np.array([b]).T  # this transposition turns this python list (row vector) into a column vector
print(np.dot(a, b))  # numpy funtion for both matrix product and dot product is the same func call

print("")
inputs = [[1.0, 2.0, 3.0, 2.5],
 [2.0, 5.0, -1.0, 2.0],
 [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1.0],
 [0.5, -0.91, 0.26, -0.5],
 [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]
layer_outputs = np.dot(inputs, np.array(weights).T) + biases
print(layer_outputs)

print("--------------------------------------------------------------------------------------")
