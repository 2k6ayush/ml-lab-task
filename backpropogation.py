import pandas as pd
import numpy as np

# 1. Load Iris Dataset
df = pd.read_csv('Iris.csv')
print("Dataset Shape:", df.shape)
print(df.head())

# 2. Extract Features (4 columns) and Target (Species)
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
# Normalize features to [0, 1] range for stable sigmoid activation
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# One-hot encode the 3 classes (Iris-setosa, Iris-versicolor, Iris-virginica)
species_labels = sorted(df['Species'].unique())
y = pd.get_dummies(df['Species'])[species_labels].values.astype(float)

# 3. Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivatives_sigmoid(out):
    return out * (1 - out)

# 4. Neural Network Parameters
epoch = 3000   # Setting training iterations
lr = 0.1       # Setting learning rate

inputlayer_neurons = 4    # 4 features in Iris dataset
hiddenlayer_neurons = 6   # Number of hidden layer neurons
output_neurons = 3        # 3 output neurons (one per class)

# 5. Initialize Weights and Biases
np.random.seed(42)
wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
bh = np.random.uniform(size=(1, hiddenlayer_neurons))
wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

# 6. Training with Backpropagation
for i in range(epoch):
    # Forward Propagation
    hinp = np.dot(X, wh) + bh
    hlayer_act = sigmoid(hinp)
    outinp = np.dot(hlayer_act, wout) + bout
    output = sigmoid(outinp)
    
    # Backpropagation
    EO = y - output
    outgrad = derivatives_sigmoid(output)
    d_output = EO * outgrad
    
    EH = d_output.dot(wout.T)
    hiddengrad = derivatives_sigmoid(hlayer_act)
    d_hiddenlayer = EH * hiddengrad
    
    # Update weights and biases
    wout += hlayer_act.T.dot(d_output) * lr
    wh += X.T.dot(d_hiddenlayer) * lr
    bout += np.sum(d_output, axis=0, keepdims=True) * lr
    bh += np.sum(d_hiddenlayer, axis=0, keepdims=True) * lr

# 7. Predictions & Accuracy
pred = np.argmax(output, axis=1)
actual = np.argmax(y, axis=1)
accuracy = np.mean(pred == actual) * 100
print(f"Final Accuracy: {accuracy:.2f}%")

print("\nFirst 10 Predictions:")
for i in range(10):
    print(f"Sample {i+1}: Actual = {species_labels[actual[i]]}, Predicted = {species_labels[pred[i]]}")
