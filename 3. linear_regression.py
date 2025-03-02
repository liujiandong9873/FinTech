import json
import numpy as np

# Read the JSON file
with open('house_prices.json', 'r', encoding='utf-8') as f:
    data_dict = json.load(f)

# Convert the dictionary to a numpy array
data_array = np.array([[d['卧室个数'], d['卫生间个数'], d['面积'], d['房产价格']] for d in data_dict])

# Separate the features (x) and the target (y)
x = data_array[:, :-1]  # All rows, all columns except the last
y = data_array[:, -1]   # All rows, only the last column

# Define the weight matrix w (initialized with random values)
w = np.random.randn(x.shape[1], 1)

# Define the loss function (Mean Squared Error)
def loss_function(x, y, w):
    m = len(y)
    predictions = x.dot(w)
    loss = (1 / (2 * m)) * np.sum((predictions - y.reshape(-1, 1)) ** 2)
    return loss

# Display the first few entries of x, y, and the initial loss
print("x:", x[:5])
print("y:", y[:5])
print("Initial loss:", loss_function(x, y, w))