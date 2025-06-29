import numpy as np
import sklearn as sk
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    cost = -1/m * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return cost

def gradient(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    grad = 1/m * X.T @ (h - y)
    return grad

def gradient_descent(X, y, theta, alpha, iterations):
    costs = []
    for i in range(iterations):
        theta = theta - alpha * gradient(X, y, theta)
        costs.append(compute_cost(X, y, theta))
    return theta, costs

def stochastic_gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    costs = []
    for i in range(iterations):
        idxs = [idx for idx in range(m)]
        np.random.shuffle(idxs)
        for j in range(m):
            idx = idxs[j]
            X_i = X[idx, :].reshape(1, X.shape[1])
            y_i = y[idx].reshape(1)
            theta = theta - alpha * gradient(X_i, y_i, theta)
        costs.append(compute_cost(X, y, theta))
    return theta, costs



dataset = load_breast_cancer()
X = dataset.data
y = dataset.target


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_val = np.c_[np.ones(X_val.shape[0]), X_val]


learning_rates = [0.001, 0.01, 0.1, 1]
iterations = 100
theta_init = np.zeros(X_train.shape[1])


plt.figure(figsize=(10, 6))


for lr in learning_rates:
    theta, costs = gradient_descent(X_train, y_train, theta_init, alpha=lr, iterations=iterations)
    plt.plot(costs, label=f'LR={lr}')


plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Gradient Descent: Cost vs. Iterations for Different Learning Rates')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))

for lr in learning_rates:
    theta, costs = stochastic_gradient_descent(X_train, y_train, theta_init, alpha=lr, iterations=iterations)
    plt.plot(costs, label=f'LR={lr}')

plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('SGD: Cost vs. Epochs for Different Learning Rates')
plt.legend()
plt.show()

best_lr_gd = 1
best_lr_sgd = 0.1


theta_gd, costs_gd = gradient_descent(X_train, y_train, theta_init, alpha=best_lr_gd, iterations=iterations)
theta_sgd, costs_sgd = stochastic_gradient_descent(X_train, y_train, theta_init, alpha=best_lr_sgd, iterations=iterations)


plt.figure(figsize=(10, 6))
plt.plot(costs_gd, label=f'Gradient Descent, LR={best_lr_gd}')
plt.plot(costs_sgd, label=f'SGD, LR={best_lr_sgd}')
plt.xlabel('Iterations / Epochs')
plt.ylabel('Cost')
plt.title('Best GD vs. Best SGD')
plt.legend()
plt.show()


dataset = sk.datasets.fetch_openml('mnist_784')
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)



X = X / 255.0
y = y.astype(int)

X_train, X_val = X[:60000], X[60000:]
y_train, y_val = y[:60000], y[60000:]

X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_val = np.c_[np.ones(X_val.shape[0]), X_val]

y_train_5 = (y_train == 5).astype(int)
y_val_5 = (y_val == 5).astype(int)

learning_rates = [0.001, 0.01, 0.1, 1]
epochs = 50
theta_init = np.zeros(X_train.shape[1])

plt.figure(figsize=(10, 6))

for lr in learning_rates:
    theta, costs = stochastic_gradient_descent(X_train, y_train_5, theta_init, alpha=lr, iterations=epochs)
    plt.plot(costs, label=f'LR={lr}')

plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('SGD: Cost vs. Epochs for Different Learning Rates on MNIST')
plt.legend()
plt.show()

def predict(X, theta):
    return sigmoid(X @ theta) >= 0.5

best_lr_sgd = 0.01

theta_sgd, _ = stochastic_gradient_descent(X_train, y_train_5, theta_init, alpha=best_lr_sgd, iterations=epochs)
accuracy_sgd = np.mean(predict(X_val, theta_sgd) == y_val_5)

print(f"Classification accuracy with SGD: {accuracy_sgd * 100:.2f}%")

probabilities = sigmoid(X_val @ theta_sgd)
errors = (predict(X_val, theta_sgd) != y_val_5)
confidences = np.abs(probabilities - 0.5)

confused_data = sorted(zip(X_val[errors], confidences[errors], y_val_5[errors]), key=lambda x: -x[1])

plt.figure(figsize=(12, 3))
for i, (image, _, _) in enumerate(confused_data[:8]):
    plt.subplot(1, 8, i+1)
    plt.imshow(image[1:].reshape(28, 28), cmap='gray')
    plt.axis('off')

plt.show()

# What did you find to be the best learning rate when using gradient descent? 1
# # How many iterations are required until the gradient descent method has converged? 60
# What was the best learning rate when using the stochastic gradient method? 0.1
# How many epochs are required until the stochastic gradient method has converged? 60
# Which method converges faster? SGD
# Do both methods eventually find a minimizer for the cost function? Yes.
# What was the best learning rate when using the stochastic gradient method? 0.1
# How many epochs are required until the stochastic gradient method has converged？35

# The first 5 is like b.
# The second 5 shows a slight distortion from a typical "5," with a disruption in the curve at the top.
# The third 5 starts to lose the definition in the upper half, making the 'head' of the "5" less recognizable.
# The fourth 5 is same with the third one.
# The fifth 5 has a top part that is almost entirely detached, distorting the familiar shape.
# The sixth 5 has such a disconnected upper part that it barely resembles a "5" at all, looking more abstract.
# The seventh 5 is very abstracted; the top curve is nearly a separate element.
# The eighth 5 is so heavily distorted.

# In the displayed images, we notice that the confused images are those that are either poorly written, have unusual shapes for a "5", or resemble other digits closely.
# These characteristics can make it challenging for the model to classify them correctly, leading to high confidence but incorrect predictions.
