import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, n_epochs=100):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        # Initialize weights (including bias as the last weight)
        self.weights = np.zeros(input_size + 1)

    def activation(self, x):
        # Step function
        return np.where(x >= 0, 1, 0)

    def predict(self, X):
        # Add bias term (1) to input
        X_with_bias = np.c_[X, np.ones(X.shape[0])]
        linear_output = np.dot(X_with_bias, self.weights)
        return self.activation(linear_output)

    def fit(self, X, y):
        # Add bias term (1) to input
        X_with_bias = np.c_[X, np.ones(X.shape[0])]

        for _ in range(self.n_epochs):
            for xi, target in zip(X_with_bias, y):
                output = self.activation(np.dot(xi, self.weights))
                update = self.lr * (target - output)
                self.weights += update * xi

# Example: learn the AND function
if __name__ == "__main__":
    # Training data for AND gate
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0, 0, 0, 1])

    perceptron = Perceptron(input_size=2, learning_rate=0.1, n_epochs=10)
    perceptron.fit(X, y)

    print("Final weights:", perceptron.weights)
    print("Predictions:")
    for x in X:
        print(f"{x} -> {perceptron.predict(np.array([x]))[0]}")
