import numpy as np

class Perceptron:
    def __init__(self, eta: float = 0.01, epochs: int = 100, use_bias: bool = True):
        self.eta = eta
        self.epochs = epochs
        self.use_bias = use_bias
        self.weights: np.ndarray = None  
        self.bias: float = 0.0
        self.errors_per_epoch: list = []

    def _signum(self, v: float) -> int:
        
        return 1 if v >= 0 else -1

    def _net(self, x: np.ndarray) -> float:
        net = np.dot(self.weights, x)
        if self.use_bias:
            net += self.bias
        return net

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_features = X.shape[1]
        # Initialize weights and bias 
        rng = np.random.RandomState(0)
        self.weights = rng.uniform(-0.5, 0.5, n_features)
        self.bias = rng.uniform(-0.5, 0.5) if self.use_bias else 0.0

        self.errors_per_epoch = []

        for _ in range(self.epochs):
            epoch_errors = 0
            for xi, di in zip(X, y):
                v = self._net(xi)
                yi = self._signum(v)
                error = di - yi
                if error != 0:
                    # Weight update rule
                    self.weights += self.eta * error * xi
                    if self.use_bias:
                        self.bias += self.eta * error
                    epoch_errors += 1
            self.errors_per_epoch.append(epoch_errors)

        return self

    def predict_single(self, x: np.ndarray) -> int:
        return self._signum(self._net(x))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_single(xi) for xi in X])


class Adaline:
    def __init__(self, eta: float = 0.01, epochs: int = 100,
                 mse_threshold: float = 0.01, use_bias: bool = True):
        self.eta = eta
        self.epochs = epochs
        self.mse_threshold = mse_threshold
        self.use_bias = use_bias
        self.weights: np.ndarray = None
        self.bias: float = 0.0
        self.mse_history: list = []

    def _net(self, x: np.ndarray) -> float:
        net = np.dot(self.weights, x)
        if self.use_bias:
            net += self.bias
        return net

    def _signum(self, v: float) -> int:
        return 1 if v >= 0 else -1

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_features = X.shape[1]
        rng = np.random.RandomState(0)
        self.weights = rng.uniform(-0.5, 0.5, n_features)
        self.bias = rng.uniform(-0.5, 0.5) if self.use_bias else 0.0

        self.mse_history = []

        for _ in range(self.epochs):
            squared_errors = []
            for xi, di in zip(X, y):
                # Linear activation for training
                yi = self._net(xi)
                error = di - yi
                # Weight update
                self.weights += self.eta * error * xi
                if self.use_bias:
                    self.bias += self.eta * error
                squared_errors.append(error ** 2)

            mse = np.mean(squared_errors)
            self.mse_history.append(mse)

            # Early stopping if MSE is below threshold
            if mse <= self.mse_threshold:
                break

        return self

    def predict_single(self, x: np.ndarray) -> int:
        return self._signum(self._net(x))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_single(xi) for xi in X])
