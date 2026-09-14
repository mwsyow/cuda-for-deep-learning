import time

import numpy as np

# Load MNIST dataset from binary files
# Use first 10000 training samples for faster training
X_train = np.fromfile("data/X_train.bin", dtype=np.float32).reshape(60000, 784)[:10000]
y_train = np.fromfile("data/y_train.bin", dtype=np.int32)[:10000]
X_test = np.fromfile("data/X_test.bin", dtype=np.float32).reshape(10000, 784)
y_test = np.fromfile("data/y_test.bin", dtype=np.int32)

# Normalize data using MNIST dataset statistics (mean and std computed from training set)
mean, std = 0.1307, 0.3081
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Reshape to (batch, channels, height, width) format
X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def linear_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    # x: (b, d_in), w: (d_in, d_out), b: (1, d_out)
    return x @ w + b


def linear_backward(
    dy: np.ndarray, x: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # dy: (b, d_out), x: (b, d_in), w: (d_in, d_out), b: (1, d_out)
    dx = dy @ w.T
    dw = x.T @ dy
    db = np.sum(dy, axis=0)
    return dx, dw, db


def softmax(x: np.ndarray) -> np.ndarray:
    # calculate exp per entry minus the max of the entry's corresponding row
    # axis = 0; operation across row dimension, ultimately collapse the row dimension
    # keepdims such that (n, 1) stays and not turning into (n,)
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    # divide each entry with the sum of the entry's corresponding row
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray):
    # y_pred: (b, num_classes), y_true: (b,)
    # apply softmax inside CE
    batch_size = y_true.shape[0]
    probs = softmax(y_pred)
    # advance indexing which technically translates to [probs[0,_], probs[1, _], ..., probs[batch_size, _]]: (b,)
    likelihood = probs[np.arange(batch_size), y_true]
    return -np.sum(np.log(likelihood)) / batch_size


def softmax_ce_grad(y_pred: np.ndarray, y_true: np.ndarray):
    # y_pred: (b, num_classes), y_true: (b,)
    batch_size = y_true.shape
    probs = softmax(y_pred)

    # convert y_true to OHE; y_true_ohe: (b, num_classes)
    y_true_ohe = np.zeros_like(y_pred)
    y_true_ohe[np.arange(len(y_pred)), y_true] = 1

    # calculate gradient of softmax + CE
    return (probs - y_true_ohe) / batch_size


def init_weight(d_in: int, d_out: int) -> np.ndarray:
    scale = (6.0 / d_in) ** 0.5
    w = np.random.uniform(-scale, scale, (d_in, d_out))
    return w


def init_bias(d_out: int) -> np.ndarray:
    return np.zeros((1, d_out)).astype(float)


class NeuralNetwork:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.w1 = init_weight(input_dim, hidden_dim)
        self.b1 = init_bias(hidden_dim)
        self.w2 = init_weight(hidden_dim, output_dim)
        self.b2 = init_bias(output_dim)

    def forward(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        batch_size = x.shape[0]
        # Flatten input: (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = x.reshape(batch_size, -1)
        fc1_output = linear_forward(x, self.w1, self.b1)
        relu_output = relu(fc1_output)
        fc2_output = linear_forward(relu_output, self.w2, self.b2)
        return fc2_output, (relu_output, fc1_output, x)

    def backward(
        self, grad: np.ndarray, cache: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        relu_output, fc1_output, x = cache
        drelu, dw2, db2 = linear_backward(grad, relu_output, self.w2)
        dfc1 = drelu * relu_derivative(fc1_output)
        _, dw1, db1 = linear_backward(dfc1, x, self.w1)
        return dw1, db1, dw2, db2

    def update_weights(
        self,
        dw1: np.ndarray,
        db1: np.ndarray,
        dw2: np.ndarray,
        db2: np.ndarray,
        lr: float,
    ):
        self.w1 -= lr * dw1
        self.w2 -= lr * dw2
        self.b1 -= lr * db1
        self.b2 -= lr * db2


def train_timed(
    model: NeuralNetwork,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    epochs: int,
    learning_rate: float,
):
    total_start = time.time()
    timing_stats = {
        "data_loading": 0.0,
        "forward": 0.0,
        "loss_computation": 0.0,
        "backward": 0.0,
        "weight_updates": 0.0,
        "total_time": 0.0,
    }
    for ep in range(epochs):
        ep_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            start = time.time()
            x = X_train[i : i + batch_size]
            y_true = y_train[i : i + batch_size]
            end = time.time()
            timing_stats["data_loading"] += end - start

            start = time.time()
            y_pred, cache = model.forward(x)
            end = time.time()
            timing_stats["forward"] += end - start

            start = time.time()
            ep_loss += cross_entropy_loss(y_pred, y_true)
            output_grad = softmax_ce_grad(y_pred, y_true)
            end = time.time()
            timing_stats["loss_computation"] += end - start

            start = time.time()
            dw1, db1, dw2, db2 = model.backward(output_grad, cache)
            end = time.time()
            timing_stats["backward"] += end - start

            start = time.time()
            model.update_weights(dw1, db1, dw2, db2, learning_rate)
            end = time.time()
            timing_stats["weight_updates"] += end - start

        print(f"Epoch {ep} loss: {ep_loss / (len(X_train) // batch_size):.4f}")

    # Calculate total training time
    total_end = time.time()
    timing_stats["total_time"] = total_end - total_start

    # Print detailed timing breakdown
    print("\n=== PYTHON NUMPY IMPLEMENTATION TIMING BREAKDOWN ===")
    print(f"Total training time: {timing_stats['total_time']:.1f} seconds\n")

    print("Detailed Breakdown:")
    print(
        f"  Data loading:     {timing_stats['data_loading']:6.3f}s ({100.0 * timing_stats['data_loading'] / timing_stats['total_time']:5.1f}%)"
    )
    print(
        f"  Forward pass:     {timing_stats['forward']:6.3f}s ({100.0 * timing_stats['forward'] / timing_stats['total_time']:5.1f}%)"
    )
    print(
        f"  Loss computation: {timing_stats['loss_computation']:6.3f}s ({100.0 * timing_stats['loss_computation'] / timing_stats['total_time']:5.1f}%)"
    )
    print(
        f"  Backward pass:    {timing_stats['backward']:6.3f}s ({100.0 * timing_stats['backward'] / timing_stats['total_time']:5.1f}%)"
    )
    print(
        f"  Weight updates:   {timing_stats['weight_updates']:6.3f}s ({100.0 * timing_stats['weight_updates'] / timing_stats['total_time']:5.1f}%)"
    )

    print("Training completed!")


def test_model():
    import torch
    from v1 import MLP

    batch_size = 8
    x = X_train[:batch_size].reshape(batch_size, -1)

    in_dim = 784
    hidden_dim = 256
    num_classes = 10
    state_dict = torch.load(
        "/home/mws/projects/cuda-for-deep-learning/data/initial_weights.pth"
    )

    np_model = NeuralNetwork(in_dim, hidden_dim, num_classes)
    np_model.w1 = state_dict["fc1.weight"].T.cpu().numpy()
    np_model.b1 = state_dict["fc1.bias"].T.cpu().numpy()

    torch_model = MLP(in_dim, hidden_dim, num_classes)
    torch_model.load_state_dict(state_dict)

    np_fc1_output = linear_forward(x, np_model.w1, np_model.b1)
    np_relu = relu(np_fc1_output)
    torch_fc1_output = torch_model.fc1(torch.tensor(x))
    torch_relu = torch_model.relu(torch_fc1_output)

    assert np.allclose(
        torch_fc1_output.detach().cpu().numpy(), np_fc1_output, atol=1e-4
    )
    assert np.allclose(torch_relu.detach().cpu().numpy(), np_relu, atol=1e-4)


if __name__ == "__main__":
    # test_model()
    # Network architecture parameters
    input_size = 784  # MNIST images: 28×28 = 784 pixels
    hidden_size = 256  # Number of hidden units
    output_size = 10  # Number of classes (digits 0-9)

    # Initialize neural network
    model = NeuralNetwork(input_size, hidden_size, output_size)

    # Training hyperparameters
    batch_size = 256
    epochs = 10
    learning_rate = 0.01

    # Train the model
    train_timed(
        model, X_train, y_train, X_test, y_test, batch_size, epochs, learning_rate
    )
