from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torchvision import datasets, transforms

from adversarial_lassonet.paths import get_data_root

# Portions of the tabular dataset loading logic are adapted from
# Concrete-Autoencoders:
# https://github.com/mfbalin/Concrete-Autoencoders/blob/master/experiments/generate_comparison_figures.py


DATASET_LOCATIONS = {
    "mice_csv": ("mice", "Data_Cortex_Nuclear.csv"),
    "isolet_train": ("isolet", "isolet1234.data"),
    "isolet_test": ("isolet", "isolet5.data"),
    "epileptic_csv": ("epileptic", "data.csv"),
    "coil_dir": ("coil-20-proc",),
    "activity_x_train": ("activity", "final_X_train.txt"),
    "activity_x_test": ("activity", "final_X_test.txt"),
    "activity_y_train": ("activity", "final_y_train.txt"),
    "activity_y_test": ("activity", "final_y_test.txt"),
    "torchvision_root": ("torchvision",),
}


def _dataset_path(*parts: str) -> Path:
    return get_data_root().joinpath(*parts)


def _resolve_first_existing_path(*candidates: tuple[str, ...]) -> Path:
    for parts in candidates:
        path = _dataset_path(*parts)
        if path.exists():
            return path
    joined = ", ".join(str(_dataset_path(*parts)) for parts in candidates)
    raise FileNotFoundError(
        f"Required dataset artifact not found. Tried: {joined}\n"
        "Please place the dataset under data/raw/ or set LASSONET_DATA_DIR."
    )


def _resolve_named_path(name: str) -> Path:
    path = _dataset_path(*DATASET_LOCATIONS[name])
    if not path.exists():
        raise FileNotFoundError(
            f"Required dataset artifact not found: {path}\n"
            "Please place the dataset under data/raw/ or set LASSONET_DATA_DIR."
        )
    return path


def load_mice(one_hot: bool = False):
    filling_value = -100000
    mice_csv = _resolve_named_path("mice_csv")
    X = np.genfromtxt(
        mice_csv,
        delimiter=",",
        skip_header=1,
        usecols=range(1, 78),
        filling_values=filling_value,
        encoding="UTF-8",
    )
    classes = np.genfromtxt(
        mice_csv,
        delimiter=",",
        skip_header=1,
        usecols=range(78, 81),
        dtype=None,
        encoding="UTF-8",
    )

    for i, row in enumerate(X):
        for j, val in enumerate(row):
            if val == filling_value:
                X[i, j] = np.mean(
                    [
                        X[k, j]
                        for k in range(classes.shape[0])
                        if np.all(classes[i] == classes[k])
                    ]
                )

    DY = np.zeros((classes.shape[0]), dtype=np.uint8)
    for i, row in enumerate(classes):
        for j, (val, label) in enumerate(zip(row, ["Control", "Memantine", "C/S"])):
            DY[i] += (2**j) * (val == label)

    Y = np.zeros((DY.shape[0], np.unique(DY).shape[0]))
    for idx, val in enumerate(DY):
        Y[idx, val] = 1

    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    DY = DY[indices]

    if not one_hot:
        Y = DY

    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    print(f"X shape: {X.shape}, Y shape: {Y.shape}")

    split = X.shape[0] * 4 // 5
    return (X[:split], Y[:split]), (X[split:], Y[split:])


def load_isolet():
    isolet_train = _resolve_named_path("isolet_train")
    isolet_test = _resolve_named_path("isolet_test")
    x_train = np.genfromtxt(
        isolet_train,
        delimiter=",",
        usecols=range(0, 617),
        encoding="UTF-8",
    )
    y_train = np.genfromtxt(
        isolet_train,
        delimiter=",",
        usecols=[617],
        encoding="UTF-8",
    )
    x_test = np.genfromtxt(
        isolet_test,
        delimiter=",",
        usecols=range(0, 617),
        encoding="UTF-8",
    )
    y_test = np.genfromtxt(
        isolet_test,
        delimiter=",",
        usecols=[617],
        encoding="UTF-8",
    )

    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(np.concatenate((x_train, x_test)))
    x_train = X[: len(y_train)]
    x_test = X[len(y_train) :]

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    return (x_train, y_train - 1), (x_test, y_test - 1)


def load_epileptic():
    filling_value = -100000
    epileptic_csv = _resolve_named_path("epileptic_csv")
    X = np.genfromtxt(
        epileptic_csv,
        delimiter=",",
        skip_header=1,
        usecols=range(1, 179),
        filling_values=filling_value,
        encoding="UTF-8",
    )
    Y = np.genfromtxt(
        epileptic_csv,
        delimiter=",",
        skip_header=1,
        usecols=range(179, 180),
        encoding="UTF-8",
    )

    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    print(X.shape, Y.shape)
    return (X[:8000], Y[:8000]), (X[8000:], Y[8000:])


def load_coil():
    coil_dir = _resolve_named_path("coil_dir")
    samples = []
    for i in range(1, 21):
        for image_index in range(72):
            obj_img = Image.open(coil_dir / f"obj{i}__{image_index}.png")
            rescaled = obj_img.resize((20, 20))
            pixels_values = [float(x) for x in list(rescaled.getdata())]
            sample = np.array(pixels_values + [i])
            samples.append(sample)
    samples = np.array(samples)
    np.random.shuffle(samples)
    data = samples[:, :-1]
    targets = (samples[:, -1] + 0.5).astype(np.int64)
    data = (data - data.min()) / (data.max() - data.min())

    split = data.shape[0] * 4 // 5
    train = (data[:split], targets[:split] - 1)
    test = (data[split:], targets[split:] - 1)
    print(train[0].shape, train[1].shape)
    print(test[0].shape, test[1].shape)
    return train, test


def load_data(fashion: bool = False, digit=None, normalize: bool = False):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.numpy().flatten()),
        ]
    )

    dataset_cls = datasets.FashionMNIST if fashion else datasets.MNIST
    root = _dataset_path(*DATASET_LOCATIONS["torchvision_root"])
    train_dataset = dataset_cls(root=root, train=True, download=True, transform=transform)
    test_dataset = dataset_cls(root=root, train=False, download=True, transform=transform)

    x_train = np.array([data for data, _ in train_dataset])
    y_train = np.array([label for _, label in train_dataset])
    x_test = np.array([data for data, _ in test_dataset])
    y_test = np.array([label for _, label in test_dataset])

    if digit is not None and 0 <= digit <= 9:
        train = {y: [] for y in range(10)}
        test = {y: [] for y in range(10)}
        for x, y in zip(x_train, y_train):
            train[y].append(x)
        for x, y in zip(x_test, y_test):
            test[y].append(x)
        for y in range(10):
            train[y] = np.asarray(train[y])
            test[y] = np.asarray(test[y])
        x_train = train[digit]
        x_test = test[digit]

    x_train = x_train.reshape((-1, 784)).astype(np.float32)
    x_test = x_test.reshape((-1, 784)).astype(np.float32)

    if normalize:
        X = np.concatenate((x_train, x_test))
        X = (X - X.min()) / (X.max() - X.min())
        x_train = X[: len(y_train)]
        x_test = X[len(y_train) :]

    return (x_train, y_train), (x_test, y_test)


def load_mnist():
    train, test = load_data(fashion=False, normalize=True)
    x_train, x_test, y_train, y_test = train_test_split(test[0], test[1], test_size=0.2)
    return (x_train, y_train), (x_test, y_test)


def _colorize_mnist_split(X, y, correlation):
    y_binary = (y >= 5).astype(np.int64)
    flip = (np.random.rand(y_binary.shape[0]) < (1.0 - correlation)).astype(np.int64)
    color = np.logical_xor(y_binary, flip).astype(np.int64)

    channel_0 = X.copy()
    channel_1 = X.copy()
    channel_0[color == 1] = 0.0
    channel_1[color == 0] = 0.0
    X_colored = np.concatenate([channel_0, channel_1], axis=1).astype(np.float32)
    return X_colored, y_binary


def load_colored_mnist(correlation_train=0.9, correlation_test=0.1):
    (x_train, y_train), (x_test, y_test) = load_data(fashion=False, normalize=True)
    x_train_colored, y_train_binary = _colorize_mnist_split(x_train, y_train, correlation_train)
    x_test_colored, y_test_binary = _colorize_mnist_split(x_test, y_test, correlation_test)

    print(x_train_colored.shape, y_train_binary.shape)
    print(x_test_colored.shape, y_test_binary.shape)
    return (x_train_colored, y_train_binary), (x_test_colored, y_test_binary)


def load_fashion():
    train, test = load_data(fashion=True, normalize=True)
    x_train, x_test, y_train, y_test = train_test_split(test[0], test[1], test_size=0.2)
    return (x_train, y_train), (x_test, y_test)


def load_mnist_two_digits(digit1, digit2):
    train_digit_1, _ = load_data(digit=digit1)
    train_digit_2, _ = load_data(digit=digit2)

    X_train_1, X_test_1 = train_test_split(train_digit_1[0], test_size=0.6)
    X_train_2, X_test_2 = train_test_split(train_digit_2[0], test_size=0.6)

    X_train = np.concatenate((X_train_1, X_train_2))
    y_train = np.array([0] * X_train_1.shape[0] + [1] * X_train_2.shape[0])
    shuffled_idx = np.random.permutation(X_train.shape[0])
    np.take(X_train, shuffled_idx, axis=0, out=X_train)
    np.take(y_train, shuffled_idx, axis=0, out=y_train)

    X_test = np.concatenate((X_test_1, X_test_2))
    y_test = np.array([0] * X_test_1.shape[0] + [1] * X_test_2.shape[0])
    shuffled_idx = np.random.permutation(X_test.shape[0])
    np.take(X_test, shuffled_idx, axis=0, out=X_test)
    np.take(y_test, shuffled_idx, axis=0, out=y_test)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    return (X_train, y_train), (X_test, y_test)


def load_activity():
    x_train = np.loadtxt(
        _resolve_first_existing_path(
            DATASET_LOCATIONS["activity_x_train"],
            ("dataset_uci", "final_X_train.txt"),
        ),
        encoding="UTF-8",
    )
    x_test = np.loadtxt(
        _resolve_first_existing_path(
            DATASET_LOCATIONS["activity_x_test"],
            ("dataset_uci", "final_X_test.txt"),
        ),
        encoding="UTF-8",
    )
    y_train = np.loadtxt(
        _resolve_first_existing_path(
            DATASET_LOCATIONS["activity_y_train"],
            ("dataset_uci", "final_y_train.txt"),
        ),
        encoding="UTF-8",
    ) - 1
    y_test = np.loadtxt(
        _resolve_first_existing_path(
            DATASET_LOCATIONS["activity_y_test"],
            ("dataset_uci", "final_y_test.txt"),
        ),
        encoding="UTF-8",
    ) - 1

    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(np.concatenate((x_train, x_test)))
    x_train = X[: len(y_train)]
    x_test = X[len(y_train) :]

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    return (x_train, y_train), (x_test, y_test)


def load_dataset(dataset):
    if dataset == "MNIST":
        return load_mnist()
    if dataset == "ColoredMNIST":
        return load_colored_mnist()
    if dataset == "MNIST-Fashion":
        return load_fashion()
    if dataset == "MICE":
        return load_mice()
    if dataset == "COIL":
        return load_coil()
    if dataset == "ISOLET":
        return load_isolet()
    if dataset == "Activity":
        return load_activity()
    print("Please specify a valid dataset")
    return None
