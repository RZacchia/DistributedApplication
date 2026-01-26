import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_cifar10(data_dir="./data"):
    tfm_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    tfm_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tfm_train)
    test = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tfm_test)
    return train, test

def dirichlet_partition(labels, num_clients: int, alpha: float, seed: int):
    """
    Classic Dirichlet label-skew partition:
    For each class k, draw proportions over clients ~ Dir(alpha).
    """
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    num_classes = labels.max() + 1
    idx_by_class = [np.where(labels == k)[0] for k in range(num_classes)]
    for k in range(num_classes):
        rng.shuffle(idx_by_class[k])

    client_indices = [[] for _ in range(num_clients)]
    for k in range(num_classes):
        props = rng.dirichlet(alpha * np.ones(num_clients))
        # turn proportions into counts
        counts = (props * len(idx_by_class[k])).astype(int)
        # fix rounding
        diff = len(idx_by_class[k]) - counts.sum()
        for i in rng.choice(num_clients, size=abs(diff), replace=True):
            counts[i] += 1 if diff > 0 else -1
        start = 0
        for c in range(num_clients):
            take = counts[c]
            if take > 0:
                client_indices[c].extend(idx_by_class[k][start:start+take].tolist())
            start += take
    for c in range(num_clients):
        rng.shuffle(client_indices[c])
    return client_indices

def make_loaders(train_ds, test_ds, client_indices, batch_size: int):
    client_loaders = []
    for idxs in client_indices:
        subset = Subset(train_ds, idxs)
        client_loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True))
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    return client_loaders, test_loader
