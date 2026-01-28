# Quantized FedLFP (PyTorch)

## Install Dependencies
Create a virtual environment end execute the following command in it in the directory "project"

```bash
pip install -e .["dev"]
```


## Commands

```bash
fedLFP --krono --data_dir ./data/kronodroid_npz/ --runs 5
```
Possible Arguments:

```python
    "--data_dir", type=str, required=True
    "--device", type=str, default="cuda"
    "--num_clients", type=int, default=20
    "--alpha", type=float, default=0.2
    "--batch_size", type=int, default=64
    "--runs", type=int, default=3, help="paper-style: repeat and average best-over-rounds"
    "--seed", type=int, default=0
    "--quantize", type=bool, default=True

    # mutually_exclusive
    "--krono", action="store_true"
    "--cifar10", action="store_true"
```


## Experiment Run

5 quantized and 5 not quantized runs with datasets kronodroid and cifar10

```bash
./experimenta_run.sh
```