#!/bin/bash

RUNS=5

echo starting normal runs kronodroid
fedLFP --krono --data_dir ./data/kronodroid_npz/ --runs $RUNS --no-quantize

echo starting quantized runs kronodroid
fedLFP --krono --data_dir ./data/kronodroid_npz/ --runs $RUNS



# echo starting quantized runs cifar
# fedLFP --cifar10 --data_dir ./data/ --runs $RUNS

# echo starting normal runs cifar
# fedLFP --cifar10 --data_dir ./data/ --runs $RUNS --no-quantized