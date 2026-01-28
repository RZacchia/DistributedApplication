#!/bin/bash
echo starting quantized runs kronodroid
fedLFP --krono --data_dir ./data/kronodroid_npz/ --runs 5

echo starting normal runs kronodroid
fedLFP --krono --data_dir ./data/kronodroid_npz/ --runs 5 --quantize False


echo starting quantized runs cifar
fedLFP --cifar10 --data_dir ./data/ --runs 5

echo starting normal runs cifar
fedLFP --cifar10 --data_dir ./data/ --runs 5 --quantize False