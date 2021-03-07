#!/bin/bash

# TODO description.

# Author: Spencer M. Richards
#         Autonomous Systems Lab (ASL), Stanford
#         (GitHub: spenrich)

for seed in {0..9}
do
    for M in 2 5 10 20 30 40 50
    do
        echo "seed = $seed, M = $M"
        python test_all.py $seed $M
    done
done
