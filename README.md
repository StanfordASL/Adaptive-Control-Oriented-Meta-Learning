# Adaptive-Control-Oriented Meta-Learning for Nonlinear Systems

This repository accompanies the paper ["Adaptive-Control-Oriented Meta-Learning for Nonlinear Systems"](https://arxiv.org/abs/2103.04490) [1].


## Getting started

Ensure you are using Python 3. Clone this repository and install the packages listed in `requirements.txt`. In particular, this code uses [JAX](https://github.com/google/jax).


## Reproducing results

Training data, trained parameters, and test results are all conveniently saved in this repository, since it can take a while to re-generate them. To simply produce Figures 2, 3, and 4 in [1], run the command `python plots.py`.

Training data can be generated with the command `python generate_data.py`.

Parameters can then be trained (for multiple training set sizes and random seeds) with the command `./train.sh`. This will take a while.

Finally, test results for Figures 3 and 4 in [1] can be produced with the commands `python test_single.py` and `./test.sh`, respectively. This may also take a while.


## Citing this work

Please use the following bibtex entry to cite this work.
```
@INPROCEEDINGS{RichardsAzizanEtAl2021,
author    = {Richards, S. M. and Azizan, N. and Slotine, J.-J. E. and Pavone, M.},
title     = {Adaptive-control-oriented meta-learning for nonlinear systems},
booktitle = {Robotics: Science and Systems},
year      = {2021},
note      = {In press. Available at \url{https://arxiv.org/abs/2103.04490}},
}
```


## References
[1] S. M. Richards, N. Azizan, J.-J. E. Slotine, and M. Pavone. Adaptive-control-oriented meta-learning for nonlinear systems. In *Robotics: Science and Systems*, 2021. In press. Available at <https://arxiv.org/abs/2103.04490>.
