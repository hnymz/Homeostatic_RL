#!/bin/bash

conda activate RL
which python
python ../RL/PPO_train.py "$1" "$2" "$3" "$4" "$5" "$6"