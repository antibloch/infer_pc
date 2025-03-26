#!/bin/bash

python infer_kitti_kp.py --ply "$1"

python infer_kitti_rand.py --ply "$1"

python infer_paris_kp.py --ply "$1"

python infer_paris_rand.py --ply "$1"

python infer_semantic3d_rand.py --ply "$1"

python infer_toronto_kp.py --ply "$1"