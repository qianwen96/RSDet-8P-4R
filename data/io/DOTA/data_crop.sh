
srun --mpi=pmi2 -p VI_UC_1080TI -n1 --gres=gpu:0 --ntasks-per-node=1 python data_crop.py