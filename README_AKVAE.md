

## An example to run AK-VAE

run LightGCN on **Gowalla** dataset:

* change base directory

Change `ROOT_PATH` in `code/Others/world.py`

* command

` python main.py --dataset video_games --topks=[20] --model VAEKernel --epoch 400 --lr 0.001 --tau_model2 0.08 --dropout_model2 0.5 --reg_model2 0.0001 --kl_anneal 0.3 --cuda 0`

