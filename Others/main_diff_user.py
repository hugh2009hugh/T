import os

from torch.serialization import save
import world
import utils
from world import cprint
import torch
from torch import nn
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sn
sn.set()
import joblib
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# from torch.multiprocessing import multiprocessing
# multiprocessing.set_start_method('spawn')
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register as register
from register import dataset

if __name__ == '__main__':
    Recmodel = register.MODELS[world.model_name](world.config, dataset)
    if isinstance(Recmodel, nn.Module):
        Recmodel = Recmodel.to(world.device)

    if world.model_name == 'lgn':
        bpr = utils.BPRLoss(Recmodel, world.config)
    else:
        pass

    weight_file = utils.getFileName()
    print(f"load and save to {weight_file}")
    if world.LOAD:
        try:
            Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
            world.cprint(f"loaded model weights from {weight_file}")
        except FileNotFoundError:
            print(f"{weight_file} not exists, start from beginning")
    Neg_k = 1

    # init tensorboard
    if world.tensorboard:
        w : SummaryWriter = SummaryWriter(
                                        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                        )
    else:
        w = None
        world.cprint("not enable tensorflowboard")

    best_results = {'precision': np.zeros(len(world.topks)),
                    'recall': np.zeros(len(world.topks)),
                    'ndcg': np.zeros(len(world.topks))}
    best_epoch = 0

    try:
        if(world.simple_model != 'none'):
            epoch = 0
            cprint("[TEST]")
            adj_mat = dataset.UserItemNet.tolil()
            import model
            if(world.simple_model == 'lgn-ide'):
                lm = model.LGCN_IDE(adj_mat)
                lm.train()
            elif(world.simple_model == 'gf-cf'):
                lm = model.GF_CF(adj_mat)
                lm.train()
            def ensure_dirs(path):
                if not os.path.exists(path):
                    os.makedirs(path)
            #Procedure.Test(dataset, lm, epoch, w, world.config['multicore'])
            results_groups, group_line= Procedure.Test_Diff_Users(dataset, lm, epoch, w, world.config['multicore'])
            sparsity_path = f'./user_sparsity_analysis/{world.dataset}/{world.simple_model}/'
            ensure_dirs(sparsity_path)
            joblib.dump(results_groups, sparsity_path + 'results_group.pkl', compress=3)
            joblib.dump(group_line, sparsity_path + 'group_line.pkl', compress=3)
        else:
            def mplot(data, name):
                plt.xlabel('epoch')
                plt.ylabel(f'{name}')
                plt.title(f'{world.model_name}-{world.dataset}-{name}')
                plt.plot(data)
                if len(eval(world.args.enc_dims)) > 1:
                    save_name = f'{world.model_name}-{world.dataset}-{name}-{world.args.enc_dims}-{world.args.act_vae}.jpg'
                else:
                    save_name = f'{world.model_name}-{world.dataset}-{name}-{world.args.enc_dims}.jpg'
                plt.savefig(save_name)
                plt.clf()
            def save_data(data, path, name):
                save_name = path + f'{world.model_name}-{world.dataset}-{name}-{world.args.enc_dims}.pkl'
                joblib.dump(data, save_name, compress=3)

            def ensure_dirs(path):
                if not os.path.exists(path):
                    os.makedirs(path)

            epoch = 0
            if isinstance(Recmodel, nn.Module):        
                Recmodel.load_state_dict(torch.load(weight_file))
                # from Procedure import Test_Embeddings
                # results = Test_Embeddings(dataset, Recmodel, epoch, w, world.config['multicore'])
                results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
                print(results)
                results_groups, group_line= Procedure.Test_Diff_Users(dataset, Recmodel, epoch, w, world.config['multicore'])
                sparsity_path = f'./user_sparsity_analysis/{world.dataset}/{world.model_name}/'
                ensure_dirs(sparsity_path)
                joblib.dump(results_groups, sparsity_path + 'results_group.pkl', compress=3)
                joblib.dump(group_line, sparsity_path + 'group_line.pkl', compress=3)
                print("results_groups:", results_groups)
                for key in results_groups:
                    result = results_groups[key]
                    print("[%.5f, %.5f, %.5f],"%(result['precision'], result['recall'], result['ndcg']))

                # mplot(neg_list, 'negative_log_likelihood')
                # mplot(kl_list, 'kl_divergence')
                # mplot(reg_list, 'regularization_loss')
                # mplot(ndcg_list, 'ndcg@20')
                # path = './pakdd/plot_data/'
                # ensure_dirs(path)
                # save_data(neg_list, path, 'negll-norm')
                # save_data(kl_list, path, 'kl-norm')
                # save_data(ndcg_list, path, 'ndcg-norm')

    finally:
        if world.tensorboard:
            w.close()
