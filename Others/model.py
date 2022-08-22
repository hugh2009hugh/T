"""
Created on Aug 22, 2022

Define models here
"""
from numpy.core.fromnumeric import compress, std
from torch.nn import init
from torch.optim import Optimizer
from world import cprint
import world
import torch
from torch.autograd import Variable, backward
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.utils.data import DataLoader
import torch.utils.data as data
import time
from dataloader import BasicDataset
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
import numpy as np
from sparsesvd import sparsesvd
from copy import deepcopy
import math
import os
import joblib
from sklearn.cluster import KMeans
from concurrent import futures
import copy
from tqdm import tqdm
from collections import defaultdict
import pickle
import operator
from functools import reduce
from itertools import chain
from scipy.sparse import csr_matrix


class VAEKernel(nn.Module):
    """
    Implementation of Mult-VAE
    """
    def __init__(self, config, dataset):
        super(VAEKernel, self).__init__()
        self.config = config
        self.dataset = dataset

        R = self.dataset.UserItemNet.A
        self.R = torch.tensor(R).float()

        self.lam = self.config['vae_reg_param']
        self.anneal_ph = self.config['kl_anneal']
        self.act = world.config['act_vae']
        self.tau = config['tau_model2']
        self.dropout = config['dropout_model2']
        self.alpha = config['alpha_model2']
        self.normalize = config['normalize_model2']
        self.topk = config['topK_model3']
        self.FP = config['FP']
        self.FN = config['FN']
        self.weight_scale = config['weight_scale']
        self.topK = config['topK']

        enc_dims = self.config['enc_dims']
        if isinstance(enc_dims, str):
            enc_dims = eval(enc_dims)
        dec_dims = enc_dims[::-1]

        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        enc_dims = [self.num_items] + enc_dims
        dec_dims = dec_dims + [self.num_items]

        self.encoder = nn.Sequential()
        for i, (in_dim, out_dim) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
            if i == len(enc_dims) - 2:
                out_dim = out_dim * 2
            self.encoder.add_module(name='Encoder_Linear_%s'%i, module=nn.Linear(in_dim, out_dim))
            if i != len(enc_dims) - 2:
                self.encoder.add_module(name='Encoder_Activation_%s'%i, module=self.act)
                pass

        self.mapper = nn.Linear(enc_dims[-1], 1)  

        self.items = nn.parameter.Parameter(torch.randn(self.num_items, enc_dims[-1]))
        self.items_weight = nn.parameter.Parameter(torch.randn(self.num_items, enc_dims[-1]))
 
        self.init_param()
        self.init_optim()

        self.get_topk_ii()

        self.gram_matrix = torch.tensor(self.gram_matrix).float().to(world.device)
        self.epoch = 0
    
    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if isinstance(m.bias, torch.Tensor):
                    nn.init.trunc_normal_(m.bias, std=0.001)
        nn.init.xavier_normal_(self.items)
        
    
    def init_optim(self):
        self.optim = optim.Adam([param for param in self.parameters() if param.requires_grad], self.config['vae_lr'])
    
    def forward(self, rating_matrix_batch):

        batch_input0 = F.normalize(rating_matrix_batch, p=2, dim=1)
        batch_input0 = F.dropout(batch_input0, p=self.dropout, training=self.training)

        x = self.encoder(batch_input0)
        mean, logvar = x[:, :(len(x[0])//2)], x[:, (len(x[0])//2):]
        stddev = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(stddev)
        if self.training:
            z = mean + epsilon * stddev
        else:
            z = mean
        dot_product = z @ self.items.T

        if self.normalize:
            z = F.normalize(z)
            items = F.normalize(self.items)
            out = z @ items.T  / self.tau #- self.popularity
        
        else:
            out = z @ self.items.T 

        var_square = torch.exp(logvar)
        kl = 0.5 * torch.mean(torch.sum(mean ** 2 + var_square - 1. - logvar, dim=-1))

        return z, out, kl, batch_input0  #out_weight 对应每个样本的不确定度

    #核心方法：先整理出新的input，再做forward计算，再根据新的input重新整理结果
    def forward_kernel(self, rating_matrix_batch):
        batch_input0 = F.normalize(rating_matrix_batch, p=2, dim=1)
        batch_input0 = F.dropout(batch_input0, p=self.dropout, training=self.training)
        batch_input0_bak = batch_input0.detach()

        zeros = torch.zeros(rating_matrix_batch.shape[1]).to(world.device)
        ones = torch.ones(rating_matrix_batch.shape[1]).to(world.device)

        #将batch_input0转为只有0/1元素
        batch_input01 = torch.where(batch_input0>0, ones, zeros)
        #为每个用户进行处理
        batch_input_arr = []   #记录每个用户下多个新input
        batch_input_num = []   #记录每个用户下多个新input的 个数
        for user in range(batch_input01.shape[0]):
            user_input = batch_input01[user]
            items = torch.nonzero(user_input)
            #取出每个item对应的gram matrix(行) 中与其他商品的相似度， 并与batch_input01进行处理
            for item in items:
                item_similars = self.gram_matrix[item]  #整个gram matrix放不下，只是每次调用时，才放入gpu
                input_item = item_similars * user_input
                input_item = F.normalize(input_item, p=1, dim=1)
                batch_input_arr.append(input_item)
            
            batch_input_num.append(len(items))
            # print(batch_input_arr, batch_input_num)
            # input("Next?")

        new_input = torch.cat(batch_input_arr, dim =0)
        batch_input0 = F.normalize(new_input, p=2, dim=1)

        #沿用旧的encoder 和 decoder
        x = self.encoder(batch_input0)
        mean, logvar = x[:, :(len(x[0])//2)], x[:, (len(x[0])//2):]
        stddev = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(stddev)
        if self.training:
            z = mean + epsilon * stddev
        else:
            z = mean
        dot_product = z @ self.items.T

        if self.normalize:
            z = F.normalize(z)
            items = F.normalize(self.items)
            out = z @ items.T  / self.tau #- self.popularity
        
        else:
            out = z @ self.items.T 

        #这里对z重新进行整合
        _index = 0
        new_output = []
        out = torch.exp(out)
        for inner_num in batch_input_num: 
            if inner_num!= 0:
                start_index = _index
                end_index = _index + inner_num   #有些可能 inner_num 为0 ，要处理该异常
                inner_out = torch.mean(out[start_index:end_index, :], dim = 0)  #对一个用户下的各个out求平均
                inner_out = torch.log(inner_out)
                #print("inner_out:", inner_out)
                new_output.append(inner_out.unsqueeze(0))
                #print(start_index,end_index, torch.isnan(inner_out).int().sum() !=0)
                _index = end_index
            else:    #有些可能 inner_num 为0 ，要处理该异常
                new_output.append(zeros.unsqueeze(0))

        new_output = torch.cat(new_output, dim=0)
        #print(new_output.shape)

        var_square = torch.exp(logvar)
        kl = 0.5 * torch.mean(torch.sum(mean ** 2 + var_square - 1. - logvar, dim=-1))

        #对比原始的，以及汇总版本的
        # if self.epoch >100:
            
        #     x = self.encoder(batch_input0_bak)
        #     mean, logvar = x[:, :(len(x[0])//2)], x[:, (len(x[0])//2):]
        #     stddev = torch.exp(0.5 * logvar)
        #     epsilon = torch.randn_like(stddev)
        #     if self.training:
        #         z = mean + epsilon * stddev
        #     else:
        #         z = mean
        #     dot_product = z @ self.items.T

        #     if self.normalize:
        #         z = F.normalize(z)
        #         items = F.normalize(self.items)
        #         out = z @ items.T  / self.tau #- self.popularity

        #     #旧版救过
        #     print((torch.nn.Softmax(dim=1)(out)*batch_input0_bak)[0])
        #     print(torch.nonzero((torch.nn.Softmax(dim=1)(out)*batch_input0_bak)[0]))
        #     print((torch.nn.LogSoftmax(dim=1)(out)*batch_input0_bak)[0,torch.nonzero((torch.nn.LogSoftmax(dim=1)(out)*batch_input0_bak)[0])])
        #     print(torch.sum((torch.nn.LogSoftmax(dim=1)(out)*batch_input0_bak)[0,torch.nonzero((torch.nn.LogSoftmax(dim=1)(out)*batch_input0_bak)[0])]))

        #     #分合版结果
        #     print((torch.nn.Softmax(dim=1)(new_output)*batch_input0_bak)[0])
        #     print(torch.nonzero((torch.nn.Softmax(dim=1)(new_output)*batch_input0_bak)[0]))
        #     print((torch.nn.LogSoftmax(dim=1)(new_output)*batch_input0_bak)[0,torch.nonzero((torch.nn.LogSoftmax(dim=1)(new_output)*batch_input0_bak)[0])])
        #     print(torch.sum((torch.nn.LogSoftmax(dim=1)(new_output)*batch_input0_bak)[0,torch.nonzero((torch.nn.LogSoftmax(dim=1)(new_output)*batch_input0_bak)[0])]))
        #     input("Next?")


        return z, new_output, kl, batch_input0  #out_weight 对应每个样本的不确定度

    def getUsersRating(self, users):
        self.eval()
        users = users.cpu()
        rating_matrix_batch = self.R[users].to(world.device)
        _, predict_out, _, _ = self.forward_kernel(rating_matrix_batch)
        return predict_out

    @staticmethod
    def calculate_mult_log_likelihood(prediction, label, users, items, ii_sim):
        log_softmax_output = torch.nn.LogSoftmax(dim=-1)(prediction)
        log_likelihood_O = -torch.mean(torch.sum(log_softmax_output * label, 1))
        log_likelihood = log_likelihood_O
        log_likelihood_I = -torch.sum(log_softmax_output[(users, items)] * ii_sim) / prediction.shape[0]
        log_likelihood += log_likelihood_I
        return log_likelihood, log_likelihood_O, log_likelihood_I

    #考虑权重的版本  #0304重新考虑multnomial distribution likelihood , 结合negative sampling and SimpleX
    @staticmethod
    def calculate_mult_log_likelihood_simple(prediction, label):        
        
        #print(prediction, torch.isnan(prediction).int().sum() !=0)
        log_softmax_output = torch.nn.LogSoftmax(dim=-1)(prediction)        
        log_likelihood = -torch.mean(torch.sum(log_softmax_output * label, 1))
    
        return log_likelihood, log_likelihood, log_likelihood*0
    
    
    def reg_loss(self):
        """
        Return the L2 regularization of weights. Code is implemented according to tensorflow.
        """
        reg_list = [0.5 * torch.sum(m.weight ** 2) for m in self.modules() if isinstance(m, nn.Linear)]
        reg_list = reg_list + [0.5 * torch.sum(self.items ** 2)] # + [0.5 * torch.sum(self.items_weight ** 2)]
        reg = 0.
        for val in reg_list:
            reg += val
        # return val # Experiments in PAKDD, I have written the wrong code. But it doesn't matter.
        return reg

    def train_one_epoch(self):
        self.train()
        users = np.arange(self.num_users)
        #np.random.shuffle(users)
        batch_size = self.config['vae_batch_size']
        n_batch = math.ceil(self.num_users / batch_size)
        loss_dict = {}
        neg_ll_list = []
        kl_list = []
        reg_list = []

        self.epoch += 1

        torch.autograd.set_detect_anomaly(True)
        for idx in range(n_batch):

            start_idx = idx * batch_size
            end_idx = min(start_idx + batch_size, self.num_users)
            batch_users = users[start_idx:end_idx]
            rating_matrix_batch = self.R[batch_users].to(world.device)


            _, predict_out, kl, _ = self.forward_kernel(rating_matrix_batch)
            
            neg_ll, log_likelihood_O, log_likelihood_I = self.calculate_mult_log_likelihood_simple(predict_out, rating_matrix_batch)  #self.PRINT 传递epoch信息
            
            #############################################################################
            loss = neg_ll + self.anneal_ph * kl + self.lam * self.reg_loss()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if idx == n_batch - 1:
                print('log_likelihood_O: %.4f'%(log_likelihood_O.item()-log_likelihood_I.item()), end=", ")
                print('log_likelihood_I: %.4f'%log_likelihood_I.item(), end=", ")
                print("anneal_ph *KL: %.4f"%(self.anneal_ph *kl.item()), end=", ")
                print("lam * reg_loss: %.4f"%(self.lam *self.reg_loss().item()))

            neg_ll_list.append(neg_ll.item())
            kl_list.append(kl.item())
            reg_list.append(self.reg_loss().item())

        loss_dict['neg_ll'] = neg_ll_list
        loss_dict['kl'] = kl_list
        loss_dict['reg'] =  reg_list
        return loss_dict

    def get_topk_ii(self):
        """
        For every item, get its topk similar items according to the co-occurrent matrix.
        """
        save_path = f'./pretrained/{world.dataset}/{world.model_name}'
        ii_sim_mat_path = save_path + '/ii_sim_mat_'+ str(self.topk) +'.pkl'
        ii_sim_idx_mat_path = save_path + '/ii_sim_idx_mat_'+ str(self.topk) +'.pkl'
        gram_matrix_path = save_path + '/gram_matrix.pkl'
        if not os.path.exists(ii_sim_mat_path):
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            adj_mat = self.dataset.UserItemNet
            row_sum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(row_sum, -0.5).flatten()
            d_inv[np.isposinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            norm_mat = d_mat.dot(adj_mat)
            col_sum = np.array(adj_mat.sum(axis=0))
            d_inv = np.power(col_sum, -0.5).flatten()
            d_inv[np.isposinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            norm_mat = norm_mat.dot(d_mat).astype(np.float32)
            gram_matrix = norm_mat.T.dot(norm_mat).toarray()
            print("Successfully created the co-occurrence matrix!")
            self.ii_sim_mat = torch.zeros(self.num_items, self.topk)
            self.ii_sim_idx_mat = torch.zeros(self.num_items, self.topk)
            for iid in range(self.num_items):
                row = torch.from_numpy(gram_matrix[iid])
                sim, idx = torch.topk(row, self.topk)
                self.ii_sim_mat[iid] = sim
                self.ii_sim_idx_mat[iid] = idx
                if iid % 15000 == 0:
                    print('Getting {} items\' topk done'.format(iid))
            self.ii_sim_mat = self.ii_sim_mat
            self.ii_sim_idx_mat = self.ii_sim_idx_mat.numpy()
            self.gram_matrix = gram_matrix
            joblib.dump(self.ii_sim_mat, ii_sim_mat_path, compress=3)
            joblib.dump(self.ii_sim_idx_mat, ii_sim_idx_mat_path, compress=3)
            joblib.dump(gram_matrix, gram_matrix_path, compress=3)
        else:
            self.ii_sim_mat = joblib.load(ii_sim_mat_path)
            self.ii_sim_idx_mat = joblib.load(ii_sim_idx_mat_path)
            self.gram_matrix = joblib.load(gram_matrix_path)

        print(world.LOAD)
        if world.LOAD ==1:  #如果加载预训练的模型
            #gram_matrix = self.items.mm(self.items.T)
            #print(self.encoder, self.encoder[0])
            weight_len = self.encoder[0].weight.shape[0]//2
            print(weight_len)
            gram_matrix = self.encoder[0].weight[:weight_len,:].T.mm(self.encoder[0].weight[:weight_len,:])
            
            # #保存encoder中item的embed
            # fout = open('encoder_item.pkl','wb')
            # pickle.dump(self.encoder[0].weight[:weight_len,:].T.detach().numpy(), fout)
            # fout.close()

            #保存encoder中item的embed
            fout = open('decoder_item.pkl','wb')
            pickle.dump(self.items.detach().numpy(), fout)
            fout.close()

            f = open("items_embedding_VAE++.pkl",'rb+')

            try:
                item_embedding = pickle.load(f)
            except EOFError: #捕获异常EOFError 后返回None
                print("EOFError!")

            gram_matrix = torch.from_numpy(item_embedding.dot(item_embedding.T))
        
            self.gram_matrix = gram_matrix

            print(gram_matrix.cpu().shape, self.items.shape)
            #print(self.encoder['Encoder_Linear_0'].data, self.encoder['Encoder_Linear_0'].data.mul(self.encoder['Encoder_Linear_0'].data.T))



