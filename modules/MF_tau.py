'''
Created on March 1st, 2023

@author: Junkang Wu (jkwu0909@gmail.com)
'''
from tarfile import POSIX_MAGIC
from numpy.core.fromnumeric import size
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import sys
sys.path.append("..")
from utils import losses
from scipy.special import lambertw

class MF(nn.Module):
    def __init__(self, data_config, args_config, adj_mat, logger=None):
        super(MF, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat

        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.logger = logger
        if self.mess_dropout:
            self.dropout = nn.Dropout(args_config.mess_dropout_rate)
        self.edge_dropout = args_config.edge_dropout
        self.edge_dropout_rate = args_config.edge_dropout_rate
        self.pool = args_config.pool
        self.n_negs = args_config.n_negs

        self.temperature = args_config.temperature
        self.temperature_2 = args_config.temperature_2

        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")
     
        # param for norm
        self.u_norm = args_config.u_norm
        self.i_norm = args_config.i_norm
        self.tau_mode = args_config.tau_mode
        # init  setting
        self._init_weight()
        self.user_embed = nn.Parameter(self.user_embed)
        self.item_embed = nn.Parameter(self.item_embed)
        self.loss_name = args_config.loss_fn
        self.generate_mode = args_config.generate_mode
        # define loss function
        if args_config.loss_fn == "Adap_tau_Loss":
            print(self.loss_name)
            print("start to make tables")
            self.lambertw_table = torch.FloatTensor(lambertw(np.arange(-1, 1002, 1e-4))).to(self.device)
            self.loss_fn = losses.Adap_tau_Loss()
        elif args_config.loss_fn == "SSM_Loss":
            print(self.loss_name)
            self.loss_fn = losses.SSM_Loss(self._margin, self.temperature, self._negativa_weight, args_config.pos_mode)
        else:
            raise NotImplementedError("loss={} is not support".format(args_config.loss_fn))

        self.register_buffer("memory_tau", torch.full((self.n_users,), 1 / 0.10))
        self.register_buffer("memory_tau_i", torch.full((self.n_users,), 1 / 0.10))

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.user_embed = initializer(torch.empty(self.n_users, self.emb_size))
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))

    def _update_tau_memory(self, x, x_i):
        # x: std [B]
        # y: update position [B]
        with torch.no_grad():
            x = x.detach()
            x_i = x_i.detach()
            self.memory_tau = x
            self.memory_tai_i = x_i

    # def _loss_to_tau(self, x, x_all):
    #     t_0 = x_all
    #     if x is None:
    #         tau = t_0 * torch.ones_like(self.memory_tau, device=self.device)
    #     else:
    #         base_laberw = torch.mean(x)
    #         laberw_data = torch.clamp((x - base_laberw) / self.temperature_2,
    #                                 min=-np.e ** (-1), max=1000)
    #         laberw_data = self.lambertw_table[((laberw_data + 1) * 1e4).long()]
    #         tau = (t_0 * torch.exp(-laberw_data)).detach()
    #     return tau

    def _loss_to_tau(self, x, x_all):
        if self.tau_mode == "weight_v0":
            t_0 = x_all
            tau = t_0 * torch.ones_like(self.memory_tau, device=self.device)
        elif self.tau_mode == "weight_ratio":
            t_0 = x_all
            if x is None:
                tau = t_0 * torch.ones_like(self.memory_tau, device=self.device)
            else:
                base_laberw = torch.quantile(x, self.temperature)
                laberw_data = torch.clamp((x - base_laberw) / self.temperature_2,
                                        min=-np.e ** (-1), max=1000)
                laberw_data = self.lambertw_table[((laberw_data + 1) * 1e4).long()]
                tau = (t_0 * torch.exp(-laberw_data)).detach()
        elif self.tau_mode == "weight_mean":
            t_0 = x_all
            if x is None:
                tau = t_0 * torch.ones_like(self.memory_tau, device=self.device)
            else:
                # Calculate norms
                user_norms = torch.norm(self.user_embed, p=2, dim=1)
                item_norms = torch.norm(self.item_embed, p=2, dim=1)
                embedding_norms = torch.cat([user_norms, item_norms])

                # Calculate Median Absolute Deviation (MAD)
                median = torch.median(embedding_norms)
                mad = torch.median(torch.abs(embedding_norms - median))
                variance_norms = torch.var(embedding_norms)

                #kappa = 1.0  # Sensitivity to MAD changes
                mad_factor = 0.1 * mad
                
                var = 1.0 * variance_norms

                base_laberw = torch.quantile(x, self.temperature)
                #base_laberw = torch.mean(x)
                
                laberw_input = torch.clamp((x  - base_laberw + mad_factor + var) / self.temperature_2,
                                           min=-np.e ** (-1), max=1000)
                laberw_data = self.lambertw_table[((laberw_input + 1) * 1e4).long()]
                tau = (t_0 * torch.exp(-laberw_data)).detach()
        return tau
    
    def add_noise(self, emb, eps=0.03):
        random_noise = torch.rand_like(emb).cuda()
        emb_ = emb + torch.sign(emb) * F.normalize(random_noise, dim=-1) * eps
        return emb_
    
    def InfoNCE(self, view1, view2, temperature: float, b_cos: bool = True):
        """
        Args:
            view1: (torch.Tensor - N x D)
            view2: (torch.Tensor - N x D)
            temperature: float
            b_cos (bool)

        Return: Average InfoNCE Loss
        """
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

        pos_score = (view1 @ view2.T) / temperature
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()
    
    def cal_cl_loss(self,user_view1,user_view2,item_view1,item_view2,temperature_u, temperature_i):
        
        user_cl_loss = self.InfoNCE(user_view1, user_view2, temperature_u.unsqueeze(1))
        item_cl_loss = self.InfoNCE(item_view1, item_view2, temperature_i.unsqueeze(1))
        
        return (user_cl_loss + item_cl_loss)
    
    def forward(self, batch=None, loss_per_user=None, loss_per_ins=None, epoch=None, w_0=None, s=0):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']
        
        cl_user_emb = self.add_noise(self.user_embed)
        cl_item_emb = self.add_noise(self.item_embed)

        if s == 0 and w_0 is not None:
            tau_user = self._loss_to_tau(loss_per_user, w_0)
            tau_item = self._loss_to_tau(loss_per_ins, w_0)
            self._update_tau_memory(tau_user, tau_item)
         
        return self.Uniform_loss(self.user_embed[user], self.item_embed[pos_item], self.item_embed[neg_item], cl_user_emb[user], cl_item_emb[pos_item], user, w_0)

    def gcn_emb(self):
        user_gcn_emb, item_gcn_emb = self.user_embed, self.item_embed
        # user_gcn_emb, item_gcn_emb = self.pooling(user_gcn_emb), self.pooling(item_gcn_emb)
        return user_gcn_emb.detach(), item_gcn_emb.detach()
    
    def generate(self, mode='test', split=True):
        user_gcn_emb = self.user_embed
        item_gcn_emb = self.item_embed
        if self.generate_mode == "cosine":
            if self.u_norm:
                user_gcn_emb = F.normalize(user_gcn_emb, dim=-1)
            if self.i_norm:
                item_gcn_emb = F.normalize(item_gcn_emb, dim=-1)
        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    # 对比训练loss，仅仅计算角度
    def Uniform_loss(self, user_gcn_emb, pos_gcn_emb, neg_gcn_emb, cl_user_emb, cl_item_emb, user, w_0=None):
        batch_size = user_gcn_emb.shape[0]
        u_e = user_gcn_emb  # [B, F]
        if self.mess_dropout:
            u_e = self.dropout(u_e)
        pos_e = pos_gcn_emb # [B, F]
        neg_e = neg_gcn_emb # [B, M, F]

        item_e = torch.cat([pos_e.unsqueeze(1), neg_e], dim=1) # [B, M+1, F]
        if self.u_norm:
            u_e = F.normalize(u_e, dim=-1)
            
        if self.i_norm:
            item_e = F.normalize(item_e, dim=-1)

        y_pred = torch.bmm(item_e, u_e.unsqueeze(-1)).squeeze(-1) # [B M+1]
        # cul regularizer
        regularize = (torch.norm(user_gcn_emb[:, :]) ** 2
                       + torch.norm(pos_gcn_emb[:, :]) ** 2
                       + torch.norm(neg_gcn_emb[:, :, :]) ** 2) / 2  # take hop=0
        emb_loss = self.decay * regularize / batch_size

        if self.loss_name == "Adap_tau_Loss":
            mask_zeros = None
            tau = torch.index_select(self.memory_tau, 0, user).detach()
            tau_i = torch.index_select(self.memory_tau_i, 0, user).detach()
            loss, loss_ = self.loss_fn(y_pred, tau, w_0)
            cl_loss = 0.2 * self.cal_cl_loss(u_e, cl_user_emb, pos_e, cl_item_emb, tau, tau_i)
            return loss.mean() + emb_loss, loss_, emb_loss, tau
        elif self.loss_name == "SSM_Loss":
            loss, loss_ = self.loss_fn(y_pred)
            return loss.mean() + emb_loss, loss_, emb_loss, y_pred
        else:
            raise NotImplementedError("loss={} is not support".format(self.loss_name))

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0

    # negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask
