'''
Created on July 1st, 2024

@author: Li Haofeng (2152498@tongji.edu.cn)
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




class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, n_hops, n_users, interact_mat,
                 edge_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.interact_mat = interact_mat
        self.n_users = n_users
        self.n_hops = n_hops
        self.edge_dropout_rate = edge_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))
    
    

    def forward(self, user_embed, item_embed,
                mess_dropout=True, edge_dropout=True, perturbed=False, eps=0.03):
        # user_embed: [n_users, channel]
        # item_embed: [n_items, channel]

        # all_embed: [n_users+n_items, channel]
        all_embed = torch.cat([user_embed, item_embed], dim=0)
        
        agg_embed = all_embed
        embs = [all_embed]
        all_embeddings_cl = agg_embed
        for hop in range(self.n_hops):
            interact_mat = self._sparse_dropout(self.interact_mat,
                                                self.edge_dropout_rate) if edge_dropout \
                                                                        else self.interact_mat

            agg_embed = torch.sparse.mm(interact_mat, agg_embed)
            if mess_dropout:
                agg_embed = self.dropout(agg_embed)
            # agg_embed = F.normalize(agg_embed)
            else:
                if perturbed:
                    random_noise = torch.rand_like(agg_embed).cuda()
                    agg_embed += torch.sign(agg_embed) * F.normalize(random_noise, dim=-1) * eps
            embs.append(agg_embed)
            if hop==0:
                all_embeddings_cl = agg_embed
        embs = torch.stack(embs, dim=1)  # [n_entity, n_hops+1, emb_size]
        if perturbed:
            return embs[:self.n_users, :], embs[self.n_users:, :],all_embeddings_cl[:self.n_users, :], all_embeddings_cl[self.n_users:, :]
        return embs[:self.n_users, :], embs[self.n_users:, :]

class lgn_frame(nn.Module):
    def __init__(self, data_config, args_config, adj_mat, logger=None):
        super(lgn_frame, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat
        
        self.func = args_config.func
        self.func_origin = args_config.func_origin
        self.cl_rate = args_config.cl_rate
        

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
        
        
        
        self.prev_loss = None
        self.prev_std = None
        self.base_laberw = None
        self.P = 0.7  # Proportional coefficient
        self.I = 0.01  # Integral coefficient
        self.D = 0.05  # Derivative coefficient
        self.beta = 2.0  # Scaling factor for tau adjustment
        self.previous_variance_norms = None

        # Initialize variables for PID control
        self.integral_error = 0
        self.prev_error = 0
        self.momentum = 0.1
        
        self.total_epoch = args_config.epoch
      
        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")
       
        # param for norm
        self.u_norm = args_config.u_norm
        self.i_norm = args_config.i_norm
        self.tau_mode = args_config.tau_mode
       
        self._init_weight()
        self.user_embed = nn.Parameter(self.user_embed)
        self.item_embed = nn.Parameter(self.item_embed)
        self.base_kappa = nn.Parameter(torch.tensor(0.1))  # Example starting sensitivity value
       
        self.loss_name = args_config.loss_fn
        
        self.generate_mode = args_config.generate_mode

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
        self.register_buffer("memory_tau_u", torch.full((self.n_users,), 1 / 0.10))
        self.gcn = self._init_model()
        self.sampling_method = args_config.sampling_method
        
        self.u_target_his = torch.randn((self.n_users, self.emb_size), requires_grad=False).to(self.device)
        self.i_target_his = torch.randn((self.n_items, self.emb_size), requires_grad=False).to(self.device)
        

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.user_embed = initializer(torch.empty(self.n_users, self.emb_size))
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))
        
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(n_hops=self.context_hops,
                         n_users=self.n_users,
                         interact_mat=self.sparse_norm_adj,
                         edge_dropout_rate=self.edge_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)
    
    

    def _update_tau_memory(self, x, u_cl, i_cl):
        # x: std [B]
        # y: update position [B]
        with torch.no_grad():
            x = x.detach()
            u_cl = u_cl.detach()
            i_cl = i_cl.detach()
            self.memory_tau = x
            self.memory_tau_i = i_cl
            self.memory_tau_u = u_cl

    def get_decay_factor(self, epoch):
        return 1 / (1 + epoch * 0.01)  # Example decay function; adjust as needed
    
    def _loss_to_tau_cl(self, x, x_all):
        if self.tau_mode == "weight_v0":
            t_0 = x_all
            tau = t_0 * torch.ones_like(self.memory_tau, device=self.device)
        elif self.tau_mode == "weight_ratio":
            t_0 = x_all
            if x is None:
                tau = t_0 * torch.ones_like(self.memory_tau, device=self.device)
            else:
                # base_laberw = torch.quantile(x, self.temperature)
                base_laberw = torch.mean(x)
                laberw_data = torch.clamp((x - base_laberw) / self.temperature_2,
                                        min=-np.e ** (-1), max=1000)
                laberw_data = self.lambertw_table[((laberw_data + 1) * 1e4).long()]
                tau = (t_0 * torch.exp(-laberw_data)).detach()
        elif self.tau_mode == "weight_mean":
            t_0 = x_all
            if x is None:
                tau = t_0 * torch.ones_like(self.memory_tau, device=self.device)
            else:
                base_laberw = torch.mean(x)
                laberw_data = torch.clamp((x - base_laberw) / self.temperature_2,
                                        min=-np.e ** (-1), max=1000)
                laberw_data = self.lambertw_table[((laberw_data + 1) * 1e4).long()]
                tau = (t_0 * torch.exp(-laberw_data)).detach()                
        return tau
            
    def _loss_to_tau(self, x, x_all, current_epoch, total_epochs, u_e, pos_e):
        if self.tau_mode == "weight_v0":
            t_0 = x_all
            tau = t_0 * torch.ones_like(self.memory_tau, device=self.device)
        elif self.tau_mode == "weight_ratio":
            t_0 = x_all
            if x is None:
                tau = t_0 * torch.ones_like(self.memory_tau, device=self.device)
            else:
                # base_laberw = torch.quantile(x, self.temperature)
                base_laberw = torch.mean(x)
                laberw_data = torch.clamp((x - base_laberw) / self.temperature_2,
                                        min=-np.e ** (-1), max=1000)
                laberw_data = self.lambertw_table[((laberw_data + 1) * 1e4).long()]
                tau = (t_0 * torch.exp(-laberw_data)).detach()
        elif self.tau_mode == "weight_mean":
            t_0 = x_all
            if x is None:
                tau = t_0 * torch.ones_like(self.memory_tau, device=self.device)
            else:
                # base_laberw = torch.mean(x)
                # laberw_data = torch.clamp((x - base_laberw) / self.temperature_2,
                #                         min=-np.e ** (-1), max=1000)
                # laberw_data = self.lambertw_table[((laberw_data + 1) * 1e4).long()]
                # tau = (t_0 * torch.exp(-laberw_data)).detach()
                #user_embed, item_embed = self.gcn_emb()
                #u_e1, pos_e1 = self.pooling(u_e), self.pooling(pos_e)
                if self.u_norm:
                    u_e1 = F.normalize(u_e, dim=-1)
                    
                    
                if self.i_norm:
                    pos_e1 = F.normalize(pos_e, dim=-1)
                user_norms = torch.norm(u_e1, p=2, dim=1)
                item_norms = torch.norm(pos_e1, p=2, dim=1)
                embedding_norms = torch.cat([user_norms, item_norms])

                # Calculate Median Absolute Deviation (MAD)
                median = torch.median(embedding_norms)
                mad = torch.median(torch.abs(embedding_norms - median))
                variance_norms = torch.var(embedding_norms)

                #kappa = 1.0  # Sensitivity to MAD changes
                mad_factor = 0.5 * mad
                
                var = 2.0 * variance_norms
                func = mad_factor + var
                
                user_norms_origin = torch.norm(self.user_embed, p=2, dim=1)
                item_norms_origin = torch.norm(self.item_embed, p=2, dim=1)
                embedding_norms_origin = torch.cat([user_norms_origin, item_norms_origin])

                # Calculate Median Absolute Deviation (MAD)
                median_origin = torch.median(embedding_norms_origin)
                mad_origin = torch.median(torch.abs(embedding_norms_origin - median_origin))
                variance_norms_origin = torch.var(embedding_norms_origin)

                #kappa = 1.0  # Sensitivity to MAD changes
                mad_factor_origin = 0.1 * mad_origin
                
                var_origin = 1.0 * variance_norms_origin

                base_laberw = torch.mean(x)
                
                func_origin = var_origin + mad_factor_origin
                
                laberw_input = torch.clamp((x  - base_laberw + func_origin * self.func_origin + func * self.func) / self.temperature_2,
                                           min=-np.e ** (-1), max=1000)
                laberw_data = self.lambertw_table[((laberw_input + 1) * 1e4).long()]
                tau = (t_0 * torch.exp(-laberw_data)).detach()                             
        return tau
        

       

    

    def forward(self, batch=None, loss_per_user=None, loss_per_ins=None, epoch=None, w_0=None, s=0):
        user = batch['users']
        pos_item = batch['pos_items']
        user_gcn_emb, item_gcn_emb, cl_user_emb, cl_item_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=self.edge_dropout,
                                              mess_dropout=self.mess_dropout, perturbed=True)
        # neg_item = batch['neg_items']  # [batch_size, n_negs * K]
        if s == 0 and w_0 is not None:
            # self.logger.info("Start to adjust tau with respect to users")
            tau_user = self._loss_to_tau(loss_per_user, w_0, epoch, self.total_epoch, user_gcn_emb[user], item_gcn_emb[pos_item])
            #tau_item = self._loss_to_tau(loss_per_ins, w_0, epoch, self.total_epoch, user_gcn_emb[user], item_gcn_emb[pos_item])
            tau_user_cl = self._loss_to_tau_cl(loss_per_user, w_0)
            tau_item_cl = self._loss_to_tau_cl(loss_per_ins, w_0)
            self._update_tau_memory(tau_user, tau_user_cl, tau_item_cl)
            
        if self.sampling_method == "no_sample":
            return self.NO_Sample_Uniform_loss(user_gcn_emb[user], item_gcn_emb[pos_item], cl_user_emb[user], cl_item_emb[pos_item], user, pos_item, w_0)
        else:
            neg_item = batch['neg_items']
            return self.Uniform_loss(user_gcn_emb[user], item_gcn_emb[pos_item], item_gcn_emb[neg_item], user, w_0)
       

    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.pool == 'mean':
            return embeddings.mean(dim=-2)
        elif self.pool == 'sum':
            return embeddings.sum(dim=-2)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def gcn_emb(self):
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=False,
                                              mess_dropout=False)
        user_gcn_emb, item_gcn_emb = self.pooling(user_gcn_emb), self.pooling(item_gcn_emb)
        return user_gcn_emb.detach(), item_gcn_emb.detach()
    
    
    def generate(self, mode='test', split=True):
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=False,
                                              mess_dropout=False)
        user_gcn_emb, item_gcn_emb = self.pooling(user_gcn_emb), self.pooling(item_gcn_emb)
        if self.generate_mode == "cosine":
            user_gcn_emb = F.normalize(user_gcn_emb, dim=-1)
            item_gcn_emb = F.normalize(item_gcn_emb, dim=-1)
                
        elif self.generate_mode == "reweight":
            # reweight focus on items
            item_norm = torch.norm(item_gcn_emb, p=2, dim=-1)
            mean_norm = item_norm.mean()
            item_gcn_emb = item_gcn_emb / item_norm.unsqueeze(1)  * mean_norm * self.reweight.unsqueeze(1)
        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())
    
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

    # 对比训练loss，仅仅计算角度
    def Uniform_loss(self, user_gcn_emb, pos_gcn_emb, neg_gcn_emb, user, w_0=None):
        batch_size = user_gcn_emb.shape[0]
        u_e = self.pooling(user_gcn_emb)  # [B, F]
        pos_e = self.pooling(pos_gcn_emb) # [B, F]
        neg_e = self.pooling(neg_gcn_emb) # [B, M, F]

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
            loss, loss_ = self.loss_fn(y_pred, tau, w_0)
            return loss.mean() + emb_loss, loss_, emb_loss, tau
        elif self.loss_name == "SSM_Loss":
            loss, loss_ = self.loss_fn(y_pred)
            return loss.mean() + emb_loss, loss_, emb_loss, y_pred
        else:
            raise NotImplementedError("loss={} is not support".format(self.loss_name))


    def loss_cf(self, p, z):  # negative cosine similarity
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    
    def NO_Sample_Uniform_loss(self, user_gcn_emb, pos_gcn_emb, cl_user_emb, cl_item_emb, user, pos_item, w_0=None):
        batch_size = user_gcn_emb.shape[0]
        u_e = self.pooling(user_gcn_emb)  # [B, F]
        pos_e = self.pooling(pos_gcn_emb) # [B, F]

        if self.u_norm:
            u_e = F.normalize(u_e, dim=-1)
            u_e_cl = F.normalize(cl_user_emb, dim=-1)
            
        if self.i_norm:
            pos_e = F.normalize(pos_e, dim=-1)
            item_e_cl = F.normalize(cl_item_emb, dim=-1)
            
        
        # contrust y_pred framework
        row_swap = torch.cat([torch.arange(batch_size).long(), torch.arange(batch_size).long()]).to(self.device)
        col_before = torch.cat([torch.arange(batch_size).long(), torch.zeros(batch_size).long()]).to(self.device)
        col_after = torch.cat([torch.zeros(batch_size).long(), torch.arange(batch_size).long()]).to(self.device)
        y_pred = torch.mm(u_e, pos_e.t().contiguous())
        y_pred[row_swap, col_before] = y_pred[row_swap, col_after]
        # cul regularizer
        regularize = (torch.norm(user_gcn_emb[:, :]) ** 2
                       + torch.norm(pos_gcn_emb[:, :]) ** 2) / 2  # take hop=0
        emb_loss = self.decay * regularize / batch_size
        
        if self.loss_name == "Adap_tau_Loss":
            mask_zeros = None
            tau = torch.index_select(self.memory_tau, 0, user).detach()
            tau_i = torch.index_select(self.memory_tau_i, 0, user).detach()
            tau_u = torch.index_select(self.memory_tau_u, 0, user).detach()

            loss, loss_ = self.loss_fn(y_pred, tau, w_0)    
            cl_loss = self.cl_rate * self.cal_cl_loss(u_e, u_e_cl, pos_e, item_e_cl, tau_u, tau_i)
            return loss.mean() + emb_loss + cl_loss, loss_, emb_loss, tau
            
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