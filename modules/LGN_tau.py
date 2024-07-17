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



class CustomInfoNCELoss(nn.Module):
    def __init__(self):
        super(CustomInfoNCELoss, self).__init__()

    def forward(self, user_embed, positive_embed, temperature, w_0):
        batch_size = user_embed.size(0)
        device = user_embed.device

        # Add Gaussian noise to embeddings
        noise = torch.normal(mean=0, std=0.1, size=user_embed.size()).to(device)
        noisy_user_embed = user_embed + noise

        # Construct y_pred framework
        row_swap = torch.cat([torch.arange(batch_size).long(), torch.arange(batch_size).long()]).to(device)
        col_before = torch.cat([torch.arange(batch_size).long(), torch.zeros(batch_size).long()]).to(device)
        col_after = torch.cat([torch.zeros(batch_size).long(), torch.arange(batch_size).long()]).to(device)

        # Compute similarity matrix for both user_embed and noisy_user_embed
        y_pred_clean = torch.mm(user_embed, positive_embed.t().contiguous())
        y_pred_noisy = torch.mm(noisy_user_embed, positive_embed.t().contiguous())

        # Swap rows and columns as needed
        y_pred_clean[row_swap, col_before] = y_pred_clean[row_swap, col_after]
        y_pred_noisy[row_swap, col_before] = y_pred_noisy[row_swap, col_after]

        # Positive and Negative Logits for clean embeddings
        pos_logits_clean = torch.exp(y_pred_clean[:, 0] / w_0)  # B
        neg_logits_clean = torch.exp(y_pred_clean[:, 1:] / temperature.unsqueeze(1))  # B M

        # Positive and Negative Logits for noisy embeddings
        pos_logits_noisy = torch.exp(y_pred_noisy[:, 0] / w_0)  # B
        neg_logits_noisy = torch.exp(y_pred_noisy[:, 1:] / temperature.unsqueeze(1))  # B M

        # Compute L_q and L_g for both clean and noisy embeddings
        L_q_clean = -torch.log(pos_logits_clean / torch.exp(y_pred_clean).sum(dim=1))
        L_g_noisy = -torch.log(pos_logits_noisy / torch.exp(y_pred_noisy).sum(dim=1))

        # Final loss
        loss = (L_q_clean.mean() + L_g_noisy.mean()) / 2

        # Compute the loss without temperature scaling for logging
        pos_logits_clean_ = torch.exp(y_pred_clean[:, 0])  # B
        neg_logits_clean_ = torch.exp(y_pred_clean[:, 1:])  # B M
        Ng_clean_ = neg_logits_clean_.sum(dim=-1)
        loss_clean_ = (-torch.log(pos_logits_clean_ / Ng_clean_)).mean().detach()

        pos_logits_noisy_ = torch.exp(y_pred_noisy[:, 0])  # B
        neg_logits_noisy_ = torch.exp(y_pred_noisy[:, 1:])  # B M
        Ng_noisy_ = neg_logits_noisy_.sum(dim=-1)
        loss_noisy_ = (-torch.log(pos_logits_noisy_ / Ng_noisy_)).mean().detach()

        return loss, (loss_clean_ + loss_noisy_) / 2
    
class NegNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(NegNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, user_embed, positive_embed):
        # Normalize embeddings
        user_embed = F.normalize(user_embed, dim=1)
        positive_embed = F.normalize(positive_embed, dim=1)
        
        # Compute cosine similarity
        similarity_matrix = torch.matmul(user_embed, positive_embed.T) / self.temperature
        
        # Diagonal elements are positive pairs
        pos_pairs = torch.diag(similarity_matrix)
        
        # Compute L_q
        l_q = -torch.log(torch.exp(pos_pairs) / torch.exp(similarity_matrix).sum(dim=1))
        
        # Compute L_g
        l_g = -torch.log(torch.exp(pos_pairs) / torch.exp(similarity_matrix).sum(dim=0))
        
        # Final loss
        loss = (l_q.mean() + l_g.mean()) / 2
        return loss

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
        self.negnce_loss = NegNCELoss()
        self.InfoNCE = CustomInfoNCELoss()
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
        self.gcn = self._init_model()
        self.sampling_method = args_config.sampling_method
        

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

    def _update_tau_memory(self, x, x_i):
        # x: std [B]
        # y: update position [B]
        with torch.no_grad():
            x = x.detach()
            x_i = x_i.detach()
            self.memory_tau = x
            self.memory_tai_i = x_i

    def get_decay_factor(self, epoch):
        return 1 / (1 + epoch * 0.01)  # Example decay function; adjust as needed
        
            
    def _loss_to_tau(self, x, x_all, current_epoch, total_epochs):
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
                base_laberw = torch.mean(x)
                # Assuming user_embed and item_embed are your user and item embeddings
#                 user_norms = torch.norm(self.user_embed, p=2, dim=1)
#                 item_norms = torch.norm(self.item_embed, p=2, dim=1)

#                 # Combine user and item norms into a single tensor
#                 embedding_norms = torch.cat([user_norms, item_norms])
#                 # Calculating rate of change in variance of embeddings
#                 current_variance_norms = torch.var(embedding_norms)
#                 if hasattr(self, 'previous_variance_norms'):
#                     variance_change_rate = (current_variance_norms - self.previous_variance_norms) / self.previous_variance_norms
#                 else:
#                     variance_change_rate = 0  # Default to no change on the first run
#                 self.previous_variance_norms = current_variance_norms

#                 # Dynamically adjust kappa based on the variance change rate
#                 dynamic_kappa = self.base_kappa * (1 + variance_change_rate)  # base_kappa is a predefined base sensitivity

#                 # Mod factor calculation with dynamically adjusted kappa
#                 mod_factor = dynamic_kappa * current_variance_norms

#                 # Continue with tau calculation as before
#                 laberw_input = torch.clamp((x - base_laberw + mod_factor) / self.temperature_2,
#                                            min=-np.e ** (-1), max=1000)
#                 laberw_data = self.lambertw_table[((laberw_input + 1) * 1e4).long()]
#                 tau = (t_0 * torch.exp(-laberw_data)).detach()

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

                base_laberw = torch.mean(x)
                
                laberw_input = torch.clamp((x  - base_laberw + mad_factor + var) / self.temperature_2,
                                           min=-np.e ** (-1), max=1000)
                laberw_data = self.lambertw_table[((laberw_input + 1) * 1e4).long()]
                tau = (t_0 * torch.exp(-laberw_data)).detach()
                

#                 user_norms = torch.norm(self.user_embed, p=2, dim=1)
#                 item_norms = torch.norm(self.item_embed, p=2, dim=1)

#                 # Combine user and item norms into a single tensor
#                 embedding_norms = torch.cat([user_norms, item_norms])
#                 base_laberw = torch.mean(x)
#                 variance_norms = torch.var(embedding_norms)  # Assuming embedding_norms are precomputed

#                 kappa = 0.1  # Sensitivity to variance changes
#                 mod_factor = kappa * variance_norms

#                 laberw_input = torch.clamp((x - base_laberw + mod_factor) / self.temperature_2,
#                                            min=-np.e ** (-1), max=1000)
#                 laberw_data = self.lambertw_table[((laberw_input + 1) * 1e4).long()]
#                 tau = (t_0 * torch.exp(-laberw_data)).detach()
                
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
            tau_user = self._loss_to_tau(loss_per_user, w_0, epoch, self.total_epoch)
            tau_item = self._loss_to_tau(loss_per_ins, w_0, epoch, self.total_epoch)
            self._update_tau_memory(tau_user, tau_item)
            
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
            loss, loss_ = self.loss_fn(y_pred, tau, w_0)
            cl_loss = 0.2 * self.cal_cl_loss(u_e, u_e_cl, pos_e, item_e_cl, tau, tau_i)
            
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
