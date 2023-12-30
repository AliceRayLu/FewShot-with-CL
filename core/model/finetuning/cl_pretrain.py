# -*- coding: utf-8 -*-

import pdb
from re import T
import torch
from torch.nn import functional as F, CrossEntropyLoss
from torch import nn

from core.utils import accuracy
from .finetuning_model import FinetuningModel
from torch.nn.utils.weight_norm import WeightNorm
import numpy as np
from torch.autograd import Variable


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        self.class_wise_learnable_norm = True  # See the issue#4&8 in the github
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0)  # split the weight update component to direction and norm

        if outdim <= 200:
            self.scale_factor = 2  # a fixed scale factor to scale the output of cos value into a reasonably large input for softmax
        else:
            self.scale_factor = 10 # in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(
            x_normalized)  # matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor * (cos_dist)

        return scores


class SpatialProjectionHead(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SpatialProjectionHead, self).__init__()
        self.fq = torch.nn.Linear(input_dim, output_dim)
        self.fk = torch.nn.Linear(input_dim, output_dim)
        self.fv = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.size()
        x = x.view(B, C, -1).permute(0, 2, 1)  # Reshape and permute dimensions
        q = self.fq(x)
        k = self.fk(x)
        v = self.fv(x)
        return q, k, v


class VectorMapModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(VectorMapModule, self).__init__()
        self.fc_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        ui = F.relu(self.fc_layer(x))
        za = self.fc_layer(x)
        return ui, za


class MLP(nn.Module):  # MLP with one hidden layer

    def __init__(self, feat_dim, output_dim):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(feat_dim, output_dim)
        self.act_func = nn.ReLU()

    def forward(self, x):
        x = self.hidden(x)
        x = self.act_func(x)
        return x

class CL_PRETRAIN(FinetuningModel):
    def __init__(self, feat_dim, num_class, inner_param, **kwargs):
        super(CL_PRETRAIN, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.inner_param = inner_param

        self.classifier = nn.Linear(feat_dim, num_class)
        self.classifier_rot = nn.Linear(feat_dim, 4)
        self.disclass = distLinear(feat_dim, num_class)
        self.loss_func = nn.CrossEntropyLoss()

        self.mlp = MLP(feat_dim, num_class)

    def set_forward(self, batch):
        image, global_target = batch
        image = image.to(self.device)
        (
            support_image,
            query_image,
            support_target,
            query_target,
        ) = self.split_by_episode(image, mode=2)
        episode_size, _, c, h, w = support_image.size()

        output_list = []
        for i in range(episode_size):
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            _, output = self.forward_output(episode_query_image)
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.contiguous().view(-1))
        return output, acc


    def my_forward(self, X):
        episode_size, _, c, h, w = X.size()
        output_list = []
        for i in range(episode_size):
            episode_image = X[i].contiguous().reshape(-1, c, h, w)
            _, output = self.forward_output(episode_image)
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        return output

    def set_forward_loss(self, batch):
        _, target = batch
        target = target.to(self.device)

        images, _ = batch
        image1 = images[0]
        image1 = image1.to(self.device)
        (
            X1, _, _, _
        ) = self.split_by_episode(image1, mode=2)

        image2 = images[1]
        image2 = image2.to(self.device)
        (
            X2, _, _, _
        ) = self.split_by_episode(image2, mode=2)

        X1 = self.my_forward(X1)
        X2 = self.my_forward(X2)

        X = torch.cat([X1, X2], dim=0)

        feat_extractor = self.emb_func
        # 获取全局特征
        global_feat = feat_extractor(X)

        # 定义温度参数
        tau1 = 0.1
        tau2 = 0.1
        tau3 = 0.1
        tau4 = 0.1

        # 定义权重系数
        alpha1 = 1.0
        alpha2 = 1.0
        alpha3 = 1.0

        classifier = self.classifier
        output = classifier(global_feat)
        L_CE = self.loss_func(output, target)

        l_ss_global = self.contrastive_loss(global_feat, tau1)

        input_dim = 256  # 输入特征的维度
        output_dim = 128  # 投影头输出的维度

        spatial_projection_head = SpatialProjectionHead(input_dim, output_dim)
        xa, xb = spatial_projection_head(torch.rand(32, input_dim, 8, 8))  # 32 个样本
        l_ss_local_mm = self.map_map_loss(xa, xb, tau2)

        vector_map_module = VectorMapModule(input_dim, output_dim)
        x_sample = torch.rand(32, input_dim)  # 32个样本
        ui, za = vector_map_module(x_sample)
        l_ss_local_vm = self.vec_map_loss(ui, za, tau3)

        # 计算全局监督对比损失
        l_s_global = self.supervised_contrastive_loss(global_feat, target, tau4)
        # 计算总体损失
        total_loss = L_CE + alpha1 * l_ss_global + alpha2 * (l_ss_local_mm + l_ss_local_vm) + alpha3 * l_s_global

        optimizer = self.sub_optimizer(classifier, self.inner_param["inner_optim"])
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 计算准确率
        _, predicted = torch.max(global_feat, 1)
        accuracy = (predicted == target).sum().item() / target.size(0)

        # 返回分类输出、准确率以及前向损失
        return global_feat, accuracy, total_loss

    def contrastive_loss(self, features, temperature):
        """
        计算全局自监督对比损失
        Args:
        - features: 特征张量，形状为 [2N, D]
        - temperature: 温度参数，一个标量
        Returns:
        - loss: 计算得到的损失值
        """
        features = self.projection.forward(features)

        N = features.shape[0] // 2  # 因为每个正样本对中有两个样本

        # 计算特征的归一化版本
        features = F.normalize(features, dim=1)

        # 计算相似度矩阵
        sim_matrix = torch.mm(features, features.T) / temperature

        # 为了计算损失，我们需要对每个正样本对提取相应的相似度
        sim_ij = torch.diag(sim_matrix, N) + torch.diag(sim_matrix, -N)
        sim_ij = torch.cat((sim_ij, sim_ij), dim=0)

        # 创建掩码矩阵，用于选择非对角线元素
        mask = torch.ones_like(sim_matrix)
        mask = mask.fill_diagonal_(0).bool()

        # 计算分母（即所有负样本对的相似度的和）
        neg_sim = sim_matrix.masked_select(mask).view(2 * N, -1)

        # 计算对数损失
        loss = -torch.log(torch.exp(sim_ij) / torch.exp(neg_sim).sum(dim=1))
        return loss.mean()

    def map_map_loss(self, xa, xb, temperature):
        # xa, xb: (B, HW, D)
        sim_matrix = torch.matmul(xa, xb.permute(0, 2, 1)) / temperature
        mask = torch.eye(xa.size(1)).bool()
        loss = -torch.sum(F.log_softmax(sim_matrix, dim=1)[mask]) / xa.size(0)
        return loss

    def vec_map_loss(self, ui, za, tau):
        # ui: (B, D, HW), za: (B, D)
        sim_matrix = torch.matmul(ui.permute(0, 2, 1), za.unsqueeze(-1)).squeeze() / tau
        mask = torch.eye(ui.size(2)).bool()
        loss = -torch.sum(F.log_softmax(sim_matrix, dim=1)[mask]) / ui.size(0)
        return loss

    def supervised_contrastive_loss(self, features, labels, temperature):
        """
        计算全局监督对比损失
        Args:
        - features: 特征张量，形状为 [2N, D]
        - labels: 标签张量，形状为 [2N]
        - temperature: 温度参数，一个标量
        Returns:
        - loss: 计算得到的损失值
        """
        device = features.device
        N = features.shape[0] // 2  # 因为每个正样本对中有两个样本

        # 计算特征的归一化版本
        features = F.normalize(features, dim=1)

        # 计算相似度矩阵
        sim_matrix = torch.mm(features, features.T) / temperature

        # 初始化损失值
        loss = 0.0

        # 计算损失
        for i in range(2 * N):
            # 正样本对
            pos_mask = (labels == labels[i]) & ~torch.eye(2 * N, dtype=bool, device=device)
            pos_sim = sim_matrix[i][pos_mask]

            # 所有样本（包括正负样本对）
            all_sim = sim_matrix[i]

            # 计算损失
            loss += -torch.log(torch.sum(torch.exp(pos_sim)) / torch.sum(torch.exp(all_sim)))

        return loss / (2 * N)

    def set_forward_adaptation(self, support_feat, support_target, query_feat):
        classifier = distLinear(self.feat_dim, self.test_way)
        optimizer = self.sub_optimizer(classifier, self.inner_param["inner_optim"])

        classifier = classifier.to(self.device)

        classifier.train()
        support_size = support_feat.size(0)

        for epoch in range(self.inner_param["inner_train_iter"]):
            rand_id = torch.randperm(support_size)
            for i in range(0, support_size, self.inner_param["inner_batch_size"]):
                optimizer.zero_grad()
                select_id = rand_id[i: min(i + self.inner_param["inner_batch_size"], support_size)]
                batch = support_feat[select_id]
                target = support_target[select_id]
                # print(batch.size())
                output = classifier(batch)

                loss = self.loss_func(output, target)

                loss.backward()
                optimizer.step()

        output = classifier(query_feat)
        return output

    def rot_image_generation(self, image, target):
        bs = image.shape[0]
        indices = np.arange(bs)
        np.random.shuffle(indices)
        split_size = bs // 4
        image_rot = []
        target_class = []
        target_rot_class = []

        for j in indices[0:split_size]:
            x90 = image[j].transpose(2, 1).flip(1)
            x180 = x90.transpose(2, 1).flip(1)
            x270 = x180.transpose(2, 1).flip(1)
            image_rot += [image[j], x90, x180, x270]
            target_class += [target[j] for _ in range(4)]
            target_rot_class += [torch.tensor(0), torch.tensor(1), torch.tensor(2), torch.tensor(3)]
        image_rot = torch.stack(image_rot, 0).to(self.device)
        target_class = torch.stack(target_class, 0).to(self.device)
        target_rot_class = torch.stack(target_rot_class, 0).to(self.device)
        image_rot = torch.tensor(image_rot).to(self.device)
        target_class = torch.tensor(target_class).to(self.device)
        target_rot_class = torch.tensor(target_rot_class).to(self.device)
        return image_rot, target_class, target_rot_class
