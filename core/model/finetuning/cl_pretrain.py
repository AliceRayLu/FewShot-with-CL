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

        # 定义 MLP
        mlp_hidden_dim = 128  # 请根据需要调整隐藏层维度
        self.mlp = MLP(feat_dim, mlp_hidden_dim, feat_dim)

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
            X1
        ) = self.split_by_episode(image1, mode=2)

        image2 = images[1]
        image2 = image2.to(self.device)
        (
            X2
        ) = self.split_by_episode(image2, mode=2)

        X1 = self.my_forward(X1)
        X2 = self.my_forward(X2)

        # 假设已经有一个特征提取函数和分类器，这里用 feat_extractor 和 classifier 表示 cnn
        feat_extractor = self.emb_func
        classifier = self.cls_classifier

        # 获取全局特征
        global_feat = feat_extractor(X1)
        global_output = classifier(global_feat)

        L_CE = self.loss_func(global_output, target)

        # 定义温度参数
        tau1 = 0.1
        tau2 = 0.1
        tau3 = 0.1
        tau4 = 0.1

        # 计算全局自监督对比损失
        l_ss_global = self.global_contrastive_loss(global_feat, tau1)

        # 示例用法
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
        l_s_global = self.supervised_contrastive_loss(global_output, target, tau4)

        # 定义权重系数
        alpha1 = 1.0
        alpha2 = 1.0
        alpha3 = 1.0

        # 计算总体损失
        total_loss = L_CE + alpha1 * l_ss_global + alpha2 * (l_ss_local_mm + l_ss_local_vm) + alpha3 * l_s_global

        # 计算准确率
        _, predicted = torch.max(global_output, 1)
        accuracy = (predicted == target).sum().item() / target.size(0)

        return global_output, accuracy, total_loss

    def global_contrastive_loss(self, x, temperature):
        x = self.projection.forward(x)

        # x: (2N, D)
        N, D = x.shape
        # 对投影向量进行标准化
        x = F.normalize(x, dim=1)
        # 计算相似性矩阵
        sim_matrix = torch.matmul(x, x.t()) / temperature
        # 构造与公式中∑中的 \mathbbm1 部分相对应的 mask
        mask = torch.eye(2 * N).bool()
        # 计算损失
        loss = -torch.sum(F.log_softmax(sim_matrix, dim=1)[mask]) / N

        return loss

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

    def supervised_contrastive_loss(self, x, target, temperature):
        x = self.projection.forward(x)

        # x: (2N, D), target: (2N,)
        N, D = x.shape
        x = F.normalize(x, dim=1)

        sim_matrix = torch.matmul(x, x.t()) / temperature

        loss = 0
        for i in range(2 * N):
            positive_samples = (target == target[i]).nonzero().squeeze()

            numerator = torch.exp(sim_matrix[i, positive_samples])
            denominator = torch.sum(torch.exp(sim_matrix[i, :]))

            loss += -torch.log(numerator / denominator)

        loss /= 2 * N  # Normalize by the number of samples

        return loss

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
