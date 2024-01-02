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

class SpatialProjectionHead(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SpatialProjectionHead, self).__init__()
        self.fq = torch.nn.Linear(input_dim, output_dim)
        self.fk = torch.nn.Linear(input_dim, output_dim)
        self.fv = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C = x.size()
        x = x.view(B, C, -1).permute(0, 2, 1)  # Reshape and permute dimensions
        q = self.fq(x)
        k = self.fk(x)
        v = self.fv(x)
        return q, k, v

class GlobalSSContrastiveLoss(torch.nn.Module):
    #全局自监督对比损失
    def __init__(self, temperature,device):
        super(GlobalSSContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, z_i, z_i_prime):
        # 计算相似度分数
        scores = torch.einsum("bi,bi->b", z_i, z_i_prime) / self.temperature

        # 计算正样本对的logits
        exp_scores_pos = torch.exp(scores)

        # 计算负样本对的logits
        exp_scores_neg = torch.sum(torch.exp(scores), dim=-1, keepdim=True) - exp_scores_pos

        # 添加indicator function
        # 使用掩码矩阵排除对角线元素
        mask = torch.eye(len(scores), dtype=torch.bool)
        mask = mask.to(self.device)
        exp_scores_neg = exp_scores_neg.masked_fill(mask, 0)

        # 计算对比损失
        loss = -torch.log(exp_scores_pos / (exp_scores_pos + exp_scores_neg))

        return loss.mean()

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
        self.loss_func = nn.CrossEntropyLoss()

        self.projection = MLP(feat_dim, num_class)
    
    def forward_output(self, x):
        feat_wo_head = self.emb_func(x)
        return feat_wo_head

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
            output = self.forward_output(episode_query_image)
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.contiguous().view(-1))
        return output, acc


    # def my_forward(self, X):
    #     episode_size, _, c, h, w = X.size()
    #     output_list = []
    #     for i in range(episode_size):
    #         episode_image = X[i].contiguous().reshape(-1, c, h, w)
    #         output = self.forward_output(episode_image)
    #         output_list.append(output)

    #     output = torch.cat(output_list, dim=0)
    #     return output

    def set_forward_loss(self, batch):
        images, target = batch
        target = target.to(self.device)
        images = images.to(self.device)

        # images, _ = batch
        # image1 = images[0:128]
        # image1 = image1.to(self.device)
        # (
        #     X1, _, _, _
        # ) = self.split_by_episode(image1, mode=2)

        # image2 = images[128:]
        # image2 = image2.to(self.device)
        # (
        #     X2, _, _, _
        # ) = self.split_by_episode(image2, mode=2)

        # X1 = self.my_forward(X1)
        # X2 = self.my_forward(X2)

        # images = images.to(self.device)
        # (
        #     X,_,support_target,query_target
        # ) = self.split_by_episode(images, mode=3)
        # print(X.shape)
        feat_extractor = self.emb_func
        # 获取全局特征
        global_feat = feat_extractor(images)
        print(global_feat.shape)

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

        input_dim = 256  # 输入特征的维度
        output_dim = 128  # 投影头输出的维度

        #新的计算全局自监督对比损失

        # 使用 MLP 投影头得到 z_i 和 z_i_prime
        z_i = self.projection.forward(global_feat)
        z_i_prime = self.projection.forward(global_feat)

        ss_contrastive_loss = GlobalSSContrastiveLoss(temperature=tau1,device=self.device)
        l_ss_global = ss_contrastive_loss(z_i, z_i_prime)

 # 旧的      l_ss_global = self.contrastive_loss(global_feat, tau1)

        spatial_projection_head = SpatialProjectionHead(input_dim, output_dim)
        xa, xb = spatial_projection_head(global_feat)
        l_ss_local_mm = self.map_map_loss(xa, xb, tau2)

        vector_map_module = VectorMapModule(input_dim, output_dim)
        ui, za = vector_map_module(global_feat)
        l_ss_local_vm = self.vec_map_loss(ui, za, tau3)

        # 计算全局监督对比损失
        l_s_global = self.supervised_contrastive_loss(global_feat, target, tau4)
        # 计算总体损失
        total_loss = L_CE + alpha1 * l_ss_global + alpha2 * (l_ss_local_mm + l_ss_local_vm) + alpha3 * l_s_global

        # 计算准确率
        _, predicted = torch.max(global_feat, 1)
        accuracy = (predicted == target).sum().item() / target.size(0)

        # 返回分类输出、准确率以及前向损失
        return global_feat, accuracy, total_loss


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
        features = self.projection.forward(features)

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

    def map_map_loss(self, local_features_q, local_features_k, temperature):
        B, N, D = local_features_q.shape
        local_features_q = F.normalize(local_features_q, dim=2)  # 归一化
        local_features_k = F.normalize(local_features_k, dim=2)
        sim_matrix = torch.bmm(local_features_q, local_features_k.transpose(1, 2)) / temperature
        sim_matrix = torch.exp(sim_matrix)  # 指数化
        mask = ~torch.eye(N, dtype=bool, device=local_features_q.device)  # 排除自身比较
        denom = sim_matrix.masked_fill(~mask, 0).sum(dim=2, keepdim=True)
        pos_sim = torch.exp(torch.sum(local_features_q * local_features_k, dim=2) / temperature)
        loss = -torch.log(pos_sim / denom.squeeze()).mean()
        return loss


    def set_forward_adaptation(self, support_feat, support_target, query_feat):
        # 创建一个分类器，可以是模型的一部分，也可以是单独的模型
        classifier = self.classifier()  # 需要实现 create_classifier 方法
        # 将分类器移到设备上
        classifier = classifier.to(self.device)
        # 设置为训练模式
        classifier.train()
        # 获取支持集的大小
        support_size = support_feat.size(0)
        # 设置内部训练的优化器
        optimizer = self.sub_optimizer(classifier, self.inner_param["inner_optim"])
        # 迭代进行多次梯度更新
        for epoch in range(self.inner_param["inner_train_iter"]):
            # 随机排列索引
            rand_id = torch.randperm(support_size)
            # 按批次更新梯度
            for i in range(0, support_size, self.inner_param["inner_batch_size"]):
                optimizer.zero_grad()
                select_id = rand_id[i: min(i + self.inner_param["inner_batch_size"], support_size)]
                batch = support_feat[select_id]
                # 计算损失
                _, _, loss = self.set_forward_loss(batch)
                # 反向传播和梯度更新
                loss.backward()
                optimizer.step()

        # 在查询集上进行前向传播
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

