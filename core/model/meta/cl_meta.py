import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from .meta_model import MetaModel
from core.utils import accuracy

import math


class MLP(nn.Module):  # MLP with one hidden layer

    def __init__(self, feat_dim, output_dim):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(feat_dim, output_dim)
        self.act_func = nn.ReLU()

    def forward(self, x):
        x = self.hidden(x)
        x = self.act_func(x)
        return x


class Attn(nn.Module):  # Multi-head attention with heads fixed to 1

    def __init__(self, embed_dim):
        super(Attn, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, 1)

    def forward(self, support_img, query_img, support_target):
        print(nn.functional.one_hot(support_target))
        output, weights = self.attn(query_img,nn.functional.one_hot(support_target), support_img)
        return output


class CL_META(MetaModel):
    # TODO : emb_func(include GAP)
    def __init__(self, feat_dim, class_num, way_num, **kwargs):
        super(CL_META, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.projection = MLP(feat_dim, 256)  # TODO: output dimention
        self.way_num = way_num
        self.attention = Attn(feat_dim)

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
        support_size, _, c, h, w = support_image.size()
        query_size, _, c, h, w = query_image.size()

        support_output = self.emb_func(support_image)
        prototype = self.crk(support_output, support_target)
        output_list = []
        for i in range(query_size):  # 对每一条query，找到最类似的某一类的prototype，然后预测
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            vec = self.emb_func(episode_query_image)
            maxp = 0
            for j in range(support_size):
                curp = self.P(vec, support_target[j], prototype)
                if curp > maxp:
                    maxp = curp
                    output = support_target[j]
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.contiguous().view(-1))
        return output, acc

    def set_forward_loss(self, batch):
        images, _ = batch
        print(images.shape)
        image1 = images[0:images.shape[0]//2]
        print(image1.shape)
        image1 = image1.to(self.device)
        (
            support_X1,
            query_X1,
            support_y1,
            query_y1,
        ) = self.split_by_episode(image1, mode=2)

        image2 = images[images.shape[0]//2:]
        image2 = image2.to(self.device)
        (
            support_X2,
            query_X2,
            support_y2,
            query_y2,
        ) = self.split_by_episode(image2, mode=2)

        episode_size,_, c, h, w = support_X1.size()

        output_list = []
        loss = []

        for i in range(episode_size):
            episode_support_X1 = support_X1[i].contiguous().reshape(-1, c, h, w)
            episode_support_X2 = support_X2[i].contiguous().reshape(-1, c, h, w)
            episode_query_X1 = query_X1[i].contiguous().reshape(-1, c, h, w)
            episode_query_X2 = query_X2[i].contiguous().reshape(-1, c, h, w)
            episode_support_y1 = support_y1[i].reshape(-1)
            episode_support_y2 = support_y2[i].reshape(-1)
            episode_query_y1 = query_y1[i].reshape(-1)
            episode_query_y2 = query_y2[i].reshape(-1)

            cur_loss, output1, output2 = self.set_forward_adaptation(episode_support_X1, episode_query_X1,
                                                                     episode_support_y1, episode_query_y1,
                                                                     episode_support_X2, episode_query_X2,
                                                                     episode_support_y2, episode_query_y2)
            loss.append(cur_loss)

            output_list.append(torch.cat((output1, output2)), dim=0)

        output = torch.cat(output_list, dim=0)

        loss = sum(loss) / len(loss)

        return output, 1, loss

    def set_forward_adaptation(self, support_X1, query_X1, support_y1, query_y1,
                               support_X2, query_X2, support_y2, query_y2):
        # 这里传入的是单个episode！！
        fast_parameters = list(item[1] for item in self.named_parameters())
        for parameter in self.parameters():
            parameter.fast = None
        self.emb_func.train()
        # self.classifier.train()

        features_support1 = self.forward_output(support_X1)
        features_support2 = self.forward_output(support_X2)
        features_query1 = self.forward_output(query_X1)
        features_query2 = self.forward_output(query_X2)

        tau5 = 0.1
        beta = 0.01

        L_meta = self.L_meta(features_support1, features_query1, support_y1, query_y1,
                             features_support2, features_query2, support_y2, query_y2)
        L_info = self.L_info(features_support1, features_query1, support_y1, query_y1,
                             features_support2, features_query2, support_y2, query_y2, tau5)

        loss = L_meta + beta * L_info
        grad = torch.autograd.grad(
            loss, fast_parameters, create_graph=True, allow_unused=True
        )
        fast_parameters = []

        for k, weight in enumerate(self.named_parameters()):
            if grad[k] is None:
                continue
            lr = 0.01
            if weight[1].fast is None:
                weight[1].fast = weight[1] - lr * grad[k]
            else:
                weight[1].fast = weight[1].fast - lr * grad[k]
            fast_parameters.append(weight[1].fast)

        return loss, features_support1, features_support2

    def crk(self, X, y):
        # 输入一个经过GAP的 **support set**

        class_values = {}
        class_counts = {}

        for i in range(len(X)):
            target = y[i].item()  # 将PyTorch张量转换为标量
            value = X[i]

            if target not in class_values:
                class_values[target] = value
                class_counts[target] = 1
            else:
                class_values[target] += value
                class_counts[target] += 1

        class_means = {}
        for target in class_values:
            class_means[target] = class_values[target] / class_counts[target]

        result_array = [class_means.get(target, 0) for target in range(max(y) + 1)]

        res = torch.tensor([item.cpu().detach().numpy() for item in result_array]).cuda()

        return res

    def P(self, X, y, c):
        # 输入单条样本，计算一个label为y的概率
        # 利用crk函数结果中的相应类别(此时已经经过attention)

        pdist = nn.PairwiseDistance(p=2)
        # 使用广播计算和各个类别的欧氏距离
        distances = pdist(X, c)

        up = math.exp(-distances[:, y])
        # 对每个类别的距离取指数并相加
        down = math.exp(-distances).sum()

        p = up / down
        return p

    def L_mn(self, X, y, c):
        # 输入一个 **query set**
        # 对于X，y中的每个样本都调用P函数，将结果取log并相加
        # 此处(X,y)与c属于相同或不同的augmentation方法！！！
        log_probs = 0.0
        for i in range(len(X)):
            # 调用P函数计算概率
            p = self.P(X[i], y[i], c)
            # 取对数并相加
            log_probs -= torch.log(p)

        return log_probs / X.shape[0]

    def L_meta(self, support_X1, query_X1, support_y1, query_y1,
               support_X2, query_X2, support_y2, query_y2):
        # 该部分所有X都经过GAP

        c1 = self.crk(support_X1, support_y1)
        c2 = self.crk(support_X2, support_y2)

        # TODO : attention how to forward
        c1 = self.attention.forward(c1, query_X1, support_y1)
        c2 = self.attention.forward(c2, query_X2, support_y2)

        Lmeta = self.L_mn(query_X1, query_y1, c1) + self.L_mn(query_X1, query_y1, c2) + \
                self.L_mn(query_X2, query_y2, c1) + self.L_mn(query_X2, query_y2, c2)

        return Lmeta / 4

    def L_z(self, X, H, A, tau):
        # 这里输入的X应该是经过projection的单条样本

        # X与A中每个向量点乘，除以tau，取指数
        exp_a = torch.exp(torch.matmul(A, X) / tau)

        pdist = nn.PairwiseDistance(p=2)
        dists_a = pdist(X, A)

        lamb_a = torch.from_numpy(2 - dists_a.numpy())  # 如果distances是PyTorch张量，需要转换成NumPy数组

        down = torch.sum(exp_a * lamb_a)

        exp_h = torch.exp(torch.matmul(H, X) / tau)

        pdist = nn.PairwiseDistance(p=2)
        dists_h = pdist(X, H)

        lamb_h = torch.from_numpy(2 - dists_h.numpy())  # 如果distances是PyTorch张量，需要转换成NumPy数组

        up = exp_h * lamb_h

        return -torch.sum(torch.log(up / down))

    def L_info(self, support_X1, query_X1, support_y1, query_y1,
               support_X2, query_X2, support_y2, query_y2, tau):

        # TODO : projection
        support_X1 = self.projection.forward(support_X1)
        support_X2 = self.projection.forward(support_X2)
        query_X1 = self.projection.forward(query_X1)
        query_X2 = self.projection.forward(query_X2)

        merged_query_X = torch.cat([query_X1, query_X2], dim=0)
        merged_query_y = torch.cat([query_y1, query_y2], dim=0)

        A = torch.cat([support_X1, support_X2,
                       self.crk(support_X1, support_y1),
                       self.crk(support_X2, support_y2)])

        merged_support_X = torch.cat([support_X1, support_X2], dim=0)
        merged_support_y = torch.cat([support_y1, support_y2], dim=0)

        # 获取类别的数量
        num_classes = torch.max(merged_support_y) + 1

        # 初始化一个空列表，用于存储每个类别的样本
        H = []

        # 遍历每个类别
        for class_idx in range(num_classes):
            # 找到同类样本的索引
            class_indices = torch.nonzero(merged_support_y == class_idx, as_tuple=True)[0]

            # 使用索引获取同类样本，并添加到结果列表
            class_samples = merged_support_X[class_indices]
            H.append(class_samples)

        # 将结果列表合并成一个张量
        H = torch.cat(H, dim=0)

        res = 0
        for i in range(len(merged_query_X)):
            res += self.L_z(merged_query_X[i], H[merged_query_y[i]], A, tau) \
                   / H[merged_query_y[i]].shape[0]

        return res
