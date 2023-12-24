import torch
from torch import nn

from core.model import convert_maml_module, MetaModel
from core.utils import accuracy


class MLP(nn.Module):  # MLP with one hidden layer

    def __init__(self, feat_dim, output_dim):
        super(MLP,self).__init__()
        self.hidden = nn.Linear(feat_dim,output_dim)
        self.act_func = nn.ReLU()

    def forward(self, x):
        x = self.hidden(x)
        x = self.act_func(x)
        return x


class CL_META(MetaModel):
    # TODO : loss_func  emb_func(include GAP)
    def __init__(self, feat_dim, loss_func, inner_param, **kwargs):
        super(CL_META, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.inner_param = inner_param
        self.loss_func = loss_func
        # self.classifier = FXXKLayer(feat_dim, way_num=self.way_num)
        self.inner_param = inner_param

        convert_maml_module(self)

    def forward_output(self, x):
        feat_wo_head = self.emb_func(x)
        feat_w_head = self.classifier(feat_wo_head)
        return feat_wo_head, feat_w_head

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

    def set_forward_loss(self, batch):
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
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            self.set_forward_adaptation(episode_support_image, episode_support_target)

            features, output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        loss = self.loss_func(output, query_target.contiguous().view(-1)) / episode_size
        acc = accuracy(output, query_target.contiguous().view(-1))
        return output, acc, loss

    def set_forward_adaptation(self, support_set, support_target):
        # 分类头和特征提取器采取不同的learning rate
        extractor_lr = self.inner_param["extractor_lr"]
        classifier_lr = self.inner_param["classifier_lr"]
        fast_parameters = list(item[1] for item in self.named_parameters())
        for parameter in self.parameters():
            parameter.fast = None
        self.emb_func.train()
        self.classifier.train()
        features, output = self.forward_output(support_set)
        loss = self.loss_func(output, support_target)
        grad = torch.autograd.grad(
            loss, fast_parameters, create_graph=True, allow_unused=True
        )
        fast_parameters = []

        for k, weight in enumerate(self.named_parameters()):
            if grad[k] is None:
                continue
            # TODO : linear要设置吗 ???
            lr = classifier_lr if "Linear" in weight[0] else extractor_lr
            if weight[1].fast is None:
                weight[1].fast = weight[1] - lr * grad[k]
            else:
                weight[1].fast = weight[1].fast - lr * grad[k]
            fast_parameters.append(weight[1].fast)
    def crk(X, y):
        # 输入一个经过GAP的 **support set**

        # 创建一个字典用于存储每个类别的值和计数
        class_values = {}
        class_counts = {}

        # 遍历s_image1中的元素
        for i in range(len(X)):
            target = y[i].item()  # 将PyTorch张量转换为标量
            value = X[i]

            # 如果目标类别不在字典中，添加新的键值对
            if target not in class_values:
                class_values[target] = value
                class_counts[target] = 1
            else:
                # 如果目标类别已经在字典中，累加值和计数
                class_values[target] += value
                class_counts[target] += 1

        # 计算每个类别的均值
        class_means = {}
        for target in class_values:
            class_means[target] = class_values[target] / class_counts[target]

        # 将结果存储在长度为类别数量的数组中
        result_array = [class_means.get(target, 0) for target in range(max(y) + 1)]

        return torch.tensor(result_array)

    def P(X, y, c):
        # 输入单条样本，计算一个label为y的概率
        # 利用crk函数结果中的相应类别(此时已经经过attention)

        pdist = nn.PairwiseDistance(p=2)
        # 使用广播计算和各个类别的欧氏距离
        distances = pdist(X, c)

        up = exp(-distances[:, y])
        # 对每个类别的距离取指数并相加
        down = exp(-distances).sum()

        p = up / down
        return p

    def L_mn(X, y, c):
        # 输入一个 **query set**
        # 对于X，y中的每个样本都调用P函数，将结果取log并相加
        # 此处(X,y)与c属于相同或不同的augmentation方法！！！
        log_probs = 0.0
        for i in range(len(X)):
            # 调用P函数计算概率
            p = P(X[i], y[i], c)
            # 取对数并相加
            log_probs -= torch.log(p)

        return log_probs / X.shape[0]

    def L_meta(image1, image2):
        (
            support_X1,
            query_X1,
            support_y1,
            query_y1,
        ) = self.split_by_episode(image1, mode=2)

        (
            support_X2,
            query_X2,
            support_y2,
            query_y2,
        ) = self.split_by_episode(image2, mode=2)

        c1 = crk(support_X1, support_y1)
        c2 = crk(support_X2, support_y2)

        c1 = self.attention(c1)
        c2 = self.attention(c2)

        Lmeta = L_mn(query_X1, query_y1, c1) + L_mn(query_X1, query_y1, c2) + \
                L_mn(query_X2, query_y2, c1) + L_mn(query_X2, query_y2, c2)

        return Lmeta / 4