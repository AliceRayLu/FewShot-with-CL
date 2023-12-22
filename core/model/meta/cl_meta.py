import nn
from einops.layers import torch

from core.model import convert_maml_module, MetaModel
from core.model.meta.boil import BOILLayer
from core.utils import accuracy


class FXXKLayer(nn.Module):

    # N-way 类别数量 K-shot 类别中样本数量
    def __init__(self, feat_dim=64, way_num=5):
        super(BOILLayer, self).__init__()
        self.layers = nn.Sequential(nn.Linear(feat_dim, way_num))

    def forward(self, x):
        return self.layers(x)



class FXXK(MetaModel):
    # TODO : loss_func  emb_func(include GAP)
    def __init__(self, feat_dim, loss_func, inner_param, **kwargs):
        super(FXXK, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.inner_param = inner_param
        self.loss_func = loss_func
        self.classifier = FXXKLayer(feat_dim, way_num=self.way_num)
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
