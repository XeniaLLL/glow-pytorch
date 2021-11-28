import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la

logabs = lambda x: torch.log(torch.abs(x))


# three main components of GLOW
# flow, inverse of flow and log-determinants
class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        '''
        BN to alleviate the problems encountered when training deep models
        however, the variance of activations noise added by BN -> inversely proportional to minibatch size per PU(处理单元)
        --> performance degrade for small mini-batch size in PU
        actnorm--> affine transformation of the activations using a scale and bias param per channel --> similar to BN
        initialize actnrom: 0 mean and unit variance given an initial minibatch of data --> data dependent init
        --> after init --> data-independent
        :param in_channel:
        :param logdet:
        '''
        super(ActNorm, self).__init__()
        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))
        self.register_buffer("initialized", torch.tensor(0, type=torch.unit8))
        self.logdet = logdet

    def initialize(self, input):
        '''
        a minibatch data is used for init
        :param input:
        :return:
        '''
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (flatten.mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3))  # todo 为什么会产生这么多的1的维度
            std = (flatten.std(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3))
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape
        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)
        logdet = height * width * torch.sum(log_abs)
        if self.logdet:
            return self.scale * (input + self.loc), logdet
        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        '''
        a flow contains the equivalent of a permutation that reverses the ordering of the channels
        replact the fixed permutation with a (learned) invertible 1x1 conv, weigth matrix is initialized as a random rotation matrix
        Note: 1x1 conv with equal number of input and output channels --> generalization of a permutation operation
        :param in_channel:
        '''
        super(InvConv2d, self).__init__()
        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)  # QR decomposition,得到的是一个正交基矩阵Q 没有利用上三角矩阵R
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape
        out = F.conv2d(input, self.weight)
        logdet = (height * width * torch.slogdet(self.weight.squeeze().double())[1].float())
        # slogdet() compute the sign and the natural log of the absolute value of the determinant of square matrix
        return out, logdet

    def reverse(self, output):
        return F.conv2d(output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super(InvConv2dLU, self).__init__()
        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))  # 置换矩阵, 上三角矩阵, 下三角矩阵
        w_s = np.diag(w_u)  # 对角线的矩阵
        w_u = np.triu(w_u, 1)  # 得到上三角矩阵, k指定了对角线的为0的位置,以第一个元素的对角线为0坐标
        u_mask = np.triu(np.ones_like(w_u), 1)  # 过滤对角线的值
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer('w_p', w_p)  # no updating--> register --> move .to(device)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)  # nn.Param 是optim.step 会更新的部分
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape
        weight = self.calc_weight()
        out = F.conv2d(input, weight)
        logdet = height * width * torch.sum(self.w_s)
        return out, logdet

    def cal_weight(self):
        weight = (self.w_p @ (self.w_l * self.l_mask + self.l_eye) @ (
                (self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s))))
        return weight.unsqueeze(2).unsqueeze(3)


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super(ZeroConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        # [1,1,1,1] param pad: m-elements tuple (padding_left,padding_right, padding_top, padding_bottom) for 2-dim input tensor
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)
        return out


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True):
        '''
        computationally efficient
        additive coupling layer: special case with s=1 and log-det of 0
        trick1: zero initialization-> (init the last conv of each NN) --> perform an identity function--> help training
        trick2: split + concatenation --> split(h) along the channel dim, concat() for the reverse operation
        trick3: permutation--> ensure each of dimension is affected in the steps of flow--> equal to reverse the ordering of the channels before  additive coupling
                           --> random permutation, 1x1 conv
        :param in_channel:
        :param filter_size:
        :param affine:
        '''
        super(AffineCoupling, self).__init__()
        self.affine = affine
        self.net = nn.Sequential(nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(filter_size, filter_size, 1),
                                 nn.ReLU(inplace=True),
                                 ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2)
                                 )
        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()
        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zeor_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)  # 实现split
        # number of chunks to return--> split the tensor into the specified number of chunks --> each is just a view
        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            s = F.sigmoid(log_s + 2)  # additive
            out_b = (in_b + t) * s
            # todo check the difference
            # s=torch.exp(log_s)
            # out_a=s*in_a+t
            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)
        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None
        return torch.cat([in_a, out_b], 1)

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)
        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            s = F.sigmoid(log_s + 2)  # todo +2?
            in_b = out_b / s - t
            # todo compare the difference between in_b and out_a
            # s=torch.exp(log_s)
            # in_a=(out_a-t)/s

        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out
        return torch.cat([out_a, out_b], 1)


class Flow(nn.Module):
    def __init__(self, in_channel, affine=True, conv_lu=True):
        super(Flow, self).__init__()
        self.actnorm = ActNorm(in_channel)
        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)
        else:
            self.invconv = InvConv2d(in_channel)

        self.coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, input):
        out, logdet = self.actnorm(input)
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out)
        logdet += det1
        if det2 is not None:
            logdet += det2
        return out, logdet

    def reverse(self, output):
        input = self.coupling.reverse(output)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)
        return input


def gaussian_log_p(x, mean, log_std):
    return -0.5 * log(2 * pi) - log_std - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_std)


def gaussian_sample(eps, mean, log_std):
    return mean + torch.exp(log_std) * eps


class GlowBlock(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, affine=True, conv_lu=True):
        '''
        composite three types of flow modules
        :param in_channel:
        :param n_flow:
        :param split:
        :param affine:
        :param conv_lu:
        '''
        super(GlowBlock, self).__init__()
        squeeze_dim = in_channel * 4
        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))
        self.split = split
        if split:
            self.prior = ZeroConv2d(in_channel * 2, in_channel * 4)
        else:
            self.prior = ZeroConv2d(in_channel * 4, in_channel * 8)

    def forward(self, input):
        b_size, n_channel, height, width = input.shape
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)
        logdet = 0
        for flow in self.flows:
            out, det = flow(out)
            logdet += det

        if self.split:
            out, z_new = out.chunck(2, 1)
            mean, log_std = self.prior(out).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_std)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out
        return out, logdet, log_p, z_new

    def reverse(self, output, eps=None, reconstruct=False):
        input = output

        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)
            else:
                input = eps

        else:
            if self.split:
                mean, log_std = self.prior(input).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_std)
                input = torch.cat([output, z], 1)

            else:
                zero = torch.zeros_like(input)
                mean, log_std = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_std)
                input = z
        for flow in self.flows[::-1]:
            input = flow.reverse(input)
        b_size, n_channel, height, width = input.shape
        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(b_size, n_channel // 4, height * 2, width * 2)
        return unsqueezed


class Glow(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, affine=True, conv_lu=True):
        '''
        add more than one Glow module for training
        :param in_channel:
        :param n_flow:
        :param n_block:
        :param affine:
        :param conv_lu:
        '''
        super(Glow, self).__init__()
        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(GlowBlock(in_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 2

        self.blocks.append(GlowBlock(n_channel, n_flow, split=False, affine=affine))

    def forward(self, input):
        log_p_sum = 0
        logdet = 0
        out = input
        z_outs = []
        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            logdet = logdet + det
            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, logdet, z_outs

    def reverse(self, z_list, reconstruct=False):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct)
            else:
                input = block.reverse(input, z_list[-(i + 1)], reconstruct=reconstruct)
        return input
