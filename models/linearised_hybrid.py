import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


class LHFM(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1,
                 focusing_factor=3):
        super(LHFM, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation

        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head


        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)


        self.focusing_factor = focusing_factor
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, out_planes)))
        self.dwc = nn.Conv2d(in_channels=self.head_dim, out_channels=self.head_dim,
                             kernel_size=5, groups=self.head_dim, padding=2)


        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1,
                                  stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)

        device = self.rate1.device
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv, device=device)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

        nn.init.constant_(self.scale, 0.)
        nn.init.kaiming_normal_(self.dwc.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        b, c, h, w = q.shape

        h_out = h // self.stride if self.stride > 1 else h
        w_out = w // self.stride if self.stride > 1 else w

        f_all = self.fc(torch.cat([
            q.view(b, self.head, self.head_dim, h * w),
            k.view(b, self.head, self.head_dim, h * w),
            v.view(b, self.head, self.head_dim, h * w)
        ], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(b, -1, h, w)
        out_conv = self.dep_conv(f_conv)

        _, _, h_out, w_out = out_conv.shape


        q_att = F.adaptive_avg_pool2d(q, (h_out, w_out)) if self.stride > 1 else q
        # q_att = F.avg_pool2d(q, self.stride, self.stride) if self.stride > 1 else q
        k_att, v_att = k, v

        q_flat = q_att.flatten(2).transpose(1, 2)
        k_flat = k_att.flatten(2).transpose(1, 2)
        v_flat = v_att.flatten(2).transpose(1, 2)

        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)

        q_flat, k_flat = kernel_function(q_flat) + 1e-6, kernel_function(k_flat) + 1e-6
        q_flat, k_flat = q_flat / scale, k_flat / scale

        q_norm = q_flat.norm(dim=-1, keepdim=True)
        k_norm = k_flat.norm(dim=-1, keepdim=True)

        q_flat = q_flat ** self.focusing_factor
        k_flat = k_flat ** self.focusing_factor

        q_flat = (q_flat / q_flat.norm(dim=-1, keepdim=True)) * q_norm
        k_flat = (k_flat / k_flat.norm(dim=-1, keepdim=True)) * k_norm

        q_flat, k_flat, v_flat = (rearrange(tensor, "b n (h c) -> (b h) n c", h=self.head) for tensor in
                                  [q_flat, k_flat, v_flat])

        z = 1 / (torch.einsum("b i c, b c -> b i", q_flat, k_flat.sum(dim=1)) + 1e-6)
        kv = torch.einsum("b j c, b j d -> b c d", k_flat, v_flat)
        out_att_flat = torch.einsum("b i c, b c d, b i -> b i d", q_flat, kv, z)

        v_for_dwc = F.adaptive_avg_pool2d(v_att, (h_out, w_out)) if self.stride > 1 else v_att
        feature_map = rearrange(v_for_dwc, "b (h c) h_out w_out -> (b h) c h_out w_out", h=self.head)
        feature_map = rearrange(self.dwc(feature_map), "(b h) c h_out w_out -> (b h) (h_out w_out) c", h=self.head)

        out_att_flat = out_att_flat + feature_map
        out_att_flat = rearrange(out_att_flat, "(b h) (h_out w_out) c -> b (h c) h_out w_out", h=self.head, h_out=h_out,
                                 w_out=w_out)

        return self.rate1 * out_att_flat + self.rate2 * out_conv