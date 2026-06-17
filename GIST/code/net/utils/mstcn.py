import torch
import torch.nn as nn

class ConvTemporalGraphical(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True,
                 dropout_prob=0.2,
                 kernel_sizes=(3, 5, 7)):  # ✅ 多 kernel size 分支
        super().__init__()

        self.kernel_size = kernel_size

        # === 多 kernel size temporal convolution 分支（channel 減半）
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            padding = (k // 2, 0)  # temporal padding 自動對齊
            branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels // 2, kernel_size=(k, 1), padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=(k, 1), padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob)
            )
            self.branches.append(branch)

        # 分支輸出通道總數
        merged_channels = (out_channels // 2) * len(kernel_sizes)

        self.merge_conv = nn.Conv2d(
            merged_channels,
            out_channels * kernel_size,
            kernel_size=(1, 1)
        )
        self.bn_after_merge = nn.BatchNorm2d(out_channels * kernel_size)

        # shortcut 直接升維
        self.input_shortcut = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(1, 1),
            bias=False
        )

        self.smooth_conv = nn.Sequential(
            nn.Conv2d(out_channels * kernel_size, out_channels * kernel_size, kernel_size=(3,1), padding=(1,0), bias=False),
            nn.BatchNorm2d(out_channels * kernel_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        identity = self.input_shortcut(x)

        # 多 kernel 分支
        outs = [branch(x) for branch in self.branches]
        x = torch.cat(outs, dim=1)

        x = self.merge_conv(x)
        x = self.bn_after_merge(x)
        x = torch.relu(x + identity)

        x = self.smooth_conv(x)

        # 標準 graph conv
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A
