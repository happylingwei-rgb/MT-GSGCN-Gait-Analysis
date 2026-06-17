import torch
import torch.nn as nn
import torch.nn.functional as F
# from net.utils.tgcn import ConvTemporalGraphical
from net.utils.mstcn import ConvTemporalGraphical  # 多尺度 TCN 模組
from net.utils.gsgcn_graph7 import Graph


# =========================
# CBAM 注意力模組（輸入 shape=(N,C,T,V)）
# Convolutional Block Attention Module (CBAM)
# =========================
class CBAM2d(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        assert reduction >= 1
        mid = max(1, channels // reduction)

        # 共享 MLP（1x1 conv 版）：C -> mid -> C
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
        )

        pad = spatial_kernel // 2
        self.spatial_conv = nn.Conv2d(
            in_channels=2, out_channels=1,
            kernel_size=(spatial_kernel, spatial_kernel),
            padding=(pad, pad), bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (N, C, T, V)
        # ----- Channel attention -----
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)              # (N,C,1,1)
        max_pool, _ = torch.max(x, dim=2, keepdim=True)                 # (N,C,1,V)
        max_pool, _ = torch.max(max_pool, dim=3, keepdim=True)          # (N,C,1,1)
        ca = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        x = x * ca

        # ----- Spatial attention -----
        avg_map = torch.mean(x, dim=1, keepdim=True)                    # (N,1,T,V)
        max_map, _ = torch.max(x, dim=1, keepdim=True)                  # (N,1,T,V)
        sa = torch.sigmoid(self.spatial_conv(torch.cat([avg_map, max_map], dim=1)))
        return x * sa


class Model(nn.Module):
    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting=True, **kwargs):
        super().__init__()

        # Load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # ST-GCN parameters
        spatial_kernel_size = A.size(0)            # K 分支圖數（如 3）
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * self.graph.num_node)  # (C*V)

        # GCN layers（最後一層輸出 256，交由 FC 分類）
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        self.st_gcn_networks = nn.ModuleList((
            # block 1：不使用 CBAM
            st_gcn(in_channels, 64,  kernel_size, stride=1, residual=True, **kwargs0,
                   use_cbam=False),
            # block 2：使用 CBAM
            st_gcn(64,         64,  kernel_size, stride=1, **kwargs,
                   use_cbam=True, cbam_reduction=16, cbam_kernel=7),
            # block 3：使用 CBAM、降採樣
            st_gcn(64,         128, kernel_size, stride=2, **kwargs,
                   use_cbam=True, cbam_reduction=16, cbam_kernel=7),
            # block 4：使用 CBAM、再降採樣；輸出固定為 256
            st_gcn(128,        256, kernel_size, stride=2, **kwargs,
                   use_cbam=True, cbam_reduction=16, cbam_kernel=7),
        ))

        # Edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size())) for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # Fully connected layer（分類頭，1×1 Conv）
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):
        # x: (N, C, T, V, M)
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()        # (N, M, V, C, T)
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous()  # (N, M, C, T, V)
        x = x.view(N * M, C, T, V)

        # ST-GCN backbone
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # GAP over (T, V)
        x = F.avg_pool2d(x, x.size()[2:])                # (N*M, 256, 1, 1)
        x = x.view(N, M, -1, 1, 1).mean(dim=1)           # (N, 256, 1, 1)

        # FC head
        x = self.fcn(x)                                   # (N, num_class, 1, 1)
        x = x.view(x.size(0), -1)                         # (N, num_class)
        return x

    def extract_feature(self, x):
        # 與 forward 相同，但回傳：
        #   feature：最後一個 block 的特徵（未池化，N,C,T,V,M 排列）
        #   output ：分類 logits 的時空圖（未池化，N,num_class,T,V,M）
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # 取最後特徵（未池化）
        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)  # (N, C, T, V, M)

        # 產生分類 logits 的時空圖（先過 FC，再還原維度）
        logits_map = self.fcn(x)                                 # (N*M, num_class, T, V)
        output = logits_map.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)  # (N, num_class, T, V, M)

        return output, feature


class st_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dropout=0, residual=True,
                 use_cbam: bool = False, cbam_reduction: int = 16, cbam_kernel: int = 7):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        # 空間支路：MSTCN + K 分支圖聚合（由 ConvTemporalGraphical 實作）
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        # 時間支路：Temporal Convolutional Network (TCN)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        # CBAM（可選）
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM2d(out_channels, reduction=cbam_reduction, spatial_kernel=cbam_kernel)

        # 殘差分支
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)
        self.output_bn = nn.BatchNorm2d(out_channels)  # 每個 block 輸出後的 BN

    def forward(self, x, A):
        res = self.residual(x)

        # 主幹：圖卷積 + 時間卷積
        x, A = self.gcn(x, A)
        x = self.tcn(x)

        # 在與殘差相加前套 CBAM（常見於 ResNet 放置方式）
        if self.use_cbam:
            x = self.cbam(x)

        x = x + res
        x = self.relu(x)
        x = self.output_bn(x)
        return x, A
