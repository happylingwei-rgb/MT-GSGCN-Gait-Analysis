# ✅ Spine強化版骨架結構

import numpy as np

class Graph:
    def __init__(self,
                 layout='custom_7joints_spine',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)  # 會建立 self.A, self.A_root, self.A_struc, self.A_gait

    def __str__(self):
        return str(self.A)

    def get_edge(self, layout):
        if layout == 'custom_7joints_spine':
            # 7 joints: [18, 17, 16, 0, 12, 13, 14] mapped to [0~6]
            # 0: LA, 1: LK, 2: LH, 3: Spine, 4: RH, 5: RK, 6: RA
            self.num_node = 7
            self_link = [(i, i) for i in range(self.num_node)]

            # (b) Trajectory Subset（自連邊）
            self.edge_indices_trajectory = self_link

            # (c) Angle Subset（同側骨架邊）
            self.edge_indices_angle = [(0, 1), (1, 2),  # 左腿
                                       (4, 5), (5, 6)]  # 右腿

            # (d) Gait Link Subset（跨腳對稱）
            self.edge_indices_gait_link = [(0, 6), (1, 5), (2, 4)]

            # (e) Spine強化連結（Spine ↔ LH / RH）
            self.edge_indices_spine_links = [(3, 2), (3, 4)]

            # 主圖包含所有邊
            self.edge = self_link + self.edge_indices_angle + self.edge_indices_gait_link + self.edge_indices_spine_links
            self.center = 3  # Spine base
        else:
            raise ValueError(f"Layout '{layout}' is not supported.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)

        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            self.A = np.stack(A)
        else:
            raise ValueError("Unknown strategy")

        # === 建立三個子圖 ===
        self.A_root = normalize_digraph(edge_to_matrix(self.edge_indices_trajectory, self.num_node))
        self.A_struc = normalize_digraph(edge_to_matrix(self.edge_indices_angle + self.edge_indices_spine_links, self.num_node))
        self.A_gait = normalize_digraph(edge_to_matrix(self.edge_indices_gait_link, self.num_node))


# 其他輔助函數

def edge_to_matrix(edge_list, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in edge_list:
        A[i, j] = 1
        A[j, i] = 1  # 無向邊
    return A

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[i, j] = 1
        A[j, i] = 1

    hop_dis = np.full((num_node, num_node), np.inf)
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def normalize_digraph(A):
    Dl = np.sum(A, axis=0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    return np.dot(A, Dn)

def normalize_undigraph(A):
    Dl = np.sum(A, axis=0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)
    return np.dot(np.dot(Dn, A), Dn)
