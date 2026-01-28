import torch
import torch.nn as nn
from  zz.all_nn.zz_attention import LearnableFourierPositionalEncoding,TransformerLayer
import torch.nn.functional as F

class RKHS_filter_Model(nn.Module):
    def __init__(self, dim=None, att_layer=None, init_lambda=0.1, init_beta=1.0):
        super(RKHS_filter_Model, self).__init__()
        self.dim = dim
        self.use_mlp_weight = False
        self.att_layer = att_layer
        self.num_heads = 4

        self.lambda_param = nn.Parameter(torch.tensor(float(init_lambda), dtype=torch.float32))
        self.beta_param = nn.Parameter(torch.tensor(float(init_beta), dtype=torch.float32))

        if self.use_mlp_weight:
            self.weight_mlp = nn.Sequential(
                nn.Linear(dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        self.predictor = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        head_dim = self.dim // self.num_heads
        self.posenc = LearnableFourierPositionalEncoding(2, head_dim, head_dim)
        self.transformers = nn.ModuleList(
            [TransformerLayer(self.dim, self.num_heads) for _ in range(self.att_layer)]
        )

    def gaussian_kernel(self, X, Y):
        beta = F.softplus(self.beta_param)
        dist = torch.cdist(X, Y, p=2) ** 2
        return torch.exp(-beta * dist)

    def solve_weighted_rkhs(self, K, f, w=None):
        """K: [N, N], f: [N, D], w: [N] or None → C: [N, D]"""
        N, D = f.shape
        dtype = K.dtype
        device = K.device
        lambda_reg = F.softplus(self.lambda_param).to(dtype)
        eye = torch.eye(N, device=device, dtype=dtype)

        if w is None:
            A = K + lambda_reg * eye
            C = torch.linalg.lstsq(A, f).solution
        else:
            W = torch.diag(w.to(dtype))
            A = W @ K @ W + lambda_reg * eye
            Bm = W @ f
            C = torch.linalg.lstsq(A, Bm).solution
            
        return C
    def solve_weighted_rkhs(self, K, f, w=None, eps=1e-6):
        """
        K: [N, N] (symmetric PSD)
        f: [N, D]
        w: [N] or None
        return: C [N, D]
        """
        N, D = f.shape
        dtype = K.dtype
        device = K.device

        lambda_reg = F.softplus(self.lambda_param).to(dtype)

        # 构造 A
        if w is None:
            A = K
            B = f
        else:
            w = w.to(dtype)
            W = torch.diag(w)
            A = W @ K @ W
            B = W @ f

        # 数值稳定：λI + εI
        A = A + (lambda_reg + eps) * torch.eye(N, device=device, dtype=dtype)

        # Cholesky 分解
        L = torch.linalg.cholesky(A)          # A = L L^T
        C = torch.cholesky_solve(B, L)        # 解 A C = B

        return C

    def forward(self, features_a, features_b, matches, kpts_a, kpts_b):
        B, K_max, _ = matches.shape
        device = matches.device
        dtype = features_a.dtype

        # 初始化输出 logits（全 -1 或 0，由外部 mask 忽略）
        logits = torch.zeros(B, K_max, device=device, dtype=dtype)

        # 对每个样本独立处理（B 通常较小，如 ≤ 8）
        for b in range(B):
            match_b = matches[b]  # [K_max, 2]
            valid_mask_b = match_b[:, 0] >= 0
            if not valid_mask_b.any():
                continue  # 全无效，logits[b] 保持 0

            # 提取有效索引
            idx_a_b = match_b[valid_mask_b, 0]  # [K_valid]
            idx_b_b = match_b[valid_mask_b, 1]

            # 提取有效特征和关键点（安全，无 -1）
            feat_a_b = features_a[b].index_select(0, idx_a_b)  # [K_valid, dim]
            feat_b_b = features_b[b].index_select(0, idx_b_b)
            kpts0_b = kpts_a[b].index_select(0, idx_a_b)       # [K_valid, C]
            kpts1_b = kpts_b[b].index_select(0, idx_b_b)

            f_i_b = feat_b_b - feat_a_b  # [K_valid, dim]

            # 高斯核（仅有效点）
            kpts_xy_b = kpts0_b[:, :2]  # [K_valid, 2]
            K_b = self.gaussian_kernel(kpts_xy_b.unsqueeze(0), kpts_xy_b.unsqueeze(0)).squeeze(0)  # [K_valid, K_valid]

            # 权重
            w_b = None
            if self.use_mlp_weight:
                w_b = self.weight_mlp(f_i_b).squeeze(-1)  # [K_valid]

            # RKHS 求解
            C_b = self.solve_weighted_rkhs(K_b, f_i_b, w_b)  # [K_valid, dim]

            # 全局上下文
            K_norm_b = K_b / (K_b.sum(dim=-1, keepdim=True) + 1e-6)
            f_global_b = K_norm_b @ C_b  # [K_valid, dim]

            # 注意力
            desc0_b = f_i_b.unsqueeze(0)
            desc1_b = f_global_b.unsqueeze(0)
            encoding0_b = self.posenc(kpts0_b[:, :2].unsqueeze(0)) # [K_valid, ...]
            encoding1_b = self.posenc(kpts1_b[:, :2].unsqueeze(0))

            for i in range(self.att_layer):
                if desc0_b.shape[0] == 0:
                    break
                desc0_b, desc1_b, _, _, _, _ = self.transformers[i](
                    desc0_b, desc1_b,
                    encoding0_b, encoding1_b
                )
                # desc0_b = desc0_b.squeeze(0)
                # desc1_b = desc1_b.squeeze(0)

            # 预测
            f_fusion_b = desc0_b  # [K_valid, dim]
            logits_b = self.predictor(f_fusion_b).squeeze(-1)  # [K_valid]

            # Scatter 回原位置
            valid_indices = torch.nonzero(valid_mask_b, as_tuple=True)[0]  # [K_valid]
            logits[b, valid_indices] = logits_b

        return logits  # [B, K_max], 无效位置为 0（由外部 mask 忽略）



class RKHS_filter_Model_test(nn.Module):
    def __init__(self, dim=None,att_layer=None, init_lambda=0.1, init_beta=1.0):
        super(RKHS_filter_Model_test, self).__init__()
        self.dim = dim
        self.use_mlp_weight = False
        self.att_layer=att_layer
        self.num_heads=4
        # self.out_dim=256

        # 可学习的 lambda_reg (必须 > 0)
        self.lambda_param = nn.Parameter(torch.tensor(float(init_lambda), dtype=torch.float32))
        # 可学习的 beta (必须 > 0)
        self.beta_param = nn.Parameter(torch.tensor(float(init_beta), dtype=torch.float32))

        if self.use_mlp_weight:
            self.weight_mlp = nn.Sequential(
                nn.Linear(dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        self.predictor = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        head_dim = self.dim //self.num_heads
        self.posenc = LearnableFourierPositionalEncoding(
            2 , head_dim, head_dim
        )
        self.transformers = nn.ModuleList(
            [TransformerLayer(self.dim , self.num_heads) for _ in range(self.att_layer)]
        )

    def gaussian_kernel(self, X, Y):
        """高斯核，使用可学习 beta"""
        beta = F.softplus(self.beta_param)  # 保证 beta > 0
        dist = torch.cdist(X, Y, p=2)** 2
        return torch.exp(-beta * dist)

    # def solve_weighted_rkhs(self, K, f, w=None):
    #     N = K.shape[0]
    #     dtype = K.dtype
    #     device = K.device
    #     lambda_reg = F.softplus(self.lambda_param).to(dtype)

    #     if w is None:
    #         A = K + lambda_reg * torch.eye(N, device=device, dtype=dtype)
    #         # 替换 solve → lstsq
    #         # C, _ = torch.linalg.lstsq(A, f)  # 返回 (solution, residuals)
    #         C = torch.linalg.lstsq(A, f)[0]
    #     else:
    #         W = torch.diag(w.to(dtype))
    #         A = W @ K @ W + lambda_reg * torch.eye(N, device=device, dtype=dtype)
    #         B = W @ f
    #         C, _ = torch.linalg.lstsq(A, B)
    #     return C
    def solve_weighted_rkhs(self, K, f, w=None):
        """
        K: [B, N, N]
        f: [B, N, D]
        w: [B, N] or None
        """
        B, N, _ = K.shape
        dtype = K.dtype
        device = K.device

        lambda_reg = F.softplus(self.lambda_param).to(dtype)
        eye = torch.eye(N, device=device, dtype=dtype)

        C_all = []

        for b in range(B):
            Kb = K[b]
            fb = f[b]

            if w is None:
                A = Kb + lambda_reg * eye
                Cb = torch.linalg.lstsq(A, fb).solution
            else:
                wb = w[b].to(dtype)
                W = torch.diag(wb)
                A = W @ Kb @ W + lambda_reg * eye
                Bm = W @ fb
                Cb = torch.linalg.lstsq(A, Bm).solution

            C_all.append(Cb)

        return torch.stack(C_all, dim=0)  # [B, N, D]
       
    def forward(self, features_a, features_b, matches, kpts_a, kpts_b):
        """
        注意：当前实现只处理 batch 中的第一个样本（b=0），
        若需处理完整 batch，请恢复 for b in range(B) 循环。
        """
        all_outputs = []
        
        # 只处理 batch 第一个样本（与你当前代码一致）
        match = matches[0]  # [K, 2]
        if match.numel() == 0:
            logits = torch.zeros(0, device=features_a.device)
            all_outputs.append(logits)
        else:
            # 过滤有效匹配（非 padding）
            valid_mask = match[:, 0] >= 0
            match = match[valid_mask]
            if match.numel() == 0:
                logits = torch.zeros(0, device=features_a.device)
                all_outputs.append(logits)
            else:
                idx_a = match[:, 0]
                idx_b = match[:, 1]

                feat_a = features_a[:,idx_a]  # [K, dim]
                feat_b = features_b[:,idx_b]
                kpts0= kpts_a[:,idx_a]# [K, dim]
                kpts1= kpts_b[:,idx_b]
                f_i = feat_b - feat_a          # [K, dim]

                # 只使用 xy 位置（你已改为 kpts_xy）
                kpts_xy = kpts0[:,:,:2]  # [K, 2]

                K = self.gaussian_kernel(kpts_xy, kpts_xy)  # [K, K]

                w = None
                if self.use_mlp_weight:
                    w = self.weight_mlp(f_i).squeeze(-1)  # [K]

                C = self.solve_weighted_rkhs(K, f_i, w)  # [K, dim]
                K_norm = K / (K.sum(dim=-1, keepdim=True) + 1e-6)
                f_global = K_norm @ C
                ##########################搞个注意力把
                f_origin=feat_b - feat_a 
                desc0=f_i
                desc1=f_global
                
                encoding0 = self.posenc(kpts0[:,:,:2])
                encoding1 = self.posenc(kpts1[:,:,:2])

                attn01_all = []
                attn10_all = []
                self_attn0_all = []
                self_attn1_all = []
                for i in range( self.att_layer):
                    if desc0.shape[1] == 0 or desc1.shape[1] == 0:  # no keypoints
                        break
                    desc0, desc1 ,attn01,attn10,self_attn0,self_attn1 = self.transformers[i](
                        desc0, desc1, encoding0, encoding1
                    )
                    attn01_all.append(attn01)
                    attn10_all.append(attn10)
                    self_attn0_all.append(self_attn0)
                    self_attn1_all.append(self_attn1) 
                    if i == self.att_layer- 1:
                        continue  #
                ####################################################
                f_fusion=desc0
                logits = self.predictor(f_fusion).squeeze(-1)        # [1, 6]    # [21, 2]
                target_len = matches[0].shape[0]     # 21

                # 计算需要 padding 的数量
                current_len = logits.shape[1]       # 6
                pad_len = target_len - current_len  # 15

                # 在最后一个维度（dim=-1）右侧填充 pad_len 个 -1
                logits_padded = F.pad(logits, pad=(0, pad_len), value=-1.0)  
                # logits_padded =logits 
                 # [K]
                # all_outputs.append(logits)
        return logits_padded,f_fusion,f_origin,f_global
        # return logits_padded



######################之间相减不要注意力聚合


class RKHS_filter_Model_jian(nn.Module):
    def __init__(self, dim=None,att_layer=None, init_lambda=0.1, init_beta=1.0):
        super(RKHS_filter_Model_jian, self).__init__()
        self.dim = dim
        self.use_mlp_weight = False
        self.att_layer=att_layer
        self.num_heads=4
        # self.out_dim=256

        # 可学习的 lambda_reg (必须 > 0)
        self.lambda_param = nn.Parameter(torch.tensor(float(init_lambda), dtype=torch.float32))
        # 可学习的 beta (必须 > 0)
        self.beta_param = nn.Parameter(torch.tensor(float(init_beta), dtype=torch.float32))

        if self.use_mlp_weight:
            self.weight_mlp = nn.Sequential(
                nn.Linear(dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        self.predictor = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        head_dim = self.dim //self.num_heads
        self.posenc = LearnableFourierPositionalEncoding(
            2 , head_dim, head_dim
        )
        self.transformers = nn.ModuleList(
            [TransformerLayer(self.dim , self.num_heads) for _ in range(self.att_layer)]
        )

    def gaussian_kernel(self, X, Y):
        """高斯核，使用可学习 beta"""
        beta = F.softplus(self.beta_param)  # 保证 beta > 0
        dist = torch.cdist(X, Y, p=2)** 2
        return torch.exp(-beta * dist)

    # def solve_weighted_rkhs(self, K, f, w=None):
    #     N = K.shape[0]
    #     dtype = K.dtype
    #     device = K.device
    #     lambda_reg = F.softplus(self.lambda_param).to(dtype)

    #     if w is None:
    #         A = K + lambda_reg * torch.eye(N, device=device, dtype=dtype)
    #         # 替换 solve → lstsq
    #         # C, _ = torch.linalg.lstsq(A, f)  # 返回 (solution, residuals)
    #         C = torch.linalg.lstsq(A, f)[0]
    #     else:
    #         W = torch.diag(w.to(dtype))
    #         A = W @ K @ W + lambda_reg * torch.eye(N, device=device, dtype=dtype)
    #         B = W @ f
    #         C, _ = torch.linalg.lstsq(A, B)
    #     return C
    def solve_weighted_rkhs(self, K, f, w=None):
        """
        K: [B, N, N]
        f: [B, N, D]
        w: [B, N] or None
        """
        B, N, _ = K.shape
        dtype = K.dtype
        device = K.device

        lambda_reg = F.softplus(self.lambda_param).to(dtype)
        eye = torch.eye(N, device=device, dtype=dtype)

        C_all = []

        for b in range(B):
            Kb = K[b]
            fb = f[b]

            if w is None:
                A = Kb + lambda_reg * eye
                Cb = torch.linalg.lstsq(A, fb).solution
            else:
                wb = w[b].to(dtype)
                W = torch.diag(wb)
                A = W @ Kb @ W + lambda_reg * eye
                Bm = W @ fb
                Cb = torch.linalg.lstsq(A, Bm).solution

            C_all.append(Cb)

        return torch.stack(C_all, dim=0)  # [B, N, D]
       
    def forward(self, features_a, features_b, matches, kpts_a, kpts_b):
        """
        注意：当前实现只处理 batch 中的第一个样本（b=0），
        若需处理完整 batch，请恢复 for b in range(B) 循环。
        """
        all_outputs = []
        
        # 只处理 batch 第一个样本（与你当前代码一致）
        match = matches[0]  # [K, 2]
        if match.numel() == 0:
            logits = torch.zeros(0, device=features_a.device)
            all_outputs.append(logits)
        else:
            # 过滤有效匹配（非 padding）
            valid_mask = match[:, 0] >= 0
            match = match[valid_mask]
            if match.numel() == 0:
                logits = torch.zeros(0, device=features_a.device)
                all_outputs.append(logits)
            else:
                idx_a = match[:, 0]
                idx_b = match[:, 1]

                feat_a = features_a[:,idx_a]  # [K, dim]
                feat_b = features_b[:,idx_b]
                kpts0= kpts_a[:,idx_a]# [K, dim]
                kpts1= kpts_b[:,idx_b]
                f_i = feat_b - feat_a          # [K, dim]

                # 只使用 xy 位置（你已改为 kpts_xy）
                kpts_xy = kpts0[:,:,:2]  # [K, 2]

                K = self.gaussian_kernel(kpts_xy, kpts_xy)  # [K, K]

                w = None
                if self.use_mlp_weight:
                    w = self.weight_mlp(f_i).squeeze(-1)  # [K]

                C = self.solve_weighted_rkhs(K, f_i, w)  # [K, dim]
                K_norm = K / (K.sum(dim=-1, keepdim=True) + 1e-6)
                f_global = K_norm @ C
                ##########################搞个注意力把
                f_origin=feat_b - feat_a 
                ####################################################
                f_fusion=f_origin-f_global 
                logits = self.predictor(f_fusion).squeeze(-1)        # [1, 6]    # [21, 2]
                target_len = matches[0].shape[0]     # 21

                # 计算需要 padding 的数量
                current_len = logits.shape[1]       # 6
                pad_len = target_len - current_len  # 15

                # 在最后一个维度（dim=-1）右侧填充 pad_len 个 -1
                logits_padded = F.pad(logits, pad=(0, pad_len), value=-1.0)  
                # logits_padded =logits 
                 # [K]
                # all_outputs.append(logits)
        return logits_padded,f_fusion,f_origin,f_global
        # return logits_padded

############################################


######无矫正   
class RKHS_filter_Model1(nn.Module):
    def __init__(self, dim=None,att_layer=None, init_lambda=0.1, init_beta=1.0):
        super(RKHS_filter_Model1, self).__init__()
        self.dim = dim
        self.use_mlp_weight = False
        self.att_layer=att_layer
        self.num_heads=4
        # self.out_dim=256

        # 可学习的 lambda_reg (必须 > 0)
        self.lambda_param = nn.Parameter(torch.tensor(float(init_lambda), dtype=torch.float32))
        # 可学习的 beta (必须 > 0)
        self.beta_param = nn.Parameter(torch.tensor(float(init_beta), dtype=torch.float32))

        if self.use_mlp_weight:
            self.weight_mlp = nn.Sequential(
                nn.Linear(dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        self.predictor = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        head_dim = self.dim //self.num_heads
        self.posenc = LearnableFourierPositionalEncoding(
            2 , head_dim, head_dim
        )
        self.transformers = nn.ModuleList(
            [TransformerLayer(self.dim , self.num_heads) for _ in range(self.att_layer)]
        )

    def gaussian_kernel(self, X, Y):
        """高斯核，使用可学习 beta"""
        beta = F.softplus(self.beta_param)  # 保证 beta > 0
        dist = torch.cdist(X, Y, p=2)** 2
        return torch.exp(-beta * dist)

    # def solve_weighted_rkhs(self, K, f, w=None):
    #     N = K.shape[0]
    #     dtype = K.dtype
    #     device = K.device
    #     lambda_reg = F.softplus(self.lambda_param).to(dtype)

    #     if w is None:
    #         A = K + lambda_reg * torch.eye(N, device=device, dtype=dtype)
    #         # 替换 solve → lstsq
    #         # C, _ = torch.linalg.lstsq(A, f)  # 返回 (solution, residuals)
    #         C = torch.linalg.lstsq(A, f)[0]
    #     else:
    #         W = torch.diag(w.to(dtype))
    #         A = W @ K @ W + lambda_reg * torch.eye(N, device=device, dtype=dtype)
    #         B = W @ f
    #         C, _ = torch.linalg.lstsq(A, B)
    #     return C
    def solve_weighted_rkhs(self, K, f, w=None):
        """
        K: [B, N, N]
        f: [B, N, D]
        w: [B, N] or None
        """
        B, N, _ = K.shape
        dtype = K.dtype
        device = K.device

        lambda_reg = F.softplus(self.lambda_param).to(dtype)
        eye = torch.eye(N, device=device, dtype=dtype)

        C_all = []

        for b in range(B):
            Kb = K[b]
            fb = f[b]

            if w is None:
                A = Kb + lambda_reg * eye
                Cb = torch.linalg.lstsq(A, fb).solution
            else:
                wb = w[b].to(dtype)
                W = torch.diag(wb)
                A = W @ Kb @ W + lambda_reg * eye
                Bm = W @ fb
                Cb = torch.linalg.lstsq(A, Bm).solution

            C_all.append(Cb)

        return torch.stack(C_all, dim=0)  # [B, N, D]
       
    def forward(self, features_a, features_b, matches, kpts_a, kpts_b):
        """
        注意：当前实现只处理 batch 中的第一个样本（b=0），
        若需处理完整 batch，请恢复 for b in range(B) 循环。
        """
        all_outputs = []
        
        # 只处理 batch 第一个样本（与你当前代码一致）
        match = matches[0]  # [K, 2]
        if match.numel() == 0:
            logits = torch.zeros(0, device=features_a.device)
            all_outputs.append(logits)
        else:
            # 过滤有效匹配（非 padding）
            valid_mask = match[:, 0] >= 0
            match = match[valid_mask]
            if match.numel() == 0:
                logits = torch.zeros(0, device=features_a.device)
                all_outputs.append(logits)
            else:
                idx_a = match[:, 0]
                idx_b = match[:, 1]

                feat_a = features_a[:,idx_a]  # [K, dim]
                feat_b = features_b[:,idx_b]
                kpts0= kpts_a[:,idx_a]# [K, dim]
                kpts1= kpts_b[:,idx_b]
                f_i = feat_b - feat_a          # [K, dim]

                # 只使用 xy 位置（你已改为 kpts_xy）
                kpts_xy = kpts0[:,:,:2]  # [K, 2]

                K = self.gaussian_kernel(kpts_xy, kpts_xy)  # [K, K]

                w = None
                if self.use_mlp_weight:
                    w = self.weight_mlp(f_i).squeeze(-1)  # [K]

                C = self.solve_weighted_rkhs(K, f_i, w)  # [K, dim]
                K_norm = K / (K.sum(dim=-1, keepdim=True) + 1e-6)
                f_global = K_norm @ C
                ##########################搞个注意力把
                f_origin=feat_b - feat_a 
                desc0=f_i
                desc1=f_global
                
                encoding0 = self.posenc(kpts0[:,:,:2])
                encoding1 = self.posenc(kpts1[:,:,:2])

                attn01_all = []
                attn10_all = []
                self_attn0_all = []
                self_attn1_all = []
                # for i in range( self.att_layer):
                #     if desc0.shape[1] == 0 or desc1.shape[1] == 0:  # no keypoints
                #         break
                #     desc0, desc1 ,attn01,attn10,self_attn0,self_attn1 = self.transformers[i](
                #         desc0, desc1, encoding0, encoding1
                #     )
                #     attn01_all.append(attn01)
                #     attn10_all.append(attn10)
                #     self_attn0_all.append(self_attn0)
                #     self_attn1_all.append(self_attn1) 
                #     if i == self.att_layer- 1:
                #         continue  #
                ####################################################
                f_fusion=desc0
                logits = self.predictor(f_fusion).squeeze(-1)        # [1, 6]    # [21, 2]
                target_len = matches[0].shape[0]     # 21

                # 计算需要 padding 的数量
                current_len = logits.shape[1]       # 6
                pad_len = target_len - current_len  # 15

                # 在最后一个维度（dim=-1）右侧填充 pad_len 个 -1
                logits_padded = F.pad(logits, pad=(0, pad_len), value=-1.0)  
                # logits_padded =logits 
                 # [K]
                # all_outputs.append(logits)
        return logits_padded,f_fusion,f_origin,f_global
        # return logits_padded
