from json import decoder
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys 
import os
from timm.models.layers import DropPath, trunc_normal_

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from modules.point_4d_convolution import *
from modules.transformer import *



####
# P4Encoder+TransformerDecoder
# Decoder部分与PointMAE相同，Encoder部分使用P4的结构


class P4TransformerEncoder(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 mlp_dim, mask_ratio):                                                 # output
        super().__init__()

        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.emb_relu = nn.ReLU() if emb_relu else False

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, mlp_dim),
        #     nn.GELU(),
        #     nn.Linear(mlp_dim, num_classes),
        # )
        
        self.mask_ratio = mask_ratio


    # def _mask_choice_16(self, embedding):
    #     B, T, C, N = embedding.shape
    #     mask_idx = []
    #     list1 = list(range(10,40))
    #     list2 = list(range(50,80))
    #     list3 = list(range(90,120))
    #     list4 = list(range(130,160))
    #     list_ = list1 + list2 + list3 + list4
    #     mask = torch.zeros(160)
    #     mask[list_] = 1
    #     mask_idx.append(mask.bool())

    #     bool_masked_pos = torch.stack(mask_idx)
    #     bool_masked_pos = bool_masked_pos.expand(B,160).to(center.device)
    #     return bool_masked_pos

    # def _mask_center_block(self, center, noaug=False):
    #     '''
    #         center : B G 3
    #         --------------
    #         mask : B G (bool)
    #     '''
    #     # skip the mask
    #     if noaug or self.mask_ratio == 0:
    #         return torch.zeros(center.shape[:2]).bool()
    #     # mask a continuous part
    #     mask_idx = []
    #     for points in center:
    #         # G 3
    #         points = points.unsqueeze(0)  # 1 G 3
    #         index = random.randint(0, points.size(1) - 1)
    #         distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
    #                                      dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

    #         idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
    #         ratio = self.mask_ratio
    #         mask_num = int(ratio * len(idx))
    #         mask = torch.zeros(len(idx))
    #         mask[idx[:mask_num]] = 1
    #         mask_idx.append(mask.bool())

    #     bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

    #     return bool_masked_pos

    # def _mask_rand(self, features, noaug = False):
    #     '''
    #         features : P4Conv的输出交换维度后的结果，B T* (N//spatial stride) C
    #         --------------
    #         mask : B T* (N//spatial stride) (bool)
    #     '''
    #     B, TN, C = features.shape
    #     # # skip the mask
    #     # if noaug or self.mask_ratio == 0:
    #     #     return torch.zeros(center.shape[:2]).bool()

    #     self.num_mask = int(self.mask_ratio * TN)

    #     overall_mask = np.zeros([B, TN])
    #     for i in range(B):
    #         mask = np.hstack([
    #             np.zeros(TN-self.num_mask),
    #             np.ones(self.num_mask),
    #         ])
    #         np.random.shuffle(mask)
    #         overall_mask[i, :] = mask
    #     overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

    #     return overall_mask.to(features.device) # B T*N
    
    
    def _mask_rand(self, features, noaug = False):
        B, TN, C = features.shape
        # # skip the mask
        # if noaug or self.mask_ratio == 0:
        #     return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * TN)
        len_keep = int(TN*(1-self.mask_ratio))
        
        noise = torch.rand(B,TN,device=features.device)
        
        ids_shuffle = torch.argsort(noise,dim=1)
        ids_restore = torch.argsort(ids_shuffle,dim=1)
        print(ids_restore.shape)

        overall_mask = torch.ones([B,TN],device=features.device)
        overall_mask[:,len_keep] = 0
        print(len_keep)
        mask = torch.gather(overall_mask,dim=1,index=ids_restore).to(torch.bool)
        num=0
        for i in range(B):
            for j in range(TN):
                if mask[i][j]==0:
                    num += 1
        print("num:",num)
        return mask,ids_restore
    

    def forward(self, input):                                                                                                               # [B, L, N, 3]
        device = input.get_device()
        xyzs, features = self.tube_embedding(input)                                                                                         # [B, L, n, 3], [B, L, C, n] 
        xyz_points = input.clone()
        xyzts = []
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]
        for t, xyz in enumerate(xyzs):
            t = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t+1)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzts.append(xyzt)
        xyzts = torch.stack(tensors=xyzts, dim=1)
        xyzts = torch.reshape(input=xyzts, shape=(xyzts.shape[0], xyzts.shape[1]*xyzts.shape[2], xyzts.shape[3]))                           # [B, L*n, 4]

        xyzts_copy = xyzts.clone()
        
        features = features.permute(0, 1, 3, 2)                                                                                             # [B, L,   n, C]
        print("features:",features.shape)
        features = torch.reshape(input=features, shape=(features.shape[0], features.shape[1]*features.shape[2], features.shape[3]))         # [B, L*n, C]
        B, TN, C = features.shape
        
        mask,ids_restore = self._mask_rand(features)
        
        feature_visible = features[~mask].reshape(B, -1, C)
        xyzts = xyzts[~mask].reshape(B, -1, 4)
        
        xyzts_visible = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

        embedding = xyzts_visible + feature_visible

        if self.emb_relu:
            embedding = self.emb_relu(embedding)

        output = self.transformer(embedding)
        # output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        # output = self.mlp_head(output)

        return output, xyzts_copy, mask, ids_restore, xyz_points


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DecoderAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = DecoderAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # def forward(self, x, pos, return_token_num):
    #     for _, block in enumerate(self.blocks):
    #         x = block(x + pos)

    #     x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
    #     return x
    
    def forward(self, x, pos, mask):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(self.norm(x[mask].reshape(x.shape[0],-1,x.shape[2])))  # only return the mask tokens predict pixel
        return x
    

class CDLoss(nn.Module):
    """
    CD Loss.
    """

    def __init__(self):
        super(CDLoss, self).__init__()
    
    def forward(self, prediction, ground_truth):
        batch_size, num, _ = prediction.size()
        print(prediction.shape)
        cd_loss = torch.tensor(0, dtype=torch.float32, device=prediction.device)
        prediction = prediction.view(batch_size, num, 1, 3)
        ground_truth = ground_truth.view(batch_size, 1, num, 3)
        T = torch.sum((prediction - ground_truth) ** 2, dim = -1).squeeze()
        print(T.shape)
        cd_loss += torch.sum(T.min(dim = 1)[0])
        cd_loss += torch.sum(T.min(dim = 2)[0])
        return cd_loss / batch_size
    
    
class P4mask(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, encoder_depth, heads, dim_head,                                   # transformer
                 mlp_dim, mask_ratio,                                      #output
                 transdim, decoder_depth=4, drop_path_rate=0.1):                        #decoder_parameters
        super().__init__()
        self.Encoder = P4TransformerEncoder(radius, nsamples, spatial_stride,             # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, encoder_depth, heads, dim_head,                                           # transformer
                 mlp_dim, mask_ratio)
                
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, decoder_depth)]
        
        self.transdim = transdim
        
        self.Decoder = TransformerDecoder(self.transdim, decoder_depth, drop_path_rate = dpr)
        
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(4, 128),
            nn.GELU(),
            nn.Linear(128, self.transdim)
        )
        
        # self.increase_dim = nn.Sequential(
        #     # nn.Conv1d(self.trans_dim, 1024, 1),
        #     # nn.BatchNorm1d(1024),
        #     # nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(self.transdim, 3*self.group_size, 1)  #############output dimension????
        # )
        
        self.loss_fn = CDLoss()
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.transdim))
        
        trunc_normal_(self.mask_token, std=.02)
        
    def forward(self, input):
        print("input:",input.shape)
        visible_output, xyzts_all, mask, ids_restore, all_points = self.Encoder(input)
        B, TN, C = visible_output.shape
        print("visible_output_encoder:",visible_output.shape)
        adjust_dimension = nn.Linear(C, self.transdim) #####调整Encoder的输出维度，与trandim相同        
        adjust_dimension = adjust_dimension.to(visible_output.device)
        visible_output = adjust_dimension(visible_output)
        print("visible_output_adjust_dimension:",visible_output.shape)
        
        pos_emb_vis = self.decoder_pos_embed(xyzts_all[~mask]).reshape(B, -1, self.transdim) #####permute????
        pos_emb_mask = self.decoder_pos_embed(xyzts_all[mask]).reshape(B, -1, self.transdim) #####permute????
        print("Decoder:")
        print("pos_emb_vis:",pos_emb_vis.shape)    
        print("pos_emb_mask:",pos_emb_mask.shape)   
        _, N_mask, _ = pos_emb_mask.shape
        print("N_mask:",N_mask)
        mask_token = self.mask_token.expand(B, N_mask ,-1)
        decoder_input = torch.cat([visible_output, mask_token],dim=1)
        decoder_pos_emb = torch.cat([pos_emb_vis, pos_emb_mask], dim=1)
        
        decoder_input = torch.gather(decoder_input,dim=1,index=ids_restore.unsqueeze(-1).repeat(1,1,visible_output.shape[2]))
        decoder_pos_emb = torch.gather(decoder_pos_emb,dim=1,index=ids_restore.unsqueeze(-1).repeat(1,1,visible_output.shape[2]))
        
        print("Dimension for decoder_input&decoder_pos_emb:",decoder_input.shape)
        # decoder_output = self.Decoder(decoder_input, decoder_pos_emb, N_mask)
        decoder_output = self.Decoder(decoder_input, decoder_pos_emb, mask)
        
        B, M, C = decoder_output.shape
        print("decoder_output:",decoder_output.shape)
        all_points = torch.reshape(input=all_points,shape=(all_points.shape[0],all_points.shape[1]*all_points.shape[2],all_points.shape[3]))
        # rebuild_points = self.increase_dim(decoder_output.transpose(1,2)).transpose(1,2).reshape(B*M, -1, 3) #####时间维度怎么办？？？？？？
        gt_points = all_points[mask].reshape(B*M, -1, 3)
        print("gt_points:",gt_points.shape)
        rebuild_points = gt_points ####Just For Debugging
        loss = self.loss_fn(rebuild_points, gt_points)
        return loss
    
        """
        现在用的是pointnet2_utils_mine.py，里面是写好的函数；P4给的代码使用的是pointnet2_utils(跑不起来)
        B:14
        mask_ratio: 0.75
        encoder输出的维度：1024
        当前example中点云总数：768
        可见点云数：192
        N_mask：576
        transdim:Decoder input 的维度  384
        
        TODO：
        Encoder：
        输入为[14,24,2048,3]->[B,T,N,3]
        输出为[14,12,64,1024]->[B,T//temporal_stride,N//saptial_stride,C]
        
        CDLoss算的不对，应该是维度出了点问题，主要问题还是在GTPOINTS
        GT_POINTS有点问题，输出的应该是原始点云中被mask的feature对应的点，但是mask矩阵的大小是根据feature大小定的，怎么找到对应的gtpoints？
        
        pointnet2_utils里面的three_interpolate没实现
        """
        
        
        