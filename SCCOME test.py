#
# Author: Haoshiang Liao
# Date: 2025/4/28
#
# tensorboard --logdir=logs --port=6006
# CUDA_VISIBLE_DEVICES=1 python VSFA.py --database=KoNViD-1k --exp_id=0

from argparse import ArgumentParser
import os
import h5py
import torch
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random
from scipy import stats
from tensorboardX import SummaryWriter
import datetime
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import time
from ptflops import get_model_complexity_info

class VQADataset(Dataset):
    def __init__(self, features_dir='CNN_features_K', index=None, max_len=2, feat_dim=11952, scale=1, batch_size=500):
        super(VQADataset, self).__init__()
        self.features = []
        self.length = []
        self.mos = []

        for batch_start in range(0, len(index), batch_size):
            batch_end = min(batch_start + batch_size, len(index))
            batch_indices = index[batch_start:batch_end]

            batch_features = []
            batch_lengths = []
            batch_mos = []

            for idx in batch_indices:
                features = np.load(features_dir + str(idx) + '_RGBcannyOptreplacedconvnext_3Dmaxmeanstd_features.npy')

                # Adjust feature dimension to match feat_dim by padding with zeros if needed
                if features.shape[1] < feat_dim:
                    pad_width = feat_dim - features.shape[1]
                    features = np.pad(features, ((0, 0), (0, pad_width)), mode='symmetric')  # Symmetric padding
                elif features.shape[1] > feat_dim:
                    raise ValueError(f"Feature dimension {features.shape[1]} exceeds expected dimension {feat_dim}")

                # Truncate or pad the temporal length
                if features.shape[0] > max_len:
                    features = features[:max_len, :]
                else:
                    pad_length = max_len - features.shape[0]
                    features = np.pad(features, ((0, pad_length), (0, 0)), mode='constant')

                batch_features.append(features)
                batch_lengths.append(features.shape[0])
                batch_mos.append(np.load(features_dir + str(idx) + '_score.npy'))

            self.features.append(np.array(batch_features))
            self.length.append(np.array(batch_lengths))
            self.mos.append(np.array(batch_mos))

        # Concatenate all batches
        self.features = np.concatenate(self.features, axis=0)
        self.length = np.concatenate(self.length, axis=0)
        self.mos = np.concatenate(self.mos, axis=0)

        self.scale = scale
        self.label = self.mos / self.scale  # Label normalization

    def __len__(self):
        return len(self.mos)

    def __getitem__(self, idx):
        sample = self.features[idx], self.length[idx], self.label[idx]
        return sample
# class VQADataset(Dataset):
#     def __init__(self, features_dir='CNN_features_KoNViD-1k', index=None, max_len=None, feat_dim=11952, scale=1):
#         super(VQADataset, self).__init__()
#         self.features_dir = features_dir
#         self.index = index
#         self.max_len = max_len
#         self.feat_dim = feat_dim
#         self.scale = scale
#         self.length = np.zeros((len(index), 1))
#         self.mos = np.zeros((len(index), 1))
#         self.file_names = []  # 用於存儲文件名稱
#
#         # 動態檢測 max_len
#         if self.max_len is None:
#             self.max_len = self._detect_max_len()
#
#     def _detect_max_len(self):
#         max_len = 0
#         for idx in self.index:
#             feature_path = self.features_dir + str(idx) + '_RGBcannyOptreplacedconvnext_3Dmaxmeanstd_features.npy'
#             if not os.path.exists(feature_path):
#                 continue
#             features = np.load(feature_path)
#             max_len = max(max_len, features.shape[0])  # 更新最大時間長度
#         return max_len
#
#     def __len__(self):
#         return len(self.mos)
#
#     def __getitem__(self, idx):
#         feature_path = self.features_dir + str(
#             self.index[idx]) + '_RGBcannyOptreplacedconvnext_3Dmaxmeanstd_features.npy'
#         label_path = self.features_dir + str(self.index[idx]) + '_score.npy'
#
#         if not os.path.exists(feature_path) or not os.path.exists(label_path):
#             raise FileNotFoundError(f"File not found: {feature_path} or {label_path}")
#
#         features = np.load(feature_path)
#         label = np.load(label_path) / self.scale
#
#         if features.shape[1] < self.feat_dim:
#             pad_width = self.feat_dim - features.shape[1]
#             features = np.pad(features, ((0, 0), (0, pad_width)), mode='symmetric')
#         elif features.shape[1] > self.feat_dim:
#             raise ValueError(f"Feature dimension {features.shape[1]} exceeds expected dimension {self.feat_dim}")
#
#         if features.shape[0] > self.max_len:
#             features = features[:self.max_len, :]
#         else:
#             pad_length = self.max_len - features.shape[0]
#             features = np.pad(features, ((0, pad_length), (0, 0)), mode='constant')
#
#         self.length[idx] = features.shape[0]
#         sample = (features, self.length[idx], label)
#         return sample
class Mlp(nn.Module):
    def __init__(self, input_features, hidden_features=256, out_features=1, drop_rate=0.2, act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden_features)
        # self.bn1 = nn.BatchNorm1d(hidden_features)
        self.act1 = act_layer()
        self.drop1 = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(hidden_features, hidden_features // 2)
        self.act2 = act_layer()
        self.drop2 = nn.Dropout(drop_rate)
        self.fc3 = nn.Linear(hidden_features // 2, out_features)

    def forward(self, input_feature):
        x = self.fc1(input_feature)
        # x = self.bn1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop2(x)
        output = self.fc3(x)
        return output

class VQADataset_test(Dataset):
    def __init__(self, features_dir='CNN_features_LSVQ_test', index=None, max_len=2, feat_dim=11952, scale=1):
        super(VQADataset_test, self).__init__()
        self.features_dir = features_dir
        self.index = index
        self.max_len = max_len
        self.feat_dim = feat_dim
        self.scale = scale
        self.length = np.zeros((len(index), 1))
        self.mos = np.zeros((len(index), 1))
        self.file_names = []  # 用於存儲文件名稱


    def __len__(self):
        return len(self.mos)

    def __getitem__(self, idx):
        feature_path = self.features_dir + str(
            self.index[idx]) + '_RGBcannyOptreplacedconvnext_3Dstdmeanmax_features.npy'
        label_path = self.features_dir + str(self.index[idx]) + '_score.npy'

        # 檢查特徵文件是否存在
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature file not found: {feature_path}")

        features = np.load(feature_path)

        # 檢查標籤文件是否存在
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")

        label = np.load(label_path) / self.scale

        # 檢查標籤是否包含 NaN
        if np.isnan(label).any():
            raise ValueError(f"Label for index {idx} contains NaN.")

        # 調整特徵形狀
        if features.shape[1] < self.feat_dim:
            pad_width = self.feat_dim - features.shape[1]
            features = np.pad(features, ((0, 0), (0, pad_width)), mode='symmetric')
        elif features.shape[1] > self.feat_dim:
            raise ValueError(f"Feature dimension {features.shape[1]} exceeds expected dimension {self.feat_dim}")

        # Truncate or pad the temporal length
        if features.shape[0] > self.max_len:
            features = features[:self.max_len, :]
        else:
            pad_length = self.max_len - features.shape[0]
            features = np.pad(features, ((0, pad_length), (0, 0)), mode='constant')

        self.length[idx] = features.shape[0]

        # 返回特徵、長度、標籤以及檔案名稱
        file_name = os.path.basename(feature_path)
        return features, self.length[idx], label, file_name

class FeatureAttention(nn.Module):
    def __init__(self, feat_dim, hidden_dim=512):
        super(FeatureAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feat_dim),
            nn.Sigmoid()  # output shape: (batch, seq_len, feat_dim)
        )

    def forward(self, x):
        weights = self.attention(x)  # (batch, seq_len, feat_dim)
        attended = x * weights       # feature-wise attention
        return attended
class FeatureTransformer(nn.Module):
    def __init__(self, patch_size=128, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.patch_size = patch_size
        self.d_model = d_model

    def forward(self, x):
        # x: (batch, seq_len, 11952)
        batch_size, seq_len, feat_dim = x.shape

        # 檢查能不能整除 patch_size
        assert feat_dim % self.patch_size == 0, "特徵維度要能整除 patch_size"
        num_patches = feat_dim // self.patch_size  # 幾個 patch

        # 先 reshape 成 (batch * seq_len, num_patches, patch_size)
        x = x.view(batch_size * seq_len, num_patches, self.patch_size)

        # 這樣每個 patch_size 就是 embedding 的維度 d_model
        # 如果要保留 positional embedding，可以在這裡加
        # shape 改成 (num_patches, batch*seq_len, d_model) 以符合Transformer預設 (S, N, E)
        x = x.permute(1, 0, 2)  # (num_patches, batch*seq_len, d_model=patch_size)

        out = self.transformer(x)  # (num_patches, batch*seq_len, d_model)

        # 再 permute 回來 (batch*seq_len, num_patches, d_model)
        out = out.permute(1, 0, 2)

        # 你可以取最後一層 [CLS] token 的概念，也可以 flatten
        # 這裡示範 flatten:
        out = out.reshape(batch_size, seq_len, num_patches * self.d_model)  # (batch, seq_len, num_patches * d_model)


        return out
class GLUBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x):  # (B, T, D)
        x_proj = self.fc(x)
        x_out, gate = x_proj.chunk(2, dim=-1)
        return x_out * torch.sigmoid(gate)

class FC(nn.Module):
    def __init__(
        self,
        total_input_size=11952   ,
        split_3d_index=10752,
        hidden_size_non3d=4096,
        hidden_size_non3d2=2048,
        hidden_size_3d=400,
        gru_hidden_size=2048,
        reduced_size=1024,
        dropout_p=0.5,
        num_gru_layers=2
    ):
        super().__init__()
        self.split_3d_index = split_3d_index

        # MLP 處理 2D 特徵
        self.mlp_non3d = nn.Sequential(
            nn.Linear(split_3d_index, hidden_size_non3d),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size_non3d, hidden_size_non3d2),
            nn.GELU(),
            nn.Dropout(dropout_p),
        )

        # MLP 處理 3D 特徵
        three_d_input_size = total_input_size - split_3d_index
        self.mlp_3d = nn.Sequential(
            nn.Linear(three_d_input_size, hidden_size_3d),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )

        # GRU for Q
        self.gru = nn.GRU(
            input_size=hidden_size_non3d2,
            hidden_size=gru_hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_p
        )

        # Attention 投影層
        attn_dim = gru_hidden_size * 2  # Q 輸出維度
        self.query_proj = nn.Linear(attn_dim, attn_dim)
        self.key_proj = nn.Linear(attn_dim, attn_dim)
        self.value_proj = nn.Linear(attn_dim, attn_dim)
        self.glu = GLUBlock(10752, 2048)
        self.scale = attn_dim ** 0.5

        # 最後輸出層
        GRU_and_3D = attn_dim + hidden_size_3d
        self.fc_final = nn.Linear(GRU_and_3D, reduced_size)


        self.apply(self.initialize_weights)

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        # Split 2D / 3D
        non3d_feats = x[:, :, :self.split_3d_index]      # (B, T, 10752)
        three_d_feats = x[:, :, self.split_3d_index:]    # (B, T, 1200)

        # Feature projection
        # out_non3d = self.glu(non3d_feats)
        out_non3d = self.mlp_non3d(non3d_feats)          # (B, T, 2048)
        out_3d = self.mlp_3d(three_d_feats)              # (B, T, 400)
        # 重新 reshape，每 3 幀一組
        batch_size, seq_len, _ = non3d_feats.shape
        out_non3d_Q = out_non3d.view(batch_size, seq_len // 2, 2, out_non3d.shape[-1])  # (16, 80, 3, 2048)
        out_non3d_Q = out_non3d_Q.view(-1, 2, out_non3d.shape[-1])  # (16*80, 3, 2048)


        # GRU: Q
        outputs, _ = self.gru(out_non3d_Q)                 # (B, T, 4096)
        seq_group = seq_len // 2
        # # 恢復 batch 維度
        out_non3d_Q = outputs.reshape(batch_size, seq_group, 2, outputs.shape[-1])
        out_non3d_Q = out_non3d_Q.reshape(batch_size, seq_group * 2, outputs.shape[-1])
        # print(out_non3d_Q.shape)
        Q = self.query_proj(out_non3d_Q)                     # (B, T, 4096)

        # Attention: K/V from original non3d features
        K = self.key_proj(out_non3d_Q)                     # (B, T, 4096)
        V = self.value_proj(out_non3d_Q)                   # (B, T, 4096)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, T, T)
        attn_weights = torch.softmax(attn_weights, dim=-1)                # (B, T, T)
        attended = torch.matmul(attn_weights, V)                          # (B, T, 4096)

        # Combine with 3D features
        combined = torch.cat([attended, out_3d], dim=-1)  # (B, T, 4096+400)
        outputs = self.fc_final(combined)                # (B, T, 1024)
        return outputs



def TP(q, tau=12, beta=0.5):
    q = torch.unsqueeze(torch.t(q), 0)
    qm = -float('inf') * torch.ones((1, 1, tau - 1)).to(q.device)
    qp = 10000.0 * torch.ones((1, 1, tau - 1)).to(q.device)
    l = -F.max_pool1d(torch.cat((qm, -q), 2), tau, stride=1)
    m = F.avg_pool1d(torch.cat((q * torch.exp(-q), qp * torch.exp(-qp)), 2), tau, stride=1)
    n = F.avg_pool1d(torch.cat((torch.exp(-q), torch.exp(-qp)), 2), tau, stride=1)
    m = m / n
    return beta * m + (1 - beta) * l


def ranking_loss(pred, target):
    return F.margin_ranking_loss(
        pred.unsqueeze(0) - pred.unsqueeze(1),
        target.unsqueeze(0) - target.unsqueeze(1),
        torch.sign(target.unsqueeze(0) - target.unsqueeze(1)),
        margin=0.0,
    )




class TransformerModel(nn.Module):
    def __init__(self, input_size=11952, reduced_size=1024, nhead=8, num_encoder_layers=2):
        super(TransformerModel, self).__init__()
        self.feature_attention = FeatureAttention(input_size)  # <== 新增
        self.FC = FC(dropout_p=0.5)
        self.q = nn.Linear(1024, 1)
        self.attention_weights = nn.Linear(2048, 1)

    def forward(self, input, input_length, i, label, file_name=None):
        if torch.isnan(input).any() or torch.isinf(input).any():
            print(f"NaN or Inf detected in input. Batch index: {i}")
            if file_name:
                print(f"Problematic file: {file_name}")

        # Apply attention mechanism over feature dimension
        # input = self.feature_attention(input)  # <== 加上 attention
        input = self.FC(input)

        q = self.q(input)
        score = torch.zeros_like(input_length, device=q.device)
        for i in range(input_length.shape[0]):
            qi = q[i, :int(input_length[i].item())]
            qi = TP(qi)
            score[i] = torch.mean(qi)

        return score


if __name__ == "__main__":
    parser = ArgumentParser(description='SCCOME')
    parser.add_argument("--seed", type=int, default=19990417)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.00001)')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 2000)')
    parser.add_argument('--database', default='KoNViD-1k', type=str, help='database name (default: KoNViD-1k)')
    parser.add_argument('--cross', default='N', type=str, help='Y/N')
    parser.add_argument('--test_database', default='KoNViD-1k', type=str, help='database name (default: KoNViD-1k)')
    parser.add_argument('--usecheckpoint', default='N', type=str, help='Y/N')
    parser.add_argument('--model', default='score_test', type=str, help='model name (default: VSFA)')
    parser.add_argument('--exp_id', default=0, type=int, help='exp id for train-val-test splits (default: 0)')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='test ratio (default: 0.2)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='val ratio (default: 0.2)')
    parser.add_argument("--notest_during_training", action='store_true', help='flag whether to test during training')
    parser.add_argument("--disable_visualization", action='store_true', help='flag whether to enable TensorBoard visualization')
    parser.add_argument("--log_dir", type=str, default="logs", help="log directory for Tensorboard log output")
    parser.add_argument('--disable_gpu', action='store_true', help='flag whether to disable GPU')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay (default: 0.001)')
    args = parser.parse_args()

    args.decay_interval = int(args.epochs / 10)
    args.decay_ratio = 0.8

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.database == 'KoNViD-1k':
        features_dir = 'CNN_features_KoNViD-1k/'
        datainfo = 'data/KoNViD-1kinfo.mat'
    if args.database == 'CVD2014':
        features_dir = 'CNN_features_CVD2014/'
        datainfo = 'data/CVD2014info.mat'
    if args.database == 'LIVE-Qualcomm':
        features_dir = 'CNN_features_LIVE-Qualcomm/'
        datainfo = 'data/LIVE-Qualcomminfo.mat'
    if args.database == 'LIVE-VQC':
        videos_dir = 'D:/VQAdatabase/LIVE Video Quality Challenge (VQC) Database/Video/'
        features_dir = 'CNN_features_LIVE-VQC/'
        datainfo = 'data/LIVE-VQCinfo.mat'
    if args.database == 'LSVQ':
        videos_dir = 'E:/VQADatabase/LSVQ/'
        features_dir = 'CNN_features_LSVQ/'
        datainfo = 'data/LSVQ_info.mat'


    print('EXP ID: {}'.format(args.exp_id))
    print(args.database)
    print(args.model)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    Info = h5py.File(datainfo, 'r')
    index = Info['index']
    index = index[:, args.exp_id % index.shape[1]]
    ref_ids = Info['ref_ids'][0, :]
    max_len = int(Info['max_len'][0][0])
    scale = Info['scores'][0, :].max()  # label normalization factor
    if args.cross =='Y':
        # 跨資料集
        if args.test_database =='KoNViD-1k':
            features_dir_test = 'CNN_features_LSVQ_test/'
            test_index_file = 'data/LSVQ_test_info.mat'
        if args.test_database =='LIVE-Qualcomm':
            features_dir_test = 'CNN_features_LIVE-Qualcomm/'
            test_index_file = 'data/LIVE-Qualcomminfo.mat'
        if args.test_database =='LSVQ_test':
            features_dir_test = 'CNN_features_LSVQ_test/'
            test_index_file = 'data/LSVQ_test_info.mat'

        TestInfo = h5py.File(test_index_file, 'r')
        test_index = TestInfo['index'][:].flatten()
        ref_ids_test = TestInfo['ref_ids'][0, :]
        max_len_test = int(Info['max_len'][0][0])
        trainindex = index[0:int(np.ceil((1 - args.val_ratio) * len(index)))]
        val_index = index[int(np.ceil((1 - args.val_ratio) * len(index))):len(index)]
        scale_test = TestInfo['scores'][0, :].max()  # label normalization factor

        trainindex_set = set(trainindex)
        testindex_set = set(test_index)
        train_index, val_index, test_index = [], [], []
        for i in range(len(ref_ids)):
            if ref_ids[i] in trainindex_set:
                train_index.append(i)
            else:
                val_index.append(i)
        for i in range(len(ref_ids_test)):
            test_index.append(i)

        # 加載資料集
        train_dataset = VQADataset(features_dir, train_index, max_len, scale=scale)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataset = VQADataset(features_dir, val_index, max_len, scale=scale)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset)
        test_dataset = VQADataset_test(features_dir_test, test_index, max_len_test, scale=scale_test)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset)
    if args.cross =='N':
        # 相同資料集
        trainindex = index[0:int(np.ceil((1 - args.test_ratio - args.val_ratio) * len(index)))]
        testindex = index[int(np.ceil((1 - args.test_ratio) * len(index))):len(index)]
        trainindex_set = set(trainindex)
        testindex_set = set(testindex)
        train_index, val_index, test_index = [], [], []
        for i in range(len(ref_ids)):
            if ref_ids[i] in trainindex_set:
                train_index.append(i)
            elif ref_ids[i] in testindex_set:
                test_index.append(i)
            else:
                val_index.append(i)
        # 在所有 index 中檢測最大 max_len
        # all_index = train_index + val_index + test_index  # 合併所有 index
        # max_len = detect_max_len(features_dir, all_index)
        # print(f"Detected max_len: {max_len}")
        train_dataset = VQADataset(features_dir, train_index, max_len, scale=scale)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataset = VQADataset(features_dir, val_index, max_len, scale=scale)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset)
        if args.test_ratio > 0:
            test_dataset = VQADataset(features_dir, test_index, max_len, scale=scale)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset)
    model = TransformerModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.usecheckpoint =='Y':
        checkpoint = torch.load('KoNViD-1k_trained_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']  # 取得訓練到的 epoch
        best_val_criterion = checkpoint.get('best_val_criterion', -1)  # 如果有最佳指標值
    if args.usecheckpoint == 'N':
        best_val_criterion = -1  # SROCC min
        start_epoch=0


    if not os.path.exists('models'):
        os.makedirs('models')
    trained_model_file = 'models/{}-{}-EXP{}'.format(args.model, args.database, args.exp_id)
    if not os.path.exists('results'):
        os.makedirs('results')
    save_result_file = 'results/{}-{}-EXP{}'.format(args.model, args.database, args.exp_id)

    if not args.disable_visualization:  # Tensorboard Visualization
        writer = SummaryWriter(log_dir='{}/EXP{}-{}-{}-{}-{}-{}-{}'
                               .format(args.log_dir, args.exp_id, args.database, args.model,
                                       args.lr, args.batch_size, args.epochs,
                                       datetime.datetime.now().strftime("%I_%M%p on %B %d, %Y")))

    criterion = nn.L1Loss()  # L1 loss
    # optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.decay_ratio)
    # ==== Inference Profile ====
    print("🔍 Running inference profiling...")

    # Dummy input (batch=1, seq_len=240, feat_dim=11952)
    dummy_input = torch.randn(1, 240, 11952).to(device)
    dummy_length = torch.tensor([240.0]).to(device)
    dummy_label = torch.tensor([0.5]).to(device)

    # 推理時間與記憶體開銷
    model.eval()
    with torch.no_grad():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        output = model(dummy_input, dummy_length, 0, dummy_label)
        end_time = time.time()

        inference_time = end_time - start_time
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024 ** 2  # MB

    print(f"🕒 推理時間（單一樣本）: {inference_time:.6f} 秒")
    print(f"💾 峰值記憶體使用量: {peak_memory:.2f} MB")

    # 模型參數量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔢 模型參數總數: {total_params / 1e6:.2f} M")

    # FLOPs: 只對 FC 子模組計算
    fc_model = model.FC
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(fc_model, (240, 11952), as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
        print(f"⚙️ 計算量（FLOPs）: {macs}")
        print(f"🧮 參數數量（僅FC）: {params}")
    # for epoch in range(start_epoch, start_epoch + total_new_epochs):
    for epoch in range(start_epoch,args.epochs):
        # Train
        model.train()
        print("train:",epoch," epoch")
        L = 0

        for i, (features, length, label) in enumerate(train_loader):
            features = features.to(device).float()
            label = label.to(device).float()
            length = length.to(device).float()
            optimizer.zero_grad()  #
            outputs = model(features, length,i,label )
            # l1_loss = criterion(outputs, label)
            # rank_loss = ranking_loss(outputs, label)
            # loss = l1_loss + 0.1 * rank_loss
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            L = L + loss.item()
        train_loss = L / (i + 1)

        model.eval()
        # Val
        print("Val:", epoch, " epoch")
        y_pred = np.zeros(len(val_index))
        y_val = np.zeros(len(val_index))
        L = 0
        with torch.no_grad():
            for i, (features, length, label) in enumerate(val_loader):
                features = features.to(device).float()
                length = length.to(device).float()
                y_val[i] = scale * label.item()  #
                label = label.to(device).float()
                outputs = model(features, length,i,label )
                y_pred[i] = scale * outputs.item()
                loss = criterion(outputs, label)
                L = L + loss.item()
        val_loss = L / (i + 1)
        val_PLCC = stats.pearsonr(y_pred, y_val)[0]
        val_SROCC = stats.spearmanr(y_pred, y_val)[0]
        val_RMSE = np.sqrt(((y_pred - y_val) ** 2).mean())
        val_KROCC = stats.stats.kendalltau(y_pred, y_val)[0]

        # Test
        if args.test_ratio > 0 and not args.notest_during_training:
            y_pred = np.zeros(len(test_index))
            y_test = np.zeros(len(test_index))
            L = 0
            with torch.no_grad():
                for i, (features, length, label) in enumerate(test_loader):
                    y_test[i] = scale * label.item()  #
                    features = features.to(device).float()
                    label = label.to(device).float()
                    length = length.to(device).float()
                    outputs = model(features, length,i,label )
                    y_pred[i] = scale * outputs.item()
                    loss = criterion(outputs, label)
                    L = L + loss.item()

            test_loss = L / (i + 1)
            PLCC = stats.pearsonr(y_pred, y_test)[0]
            SROCC = stats.spearmanr(y_pred, y_test)[0]
            RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())
            KROCC = stats.stats.kendalltau(y_pred, y_test)[0]

        if not args.disable_visualization:  # record training curves
            writer.add_scalar("loss/train", train_loss, epoch)  #
            writer.add_scalar("loss/val", val_loss, epoch)  #
            writer.add_scalar("SROCC/val", val_SROCC, epoch)  #
            writer.add_scalar("KROCC/val", val_KROCC, epoch)  #
            writer.add_scalar("PLCC/val", val_PLCC, epoch)  #
            writer.add_scalar("RMSE/val", val_RMSE, epoch)  #
            if args.test_ratio > 0 and not args.notest_during_training:
                writer.add_scalar("loss/test", test_loss, epoch)  #
                writer.add_scalar("SROCC/test", SROCC, epoch)  #
                writer.add_scalar("KROCC/test", KROCC, epoch)  #
                writer.add_scalar("PLCC/test", PLCC, epoch)  #
                writer.add_scalar("RMSE/test", RMSE, epoch)  #

        # Update the model with the best val_SROCC
        if val_SROCC > best_val_criterion:
            print("EXP ID={}: Update best model using best_val_criterion in epoch {}".format(args.exp_id, epoch))
            print("Val results: val loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                  .format(val_loss, val_SROCC, val_KROCC, val_PLCC, val_RMSE))
            if args.test_ratio > 0 and not args.notest_during_training:
                print("Test results: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                      .format(test_loss, SROCC, KROCC, PLCC, RMSE))
                # np.save(save_result_file, (y_pred, y_test, test_loss, SROCC, KROCC, PLCC, RMSE, test_index))
                np.save('y_pred.npy', y_pred)
                np.save('y_test.npy', y_test)
                np.save('test_loss.npy', test_loss)
                np.save('SROCC.npy', SROCC)
                np.save('KROCC.npy', KROCC)
                np.save('PLCC.npy', PLCC)
                np.save('RMSE.npy', RMSE)
                np.save('test_index.npy', test_index)

            checkpoint = {
                'epoch': epoch + 1,  # 下一個 epoch
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_criterion': val_SROCC
            }
            torch.save(checkpoint, 'KoNViD-1k_trained_model.pth')  # 儲存檢查點
            torch.save(model.state_dict(), trained_model_file)
            best_val_criterion = val_SROCC  # 更新最佳驗證指標

    # Test
    if args.test_ratio > 0:

        model.load_state_dict(torch.load(trained_model_file))  #
        model.eval()
        with torch.no_grad():
            y_pred = np.zeros(len(test_index))
            y_test = np.zeros(len(test_index))
            L = 0
            for i, (features, length, label) in enumerate(test_loader):
                y_test[i] = scale * label.item()  #
                features = features.to(device).float()
                label = label.to(device).float()
                outputs = model(features, length.float(),i,label )
                y_pred[i] = scale * outputs.item()
                loss = criterion(outputs, label)
                L = L + loss.item()
        test_loss = L / (i + 1)
        PLCC = stats.pearsonr(y_pred, y_test)[0]
        SROCC = stats.spearmanr(y_pred, y_test)[0]
        RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())
        KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
        print("Test results: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
              .format(test_loss, SROCC, KROCC, PLCC, RMSE))
        # 使用關鍵字參數保存數據
        np.savez(
            save_result_file,
            y_pred=y_pred,
            y_test=y_test,
            test_loss=test_loss,
            SROCC=SROCC,
            KROCC=KROCC,
            PLCC=PLCC,
            RMSE=RMSE,
            test_index=test_index
        )
