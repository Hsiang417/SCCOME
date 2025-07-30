#
# Author: Haoshiang Liao
# Date: 2025/4/28
#
# tensorboard --logdir=logs --port=6006
# CUDA_VISIBLE_DEVICES=1 python VSFA.py --database=KoNViD-1k --exp_id=0
#
# Author: Haoshiang Liao
# Date: 2024/1/22
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

class VQADataset(Dataset):
    def __init__(self, features_dir='CNN_features_K', index=None, max_len=2, feat_dim=11952, scale=1, batch_size=500):
        super(VQADataset, self).__init__()
        self.features = []
        self.length = []
        self.mos = []
        self.feat_dim = feat_dim
        for batch_start in range(0, len(index), batch_size):
            batch_end = min(batch_start + batch_size, len(index))
            batch_indices = index[batch_start:batch_end]

            batch_features = []
            batch_lengths = []
            batch_mos = []

            for idx in batch_indices:
                features = np.load(features_dir + str(idx) + '_CCOME_features.npy')

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
class FC(nn.Module):
    def __init__(
        self,
        total_input_size=11952,  # 整體輸入維度 = 4種特徵 + 3D特徵(1200)
        split_3d_index=10752,    # 前面 14336 是「四種特徵」的維度，後面 (15536-14336)=1200 即是 3D 特徵
        hidden_size_non3d=4096,  # 用在非3D分支的隱藏維度
        hidden_size_non3d2=2048,
        hidden_size_3d=400,      # 用在3D分支的隱藏維度 (可自行調整)
        gru_hidden_size=2048,    # GRU 隱藏維度
        reduced_size=1024,       # 最終輸出的維度
        dropout_p=0.5,
        num_gru_layers=2
    ):
        super(FC, self).__init__()

        self.split_3d_index = split_3d_index

        # -----------------------------
        # 1. MLP for non-3D features
        # -----------------------------
        # 假設想要對前面 14336 維的「四種特徵」做一些前饋處理
        self.mlp_non3d = nn.Sequential(
            nn.Linear(split_3d_index, hidden_size_non3d),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size_non3d, hidden_size_non3d2),  # 再多一層
            nn.GELU(),
            nn.Dropout(dropout_p),
        )

        # -----------------------------
        # 2. MLP for 3D features
        # -----------------------------
        # 針對 1200 維的 3D 特徵，可以根據需求大小自行決定幾層
        three_d_input_size = total_input_size - split_3d_index  # 預期=1200
        self.mlp_3d = nn.Sequential(
            nn.Linear(three_d_input_size, hidden_size_3d),
            nn.GELU(),
            nn.Dropout(dropout_p),

        )

        # -----------------------------
        # 3. GRU
        # -----------------------------
        # 合併後的輸入維度 = hidden_size_non3d
        self.gru_input_dim = hidden_size_non3d2
        self.gru = nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=gru_hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_p  # 多層GRU時，官方建議透過這裡設置
        )

        # -----------------------------
        # 4. 最終輸出層
        # -----------------------------
        # 如果是雙向 GRU，輸出維度 = 2 * gru_hidden_size
        GRU_and_3D=gru_hidden_size * 2+hidden_size_3d
        self.fc_final = nn.Linear(GRU_and_3D, reduced_size)

        # 欲保留的話，可以再來一層激活或 Dropout，看需求
        # self.output_activation = nn.GELU()

        # -----------------------------
        # 初始化
        # -----------------------------
        self.apply(self.initialize_weights)
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=4)
        self.feature_transformer = FeatureTransformer(

            patch_size=128,
            d_model=128,  # 看你要不要相同
            nhead=4,
            num_layers=2
        )

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    def forward(self, x):
        """
        x 的 shape 預期為 (batch, seq_len, 15536)
        其中:
          - 前 14336 維 (split_3d_index) 對應「四種特徵」
          - 後面 1200 維 對應「3D 特徵」
        """
        # -----------------------------
        # 0. 檢查 NaN/Inf，若你仍想保留這些檢查
        # -----------------------------
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        # -----------------------------
        # 1. 分割特徵
        # -----------------------------
        non3d_feats = x[:, :, :self.split_3d_index]   # shape: (batch, seq_len, 10752)
        out_transformer = self.feature_transformer(non3d_feats)
        batch_size, seq_len, _ = x.shape

        # non3d_feats = self.transformer_encoder(non3d_feats)
        three_d_feats = x[:, :, self.split_3d_index:] # shape: (batch, seq_len, 1200)

        # -----------------------------
        # 2. 分別做前饋
        # -----------------------------

        out_non3d = self.mlp_non3d(out_transformer)    # (batch, seq_len, hidden_size_non3d)
        # out_non3d = self.transformer_encoder(out_non3d)
        # if torch.isnan(out_non3d).any() or torch.isinf(out_non3d).any():
        #     print("NaN or Inf detected in Transformer output!")
        #     out_non3d = torch.nan_to_num(out_non3d, nan=0.0, posinf=1e6, neginf=-1e6)
        out_3d = self.mlp_3d(three_d_feats)

        # (batch, seq_len, hidden_size_3d)

        # 重新 reshape，每 3 幀一組
        out_non3d = out_non3d.view(batch_size, seq_len // 3, 3, out_non3d.shape[-1])  # (16, 80, 3, 2048)
        out_non3d = out_non3d.view(-1, 3, out_non3d.shape[-1])  # (16*80, 3, 2048)


        # -----------------------------
        # 3. 特徵拼接後進入 GRU
        # -----------------------------
        outputs, _ = self.gru(out_non3d)
        seq_group = seq_len // 3
        # 恢復 batch 維度
        outputs = outputs.reshape(batch_size, seq_group, 3, outputs.shape[-1])
        outputs = outputs.reshape(batch_size, seq_group * 3, outputs.shape[-1])

        combined = torch.cat([outputs, out_3d], dim=-1)


        # -----------------------------
        # 4. 最終輸出層
        # -----------------------------
        outputs = self.fc_final(combined)  # (batch, seq_len, reduced_size)

        # 如果想要再加個 activation 或 dropout：
        # outputs = self.output_activation(outputs)

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







class TransformerModel(nn.Module):
    def __init__(self, input_size=11952, reduced_size=1024, nhead=8, num_encoder_layers=2):
        super(TransformerModel, self).__init__()
        self.FC = FC(dropout_p=0.5)



        self.GRU = nn.GRU(2048, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True)
        self.q = nn.Linear(1024, 1)
        self.attention_weights = nn.Linear(2048, 1)

    def forward(self, input, input_length, i, label, file_name=None):
        if torch.isnan(input).any() or torch.isinf(input).any():
            print(f"NaN or Inf detected in input. Batch index: {i}")
            if file_name:
                print(f"Problematic file: {file_name}")


        input = self.FC(input)
        # print(f"FC output min: {input.min()}, max: {input.max()}")
        # print(input)


        q = self.q(input)
        score = torch.zeros_like(input_length, device=q.device)
        # print(q)

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
    # (新) 這裡插入計算 FLOPs 及參數數量的程式碼
    # -------------------------------------------------------------
    from thop import profile
    from torchinfo import summary

    # 建立一個假的輸入，來進行 FLOPs/params 的估計
    # 通常 batch_size 可能設成 1 或 16 均可。
    # 記得要跟模型 forward() 所需要的參數一致
    fake_batch_size = 1
    dummy_input = torch.randn(fake_batch_size, max_len, train_dataset.feat_dim).to(device)
    dummy_length = torch.randint(1, max_len + 1, (fake_batch_size,), dtype=torch.int).to(device)
    # label 也要丟，但其實在 THOP 只要 forward() input, length, i, label
    #   i 跟 label 要給定個 placeholder，否則 forward() 會出錯。
    fake_label = torch.zeros(fake_batch_size).to(device)

    # profile 計算 FLOPs 和參數
    flops, params = profile(model, inputs=(dummy_input, dummy_length, 0, fake_label))

    print(f"\n[Model Complexity]")
    print(f"  - Total FLOPs: {flops / 1e9:.4f} GFLOPs")
    print(f"  - Total Params: {params / 1e6:.4f} M\n")

    # 也可以用 torchinfo.summary 來查看模型結構
    print("[Model Summary]")
    summary(
        model,
        input_data=(dummy_input, dummy_length, 0, fake_label),  # 這裡是一個 tuple
        col_names=("input_size", "output_size", "num_params", "params_percent"),
    )
    print("-" * 60, "\n")

    ###########################
    if args.usecheckpoint =='Y':
        checkpoint = torch.load('KoNViD-1ktest_trained_model.pth')
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
