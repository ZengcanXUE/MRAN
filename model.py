import torch
import numpy as np
import math
import torch.nn
from torch.nn import functional as F


class DistMult(torch.nn.Module):
    def __init__(self, data, ent_dim, rel_dim, **kwargs):
        super(DistMult, self).__init__()

        self.emb_e = torch.nn.Embedding(data.entities_num, ent_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(data.relations_num, rel_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(kwargs["input_dropout"])
        self.loss = torch.nn.BCELoss()

    def init(self):
        torch.nn.init.xavier_normal_(self.emb_e.weight.data)
        torch.nn.init.xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded = self.emb_e(e1)
        rel_embedded = self.emb_rel(rel)
        e1_embedded = e1_embedded.squeeze()
        rel_embedded = rel_embedded.squeeze()

        e1_embedded = self.inp_drop(e1_embedded)
        rel_embedded = self.inp_drop(rel_embedded)

        pred = torch.mm(e1_embedded * rel_embedded, self.emb_e.weight.transpose(1, 0))
        pred = torch.sigmoid(pred)

        return pred


class ComplEx(torch.nn.Module):
    def __init__(self, data, ent_dim, rel_dim, **kwargs):
        super(ComplEx, self).__init__()

        self.emb_e_real = torch.nn.Embedding(data.entities_num, ent_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(data.entities_num, ent_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(data.relations_num, rel_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(data.relations_num, rel_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(kwargs["input_dropout"])
        self.loss = torch.nn.BCELoss()

    def init(self):
        torch.nn.init.xavier_normal_(self.emb_e_real.weight.data)
        torch.nn.init.xavier_normal_(self.emb_e_img.weight.data)
        torch.nn.init.xavier_normal_(self.emb_rel_real.weight.data)
        torch.nn.init.xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, e1, rel):
        e1_embedded_real = self.emb_e_real(e1).squeeze()
        rel_embedded_real = self.emb_rel_real(rel).squeeze()
        e1_embedded_img =  self.emb_e_img(e1).squeeze()
        rel_embedded_img = self.emb_rel_img(rel).squeeze()

        e1_embedded_real = self.inp_drop(e1_embedded_real)
        rel_embedded_real = self.inp_drop(rel_embedded_real)
        e1_embedded_img = self.inp_drop(e1_embedded_img)
        rel_embedded_img = self.inp_drop(rel_embedded_img)

        # complex space bilinear product (equivalent to HolE)
        realrealreal = torch.mm(e1_embedded_real*rel_embedded_real, self.emb_e_real.weight.transpose(1,0))
        realimgimg = torch.mm(e1_embedded_real*rel_embedded_img, self.emb_e_img.weight.transpose(1,0))
        imgrealimg = torch.mm(e1_embedded_img*rel_embedded_real, self.emb_e_img.weight.transpose(1,0))
        imgimgreal = torch.mm(e1_embedded_img*rel_embedded_img, self.emb_e_real.weight.transpose(1,0))
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = torch.sigmoid(pred)

        return pred


class ConvE(torch.nn.Module):
    def __init__(self, data, ent_dim, rel_dim, **kwargs):
        super(ConvE, self).__init__()
        self.ent_dim = 200
        self.rel_dim = 200
        self.emb_e = torch.nn.Embedding(data.entities_num, self.ent_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(data.relations_num, self.rel_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_drop = torch.nn.Dropout(kwargs["hidden_dropout"])
        self.feature_map_drop = torch.nn.Dropout2d(kwargs["feature_map_dropout"])
        self.loss = torch.nn.BCELoss()
        self.emb_dim1 = 20
        self.emb_dim2 = self.ent_dim // self.emb_dim1

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=True)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(self.ent_dim)
        self.register_parameter('bias', torch.nn.Parameter(torch.zeros(data.entities_num)))
        self.fc = torch.nn.Linear(9728, self.ent_dim)

    def init(self):
        torch.nn.init.xavier_normal_(self.emb_e.weight.data)
        torch.nn.init.xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.emb_rel(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1,0))
        x += self.bias.expand_as(x)
        pred = torch.sigmoid(x)

        return pred


class MRAN(torch.nn.Module):

    def __init__(self, data, ent_dim, rel_dim, **kwargs):
        super(MRAN, self).__init__()
        self.ent_dim = ent_dim
        self.rel_dim = rel_dim
        self.reshape_H = 2
        self.reshape_W = 400

        self.in_channels = kwargs["in_channels"]
        self.out_channels = kwargs["out_channels"]
        self.filt_height = kwargs["filt_height"]
        self.filt_width = kwargs["filt_width"]

        self.enti_embedding = torch.nn.Embedding(data.entities_num, ent_dim, padding_idx=0)
        self.ent_transfer = torch.nn.Embedding(data.entities_num, ent_dim, padding_idx=0)
        self.rela_embedding = torch.nn.Embedding(data.relations_num, rel_dim, padding_idx=0)
        self.rel_transfer = torch.nn.Embedding(data.relations_num, rel_dim, padding_idx=0)

        filter_dim = self.in_channels * self.out_channels * self.filt_height * self.filt_width
        # self.fc1 = torch.nn.Linear(rel_dim, filter_dim)
        self.filter = torch.nn.Embedding(data.relations_num, filter_dim, padding_idx=0)

        self.input_drop = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_drop = torch.nn.Dropout(kwargs["hidden_dropout"])
        self.feature_map_drop = torch.nn.Dropout2d(kwargs["feature_map_dropout"])

        self.loss = torch.nn.BCELoss()

        # self.conv = torch.nn.Conv2d(self.in_channels, self.out_channels, (self.filt_height, self.filt_width),
        #                            stride=1, padding=0, bias=True)

        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.bn1 = torch.nn.BatchNorm2d(self.out_channels)
        self.bn2 = torch.nn.BatchNorm1d(ent_dim)

        self.register_parameter('bias', torch.nn.Parameter(torch.zeros(data.entities_num)))

        fc_length = (self.reshape_H - self.filt_height + 1) * (self.reshape_W - self.filt_width + 1) * self.out_channels
        self.fc = torch.nn.Linear(fc_length, ent_dim)

        # augmentation mechanism
        self.b = 1
        self.gamma = 2
        t = int(abs((math.log(self.out_channels, 2) + self.b) / self.gamma))
        # self.k_size = t if t % 2 else t + 1
        self.k_size = 3  # augmentation value, more out_channel----> more k_size
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1) #全局自适应池化
        self.conv = torch.nn.Conv1d(1, 1, kernel_size=self.k_size, padding=(self.k_size - 1) // 2, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def init(self):
        torch.nn.init.xavier_normal_(self.enti_embedding.weight.data)
        torch.nn.init.xavier_normal_(self.rela_embedding.weight.data)
        # torch.nn.init.xavier_normal_(self.ent_transfer.weight.data)
        torch.nn.init.xavier_normal_(self.rel_transfer.weight.data)
        torch.nn.init.xavier_normal_(self.filter.weight.data)

    def forward(self, entity_id, relation_id):
        batch_enti_embedding = self.enti_embedding(entity_id).reshape(-1, 1, self.ent_dim)
        batch_rela_embedding = self.rela_embedding(relation_id).reshape(-1, 1, self.rel_dim)
        batch_ent_transfer = self.ent_transfer(entity_id).reshape(-1, 1, self.ent_dim)
        batch_rel_transfer = self.rel_transfer(relation_id).reshape(-1, 1, self.rel_dim)

        # construction of relation-aware filters
        f = self.filter(relation_id)
        f = f.reshape(batch_enti_embedding.size(0) * self.in_channels * self.out_channels, 1, self.filt_height,
                      self.filt_width)

        granularity = 'complex' # Euclidean, complex(default), quaternion
        if granularity == 'Euclidean':
            e1 = batch_enti_embedding
            r1 = batch_rela_embedding
            e = e1 + e1 * r1
            r = r1 + e1 * r1
        elif granularity == 'complex':
            u = batch_enti_embedding
            x = batch_enti_embedding
            y = batch_rela_embedding
            z = batch_rela_embedding
            e = u + u * y + u * z + x * y - x * z
            r = y + u * y + u * z + x * y - x * z
        elif granularity == 'quaternion':
            u1 = batch_enti_embedding
            u2 = batch_enti_embedding
            u3 = batch_enti_embedding
            u4 = batch_enti_embedding
            x1 = batch_rela_embedding
            x2 = batch_rela_embedding
            x3 = batch_rela_embedding
            x4 = batch_rela_embedding
            e = u1 + u1 * x1 - u2 * x2 - u3 * x3 - u4 * x4 \
                   + u1 * x2 + u2 * x1 + u3 * x4 - u4 * x3 \
                   + u1 * x3 - u2 * x4 + u3 * x1 + u4 * x2 \
                   + u1 * x4 + u2 * x3 - u3 * x2 + u4 * x1
            r = x1 + u1 * x1 - u2 * x2 - u3 * x3 - u4 * x4 \
                   + u1 * x2 + u2 * x1 + u3 * x4 - u4 * x3 \
                   + u1 * x3 - u2 * x4 + u3 * x1 + u4 * x2 \
                   + u1 * x4 + u2 * x3 - u3 * x2 + u4 * x1

        x = torch.cat([e, r], 1).reshape(-1, 1, self.reshape_H, self.reshape_W)
        x = self.bn0(x)
        x = self.input_drop(x)

        # (b, out, 2-filt_h+1, 200-filt_w+1)
        x = x.permute(1, 0, 2, 3)
        # x = self.conv(x)
        x = F.conv2d(x, f, groups=batch_enti_embedding.size(0))
        x = x.reshape(batch_enti_embedding.size(0), self.out_channels, self.reshape_H - self.filt_height + 1,
                      self.reshape_W - self.filt_width + 1)

        # augmentation mechanism implementation
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of the module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        x = x * y   # squeeze-excitation-reweight

        x = self.bn1(x)
        x = torch.relu(x)
        x = self.feature_map_drop(x)

        # (b, fc_length)→ (b, ent_dim)
        x = x.view(batch_enti_embedding.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = torch.relu(x)

        x = torch.mm(x, self.enti_embedding.weight.transpose(1, 0))
        x += self.bias.expand_as(x)
        pred = torch.sigmoid(x)

        return pred
