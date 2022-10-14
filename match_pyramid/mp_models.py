import torch
import torch.nn as nn
import numpy as np


class GMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx, num_layers_gru=1, conv_sizes=None, pool_sizes=None, mp_hidden_dim=128, return_cs=False):
        super(GMEncoder, self).__init__()
        self.sentence_encoder = SentenceEncoder(vocab_size, embed_dim, hidden_dim, pad_idx, num_layers_gru)
        self.match_pyramid = MatchPyramid(conv_sizes, pool_sizes, mp_hidden_dim, return_cs)

        
    def forward(self, docs1, docs2):
        # docs1, docs2 : (batch_size, max_doc_len, max_sent_len)
        out1 = self.sentence_encoder(docs1)
        out2 = self.sentence_encoder(docs2)
        # out1, out2 : (batch_size, max_doc_len, 2*hidden_dim)
        out = self.match_pyramid(out1, out2)
               
        return out # (batch_size, 2)


class MPEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, pad_idx, conv_sizes=None, pool_sizes=None, mp_hidden_dim=128, return_cs=False):
        super(MPEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, pad_idx)
        self.match_pyramid = MatchPyramid(conv_sizes, pool_sizes, mp_hidden_dim, return_cs)

        
    def forward(self, docs1, docs2):
        # docs1, docs2 : (batch_size, max_sent_len)
        out1 = self.embed(docs1)
        out2 = self.embed(docs2)
        # out1, out2 : (batch_size, max_sent_len, embed_dim)
        out = self.match_pyramid(out1, out2)
               
        return out # (batch_size, 2)


class MPEncoder2(nn.Module):
    def __init__(self, vocab_size, embed_dim, pad_idx, hidden_dim_gru = 50, num_layers_gru=1, conv_sizes=None, pool_sizes=None, mp_hidden_dim=128, return_cs=False):
        super(MPEncoder2, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim_gru, num_layers=num_layers_gru, bias=True, batch_first=True, dropout=0, bidirectional=True)
        self.match_pyramid = MatchPyramid(conv_sizes, pool_sizes, mp_hidden_dim, return_cs)

        
    def forward(self, docs1, docs2):
        # docs1, docs2 : (batch_size, max_sent_len)
        out1, _ = self.gru(self.embed(docs1))
        out2, _ = self.gru(self.embed(docs2))
        # out1, out2 : (batch_size, max_sent_len, hidden_dim)
        out = self.match_pyramid(out1, out2)
               
        return out # (batch_size, 2)


class MPEncoder3(nn.Module):
    def __init__(self, vocab_size, embed_dim, pad_idx, hidden_dim_gru = 50, num_layers_gru=1, conv_sizes=None, pool_sizes=None, mp_hidden_dim=128, return_cs=False):
        super(MPEncoder3, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim_gru, num_layers=num_layers_gru, bias=True, batch_first=True, dropout=0, bidirectional=True)
        self.fc = nn.Linear(2*hidden_dim_gru, 2*hidden_dim_gru)
        self.cv = nn.Parameter(0.05*torch.randn(2*hidden_dim_gru), requires_grad=True)
        self.match_pyramid = MatchPyramid(conv_sizes, pool_sizes, mp_hidden_dim, return_cs)

        
    def forward(self, docs1, docs2):
        # docs1, docs2 : (batch_size, max_sent_len)
        _, max_sent_len = docs1.size()
        out1, _ = self.gru(self.embed(docs1))
        out2, _ = self.gru(self.embed(docs2))
        # out1, out2 : (batch_size, max_sent_len, hidden_dim)

        u1 = torch.tanh(self.fc(out1)) # (batch_size, max_sent_len, 2*hidden_dim)
        ip1 = torch.sum(u1*self.cv, dim=-1) # (batch_size, max_sent_len)
        att1 = torch.softmax(ip1, dim=-1)*max_sent_len # (batch_size, max_sent_len)
        out1 = out1*att1.unsqueeze(2)

        u2 = torch.tanh(self.fc(out2)) # (batch_size, max_sent_len, 2*hidden_dim)
        ip2 = torch.sum(u2*self.cv, dim=-1) # (batch_size, max_sent_len)
        att2 = torch.softmax(ip2, dim=-1)*max_sent_len # (batch_sizess, max_sent_len)
        out2 = out2*att2.unsqueeze(2)

        out = self.match_pyramid(out1, out2)
        
               
        return out # (batch_size, 2)


class SentenceEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx, num_layers_gru):
        super(SentenceEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers_gru, bias=True, batch_first=True, dropout=0, bidirectional=True)
        self.fc = nn.Linear(2*hidden_dim, 2*hidden_dim)
        self.cv = nn.Parameter(0.05*torch.randn(2*hidden_dim), requires_grad=True)
    

    def forward(self, docs):
        # text : (batch_size, max_doc_len, max_sent_len)
        batch_size, max_doc_len, max_sent_len = docs.size()
        out = docs.reshape(batch_size*max_doc_len, max_sent_len)
        out = self.embed(out) # (batch_size*max_doc_len, max_sent_len, embed_dim)
        out, _ = self.gru(out) # (batch_size*max_doc_len, max_sent_len, 2*hidden_dim)
        u = torch.tanh(self.fc(out)) # (batch_size*max_doc_len, max_sent_len, 2*hidden_dim)
        ip = torch.sum(u*self.cv, dim=-1) # (batch_size*max_doc_len, max_sent_len)
        att = torch.softmax(ip, dim=-1) # (batch_size*max_doc_len, max_sent_len)
        out = torch.sum(out*att.unsqueeze(2), dim=1)
        out = out.reshape(batch_size, max_doc_len, -1) # (batch_size, max_doc_len, 2*hidden_dim)
        
        return out


class MatchPyramid(nn.Module):
    def __init__(self, conv_sizes, pool_sizes, hidden_dim, return_cs, output_dim=2):
        super(MatchPyramid, self).__init__()

        self.conv_sizes = conv_sizes
        self.pool_sizes = pool_sizes
        self.hidden_dim = hidden_dim
        self.conv_pool_ratio = len(self.conv_sizes) // len(self.pool_sizes)
        self.return_cs = return_cs
        self.output_dim = output_dim

        self.conv = nn.ModuleList([
            nn.Conv2d(
                in_channels=self.conv_sizes[i-1][-1],
                out_channels=self.conv_sizes[i][-1],
                kernel_size=tuple(
                    self.conv_sizes[i][0:2]),
                padding=0,
                bias=True
            )
            if i > 0
            else
            nn.Conv2d(
                in_channels=1,
                out_channels=self.conv_sizes[i][-1],
                kernel_size=tuple(
                    self.conv_sizes[i][0:2]),
                padding=0,
                bias=True
            )
            for i in range(len(self.conv_sizes))
        ])

        for conv in self.conv:
            nn.init.kaiming_normal_(conv.weight)

        self.bn = nn.ModuleList([nn.BatchNorm2d(self.conv_sizes[i][-1]) for i in range(len(self.conv_sizes))])

        # self.dropout = nn.ModuleList([nn.Dropout(0.5) for _ in range(len(self.pool_sizes))])

        self.pool = nn.ModuleList([nn.AdaptiveAvgPool2d(tuple(self.pool_sizes[i])) for i in range(len(self.pool_sizes))])

        self.linear1 = nn.Linear(self.pool_sizes[-1][0] * self.pool_sizes[-1][1] * self.conv_sizes[-1][-1],
                                       self.hidden_dim, bias=True)
        nn.init.kaiming_normal_(self.linear1.weight)
        self.linear2 = nn.Linear(self.hidden_dim, self.output_dim, bias=True)
        nn.init.kaiming_normal_(self.linear2.weight)
        

    def forward(self, x1, x2):
        # x1,x2:[batch, seq_len, dim_xlm]
        batch_size, _, embed_dim = x1.size()
        # simi_img:[batch, 1, seq_len, seq_len]
        # x1_norm = x1.norm(dim=-1, keepdim=True)
        # x1_norm = x1_norm + 1e-8
        # x2_norm = x2.norm(dim=-1, keepdim=True)
        # x2_norm = x2_norm + 1e-8
        # x1 = x1 / x1_norm
        # x2 = x2 / x2_norm
        # use cosine similarity since dim is too big for dot-product
        simi_img = torch.matmul(x1, x2.transpose(1, 2)) / np.sqrt(embed_dim)

        if self.return_cs:
            return simi_img

        simi_img = simi_img.unsqueeze(1)
        # self.logger.info(simi_img.size())
        
        if self.conv_pool_ratio == 2:
            for i in range(len(self.pool_sizes)):
                simi_img = torch.relu(self.bn[2*i](self.conv[2*i](simi_img)))
                simi_img = torch.relu(self.bn[2*i+1](self.conv[2*i+1](simi_img)))
                # simi_img = self.dropout[i](simi_img)
                simi_img = self.pool[i](simi_img)
                
        elif self.conv_pool_ratio == 1:
            for i in range(len(self.pool_sizes)):
                simi_img = torch.relu(self.bn[i](self.conv[i](simi_img)))
                # simi_img = self.dropout[i](simi_img)
                simi_img = self.pool[i](simi_img)
        
        simi_img = simi_img.reshape(batch_size, -1)
        # output = self.linear1(simi_img)
        out = self.linear2(torch.relu(self.linear1(simi_img)))

        return out


class DoubleMatchPyramid(nn.Module):
    def __init__(self, vocab_size, embed_dim, pad_idx, conv_sizes_inner, pool_sizes_inner, hidden_dim_inner, output_dim_inner, conv_sizes_outer, pool_sizes_outer, hidden_dim_outer):
        super(DoubleMatchPyramid, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, pad_idx)
        self.mp_inner = MatchPyramid(conv_sizes_inner, pool_sizes_inner, hidden_dim_inner, False, output_dim_inner)
        self.conv_sizes = conv_sizes_outer
        self.pool_sizes = pool_sizes_outer
        self.hidden_dim = hidden_dim_outer
        self.conv = nn.ModuleList([
            nn.Conv2d(
                in_channels=self.conv_sizes[i-1][-1],
                out_channels=self.conv_sizes[i][-1],
                kernel_size=tuple(
                    self.conv_sizes[i][0:2]),
                padding=0,
                bias=True
            )
            if i > 0
            else
            nn.Conv2d(
                in_channels=output_dim_inner,
                out_channels=self.conv_sizes[i][-1],
                kernel_size=tuple(
                    self.conv_sizes[i][0:2]),
                padding=0,
                bias=True
            )
            for i in range(len(self.conv_sizes))
        ])

        for conv in self.conv:
            nn.init.kaiming_normal_(conv.weight)

        self.bn = nn.ModuleList([nn.BatchNorm2d(self.conv_sizes[i][-1]) for i in range(len(self.conv_sizes))])

        self.pool = nn.ModuleList([nn.AdaptiveAvgPool2d(tuple(self.pool_sizes[i])) for i in range(len(self.pool_sizes))])

        self.linear1 = nn.Linear(self.pool_sizes[-1][0] * self.pool_sizes[-1][1] * self.conv_sizes[-1][-1],
                                       self.hidden_dim, bias=True)
        nn.init.kaiming_normal_(self.linear1.weight)
        self.linear2 = nn.Linear(self.hidden_dim, 2, bias=True)
        nn.init.kaiming_normal_(self.linear2.weight)


    def forward(self, docs1, docs2):
        batch_size, doc_len, sent_len = docs1.size()
        out1 = torch.repeat_interleave(docs1, doc_len, 1).reshape(batch_size*doc_len*doc_len, sent_len)
        out2 = torch.repeat_interleave(docs2, doc_len, 0).reshape(batch_size*doc_len*doc_len, sent_len)
        out1 = self.embed(out1)
        out2 = self.embed(out2)
        out = self.mp_inner(out1, out2) # (batch_size*doc_len*doc_len, output_dim_inner)
        simi_img = out.reshape(batch_size, doc_len, doc_len, -1).permute(0,3,1,2)

        for i in range(len(self.pool_sizes)):
            simi_img = torch.relu(self.bn[i](self.conv[i](simi_img)))
            simi_img = self.pool[i](simi_img)
        
        simi_img = simi_img.reshape(batch_size, -1)
        out = self.linear2(torch.relu(self.linear1(simi_img)))

        return out
