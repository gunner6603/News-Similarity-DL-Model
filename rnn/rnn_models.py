import torch
import torch.nn as nn
import numpy as np


class GRUEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx, num_layers_gru1=1, num_layers_gru2=1):
        super(GRUEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, pad_idx)
        self.gru1 = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers_gru1, bias=True, batch_first=True, dropout=0, bidirectional=True)
        self.fc1 = nn.Linear(2*hidden_dim, 2*hidden_dim)
        self.cv1 = nn.Parameter(0.05*torch.randn(2*hidden_dim), requires_grad=True)
        self.gru2 = nn.GRU(2*hidden_dim, hidden_dim, num_layers=num_layers_gru2, bias=True, batch_first=True, dropout=0, bidirectional=True)
        self.fc2 = nn.Linear(2*hidden_dim, 2*hidden_dim)
        self.cv2 = nn.Parameter(0.05*torch.randn(2*hidden_dim), requires_grad=True)
        
    def forward(self, docs):
        # text : (batch_size, max_doc_len, max_sent_len)
        batch_size, max_doc_len, max_sent_len = docs.size()
        out = docs.view(batch_size*max_doc_len, max_sent_len)
        out = self.embed(out) # (batch_size*max_doc_len, max_sent_len, embed_dim)
        out, _ = self.gru1(out) # (batch_size*max_doc_len, max_sent_len, 2*hidden_dim)
        u1 = torch.tanh(self.fc1(out)) # (batch_size*max_doc_len, max_sent_len, 2*hidden_dim)
        # u1 = self.fc1(out)
        ip1 = torch.sum(u1*self.cv1, dim=-1) # (batch_size*max_doc_len, max_sent_len)
        att1 = torch.softmax(ip1, dim=-1) # (batch_size*max_doc_len, max_sent_len)
        out = torch.sum(out*att1.unsqueeze(2), dim=1)
        out = out.view(batch_size, max_doc_len, -1) # (batch_size, max_doc_len, 2*hidden_dim)
        out, _ = self.gru2(out) # (batch_size, max_doc_len, 2*hidden_dim)
        u2 = torch.tanh(self.fc2(out)) # (batch_size, max_doc_len, 2*hidden_dim)
        # u2 = self.fc2(out)
        ip2 = torch.sum(u2*self.cv2, dim=-1) # (batch_size, max_doc_len)
        att2 = torch.softmax(ip2, dim=-1) # (batch_size, max_doc_len)
        out = torch.sum(out*att2.unsqueeze(2), dim=1) # (batch_size, 2*hidden_dim)
        
        return out


class CNNEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx, num_layers_gru2, filter_sizes, num_filters):
        super(CNNEncoder, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim, pad_idx)
        self.gru2 = nn.GRU(sum(num_filters), hidden_dim, num_layers=num_layers_gru2, bias=True, batch_first=True, dropout=0, bidirectional=True)
        self.fc2 = nn.Linear(2*hidden_dim, 2*hidden_dim)
        self.cv2 = nn.Parameter(0.05*torch.randn(2*hidden_dim), requires_grad=True)
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
              
    def forward(self, docs):

        batch_size, max_doc_len, max_sent_len = docs.size()
        out = docs.view(batch_size*max_doc_len, max_sent_len)
        out = self.embed(out) # (batch_size*max_doc_len, max_sent_len, embed_dim)
        reshaped = out.permute(0,2,1)
        conv_list = [torch.relu(conv1d(reshaped)) for conv1d in self.conv1d_list]
        pool_list = [torch.max_pool1d(conv, kernel_size=conv.shape[2]) for conv in conv_list]
        conv_feature = torch.cat([pool.squeeze(dim=2) for pool in pool_list], dim=1) # (batch_size*max_doc_len, sum(num_filters))
        out = conv_feature.view(batch_size, max_doc_len, -1) # (batch_size, max_doc_len, sum(num_filters))
        out, _ = self.gru2(out) # (batch_size, max_doc_len, 2*hidden_dim)
        u2 = torch.tanh(self.fc2(out)) # (batch_size, max_doc_len, 2*hidden_dim)
        ip2 = torch.sum(u2*self.cv2, dim=-1) # (batch_size, max_doc_len)
        att2 = torch.softmax(ip2, dim=-1) # (batch_size, max_doc_len)
        out = torch.sum(out*att2.unsqueeze(2), dim=1) # (batch_size, 2*hidden_dim)

        return out


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, docs1, docs2, train):
        out = torch.cat([docs1, docs2], dim=1)
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(torch.dropout(out, p=0.5, train=train))
        out = torch.sigmoid(out)
        out = out.view(-1)
        
        return out


class PositionalEnc(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerNet(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_heads, n_layers, dropout, sent_len):
        super().__init__()

        # Define parameters
        self.hidden_him = hidden_dim
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.sent_len = sent_len

        # Define Layers

        # Positional encoding layer
        self.pe = PositionalEnc(embedding_dim, max_len = sent_len)

        # Encoder layer
        enc_layer = nn.TransformerEncoderLayer(embedding_dim, n_heads, hidden_dim, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers = n_layers)
      
        # Fully connected layer
        self.fc = nn.Linear(embedding_dim, output_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, embedded):

        # embedded = [batch size, sent len, emb dim]
        pos_encoded = self.pe(embedded)
        trans_out = self.encoder(pos_encoded)
        # trans_out = [batch_size, sent len, emb_dim]

        pooled = trans_out.mean(1)
        final = self.fc(self.dropout(pooled))
      
        return final


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, trans_hidden_dim, trans_output_dim, trans_n_heads, trans_n_layers, trans_dropout, pad_idx, sent_len, gru_hidden_dim, gru_num_layers, gru_dropout):
        super(TransformerEncoder, self).__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_dim, pad_idx)
        self.transformer = TransformerNet(vocab_size, embed_dim, trans_hidden_dim, trans_output_dim, trans_n_heads, trans_n_layers, trans_dropout, sent_len)
        # self.gru1 = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers_gru1, bias=True, batch_first=True, dropout=0, bidirectional=True)
        # self.fc1 = nn.Linear(2*hidden_dim, 2*hidden_dim)
        # self.cv1 = nn.Parameter(torch.randn(2*hidden_dim), requires_grad=True)
        self.gru = nn.GRU(trans_output_dim, gru_hidden_dim, num_layers=gru_num_layers, bias=True, batch_first=True, dropout=gru_dropout, bidirectional=True)
        self.fc = nn.Linear(2*gru_hidden_dim, 2*gru_hidden_dim)
        self.cv = nn.Parameter(0.05*torch.randn(2*gru_hidden_dim), requires_grad=True)
        
    def forward(self, docs):
        # text : (batch_size, max_doc_len, max_sent_len)
        batch_size, max_doc_len, max_sent_len = docs.size()
        out = docs.view(batch_size*max_doc_len, max_sent_len)
        out = self.embed(out) # (batch_size*max_doc_len, max_sent_len, embed_dim)
        out = self.transformer(out) # (batch_size*max_doc_len, trans_output_dim)

        out = out.view(batch_size, max_doc_len, -1) # (batch_size, max_doc_len, trans_output_dim)
        out, _ = self.gru(out) # (batch_size, max_doc_len, 2*gru_hidden_dim)
        u = torch.tanh(self.fc(out)) # (batch_size, max_doc_len, 2*hidden_dim)
        # u2 = self.fc2(out)
        ip = torch.sum(u*self.cv, dim=-1) # (batch_size, max_doc_len)
        att = torch.softmax(ip, dim=-1) # (batch_size, max_doc_len)
        out = torch.sum(out*att.unsqueeze(2), dim=1) # (batch_size, 2*hidden_dim)
        
        return out