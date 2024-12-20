import torch 
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset

class BlitzLSTM(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_lstm_layers: int,
                 dropout: float=0.3,
                 n_positions: int=5,
                 position_embedding_dim: int=4):
        super(BlitzLSTM, self).__init__()

        self.position_embeddings = nn.Embedding(n_positions, position_embedding_dim)
        self.norm_layer = nn.LayerNorm(input_dim + position_embedding_dim - 1)

        self.lstm = nn.LSTM(
            input_size=input_dim + position_embedding_dim - 1,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim // 2)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    
    def forward(self, 
                x, 
                seq_lens, 
                return_hidden_states: bool=False,
                return_last_hidden_state: bool=False,
                return_last_output: bool=False):

        pos_ids = x[:, :, -1].long()
        pos_embds = self.position_embeddings(pos_ids)
        x = torch.cat([x[:, :, :-1], pos_embds], axis=2)
        x = self.norm_layer(x)

        x = pack_padded_sequence(x, seq_lens, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(x)
        out, _ = pad_packed_sequence(packed_out, batch_first=True) #(B, T, M)

        out = self.fc(out) # (B, T, M/2)
        if return_hidden_states:
            return out
        if return_last_hidden_state:
            col_indices = [x - 1 for x in seq_lens]
            out = out[torch.arange(out.shape[0]), col_indices, :].squeeze() # (B, M/2)
            return out

        out = self.decoder(out).squeeze() # (B, T)
        out = torch.sigmoid(out)

        if return_last_output:
            col_indices = [x - 1 for x in seq_lens]
            out = out[torch.arange(out.shape[0]), col_indices].squeeze()

        return out

class BlitzTransformer(nn.Module):

    def __init__(self, 
                 input_dim: int, 
                 embed_dim: int, 
                 max_seq_len: int, 
                 num_heads: int, 
                 dim_feedforward: int, 
                 output_dim: int, 
                 dropout: float=0.1):
        super(BlitzTransformer, self).__init__()

        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.time_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, 2)
        self.fc_out = nn.Linear(embed_dim, output_dim)
    
    def forward(self, x, src_padding_mask=None):

        x = self.input_proj(x)

        batch_size, seq_len, _ = x.size()
        time_steps = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        time_embeds = self.time_embedding(time_steps)

        #x = x + time_embeds
        x = self.norm(x)

        # do not allow attention heads to look ahead 
        src_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        
        x = self.encoder(x, mask=src_mask, src_key_padding_mask=src_padding_mask)

        # out shape is (batch, timesteps, num_players)
        out = self.fc_out(x)
        probs = torch.sigmoid(out)

        return probs 
    
class BlitzFrameTransformer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 z_input_dim: int,
                 model_dim: int,
                 num_layers: int,
                 output_dim: int,
                 z_dim: int=8,
                 dropout: float=0.3,
                 n_positions: int=19,
                 pos_emb_dim: int=4):
        super(BlitzFrameTransformer, self).__init__()

        input_dim += pos_emb_dim
        dim_feedforward = model_dim * 4
        num_heads = min(16, max(2, 2 * round(model_dim / 64)))

        self.pos_embedding = nn.Embedding(n_positions, pos_emb_dim)

        self.norm_layer = nn.BatchNorm1d(input_dim)

        self.feat_embedding_layer = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.ReLU(),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        self.play_embedding_layer = nn.Sequential(
            nn.LayerNorm(z_input_dim),
            nn.Linear(z_input_dim, z_dim),
            nn.ReLU(),
            nn.LayerNorm(z_dim),
            nn.Dropout(dropout)
        )

        model_dim += z_dim

        self.decoder = nn.Sequential(
            nn.Linear(model_dim, model_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim // 4)
        )
        self.fc_out = nn.Linear(model_dim // 4, output_dim)

    def forward(self, x: torch.tensor, z: torch.tensor, return_embeddings: bool=False):

        # embed player positions, then combine with tracking data feats 
        pos_ids = x[:, :, -1].long()
        pos_embs = self.pos_embedding(pos_ids)
        x = torch.cat([x[:, :, :-1], pos_embs], dim=2)

        x = self.norm_layer(x.permute(0, 2, 1)).permute(0, 2, 1) 
        x = self.feat_embedding_layer(x)
        x = self.encoder(x) # (B, P, M)

        # add in play context 
        z = self.play_embedding_layer(z)
        x = torch.cat([x, z], dim=2)

        x = self.decoder(x)
        if return_embeddings:
            return x[:, :11, :]
        
        x = self.fc_out(x).squeeze()

        # remove offensive players
        x = x[:, :11]
        x = torch.sigmoid(x)

        return x
    
class PressureFrameTransformer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 z_input_dim: int,
                 model_dim: int,
                 num_layers: int,
                 output_dim: int,
                 pooling_strategy: str="average",
                 z_dim: int=8,
                 dropout: float=0.3,
                 n_positions: int=19,
                 pos_emb_dim: int=4,
                 tts_emb_dim: int=4):
        super(PressureFrameTransformer, self).__init__()

        input_dim += pos_emb_dim
        dim_feedforward = model_dim * 4
        num_heads = min(16, max(2, 2 * round(model_dim / 64)))

        self.pos_embedding = nn.Embedding(n_positions, pos_emb_dim)
        self.tts_embedding = nn.Embedding(100, tts_emb_dim)

        self.norm_layer = nn.BatchNorm1d(input_dim)

        self.feat_embedding_layer = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.ReLU(),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        if pooling_strategy == "average":
            self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        else:
            self.pooling_layer == nn.AdaptiveMaxPool1d(1)

        self.play_embedding_layer = nn.Sequential(
            nn.LayerNorm(z_input_dim),
            nn.Linear(z_input_dim, z_dim),
            nn.ReLU(),
            nn.LayerNorm(z_dim),
            nn.Dropout(dropout)
        )

        # adding play/time dimensions to model (non-player specific)
        model_dim += z_dim + tts_emb_dim

        self.decoder = nn.Sequential(
            nn.Linear(model_dim, model_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim // 4)
        )
        self.fc_out = nn.Linear(model_dim // 4, output_dim)

    def forward(self, x: torch.tensor, z: torch.tensor, return_embeddings: bool=False):

        # save tts embeddings for later 
        tts = x[:, :, -2].long()
        tts_embs = self.tts_embedding(tts)
        tts_embs = tts_embs[:, 0, :].squeeze()

        # embed player positions and then combine with tracking data feats 
        pos_ids = x[:, :, -1].long()
        pos_embs = self.pos_embedding(pos_ids)
        # not taking last two cols of x because they contain embedded vars 
        x = torch.cat([x[:, :, :-2], pos_embs], dim=2)

        x = self.norm_layer(x.permute(0, 2, 1)).permute(0, 2, 1) 
        x = self.feat_embedding_layer(x)
        x = self.encoder(x) # (B, P, M)

        x = self.pooling_layer(x.permute(0, 2, 1)).squeeze() # (B, M)

        # add in play and time context 
        z = self.play_embedding_layer(z)
        z = z[:, 0, :].squeeze()
        x = torch.cat([x, z, tts_embs], dim=1)

        x = self.decoder(x)
        # if return_embeddings:
        #     return x[:, :11, :]
        
        x = self.fc_out(x).squeeze()

        # remove offensive players
        # x = x[:, :11]

        x = torch.sigmoid(x)

        return x
    
class BlitzData(Dataset):
    def __init__(self, data: list, transform: callable=None):
        # unpack tuples 
        self.keys, self.feats, self.labels = zip(*data)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def __len__(self):
        return len(self.feats)
    
    def __getitem__(self, idx):

        key = self.keys[idx]

        X = self.feats[idx]
        seq_len = X.shape[0]
        X = torch.from_numpy(X).float()

        y = self.labels[idx]
        y = torch.from_numpy(y).float()

        return key, X.to(self.device), y.to(self.device), seq_len

class BlitzFrameData(Dataset):
    def __init__(self, data: list):
        self.keys, self.feats, self.play_feats, self.labels = zip(*data)
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):

        key = self.keys[idx]

        X = self.feats[idx]
        X = torch.from_numpy(X).float() 

        z = self.play_feats[idx]
        z = torch.from_numpy(z).float()
        z = z.unsqueeze(0).repeat(X.shape[0], 1)

        y = self.labels[idx]
        y = torch.from_numpy(y).float()

        return key, X, z, y
