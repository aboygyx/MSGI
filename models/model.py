import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer, DecoderLayerWithFourier
from models.attn import FullAttention, ProbAttention, AttentionLayer, AttentionLayerWin, AttentionLayerCrossWin
from models.embed import DataEmbedding
from models.fourier import FNetLayer, FourierMix
from models.DSAttention import DSAttention
from models.CTRGC import CTRGC
from models.GCKM import GCKM
from models.MDM import MDM
from einops import rearrange

class Informer(nn.Module):
    """
    Standard Informer architecture for long sequence time-series forecasting.
    Utilizes ProbSparse Attention to reduce computational complexity.
    """
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=128,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding: embed temporal features and positional information
        self.enc_embedding = DataEmbedding(
            enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(
            dec_in, d_model, embed, freq, dropout)
            
        # Attention mechanism selection (ProbSparse or Full Attention)
        Attn = ProbAttention if attn == 'prob' else FullAttention

        # Encoder: stacks attention layers and optional distillation (Conv1d) layers
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # Decoder: incorporates self-attention and cross-attention with encoder outputs
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # Final linear projection to map the deep features to target dimensions
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # Encoder forward pass
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # Decoder forward pass
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
            
        # Projection to the output dimension
        dec_out = self.projection(dec_out)
        
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # Shape: [B, L, D]


class InformerStack(nn.Module):
    """
    Informer with stacked encoders of different lengths to capture multi-scale temporal patterns.
    """
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=[3, 2, 1], d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding configurations
        self.enc_embedding = DataEmbedding(
            enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(
            dec_in, d_model, embed, freq, dropout)
            
        # Attention mechanism selection
        Attn = ProbAttention if attn == 'prob' else FullAttention
        
        # Encoder: Create multiple encoders for stacked multi-scale inputs
        # Example: [0, 1, 2, ...] defines the input sequence subsets
        inp_lens = list(range(len(e_layers)))
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                            d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el - 1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        
        # Decoder configurations
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # Output projection layer
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # Encoder forward pass with stacked architecture
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # Decoder forward pass
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
            
        # Final projection
        dec_out = self.projection(dec_out)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # Shape: [B, L, D]


class MSGI(nn.Module):
    """
    Integrates MDM module for marginal distribution modeling and GCKM for feature graph convolutions,
    followed by the standard Informer encoder-decoder structure.
    """
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 gamma_list=[0.1, 0.05], factor=5, d_model=512, n_heads=8,
                 e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h',
                 activation='gelu', output_attention=False, distil=True,
                 mix=True, device=torch.device('cuda:0')):

        super(DGI22, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.device = device
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.seq_len = seq_len

        # ======================================================
        # 1. MDM Module
        # ======================================================
        self.mdm = MDM(
            input_shape=(seq_len, enc_in),   # Shape: (L, D)
            k=2,
            c=2,
            layernorm=True
        )

        # ======================================================
        # 2. Feature Graph Convolution (GCKM)
        # ======================================================
        self.gckm = GCKM(gamma_list)
        self.gckm_map = nn.Linear(seq_len, seq_len)

        # ======================================================
        # 3. Embedding
        # ======================================================
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        Attn = ProbAttention if attn == 'prob' else FullAttention

        # ======================================================
        # 4. Encoder
        # ======================================================
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(False, factor, attention_dropout=dropout,
                             output_attention=output_attention),
                        d_model, n_heads, mix=False
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            [
                ConvLayer(d_model) for _ in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # ======================================================
        # 5. Decoder
        # ======================================================
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attn(True, factor, attention_dropout=dropout,
                             output_attention=False),
                        d_model, n_heads, mix=mix
                    ),
                    AttentionLayer(
                        FullAttention(False, factor,
                                      attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads, mix=False
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    # ======================================================
    # Build Feature Adjacency Matrix
    # ======================================================
    def build_feature_adj(self, x):
        """
        Calculates the adjacency matrix based on Euclidean distance between features.
        Input: x -> [B, L, D]
        Output: adj -> [B, D, D]
        """
        # Treat each feature as a node, where its attributes are the time series values
        x_feat = x.permute(0, 2, 1)  # Shape: [B, D, L]

        # Calculate pairwise L2 distance between features
        dist = torch.cdist(x_feat, x_feat, p=2)  # Shape: [B, D, D]
        adj = torch.exp(-(dist ** 2))

        return adj

    # ======================================================
    # Forward Pass
    # ======================================================
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        x_enc: input time-series tensor of shape [B, L, D]
        """
        B, L, D = x_enc.size()

        # ------------------------------------------------------
        # Step 1: MDM (Marginal Distribution Module)
        # ------------------------------------------------------
        x_mdm = self.mdm(x_enc.permute(0, 2, 1))   # Shape: [B, D, L]
        x_enc = x_mdm.permute(0, 2, 1)             # Shape: [B, L, D]

        # ------------------------------------------------------
        # Step 2: Build Feature Adjacency Matrix
        # ------------------------------------------------------
        adj = self.build_feature_adj(x_enc)        # Shape: [B, D, D]

        # ------------------------------------------------------
        # Step 3: Feature Graph Convolution (GCKM)
        # ------------------------------------------------------
        gckm_out = []
        for b in range(B):
            # Extract node features for the current batch item: [D, L]
            x_feat = x_enc[b].permute(1, 0)        # Shape: [D, L]

            # Apply GCKM kernel and aggregate features
            K = self.gckm(adj[b], x_feat)          # Shape: [D, D]
            K_agg = torch.matmul(K, x_feat)        # Shape: [D, L]
            K_map = self.gckm_map(K_agg)           # Shape: [D, L]

            # Residual connection to preserve original features
            out = x_feat + K_map
            gckm_out.append(out.permute(1, 0))     # Shape: [L, D]

        x_enc = torch.stack(gckm_out, dim=0)       # Shape: [B, L, D]

        # ------------------------------------------------------
        # Step 4: Encoder pass
        # ------------------------------------------------------
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # ------------------------------------------------------
        # Step 5: Decoder pass
        # ------------------------------------------------------
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(
            dec_out, enc_out,
            x_mask=dec_self_mask,
            cross_mask=dec_enc_mask
        )

        # Final projection to match output dimension
        dec_out = self.projection(dec_out)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]
