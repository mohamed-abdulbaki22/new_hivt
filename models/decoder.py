# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_weights


class GRUDecoder(nn.Module):

    def __init__(self,
                 local_channels: int,
                 global_channels: int,
                 future_steps: int,
                 num_modes: int,
                 uncertain: bool = True,
                 min_scale: float = 1e-3) -> None:
        super(GRUDecoder, self).__init__()
        self.input_size = global_channels
        self.hidden_size = local_channels
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.uncertain = uncertain
        self.min_scale = min_scale

        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          bias=True,
                          batch_first=False,
                          dropout=0,
                          bidirectional=False)
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))
        if uncertain:
            self.scale = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, 2))
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size + self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1))
        self.apply(init_weights)

    def forward(self,
                local_embed: torch.Tensor,
                global_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pi = self.pi(torch.cat((local_embed.expand(self.num_modes, *local_embed.shape),
                                global_embed), dim=-1)).squeeze(-1).t()
        global_embed = global_embed.reshape(-1, self.input_size)  # [F x N, D]
        global_embed = global_embed.expand(self.future_steps, *global_embed.shape)  # [H, F x N, D]
        local_embed = local_embed.repeat(self.num_modes, 1).unsqueeze(0)  # [1, F x N, D]
        out, _ = self.gru(global_embed, local_embed)
        out = out.transpose(0, 1)  # [F x N, H, D]
        loc = self.loc(out)  # [F x N, H, 2]
        if self.uncertain:
            scale = F.elu_(self.scale(out), alpha=1.0) + 1.0 + self.min_scale  # [F x N, H, 2]
            return torch.cat((loc, scale),
                             dim=-1).view(self.num_modes, -1, self.future_steps, 4), pi  # [F, N, H, 4], [N, F]
        else:
            return loc.view(self.num_modes, -1, self.future_steps, 2), pi  # [F, N, H, 2], [N, F]


class MLPDecoder(nn.Module):

    def __init__(self,
                 local_channels: int,
                 global_channels: int,
                 future_steps: int,
                 num_modes: int,
                 uncertain: bool = True,
                 min_scale: float = 1e-3) -> None:
        super(MLPDecoder, self).__init__()
        self.input_size = global_channels
        self.hidden_size = local_channels
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.uncertain = uncertain
        self.min_scale = min_scale

        self.aggr_embed = nn.Sequential(
            nn.Linear(self.input_size + self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True))
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.future_steps * 2))
        if uncertain:
            self.scale = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.future_steps * 2))
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size + self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1))
        self.apply(init_weights)

    def forward(self,
                local_embed: torch.Tensor,
                global_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pi = self.pi(torch.cat((local_embed.expand(self.num_modes, *local_embed.shape),
                                global_embed), dim=-1)).squeeze(-1).t()
        out = self.aggr_embed(torch.cat((global_embed, local_embed.expand(self.num_modes, *local_embed.shape)), dim=-1))
        loc = self.loc(out).view(self.num_modes, -1, self.future_steps, 2)  # [F, N, H, 2]
        if self.uncertain:
            scale = F.elu_(self.scale(out), alpha=1.0).view(self.num_modes, -1, self.future_steps, 2) + 1.0
            scale = scale + self.min_scale  # [F, N, H, 2]
            return torch.cat((loc, scale), dim=-1), pi  # [F, N, H, 4], [N, F]
        else:
            return loc, pi  # [F, N, H, 2], [N, F]


class TransformerDecoder(nn.Module):
    def __init__(self,
                 local_channels: int,
                 global_channels: int,
                 future_steps: int,
                 num_modes: int,
                 uncertain: bool = True,
                 min_scale: float = 1e-3) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.uncertain = uncertain
        self.min_scale = min_scale
        self.embed_dim = local_channels  # Assume local and global channels are equal

        # Mode embeddings and positional encoding
        self.mode_queries = nn.Parameter(torch.randn(num_modes, self.embed_dim))
        self.pos_encoder = PositionalEncoding(self.embed_dim, max_len=future_steps)
        
        # Context fusion
        self.context_fusion = nn.Sequential(
            nn.Linear(self.embed_dim*2, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(inplace=True)
        )

        # Transformer layers
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim, nhead=8, dim_feedforward=512, batch_first=False)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=3)

        # Output heads
        self.loc_head = nn.Linear(self.embed_dim, 2)
        if uncertain:
            self.scale_head = nn.Sequential(
                nn.Linear(self.embed_dim, 64),
                nn.LayerNorm(64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 2),
                nn.ELU(inplace=True)
            )

        # Pi network for mode probabilities
        self.pi = nn.Sequential(
            nn.Linear(2 * self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, 1)
        )
        self.apply(init_weights)

    def forward(self, local_embed: torch.Tensor, global_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Combine local (N, D) and global (F, N, D) embeddings
        local_expanded = local_embed.unsqueeze(0).expand(self.num_modes, -1, -1)  # [F, N, D]
        
        # Better context fusion
        context = self.context_fusion(torch.cat([global_embed, local_expanded], dim=-1))  # [F, N, D]

        # Prepare queries with positional encoding
        # Shape: mode queries [F, D] -> [F, H, D] where H is future_steps
        queries = self.mode_queries.unsqueeze(1).expand(-1, self.future_steps, -1)  # [F, H, D]
        
        # Reshape for transformer: [seq_len, batch, embed_dim]
        queries = queries.permute(1, 0, 2)  # [H, F, D]
        
        # Add positional encoding to queries
        queries = self.pos_encoder(queries)  # [H, F, D]
        
        # Prepare memory: [seq_len, batch, embed_dim]
        memory = context.permute(2, 0, 1)  # [D, F, N]
        memory = memory.reshape(self.embed_dim, -1).transpose(0, 1)  # [F*N, D]
        memory = memory.unsqueeze(0).expand(self.future_steps, -1, -1)  # [H, F*N, D]
        
        # Run transformer decoder
        # queries: [H, F, D], memory: [H, F*N, D]
        decoded = self.decoder(queries, memory)  # [H, F, D]
        
        # Reshape back to [F, H, N, D]
        decoded = decoded.permute(1, 0, 2)  # [F, H, D]
        decoded = decoded.unsqueeze(2).expand(-1, -1, local_embed.size(0), -1)  # [F, H, N, D]

        # Predict locations and scales
        loc = self.loc_head(decoded)  # [F, H, N, 2]
        if self.uncertain:
            scale = self.scale_head(decoded) + self.min_scale
            output = torch.cat([loc, scale], dim=-1)  # [F, H, N, 4]
        else:
            output = loc

        # Compute pi (mode probabilities)
        pi_input = torch.cat([local_embed.unsqueeze(0).expand(self.num_modes, -1, -1), global_embed], dim=-1)
        pi = self.pi(pi_input).squeeze(-1).t()  # [N, F]

        return output.permute(0, 2, 1, 3), pi  # [F, N, H, 4], [N, F]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]