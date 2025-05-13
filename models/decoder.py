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
        self.embed_dim = local_channels + global_channels

        # Learnable intention queries for each mode
        self.intention_queries = nn.Parameter(
            torch.randn(self.num_modes, self.embed_dim) * 0.02
        )

        # MLP to aggregate embeddings (unchanged)
        self.aggr_embed = nn.Sequential(
            nn.Linear(self.input_size + self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True)
        )

        # New MLP to process query-augmented inputs for trajectory predictions
        self.mode_mlp = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.future_steps * (4 if uncertain else 2))
        )

        # MLP for mode probabilities (unchanged)
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size + self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1)
        )
        self.apply(init_weights)

    def forward(self,
                local_embed: torch.Tensor,
                global_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute mode probabilities (unchanged)
        pi = self.pi(torch.cat((local_embed.expand(self.num_modes, *local_embed.shape),
                                global_embed), dim=-1)).squeeze(-1).t()  # [N, F]

        # Aggregate embeddings
        combined_embed = torch.cat((global_embed, local_embed.expand(self.num_modes, *global_embed.shape)), dim=-1)
        aggr_out = self.aggr_embed(combined_embed)  # [F, N, hidden_size]

        # Expand embeddings and queries for all modes and nodes
        num_nodes = local_embed.size(0)
        aggr_out = aggr_out.view(self.num_modes, num_nodes, -1)  # [F, N, hidden_size]
        combined_embed = combined_embed.view(self.num_modes, num_nodes, -1)  # [F, N, embed_dim]

        # Concatenate with intention queries
        queries = self.intention_queries.unsqueeze(1).expand(-1, num_nodes, -1)  # [F, N, embed_dim]
        query_input = torch.cat((combined_embed, queries), dim=-1)  # [F, N, embed_dim * 2]

        # Generate predictions for each mode
        mode_out = self.mode_mlp(query_input)  # [F, N, future_steps * (4 or 2)]
        if self.uncertain:
            mode_out = mode_out.view(self.num_modes, num_nodes, self.future_steps, 4)
            loc = mode_out[..., :2]  # [F, N, H, 2]
            scale = F.elu_(mode_out[..., 2:], alpha=1.0) + 1.0 + self.min_scale  # [F, N, H, 2]
            y_hat = torch.cat((loc, scale), dim=-1)  # [F, N, H, 4]
        else:
            y_hat = mode_out.view(self.num_modes, num_nodes, self.future_steps, 2)  # [F, N, H, 2]

        return y_hat, pi  # [F, N, H, 4 or 2], [N, F]
