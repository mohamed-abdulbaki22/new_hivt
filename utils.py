# Copyright (c) 2022-2025, Modified from Zikang Zhou's implementation.
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
import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


class TemporalData(torch.utils.data.Dataset):
    """Extended temporal data class with velocity and heading information."""
    
    def __init__(self, 
                 x: torch.Tensor,
                 positions: torch.Tensor,
                 edge_index: torch.Tensor,
                 y: Optional[torch.Tensor] = None,
                 v: Optional[torch.Tensor] = None,  # Added velocity
                 h: Optional[torch.Tensor] = None,  # Added heading
                 num_nodes: Optional[int] = None,
                 padding_mask: Optional[torch.Tensor] = None,
                 bos_mask: Optional[torch.Tensor] = None,
                 rotate_angles: Optional[torch.Tensor] = None,
                 lane_vectors: Optional[torch.Tensor] = None,
                 is_intersections: Optional[torch.Tensor] = None,
                 turn_directions: Optional[torch.Tensor] = None,
                 traffic_controls: Optional[torch.Tensor] = None,
                 lane_actor_index: Optional[torch.Tensor] = None,
                 lane_actor_vectors: Optional[torch.Tensor] = None,
                 seq_id: Optional[str] = None,
                 av_index: Optional[int] = None,
                 agent_index: Optional[int] = None,
                 city: Optional[str] = None,
                 origin: Optional[torch.Tensor] = None,
                 theta: Optional[torch.Tensor] = None,
                 **kwargs) -> None:
        self.x = x
        self.positions = positions
        self.edge_index = edge_index
        self.y = y
        self.v = v if v is not None else torch.zeros_like(x)  # Default zeros if not provided
        self.h = h if h is not None else torch.zeros(x.size(0), x.size(1))  # Default zeros if not provided
        self.num_nodes = num_nodes if num_nodes is not None else x.size(0)
        self.padding_mask = padding_mask
        self.bos_mask = bos_mask
        self.rotate_angles = rotate_angles
        self.lane_vectors = lane_vectors
        self.is_intersections = is_intersections
        self.turn_directions = turn_directions
        self.traffic_controls = traffic_controls
        self.lane_actor_index = lane_actor_index
        self.lane_actor_vectors = lane_actor_vectors
        self.seq_id = seq_id
        self.av_index = av_index
        self.agent_index = agent_index
        self.city = city
        self.origin = origin
        self.theta = theta
        
        # Store any additional kwargs
        for key, item in kwargs.items():
            self[key] = item

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
        
    def __len__(self):
        return self.num_nodes
    
    def __contains__(self, key):
        return hasattr(self, key)
    
    @property
    def keys(self):
        keys = [key for key in self.__dict__.keys() if not key.startswith('_') and key != 'keys']
        return keys
    
    def to(self, device):
        """Transfers all tensor attributes to the specified device."""
        for key, item in self.__dict__.items():
            if torch.is_tensor(item):
                self[key] = item.to(device)
            elif isinstance(item, list) and len(item) > 0 and torch.is_tensor(item[0]):
                self[key] = [i.to(device) for i in item]
        return self
    
    def cuda(self):
        """Transfers all tensor attributes to CUDA."""
        return self.to('cuda')
    
    def cpu(self):
        """Transfers all tensor attributes to CPU."""
        return self.to('cpu')


class DistanceDropEdge(object):

    def __init__(self, max_distance: Optional[float] = None) -> None:
        self.max_distance = max_distance

    def __call__(self,
                 edge_index: torch.Tensor,
                 edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.max_distance is None:
            return edge_index, edge_attr
        row, col = edge_index
        mask = torch.norm(edge_attr, p=2, dim=-1) < self.max_distance
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
        edge_attr = edge_attr[mask]
        return edge_index, edge_attr


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
