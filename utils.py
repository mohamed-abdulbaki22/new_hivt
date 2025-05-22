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
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from torch_geometric.data import Data


class TemporalData(Data):
    """Extended temporal data class with velocity and heading information for Argoverse v2."""
    
    def __init__(self, 
                 x: Optional[torch.Tensor] = None,
                 positions: Optional[torch.Tensor] = None,
                 edge_index: Optional[torch.Tensor] = None,
                 y: Optional[torch.Tensor] = None,
                 v: Optional[torch.Tensor] = None,
                 h: Optional[torch.Tensor] = None,
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
                 node_features: Optional[torch.Tensor] = None,
                 **kwargs) -> None:
        
        # Initialize all attributes as Data expects them
        super().__init__()
        
        # Core trajectory data
        self.x = x
        self.positions = positions
        self.edge_index = edge_index
        self.y = y
        self.v = v
        self.h = h
        
        # Metadata
        self.num_nodes = num_nodes
        self.seq_id = seq_id
        self.av_index = av_index
        self.agent_index = agent_index
        self.city = city
        
        # Masking and transformations
        self.padding_mask = padding_mask
        self.bos_mask = bos_mask
        self.rotate_angles = rotate_angles
        self.origin = origin
        self.theta = theta
        
        # Lane information
        self.lane_vectors = lane_vectors
        self.is_intersections = is_intersections
        self.turn_directions = turn_directions
        self.traffic_controls = traffic_controls
        self.lane_actor_index = lane_actor_index
        self.lane_actor_vectors = lane_actor_vectors
        
        # Additional features
        self.node_features = node_features
        
        # Store any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        """Define how to increment indices when batching."""
        if key == 'edge_index':
            return self.num_nodes
        elif key == 'lane_actor_index':
            return torch.tensor([[self.lane_vectors.size(0) if self.lane_vectors is not None else 0], 
                                [self.num_nodes]])
        elif key in ['av_index', 'agent_index']:
            return self.num_nodes
        else:
            return 0
    
    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        """Define concatenation dimension for batching."""
        if key in ['edge_index', 'lane_actor_index']:
            return 1
        elif key in ['seq_id', 'city']:
            return None  # Don't concatenate strings
        else:
            return 0


class DistanceDropEdge(object):
    """Drop edges based on distance threshold."""

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
    """Initialize weights for various layer types."""
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


def safe_tensor_to_device(tensor: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
    """Safely move tensor to device if it exists."""
    if tensor is not None and isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    return tensor


def validate_argoverse_data(data: TemporalData) -> bool:
    """Validate that TemporalData contains required fields for Argoverse v2."""
    required_fields = ['x', 'positions', 'edge_index', 'num_nodes']
    
    for field in required_fields:
        if not hasattr(data, field) or getattr(data, field) is None:
            return False
    
    # Check tensor shapes
    if data.x.size(0) != data.num_nodes:
        return False
    
    if data.positions.size(0) != data.num_nodes:
        return False
    
    return True
