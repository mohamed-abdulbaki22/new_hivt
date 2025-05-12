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
import os
from itertools import permutations
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from argoverse.map_representation.map_api import ArgoverseMap
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from tqdm import tqdm

from utils import TemporalData


class ArgoverseV1Dataset(Dataset):

    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 local_radius: float = 50) -> None:
        self._split = split
        self._local_radius = local_radius
        self._url = f'https://s3.amazonaws.com/argoai-argoverse/forecasting_{split}_v1.1.tar.gz'
        if split == 'sample':
            self._directory = 'forecasting_sample'
        elif split == 'train':
            self._directory = 'train'
        elif split == 'val':
            self._directory = 'val'
        elif split == 'test':
            self._directory = 'test_obs'
        else:
            raise ValueError(split + ' is not valid')
        self.root = root
        self._raw_file_names = os.listdir(self.raw_dir)
        self._processed_file_names = [os.path.splitext(f)[0] + '.pt' for f in self._raw_file_names]
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]
        super(ArgoverseV1Dataset, self).__init__(root, transform=transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'data')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def process(self) -> None:
        am = ArgoverseMap()
        for raw_path in tqdm(self.raw_paths):
            kwargs = process_argoverse(self._split, raw_path, am, self._local_radius)
            data = TemporalData(**kwargs)
            torch.save(data, os.path.join(self.processed_dir, str(kwargs['seq_id']) + '.pt'))

    def len(self) -> int:
        return len(self._raw_file_names)

    def get(self, idx) -> Data:
        return torch.load(self.processed_paths[idx])


def process_argoverse(split: str,
                      raw_path: str,
                      am: ArgoverseMap,
                      radius: float) -> Dict:
    df = pd.read_csv(raw_path)

    # Filter out actors that are unseen during the historical time steps
    timestamps = list(np.sort(df['TIMESTAMP'].unique()))
    historical_timestamps = timestamps[:30]  # 3 seconds at 10 Hz
    historical_df = df[df['TIMESTAMP'].isin(historical_timestamps)]
    actor_ids = list(historical_df['TRACK_ID'].unique())
    df = df[df['TRACK_ID'].isin(actor_ids)]
    num_nodes = len(actor_ids)

    av_df = df[df['OBJECT_TYPE'] == 'AV']  # Corrected from .iloc
    av_index = actor_ids.index(av_df.iloc[0]['TRACK_ID'])
    agent_df = df[df['OBJECT_TYPE'] == 'AGENT']  # Corrected from .iloc
    agent_index = actor_ids.index(agent_df.iloc[0]['TRACK_ID'])
    city = df['CITY_NAME'].values[0]

    # Make the scene centered at AV at the 30th timestamp
    origin = torch.tensor([av_df.iloc[29]['X'], av_df.iloc[29]['Y']], dtype=torch.float)
    av_heading_vector = origin - torch.tensor([av_df.iloc[28]['X'], av_df.iloc[28]['Y']], dtype=torch.float)
    theta = torch.atan2(av_heading_vector[1], av_heading_vector[0])
    rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                               [torch.sin(theta), torch.cos(theta)]])

    # Initialization
    x = torch.zeros(num_nodes, 50, 2, dtype=torch.float)  # 30 historical + 20 future
    edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
    padding_mask = torch.ones(num_nodes, 50, dtype=torch.bool)
    bos_mask = torch.zeros(num_nodes, 30, dtype=torch.bool)  # 30 historical steps
    rotate_angles = torch.zeros(num_nodes, dtype=torch.float)

    for actor_id, actor_df in df.groupby('TRACK_ID'):
        node_idx = actor_ids.index(actor_id)
        node_steps = [timestamps.index(timestamp) for timestamp in actor_df['TIMESTAMP']]
        padding_mask[node_idx, node_steps] = False
        if padding_mask[node_idx, 29]:  # Make no predictions if unseen at t=29
            padding_mask[node_idx, 30:] = True
        xy = torch.from_numpy(np.stack([actor_df['X'].values, actor_df['Y'].values], axis=-1)).float()
        x[node_idx, node_steps] = torch.matmul(xy - origin, rotate_mat)
        node_historical_steps = list(filter(lambda node_step: node_step < 30, node_steps))
        if len(node_historical_steps) > 1:
            heading_vector = x[node_idx, node_historical_steps[-1]] - x[node_idx, node_historical_steps[-2]]
            rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])
        else:
            padding_mask[node_idx, 30:] = True

    # bos_mask is True if time step t is valid and t-1 is invalid
    bos_mask[:, 0] = ~padding_mask[:, 0]
    bos_mask[:, 1:30] = padding_mask[:, :29] & ~padding_mask[:, 1:30]

    positions = x.clone()
    x[:, 30:] = torch.where((padding_mask[:, 29].unsqueeze(-1) | padding_mask[:, 30:]).unsqueeze(-1),
                            torch.zeros(num_nodes, 20, 2),  # 20 future steps
                            x[:, 30:] - x[:, 29].unsqueeze(-2))
    x[:, 1:30] = torch.where((padding_mask[:, :29] | padding_mask[:, 1:30]).unsqueeze(-1),
                             torch.zeros(num_nodes, 29, 2),
                             x[:, 1:30] - x[:, :29])
    x[:, 0] = torch.zeros(num_nodes, 2)

    # Get lane features at the 30th time step
    df_29 = df[df['TIMESTAMP'] == timestamps[29]]
    node_inds_29 = [actor_ids.index(actor_id) for actor_id in df_29['TRACK_ID']]
    node_positions_29 = torch.from_numpy(np.stack([df_29['X'].values, df_29['Y'].values], axis=-1)).float()
    (lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index,
     lane_actor_vectors) = get_lane_features(am, node_inds_29, node_positions_29, origin, rotate_mat, city, radius)

    y = None if split == 'test' else x[:, 30:]  # [N, 20, 2] for train/val
    seq_id = os.path.splitext(os.path.basename(raw_path))[0]

    return {
        'x': x[:, :30],  # [N, 30, 2]
        'positions': positions,  # [N, 50, 2]
        'edge_index': edge_index,  # [2, N x N - 1]
        'y': y,  # [N, 20, 2] for train/val
        'num_nodes': num_nodes,
        'padding_mask': padding_mask,  # [N, 50]
        'bos_mask': bos_mask,  # [N, 30]
        'rotate_angles': rotate_angles,  # [N]
        'lane_vectors': lane_vectors,  # [L, 2]
        'is_intersections': is_intersections,  # [L]
        'turn_directions': turn_directions,  # [L]
        'traffic_controls': traffic_controls,  # [L]
        'lane_actor_index': lane_actor_index,  # [2, E_{A-L}]
        'lane_actor_vectors': lane_actor_vectors,  # [E_{A-L}, 2]
        'seq_id': int(seq_id),
        'av_index': av_index,
        'agent_index': agent_index,
        'city': city,
        'origin': origin.unsqueeze(0),
        'theta': theta,
    }


def get_lane_features(am: ArgoverseMap,
                      node_inds: List[int],
                      node_positions: torch.Tensor,
                      origin: torch.Tensor,
                      rotate_mat: torch.Tensor,
                      city: str,
                      radius: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                              torch.Tensor]:
    lane_positions, lane_vectors, is_intersections, turn_directions, traffic_controls = [], [], [], [], []
    lane_ids = set()
    for node_position in node_positions:
        lane_ids.update(am.get_lane_ids_in_xy_bbox(node_position[0], node_position[1], city, radius))
    node_positions = torch.matmul(node_positions - origin, rotate_mat).float()
    for lane_id in lane_ids:
        lane_centerline = torch.from_numpy(am.get_lane_segment_centerline(lane_id, city)[:, :2]).float()
        lane_centerline = torch.matmul(lane_centerline - origin, rotate_mat)
        is_intersection = am.lane_is_in_intersection(lane_id, city)
        turn_direction = am.get_lane_turn_direction(lane_id, city)
        traffic_control = am.lane_has_traffic_control_measure(lane_id, city)
        lane_positions.append(lane_centerline[:-1])
        lane_vectors.append(lane_centerline[1:] - lane_centerline[:-1])
        count = len(lane_centerline) - 1
        is_intersections.append(is_intersection * torch.ones(count, dtype=torch.uint8))
        if turn_direction == 'NONE':
            turn_direction = 0
        elif turn_direction == 'LEFT':
            turn_direction = 1
        elif turn_direction == 'RIGHT':
            turn_direction = 2
        else:
            raise ValueError('turn direction is not valid')
        turn_directions.append(turn_direction * torch.ones(count, dtype=torch.uint8))
        traffic_controls.append(traffic_control * torch.ones(count, dtype=torch.uint8))
    lane_positions = torch.cat(lane_positions, dim=0)
    lane_vectors = torch.cat(lane_vectors, dim=0)
    is_intersections = torch.cat(is_intersections, dim=0)
    turn_directions = torch.cat(turn_directions, dim=0)
    traffic_controls = torch.cat(traffic_controls, dim=0)

    lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), node_inds))).t().contiguous()
    lane_actor_vectors = \
        lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_vectors.size(0), 1)
    mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < radius
    lane_actor_index = lane_actor_index[:, mask]
    lane_actor_vectors = lane_actor_vectors[mask]

    return lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index, lane_actor_vectors
