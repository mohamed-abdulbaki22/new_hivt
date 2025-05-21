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
from itertools import permutations
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from tqdm import tqdm
import pickle
from av2.map.map_api import ArgoverseStaticMap
from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario, ObjectType, TrackCategory

from utils import TemporalData


class ArgoverseV2Dataset(Dataset):

    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 local_radius: float = 50) -> None:
        self._split = split
        self._local_radius = local_radius
        
        if split == 'train':
            self._directory = 'train'
        elif split == 'val':
            self._directory = 'val'
        elif split == 'test':
            self._directory = 'test'
        else:
            raise ValueError(split + ' is not valid')
        
        self.root = root
        self._raw_file_names = [f for f in os.listdir(os.path.join(self.root, self._directory, 'scenarios')) 
                               if f.endswith('.parquet')]
        self._processed_file_names = [os.path.splitext(f)[0] + '.pt' for f in self._raw_file_names]
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]
        super(ArgoverseV2Dataset, self).__init__(root, transform=transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'scenarios')

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

    @property
    def raw_paths(self) -> List[str]:
        return [os.path.join(self.raw_dir, f) for f in self._raw_file_names]

    def process(self) -> None:
        os.makedirs(self.processed_dir, exist_ok=True)
        
        static_map_path = os.path.join(self.root, "map_files")
        
        for raw_path in tqdm(self.raw_paths):
            scenario_id = os.path.splitext(os.path.basename(raw_path))[0]
            scenario = ArgoverseScenario.from_parquet(raw_path)
            
            # Load corresponding map for this scenario
            static_map = ArgoverseStaticMap.from_json(os.path.join(static_map_path, f"{scenario.city_name}.json"))
            
            kwargs = process_argoverse_v2(self._split, scenario, static_map, self._local_radius)
            data = TemporalData(**kwargs)
            torch.save(data, os.path.join(self.processed_dir, f"{scenario_id}.pt"))

    def len(self) -> int:
        return len(self._raw_file_names)

    def get(self, idx) -> Data:
        return torch.load(self.processed_paths[idx])


def process_argoverse_v2(split: str,
                      scenario: ArgoverseScenario,
                      static_map: ArgoverseStaticMap,
                      radius: float) -> Dict:
    # Extract timestamps - Argoverse 2 uses 10Hz frequency
    timestamps = scenario.timestamps_ns
    historical_steps = 50  # Default: 5 seconds of history at 10Hz
    historical_indices = range(0, historical_steps)
    historical_timestamps = timestamps[historical_indices]
    
    # Get all track IDs that appear in the historical time steps
    track_ids = []
    for track in scenario.tracks:
        if any(ts in historical_timestamps for ts in track.timestamps_ns):
            track_ids.append(track.track_id)
    
    num_nodes = len(track_ids)
    
    # Find focal agent (equivalent to AGENT in Argoverse 1)
    focal_track = None
    for track in scenario.tracks:
        if track.category == TrackCategory.FOCAL_TRACK:
            focal_track = track
            break
    
    if focal_track is None:
        raise ValueError("No focal track found in the scenario")
    
    # The reference time in Argoverse 2 is at index 49 (end of history)
    # We need both position and heading at this time
    focal_track_timestep_indices = [np.where(focal_track.timestamps_ns == ts)[0][0] 
                                   for ts in focal_track.timestamps_ns if ts in timestamps]
    
    # Find the index of the reference time in the focal track's time steps
    ref_time_index = None
    for i, ts in enumerate(focal_track.timestamps_ns):
        if ts == timestamps[49]:  # Reference time is at index 49
            ref_time_index = i
            break
    
    if ref_time_index is None:
        raise ValueError("Reference time not found in focal track's timestamps")
    
    # Get the origin (focal agent's position at reference time)
    origin = torch.tensor([
        focal_track.xy[ref_time_index][0], 
        focal_track.xy[ref_time_index][1]
    ], dtype=torch.float)
    
    # Calculate heading from the focal agent's heading at reference time
    theta = focal_track.heading[ref_time_index]
    rotate_mat = torch.tensor([
        [torch.cos(torch.tensor(theta)), -torch.sin(torch.tensor(theta))],
        [torch.sin(torch.tensor(theta)), torch.cos(torch.tensor(theta))]
    ])
    
    # Initialize tensors
    x = torch.zeros(num_nodes, 100, 2, dtype=torch.float)  # 10 second total (100 timesteps at 10Hz)
    v = torch.zeros(num_nodes, 100, 2, dtype=torch.float)  # velocity
    h = torch.zeros(num_nodes, 100, dtype=torch.float)     # heading
    edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
    padding_mask = torch.ones(num_nodes, 100, dtype=torch.bool)
    bos_mask = torch.zeros(num_nodes, 50, dtype=torch.bool)  # 5s of history at 10Hz
    rotate_angles = torch.zeros(num_nodes, dtype=torch.float)
    
    # Map track IDs to indices
    track_id_to_index = {track_id: idx for idx, track_id in enumerate(track_ids)}
    
    # Get the indices for the focal track and AV
    focal_index = track_id_to_index[focal_track.track_id]
    
    # Find AV (if available)
    av_index = -1
    for track in scenario.tracks:
        if track.object_type == ObjectType.VEHICLE and track.track_id in track_id_to_index and track.object_category == "vehicle.car.police":
            av_index = track_id_to_index[track.track_id]
            break
    
    # If no AV is found, use the focal track as a fallback
    if av_index == -1:
        av_index = focal_index
    
    # Process each track's trajectory data
    for track in scenario.tracks:
        if track.track_id not in track_id_to_index:
            continue
            
        node_idx = track_id_to_index[track.track_id]
        
        # Map track's timestamps to scenario timesteps
        timestep_indices = []
        for i, track_ts in enumerate(track.timestamps_ns):
            try:
                ts_idx = np.where(timestamps == track_ts)[0][0]
                timestep_indices.append((i, ts_idx))
            except IndexError:
                # This timestamp might be outside the scenario's time range
                pass
        
        if not timestep_indices:
            continue
            
        # Update padding mask
        for _, scenario_idx in timestep_indices:
            padding_mask[node_idx, scenario_idx] = False
            
        # If agent is not present at reference time (t=49), don't predict future
        if padding_mask[node_idx, 49]:
            padding_mask[node_idx, 50:] = True
            
        # Extract positions, velocities, and headings
        for track_idx, scenario_idx in timestep_indices:
            # Transform position to the local coordinate system
            pos = torch.tensor([track.xy[track_idx][0], track.xy[track_idx][1]], dtype=torch.float)
            local_pos = torch.matmul(pos - origin, rotate_mat)
            x[node_idx, scenario_idx] = local_pos
            
            # Extract heading (if available) and transform to local coordinate system
            if hasattr(track, 'heading') and len(track.heading) > track_idx:
                local_heading = track.heading[track_idx] - theta
                h[node_idx, scenario_idx] = local_heading
                
            # Calculate velocities if available
            if hasattr(track, 'velocity'):
                vel_global = torch.tensor([track.velocity[track_idx][0], track.velocity[track_idx][1]], dtype=torch.float)
                # Transform velocity to local coordinate system
                vel_local = torch.matmul(vel_global, rotate_mat)
                v[node_idx, scenario_idx] = vel_local
            elif track_idx > 0 and scenario_idx > 0:
                # Calculate velocity from position differences if not provided
                prev_track_idx = track_idx - 1
                prev_scenario_idx = scenario_idx - 1
                
                # Find the previous valid position
                found_prev = False
                for i in range(len(timestep_indices)):
                    if timestep_indices[i][1] == scenario_idx:
                        if i > 0:
                            prev_track_idx, prev_scenario_idx = timestep_indices[i-1]
                            found_prev = True
                        break
                
                if found_prev:
                    dt = (track.timestamps_ns[track_idx] - track.timestamps_ns[prev_track_idx]) / 1e9  # Convert to seconds
                    if dt > 0:
                        pos_diff = x[node_idx, scenario_idx] - x[node_idx, prev_scenario_idx]
                        v[node_idx, scenario_idx] = pos_diff / dt
        
        # Calculate heading from trajectory if not available
        historical_steps = [idx for _, idx in timestep_indices if idx < 50]
        if len(historical_steps) > 1:
            # Get the two most recent historical steps
            last_step = max(historical_steps)
            prev_steps = [s for s in historical_steps if s < last_step]
            if prev_steps:
                prev_step = max(prev_steps)
                heading_vector = x[node_idx, last_step] - x[node_idx, prev_step]
                rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])
        else:
            # Make no predictions for actors with less than 2 valid historical time steps
            padding_mask[node_idx, 50:] = True
    
    # Set BOS mask (beginning of sequence)
    bos_mask[:, 0] = ~padding_mask[:, 0]
    bos_mask[:, 1:50] = padding_mask[:, :49] & ~padding_mask[:, 1:50]
    
    # Clone positions before calculating displacement vectors
    positions = x.clone()
    
    # Convert absolute positions to displacement vectors
    # Future displacements (t=50 to t=99) relative to position at t=49
    x[:, 50:] = torch.where(
        (padding_mask[:, 49].unsqueeze(-1) | padding_mask[:, 50:]).unsqueeze(-1),
        torch.zeros(num_nodes, 50, 2),
        x[:, 50:] - x[:, 49].unsqueeze(-2)
    )
    
    # Historical displacements (t=1 to t=49) relative to previous timestep
    x[:, 1:50] = torch.where(
        (padding_mask[:, :49] | padding_mask[:, 1:50]).unsqueeze(-1),
        torch.zeros(num_nodes, 49, 2),
        x[:, 1:50] - x[:, :49]
    )
    
    # First timestep has no displacement
    x[:, 0] = torch.zeros(num_nodes, 2)
    
    # Get lane features at the current time step (t=49)
    node_positions_49 = torch.zeros((num_nodes, 2), dtype=torch.float)
    node_inds_49 = []
    
    for node_idx in range(num_nodes):
        if not padding_mask[node_idx, 49]:
            node_positions_49[node_idx] = positions[node_idx, 49]
            node_inds_49.append(node_idx)
    
    (lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index,
     lane_actor_vectors) = get_lane_features_v2(static_map, node_inds_49, node_positions_49, origin, rotate_mat, scenario.city_name, radius)
    
    # Set target values (y) for training/validation
    y = None if split == 'test' else x[:, 50:]
    seq_id = scenario.scenario_id
    
    node_features = torch.cat([x[:, :50], v[:, :50], h[:, :50].unsqueeze(-1)], dim=-1)  # [N, 50, 5]
    
    return {
        'x': x[:, :50],                 # [N, 50, 2] - 5s history at 10Hz
        'v': v[:, :50],                 # [N, 50, 2] - velocity
        'h': h[:, :50],                 # [N, 50] - heading
        'positions': positions,         # [N, 100, 2]
        'edge_index': edge_index,       # [2, N x N - 1]
        'y': y,                         # [N, 50, 2] - 5s future at 10Hz
        'num_nodes': num_nodes,
        'padding_mask': padding_mask,   # [N, 100]
        'bos_mask': bos_mask,           # [N, 50]
        'rotate_angles': rotate_angles, # [N]
        'lane_vectors': lane_vectors,   # [L, 2]
        'is_intersections': is_intersections,  # [L]
        'turn_directions': turn_directions,    # [L]
        'traffic_controls': traffic_controls,  # [L]
        'lane_actor_index': lane_actor_index,  # [2, E_{A-L}]
        'lane_actor_vectors': lane_actor_vectors,  # [E_{A-L}, 2]
        'seq_id': seq_id,
        'av_index': av_index,
        'agent_index': focal_index,
        'city': scenario.city_name,
        'origin': origin.unsqueeze(0),
        'theta': torch.tensor(theta),
        'node_features': node_features, # [N, 50, 5]
    }


def get_lane_features_v2(static_map: ArgoverseStaticMap,
                      node_inds: List[int],
                      node_positions: torch.Tensor,
                      origin: torch.Tensor,
                      rotate_mat: torch.Tensor,
                      city: str,
                      radius: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                              torch.Tensor]:
    lane_positions, lane_vectors, is_intersections, turn_directions, traffic_controls = [], [], [], [], []
    lane_ids = set()
    
    # Get lane IDs within radius of each node
    for node_idx in node_inds:
        node_position = node_positions[node_idx].numpy()
        # Transform back to global coordinates
        global_position = (torch.matmul(node_positions[node_idx].unsqueeze(0), 
                          rotate_mat.transpose(0, 1)) + origin).squeeze().numpy()
        
        # Query lanes within radius using Argoverse 2 API
        nearby_lane_ids = static_map.get_lane_segments_within_radius(
            global_position[0], global_position[1], radius_m=radius
        )
        lane_ids.update(nearby_lane_ids)
    
    # Transform node positions to local coordinate system
    node_positions_local = torch.matmul(node_positions - origin, rotate_mat).float()
    
    # Process each lane
    for lane_id in lane_ids:
        lane_segment = static_map.get_lane_segment_by_id(lane_id)
        
        # Get centerline coordinates
        lane_centerline = torch.tensor(lane_segment.centerline, dtype=torch.float)[:, :2]
        
        # Transform to local coordinates
        lane_centerline_local = torch.matmul(lane_centerline - origin, rotate_mat)
        
        # Calculate lane vectors
        if len(lane_centerline_local) > 1:
            lane_positions.append(lane_centerline_local[:-1])
            lane_vectors.append(lane_centerline_local[1:] - lane_centerline_local[:-1])
            
            count = len(lane_centerline_local) - 1
            
            # Check if lane is in an intersection
            is_intersection = bool(lane_segment.is_intersection)
            is_intersections.append(is_intersection * torch.ones(count, dtype=torch.uint8))
            
            # Get turn direction
            turn_direction = 0  # Default (NONE)
            if hasattr(lane_segment, 'turn_direction'):
                if lane_segment.turn_direction == "LEFT":
                    turn_direction = 1
                elif lane_segment.turn_direction == "RIGHT":
                    turn_direction = 2
            
            turn_directions.append(turn_direction * torch.ones(count, dtype=torch.uint8))
            
            # Check traffic control
            has_traffic_control = any([
                bool(lane_segment.has_traffic_control),
                bool(lane_segment.is_intersection)
            ])
            traffic_controls.append(has_traffic_control * torch.ones(count, dtype=torch.uint8))
    
    # Concatenate lane features
    if lane_positions:
        lane_positions = torch.cat(lane_positions, dim=0)
        lane_vectors = torch.cat(lane_vectors, dim=0)
        is_intersections = torch.cat(is_intersections, dim=0)
        turn_directions = torch.cat(turn_directions, dim=0)
        traffic_controls = torch.cat(traffic_controls, dim=0)
        
        # Create lane-actor connections
        node_positions_subset = node_positions_local[node_inds]
        
        lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), 
                                                         torch.arange(len(node_inds))))).t().contiguous()
        
        # For each lane-actor pair, calculate vector from actor to lane
        lane_actor_vectors = lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions_subset.repeat(lane_vectors.size(0), 1)
        
        # Filter connections based on radius
        mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < radius
        lane_actor_index = lane_actor_index[:, mask]
        lane_actor_vectors = lane_actor_vectors[mask]
        
        # Convert second row indices (local) to global node indices
        lane_actor_index[1] = torch.tensor([node_inds[idx] for idx in lane_actor_index[1]])
        
    else:
        # Handle empty lane case
        lane_vectors = torch.zeros((0, 2), dtype=torch.float)
        is_intersections = torch.zeros(0, dtype=torch.uint8)
        turn_directions = torch.zeros(0, dtype=torch.uint8)
        traffic_controls = torch.zeros(0, dtype=torch.uint8)
        lane_actor_index = torch.zeros((2, 0), dtype=torch.long)
        lane_actor_vectors = torch.zeros((0, 2), dtype=torch.float)

    return lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index, lane_actor_vectors
