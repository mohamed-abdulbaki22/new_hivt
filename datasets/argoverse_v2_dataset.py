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
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from tqdm import tqdm
import pickle
from av2.map.map_api import ArgoverseStaticMap
from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario, ObjectType, TrackCategory
from av2.datasets.motion_forecasting import scenario_serialization

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
            self._directory = 'scenarios'  # Matches your structure
        elif split == 'val':
            self._directory = 'scenarios'
        elif split == 'test':
            self._directory = 'test'
        else:
            raise ValueError(f"{split} is not valid")
        
        self.root = root
        self._raw_file_names = []
        base_dir = os.path.join(self.root, "dataset", "train", self._directory)
        if os.path.exists(base_dir):
            for scenario_dir in os.listdir(base_dir):
                scenario_path = os.path.join(base_dir, scenario_dir)
                if os.path.isdir(scenario_path):
                    for f in os.listdir(scenario_path):
                        if f.endswith('.parquet'):
                            self._raw_file_names.append(os.path.join(scenario_dir, f))
        else:
            raise FileNotFoundError(f"Directory not found: {base_dir}")
        self._processed_file_names = [os.path.splitext(os.path.basename(f))[0] + '.pt' for f in self._raw_file_names]
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]
        super(ArgoverseV2Dataset, self).__init__(root, transform=transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, "dataset", "train", self._directory)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, "dataset", "train", self._directory, 'processed')

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
        
        for raw_path in tqdm(self.raw_paths):
            scenario_id = os.path.splitext(os.path.basename(raw_path))[0]
            df = pd.read_parquet(raw_path)
            
            # Extract unique timestamps (timesteps)
            timesteps = df["timestep"].unique()
            timesteps.sort()
            
            # Calculate timestamps_ns assuming 10Hz (100ms = 10^8 ns per step)
            start_timestamp = df["start_timestamp"].iloc[0]
            timestamps_ns = start_timestamp + (timesteps * 1e8).astype(np.int64)
            
            # Create tracks by grouping by track_id
            tracks = []
            for track_id, track_df in df.groupby("track_id"):
                track_data = {
                    "track_id": track_id,
                    "object_type": track_df["object_type"].iloc[0],
                    "category": TrackCategory.FOCAL_TRACK if track_id == df["focal_track_id"].iloc[0] else TrackCategory.SCORED_TRACK,
                    "timestamps_ns": track_df["timestep"].map(lambda x: timestamps_ns[timesteps.tolist().index(x)]).to_numpy(),
                    "xy": track_df[["position_x", "position_y"]].to_numpy(),
                    "heading": track_df["heading"].to_numpy(),
                    "velocity": track_df[["velocity_x", "velocity_y"]].to_numpy()
                }
                tracks.append(track_data)
            
            # Create ArgoverseScenario object
            city_name = df["city"].iloc[0].lower()  # Ensure lowercase for consistency
            scenario = ArgoverseScenario(
                scenario_id=scenario_id,
                timestamps_ns=timestamps_ns,  # Use the calculated timestamps_ns
                tracks=tracks,
                city_name=city_name,
                focal_track_id=df["focal_track_id"].iloc[0],
                map_id=city_name,  # Placeholder, overridden below
                slice_id=scenario_id
            )
            
            # Load corresponding map for this scenario
            scenario_dir = os.path.dirname(raw_path)
            map_files = [f for f in os.listdir(scenario_dir) if f.startswith('log_map_archive_') and f.endswith('.json')]
            if not map_files:
                raise FileNotFoundError(f"No map file found in {scenario_dir}. Available files: {os.listdir(scenario_dir)}")
            map_file_name = map_files[0]  # Take the first map file (should be unique per scenario)
            map_file = Path(scenario_dir) / map_file_name
            static_map = ArgoverseStaticMap.from_json(map_file)
            
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
    historical_steps = 50  # 5 seconds of history at 10Hz
    future_steps = 60  # 6 seconds of future at 10Hz
    total_steps = historical_steps + future_steps  # 110 total timesteps
    historical_indices = range(0, historical_steps)
    historical_timestamps = timestamps[historical_indices]
    
    # Ensure these are Python integers for slicing
    historical_steps = int(historical_steps)
    future_steps = int(future_steps)
    total_steps = int(total_steps)
    
    # Get all track IDs that appear in the historical time steps
    track_ids = []
    for track in scenario.tracks:
        if any(ts in historical_timestamps for ts in track["timestamps_ns"]):
            track_ids.append(track["track_id"])
    
    num_nodes = len(track_ids)
    
    # Find focal agent
    focal_track = None
    for track in scenario.tracks:
        if track["category"] == TrackCategory.FOCAL_TRACK:
            focal_track = track
            break
    
    if focal_track is None:
        raise ValueError("No focal track found in the scenario")
    
    # Find reference time (t=49) - Fixed the timestamp comparison
    ref_time_index = None
    ref_timestamp = timestamps[49]  # Get the actual timestamp for timestep 49
    
    for i, ts in enumerate(focal_track["timestamps_ns"]):
        if ts == ref_timestamp:
            ref_time_index = i
            break
    
    if ref_time_index is None:
        # If exact match not found, try to find the closest timestamp to timestep 49
        focal_timestamps = focal_track["timestamps_ns"]
        
        # Find the index in focal track that corresponds to timestep 49
        # This assumes the focal track has data at timestep 49
        timestep_to_focal_index = {}
        for i, focal_ts in enumerate(focal_timestamps):
            # Find which global timestep this corresponds to
            try:
                global_idx = np.where(timestamps == focal_ts)[0][0]
                timestep_to_focal_index[global_idx] = i
            except IndexError:
                continue
        
        if 49 in timestep_to_focal_index:
            ref_time_index = timestep_to_focal_index[49]
        else:
            # If timestep 49 is not available, use the last available timestep in history
            available_historical_steps = [step for step in range(50) if step in timestep_to_focal_index]
            if available_historical_steps:
                last_historical_step = max(available_historical_steps)
                ref_time_index = timestep_to_focal_index[last_historical_step]
                print(f"Warning: Using timestep {last_historical_step} instead of 49 for reference time")
            else:
                raise ValueError("No valid reference time found in focal track's historical timestamps")
    
    # Get origin and rotation
    origin = torch.tensor([
        focal_track["xy"][ref_time_index][0], 
        focal_track["xy"][ref_time_index][1]
    ], dtype=torch.float)
    
    theta = focal_track["heading"][ref_time_index]
    
    # Define track_id_to_index before using it for rotate_mat
    # This maps each track_id to its corresponding node index in track_ids
    track_id_to_index = {track_id: idx for idx, track_id in enumerate(track_ids)}
    
    # Compute rotation matrices for each node based on their heading at the reference time
    # Shape: [num_nodes, 2, 2], where each node has its own 2x2 rotation matrix
    rotate_mat = torch.stack([torch.tensor([
        [torch.cos(torch.tensor(track["heading"][ref_time_index] if ref_time_index < len(track["heading"]) else 0)),
         -torch.sin(torch.tensor(track["heading"][ref_time_index] if ref_time_index < len(track["heading"]) else 0))],
        [torch.sin(torch.tensor(track["heading"][ref_time_index] if ref_time_index < len(track["heading"]) else 0)),
         torch.cos(torch.tensor(track["heading"][ref_time_index] if ref_time_index < len(track["heading"]) else 0))]
    ], dtype=torch.float) for track in scenario.tracks if track["track_id"] in track_id_to_index], dim=0)
    
    # Initialize tensors - use total_steps for the time dimension
    x = torch.zeros(num_nodes, total_steps, 2, dtype=torch.float)
    v = torch.zeros(num_nodes, total_steps, 2, dtype=torch.float)
    h = torch.zeros(num_nodes, total_steps, dtype=torch.float)
    edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
    padding_mask = torch.ones(num_nodes, total_steps, dtype=torch.bool)
    bos_mask = torch.zeros(num_nodes, 50, dtype=torch.bool)
    rotate_angles = torch.zeros(num_nodes, dtype=torch.float)
    
    # Create mapping from track_id to node index
    track_id_to_node_idx = {track_id: idx for idx, track_id in enumerate(track_ids)}
    focal_index = track_id_to_node_idx[focal_track["track_id"]]
    
    av_index = -1
    for track in scenario.tracks:
        if track["object_type"] == "vehicle" and track["track_id"] in track_id_to_node_idx:
            av_index = track_id_to_node_idx[track["track_id"]]
            break
    
    if av_index == -1:
        av_index = focal_index
    
    # Process tracks
    for track in scenario.tracks:
        if track["track_id"] not in track_id_to_node_idx:
            continue
        
        node_idx = track_id_to_node_idx[track["track_id"]]
        timestep_indices = []
        
        for i, track_ts in enumerate(track["timestamps_ns"]):
            try:
                ts_idx = np.where(timestamps == track_ts)[0][0]
                timestep_indices.append((i, ts_idx))
            except IndexError:
                pass
        
        if not timestep_indices:
            continue
        
        # Get the rotation matrix specific to this node
        # Shape: [2, 2], used to rotate this node's position and velocity
        node_rotate_mat = rotate_mat[node_idx]
        
        for track_idx, scenario_idx in timestep_indices:
            if scenario_idx >= total_steps:  # Skip if timestep is beyond our allocated size
                continue
            padding_mask[node_idx, scenario_idx] = False
            pos = torch.tensor(track["xy"][track_idx], dtype=torch.float)
            
            # Transform position to local coordinates using the node's specific rotation matrix
            # pos - origin: [2], node_rotate_mat: [2, 2], result: [2]
            local_pos = torch.matmul(pos - origin, node_rotate_mat)
            x[node_idx, scenario_idx] = local_pos
            
            # Compute heading relative to the focal agent's heading
            h[node_idx, scenario_idx] = track["heading"][track_idx] - theta
            
            # Transform velocity to local coordinates using the node's specific rotation matrix
            # velocity: [2], node_rotate_mat: [2, 2], result: [2]
            v[node_idx, scenario_idx] = torch.matmul(
                torch.tensor(track["velocity"][track_idx], dtype=torch.float),
                node_rotate_mat
            )
    
        if padding_mask[node_idx, historical_steps-1]:
            padding_mask[node_idx, historical_steps:] = True
    
        historical_steps_list = [idx for _, idx in timestep_indices if idx < historical_steps]
        if len(historical_steps_list) > 1:
            last_step = max(historical_steps_list)
            prev_steps = [s for s in historical_steps_list if s < last_step]
            if prev_steps:
                prev_step = max(prev_steps)
                heading_vector = x[node_idx, last_step] - x[node_idx, prev_step]
                rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])
        else:
            padding_mask[node_idx, historical_steps:] = True
    
    bos_mask[:, 0] = ~padding_mask[:, 0]
    bos_mask[:, 1:historical_steps] = padding_mask[:, :historical_steps-1] & ~padding_mask[:, 1:historical_steps]
    
    positions = x.clone()
    x[:, historical_steps:] = torch.where(
        (padding_mask[:, historical_steps-1].unsqueeze(-1) | padding_mask[:, historical_steps:]).unsqueeze(-1),
        torch.zeros(num_nodes, future_steps, 2),
        x[:, historical_steps:] - x[:, historical_steps-1].unsqueeze(-2)
    )
    x[:, 1:historical_steps] = torch.where(
        (padding_mask[:, :historical_steps-1] | padding_mask[:, 1:historical_steps]).unsqueeze(-1),
        torch.zeros(num_nodes, historical_steps-1, 2),
        x[:, 1:historical_steps] - x[:, :historical_steps-1]
    )
    x[:, 0] = torch.zeros(num_nodes, 2)
    
    node_positions_49 = torch.zeros((num_nodes, 2), dtype=torch.float)
    node_inds_49 = []
    for node_idx in range(num_nodes):
        if not padding_mask[node_idx, historical_steps-1]:
            node_positions_49[node_idx] = positions[node_idx, historical_steps-1]
            node_inds_49.append(node_idx)
    
    (lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index,
     lane_actor_vectors) = get_lane_features_v2(static_map, node_inds_49, node_positions_49, origin, rotate_mat, scenario.city_name, radius)
    
    # Define y variable before using it
    y = None if split == 'test' else x[:, historical_steps:]
    seq_id = scenario.scenario_id
    node_features = torch.cat([x[:, :historical_steps], v[:, :historical_steps], h[:, :historical_steps].unsqueeze(-1)], dim=-1)
    
    return {
        'x': x[:, :historical_steps],
        'v': v[:, :historical_steps],
        'h': h[:, :historical_steps],
        'positions': positions,
        'edge_index': edge_index,
        'y': y,
        'num_nodes': num_nodes,
        'padding_mask': padding_mask,
        'bos_mask': bos_mask,
        'rotate_angles': rotate_angles,
        'lane_vectors': lane_vectors,
        'is_intersections': is_intersections,
        'turn_directions': turn_directions,
        'traffic_controls': traffic_controls,
        'lane_actor_index': lane_actor_index,
        'lane_actor_vectors': lane_actor_vectors,
        'seq_id': seq_id,
        'av_index': av_index,
        'agent_index': focal_index,
        'city': scenario.city_name,
        'origin': origin.unsqueeze(0),
        'theta': torch.tensor(theta),
        'node_features': node_features,
        'rotate_mat': rotate_mat,  # Shape: [num_nodes, 2, 2]
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
    
    # Get all lane segments from the static map
    try:
        # Try different ways to access lane segments in Argoverse 2
        if hasattr(static_map, 'vector_lane_segments'):
            all_lane_segments = static_map.vector_lane_segments
        elif hasattr(static_map, 'lane_segments'):
            all_lane_segments = static_map.lane_segments
        else:
            # Fallback: try to get lane segment IDs and build dictionary
            all_lane_segments = {}
            if hasattr(static_map, 'get_lane_segment_ids'):
                lane_ids_list = static_map.get_lane_segment_ids()
                for lane_id in lane_ids_list:
                    try:
                        lane_segment = static_map.get_lane_segment_by_id(lane_id)
                        all_lane_segments[lane_id] = lane_segment
                    except:
                        continue
    except Exception as e:
        print(f"Warning: Could not access lane segments: {e}")
        all_lane_segments = {}
    
    # Filter lanes within radius of each node
    for node_idx in node_inds:
        # Transform back to global coordinates
        global_position = (torch.matmul(node_positions[node_idx].unsqueeze(0), 
                          rotate_mat.transpose(0, 1)) + origin).squeeze().numpy()
        
        # Check each lane segment
        for lane_id, lane_segment in all_lane_segments.items():
            try:
                # Get centerline coordinates
                if hasattr(lane_segment, 'centerline'):
                    lane_centerline = np.array(lane_segment.centerline)[:, :2]
                elif hasattr(lane_segment, 'polygon'):
                    # Some versions might store centerline differently
                    lane_centerline = np.array(lane_segment.polygon.exterior.coords)[:, :2]
                else:
                    continue
                
                # Calculate distances from node to lane centerline
                distances = np.linalg.norm(lane_centerline - global_position, axis=1)
                if np.min(distances) <= radius:
                    lane_ids.add(lane_id)
            except Exception as e:
                continue
    
    # Transform node positions to local coordinate system
    node_positions_local = torch.matmul(node_positions - origin.unsqueeze(0), rotate_mat).float()
    
    # Process each lane
    for lane_id in lane_ids:
        try:
            if lane_id in all_lane_segments:
                lane_segment = all_lane_segments[lane_id]
            else:
                lane_segment = static_map.get_lane_segment_by_id(lane_id)
            
            # Get centerline coordinates
            if hasattr(lane_segment, 'centerline'):
                lane_centerline = torch.tensor(lane_segment.centerline, dtype=torch.float)[:, :2]
            else:
                continue
            
            # Transform to local coordinates
            lane_centerline_local = torch.matmul(lane_centerline - origin.unsqueeze(0), rotate_mat)
            
            # Calculate lane vectors
            if len(lane_centerline_local) > 1:
                lane_positions.append(lane_centerline_local[:-1])
                lane_vectors.append(lane_centerline_local[1:] - lane_centerline_local[:-1])
                
                count = len(lane_centerline_local) - 1
                
                # Check if lane is in an intersection
                is_intersection = bool(getattr(lane_segment, 'is_intersection', False))
                is_intersections.append(is_intersection * torch.ones(count, dtype=torch.uint8))
                
                # Get turn direction
                turn_direction = 0  # Default (NONE)
                if hasattr(lane_segment, 'turn_direction') and lane_segment.turn_direction:
                    turn_dir_str = str(lane_segment.turn_direction).upper()
                    if "LEFT" in turn_dir_str:
                        turn_direction = 1
                    elif "RIGHT" in turn_dir_str:
                        turn_direction = 2
                
                turn_directions.append(turn_direction * torch.ones(count, dtype=torch.uint8))
                
                # Check traffic control
                has_traffic_control = any([
                    bool(getattr(lane_segment, 'has_traffic_control', False)),
                    bool(getattr(lane_segment, 'is_intersection', False))
                ])
                traffic_controls.append(has_traffic_control * torch.ones(count, dtype=torch.uint8))
        except Exception as e:
            continue
    
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
