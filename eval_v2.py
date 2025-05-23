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
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import numpy as np
np.bool = np.bool_
from torch_geometric.data import DataLoader

from datasets import ArgoverseV2Dataset
from models.hivt import HiVT

"""
Evaluation script for HiVT model on Argoverse 2 Motion Forecasting dataset.

This script evaluates a trained HiVT model on the Argoverse V2 validation set,
reporting metrics such as minADE, minFDE, and MR. The model utilizes enhanced
features including position, velocity, and heading information.

Example usage:
    python eval_v2.py --root /path/to/argoverse2 --ckpt_path /path/to/checkpoint.ckpt
"""

if __name__ == '__main__':
    pl.seed_everything(2022)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, required=True)
    args = parser.parse_args()

    trainer = pl.Trainer(
        accelerator='auto',
        devices=1 if torch.cuda.is_available() else 0,
        enable_progress_bar=True,
        logger=False
    )

    model = HiVT.load_from_checkpoint(
        checkpoint_path=args.ckpt_path,
        map_location='cuda' if torch.cuda.is_available() else 'cpu',
        strict=False
    )
    val_dataset = ArgoverseV2Dataset(root=args.root, split='val', local_radius=model.hparams.local_radius)
    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                          pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
    trainer.validate(model, dataloader)
