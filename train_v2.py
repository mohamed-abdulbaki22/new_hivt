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
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodules import ArgoverseV2DataModule
from models.hivt import HiVT

"""
Training script for HiVT model on Argoverse 2 Motion Forecasting dataset.

This script adapts the original HiVT model to work with the richer features of Argoverse V2,
incorporating position, velocity, and heading information. The model maintains its
translation and rotation invariance properties while leveraging the additional data.

Example usage:
    python train_v2.py --root /path/to/argoverse2 --embed_dim 128
"""

if __name__ == '__main__':
    pl.seed_everything(2022)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--monitor', type=str, default='val_minFDE', choices=['val_minADE', 'val_minFDE', 'val_minMR'])
    parser.add_argument('--save_top_k', type=int, default=5)
    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()

    # Initialize ModelCheckpoint callback
    model_checkpoint = ModelCheckpoint(
        monitor=args.monitor,
        save_top_k=args.save_top_k,
        mode='min'
    )

    # Initialize Trainer with arguments
    trainer = pl.Trainer(
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 1,
        max_epochs=args.max_epochs,
        callbacks=[model_checkpoint]
    )

    # Initialize model and datamodule
    model = HiVT(**vars(args))
    datamodule = ArgoverseV2DataModule(
        root=args.root,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers
    )

    # Start training
    trainer.fit(model, datamodule)
