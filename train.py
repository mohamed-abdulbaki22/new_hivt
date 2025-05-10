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
import argparse
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodules import ArgoverseV1DataModule
from models.hivt import HiVT

if __name__ == '__main__':
    pl.seed_everything(2022)

    parser = argparse.ArgumentParser()
    
    # Add Trainer arguments manually
    parser.add_argument('--accelerator', type=str, default='gpu' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--default_root_dir', type=str, default='lightning_logs')

    # Your custom arguments
    parser.add_argument('--root', type=str, required=True, help='Path to the dataset root directory')
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--shuffle', action='store_true', default=True)  # Use action for boolean flags
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', action='store_true', default=True)  # Use action for boolean flags
    parser.add_argument('--persistent_workers', action='store_true', default=True)  # Use action for boolean flags
    parser.add_argument('--monitor', type=str, default='val_minFDE', choices=['val_minADE', 'val_minFDE', 'val_minMR'])
    parser.add_argument('--save_top_k', type=int, default=5)
    
    # Model-specific arguments
    parser = HiVT.add_model_specific_args(parser)
    
    args = parser.parse_args()

    # Validate dataset path
    import os
    if not os.path.exists(args.root):
        raise ValueError(f"Dataset root directory '{args.root}' does not exist. Please check the path.")

    model_checkpoint = ModelCheckpoint(
        monitor=args.monitor,
        save_top_k=args.save_top_k,
        mode='min'
    )
    
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        default_root_dir=args.default_root_dir,
        callbacks=[model_checkpoint]
    )
    
    model = HiVT(**vars(args))
    # Use manual instantiation instead of from_argparse_args for better control
    datamodule = ArgoverseV1DataModule(
        root=args.root,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        local_radius=50  # Default value; adjust if needed
    )
    trainer.fit(model, datamodule)