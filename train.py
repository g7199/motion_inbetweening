import argparse
import yaml
import torch
from types import SimpleNamespace
from models.train_loop import train  # train_loop.py import 유지
import re  # 파일명 파싱 위해 추가
import os

def main():
    parser = argparse.ArgumentParser(description="Motion Diffusion Model Training")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to the configuration .yml file")
    parser.add_argument("--resume", type=str, default=None, 
                        help="Path to checkpoint file to resume training from (e.g., checkpoints/checkpoint_epoch_XX.pt)")
    parser.add_argument("--start_epoch", type=int, default=None, 
                        help="Starting epoch for resume (overrides parsed from filename if provided)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    config = SimpleNamespace(**config_dict)
    config.resume = args.resume
    
    # 파일명에서 epoch 파싱 (if resume and no manual start_epoch)
    parsed_epoch = 0
    if args.resume and args.start_epoch is None:
        filename = os.path.basename(args.resume)  # e.g., "checkpoint_epoch_60.pt"
        match = re.search(r'checkpoint_epoch_(\d+)', filename)
        if match:
            parsed_epoch = int(match.group(1))
            print(f"Parsed start_epoch from filename: {parsed_epoch}")
        else:
            print(f"Warning: Could not parse epoch from filename '{filename}'. Using start_epoch=0.")
    
    config.start_epoch = args.start_epoch if args.start_epoch is not None else parsed_epoch  # manual override 우선
    
    train(config)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()