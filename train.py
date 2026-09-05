# Import necessary libraries
# RegTrainer: Custom regression trainer class
from utils.regression_trainer import RegTrainer
import argparse
import os

# Fix OMP_NUM_THREADS error
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import numpy as np
import random
import copy
import logging

def setup_seed(seed):
    """
    Fix random seeds for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clear_logger():
    """Clear existing logging handlers to prevent duplicate logs."""
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
args = None

def parse_args():
    """
    Parse command line arguments for the training script.
    """
    parser = argparse.ArgumentParser(description='Train')

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # ==================== Model Parameters ====================
    parser.add_argument('--model-name', default='vgg19_fpn_ppa_base', 
                        help='Model name: vgg19_fpn_ppa_base (base)')
    parser.add_argument('--data-dir', default='maize_tassel',
                        help='Path to dataset directory')
    parser.add_argument('--save-dir', default='model',
                        help='Directory to save trained models')
    parser.add_argument('--info', default='DUET_50',
                        help='Dataset category reference info')
    parser.add_argument('--save-all', type=bool, default=False,
                        help='Save all best models if True, else only the best one')

    # ==================== Randomness Control ====================
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for reproducibility')

    # ==================== Optimizer Parameters ====================
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay coefficient')

    # ==================== Training Control ====================
    parser.add_argument('--resume', default='',
                        help='Path to resume model checkpoint')
    parser.add_argument('--max-model-num', type=int, default=1,
                        help='Maximum number of saved models')
    parser.add_argument('--max-epoch', type=int, default=200,
                        help='Maximum training epochs')
    parser.add_argument('--val-epoch', type=int, default=5,
                        help='Epoch interval for validation')
    parser.add_argument('--ema-decay', type=float, default=0.999,
                        help='EMA decay rate (alpha)')
    parser.add_argument('--val-start', type=int, default=10,
                        help='Epoch to start validation')
    parser.add_argument('--unlabel-start', type=int, default=30,
                        help='Epoch to start semi-supervised learning with unlabeled data')
    parser.add_argument('--uhhm-start', type=int, default=25,
                        help='Epoch to start UHHM (Uncertainty-aware Hybrid Hidden Mapping)')

    parser.add_argument('--drf-growth-start', type=int, default=150,
                        help='Epoch to start DRF linear growth. Default 150.')

    # ==================== Data Loading ====================
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--device', default='0', 
                        help='GPU device ID')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    # ==================== Image Processing ====================
    parser.add_argument('--is-gray', type=bool, default=False,
                        help='True for grayscale input, False for RGB')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='Crop size for data augmentation')
    parser.add_argument('--downsample-ratio', type=int, default=8,
                        help='Downsample ratio for density map generation')
    parser.add_argument('--sigma', type=float, default=15.0,
                        help='Gaussian kernel standard deviation for density estimation')

    # ==================== Loss Weights ====================
    parser.add_argument('--unsup-weight', type=float, default=10,
                        help='Weight for unsupervised loss')
    parser.add_argument('--use-dynamic-unsup-weight', type=str2bool, default=True,
                        help='Enable dynamic unsupervised loss weight if True')

    # ==================== Threshold Strategies ====================
    parser.add_argument('--uhhm-thresh-mode', type=str, default='mean',
                        choices=['mean', 'mean+0.5std', 'mean-0.5std'],
                        help='UHHM threshold calculation mode')
    parser.add_argument('--drf-thresh-mode', type=str, default='mean',
                        choices=['mean', 'mean+0.5std', 'mean-0.5std'],
                        help='DRF threshold calculation mode')

    # ==================== Uncertainty Components ====================
    parser.add_argument('--uhhm-use-error', type=str2bool, default=True,
                        help='Use prediction error in UHHM if True')
    
    parser.add_argument('--drf-use-cls-unc', type=str2bool, default=True,
                        help='Use classification entropy in DRF if True')

    # ==================== Ablation Study ====================
    parser.add_argument('--use-ppa', type=str2bool, default=True,
                        help='Use PPA module in feature extraction')
    parser.add_argument('--use-cls-head', type=str2bool, default=True,
                        help='Use Classification Branch ')
    parser.add_argument('--use-unc-head', type=str2bool, default=True,
                        help='Use Uncertainty Branch . Disabling this disables UHHM/DRF.')

    # ==================== Automated Experiments ====================
    parser.add_argument('--auto-ablation-drf', type=str2bool, default=False,
                        help='Enable automated DRF growth strategy ablation experiment.')
    parser.add_argument('--stop-epoch', type=int, default=0,
                        help='Epoch to stop training early. 0 means no early stop.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    Main training script:
    1. Parse arguments
    2. Setup CUDA environment
    3. Initialize trainer and start training
    """
    # 1. Parse arguments
    args = parse_args()

    # 2. Fix random seed
    setup_seed(args.seed)

    # 3. Logging setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Random Seed set to: {args.seed}")

    # CUDA setup
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()

    # 4. Check for auto ablation experiment
    if args.auto_ablation_drf:
        print("="*80)
        print("   Starting Strict Auto Ablation Experiment for DRF Growth Strategy")
        print("   Strategy: Cascade Training to ensure strict consistency")
        print("   Checkpoints: [100, 125, 150, 175] (Growth OFF)")
        print("   Branches: Growth Start at [100, 125, 150, 175] -> 200")
        print("   Baseline: Growth OFF -> 200")
        print("="*80)

        original_args = copy.deepcopy(args)
        original_save_dir = args.save_dir

        checkpoints = [100, 125, 150, 175, 200]
        
        # Current backbone checkpoint and start epoch
        current_base_ckpt = None
        current_start_epoch = 0
        
        # Store checkpoints for each stage
        stage_checkpoints = {}

        # ==============================================================================
        # Main Backbone Training: Growth OFF
        # ==============================================================================
        
        for target_epoch in checkpoints:
            print(f"\n>>> [Backbone] Training Base Model (Growth OFF): Epoch {current_start_epoch} -> {target_epoch}...")

            base_stage_name = f'ablation_base_{current_start_epoch}_{target_epoch}'
            base_stage_dir = os.path.join(original_save_dir, base_stage_name)

            # Check for existing checkpoint
            found_ckpt = None
            if os.path.exists(base_stage_dir):
                subdirs = [os.path.join(base_stage_dir, d) for d in os.listdir(base_stage_dir) if os.path.isdir(os.path.join(base_stage_dir, d))]
                subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                
                for d in subdirs:
                    potential_ckpt = os.path.join(d, f'{target_epoch - 1}_ckpt.tar')
                    if os.path.exists(potential_ckpt):
                        found_ckpt = potential_ckpt
                        print(f"Found existing backbone checkpoint: {found_ckpt}")
                        break
            
            if found_ckpt:
                current_base_ckpt = found_ckpt
                stage_checkpoints[target_epoch] = found_ckpt
                current_start_epoch = target_epoch
                continue 

            # Configure training parameters
            clear_logger()
            args_base = copy.deepcopy(original_args)
            args_base.save_dir = base_stage_dir
            args_base.max_epoch = 200
            args_base.stop_epoch = target_epoch
            args_base.drf_growth_start = 999 

            if current_base_ckpt:
                args_base.resume = current_base_ckpt

            # Train
            trainer_base = RegTrainer(args_base)
            actual_save_dir = trainer_base.save_dir
            print(f"Backbone saving to: {actual_save_dir}")

            trainer_base.setup()
            trainer_base.train()

            # Get trained checkpoint
            ckpt_filename = f'{target_epoch - 1}_ckpt.tar'
            new_ckpt_path = os.path.join(actual_save_dir, ckpt_filename)

            if not os.path.exists(new_ckpt_path):
                files = os.listdir(actual_save_dir)
                ckpts = [f for f in files if f.endswith('_ckpt.tar')]
                if ckpts:
                    ckpts.sort(key=lambda x: int(x.split('_')[0]))
                    new_ckpt_path = os.path.join(actual_save_dir, ckpts[-1])

            print(f"Backbone stage finished. Checkpoint: {new_ckpt_path}")

            current_base_ckpt = new_ckpt_path
            stage_checkpoints[target_epoch] = new_ckpt_path
            current_start_epoch = target_epoch

            del trainer_base
            torch.cuda.empty_cache()

        # ==============================================================================
        # Branch Training: Growth ON
        # ==============================================================================

        branch_points = [100, 125, 150, 175]

        for start_val in branch_points:
            print(f"\n>>> [Branch] Training Branch: Growth Start = {start_val} (Epoch {start_val}->200)...")

            setup_seed(args.seed)

            if start_val not in stage_checkpoints:
                print(f"Error: Missing backbone checkpoint for epoch {start_val}. Skipping this branch.")
                continue

            base_ckpt = stage_checkpoints[start_val]

            branch_name = f'ablation_growth_{start_val}'
            branch_dir = os.path.join(original_save_dir, branch_name)

            skip_branch = False
            if os.path.exists(branch_dir):
                 subdirs = [os.path.join(branch_dir, d) for d in os.listdir(branch_dir) if os.path.isdir(os.path.join(branch_dir, d))]
                 for d in subdirs:
                     if os.path.exists(os.path.join(d, '199_ckpt.tar')):
                         print(f"Branch {branch_name} seems already finished. Skipping.")
                         skip_branch = True
                         break
            
            if skip_branch:
                continue

            clear_logger()
            args_branch = copy.deepcopy(original_args)
            args_branch.save_dir = branch_dir
            args_branch.resume = base_ckpt
            args_branch.max_epoch = 200
            args_branch.stop_epoch = 0
            args_branch.drf_growth_start = start_val

            trainer_branch = RegTrainer(args_branch)
            trainer_branch.setup()
            trainer_branch.train()

            del trainer_branch
            torch.cuda.empty_cache()

        print("\nAll strict ablation experiments completed successfully!")

    else:
        # Normal training
        trainer = RegTrainer(args)
        trainer.setup()
        trainer.train()
