import torch
import os
import numpy as np
from datasets.crowd_semi import Crowd
import argparse
import math
from glob import glob
from datetime import datetime
import torch.nn.functional as F

args = None

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test Resize 512x512')
    parser.add_argument('--data-dir', default='maize_tassel',
                        help='Training data directory')
    parser.add_argument('--save-dir', default='model/label_ratio_ablation/ratio_50/0210-211307',
                        help='Model directory')
    parser.add_argument('--device', default='0', help='Assign GPU device')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()

    # Load test dataset
    datasets = Crowd(os.path.join(args.data_dir, 'test'), 512, 8, is_gray=False, method='test')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=False)

    # Identify models to test
    model_list = []
    
    # Prioritize 'best_model_teacher.pth' and 'best_model_student.pth'
    best_teacher = os.path.join(args.save_dir, 'best_model_teacher.pth')
    best_student = os.path.join(args.save_dir, 'best_model_student.pth')
    
    if os.path.exists(best_teacher):
        model_list.append(best_teacher)
    if os.path.exists(best_student):
        model_list.append(best_student)
        
    # Fallback: check for 'best_model.pth' or any .pth/.tar files
    if len(model_list) == 0:
        best_model_path = os.path.join(args.save_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            model_list = [best_model_path]
        else:
            model_list = sorted(glob(os.path.join(args.save_dir, '*.pth')) + glob(os.path.join(args.save_dir, '*.tar')))
    
    if len(model_list) == 0:
        print(f"No model found in {args.save_dir}")
        exit(1)
        
    print(f"Found models: {model_list}")
    
    device = torch.device('cuda')
    
    from models.model_ppa import UncertaintyVGG19_FPN_PPA_MultiBranch, make_layers, cfg
    features = make_layers(cfg['E'])

    # Load the first checkpoint to auto-detect model configuration
    # Load to CPU first to avoid potential OOM if GPU is busy
    checkpoint = torch.load(model_list[0], map_location='cpu')

    # ==========================================
    # Auto-detect model configuration
    # ==========================================
    use_ppa = True
    use_cls_head = True
    use_unc_head = True
    model_name = 'vgg19_fpn_ppa_base' # Default model name

    # Method 1: Try to load from saved args
    if 'args' in checkpoint:
        saved_args = checkpoint['args']
        if hasattr(saved_args, 'use_ppa'): use_ppa = saved_args.use_ppa
        if hasattr(saved_args, 'use_cls_head'): use_cls_head = saved_args.use_cls_head
        if hasattr(saved_args, 'use_unc_head'): use_unc_head = saved_args.use_unc_head
        if hasattr(saved_args, 'model_name'): model_name = saved_args.model_name
            
    # Method 2: Infer from state_dict keys (more robust)
    state_dict_to_check = None
    if 'model_t_state_dict' in checkpoint:
        state_dict_to_check = checkpoint['model_t_state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict_to_check = checkpoint['model_state_dict']
    else:
        state_dict_to_check = checkpoint
        
    if state_dict_to_check:
        has_ppa_keys = any('ppa_deep' in k for k in state_dict_to_check.keys())
        has_cls_keys = any('mscb_branch' in k or 'head_cls' in k for k in state_dict_to_check.keys())
        has_unc_keys = any('lka_branch' in k or 'head_unc' in k for k in state_dict_to_check.keys())
        
        if not has_ppa_keys: use_ppa = False
        if not has_cls_keys: use_cls_head = False
        if not has_unc_keys: use_unc_head = False
        
    print(f"Final Model Config: Name={model_name}, PPA={use_ppa}, CLS={use_cls_head}, UNC={use_unc_head}")

    # Initialize model
    if model_name == 'vgg19_fpn_ppa_base':
        model = UncertaintyVGG19_FPN_PPA_MultiBranch(features, use_ppa=use_ppa, use_cls_head=use_cls_head, use_unc_head=use_unc_head)
    else:
        print(f"Unknown model name: {model_name}, defaulting to base version")
        model = UncertaintyVGG19_FPN_PPA_MultiBranch(features, use_ppa=use_ppa, use_cls_head=use_cls_head, use_unc_head=use_unc_head)
    
    log_list = []
    
    for model_path in model_list:
        name = model_path.split('/')[-1]
        
        # Load weights
        print(f"Loading weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Determine which state_dict to load
        if 'model_t_state_dict' in checkpoint:
            print("Loading Teacher model from full checkpoint...")
            model.load_state_dict(checkpoint['model_t_state_dict'])
        elif 'model_state_dict' in checkpoint:
            print("Loading Student model from full checkpoint...")
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Loading raw state_dict...")
            model.load_state_dict(checkpoint)
            
        model.to(device)
        model.eval()
        
        epoch_res = []
        epoch_sq_res = []

        print(f"Testing model: {name} with Sliding Window (Patch-based)...")

        for inputs, keypoints_batch, name in dataloader:
            inputs = inputs.to(device)
            # Inputs shape: (B, 3, H, W), e.g., (1, 3, 1024, 1024)
            b, c, h, w = inputs.shape
            
            # Crop size for sliding window
            c_size = 512
            
            # Check if puzzle method is needed (if image size not divisible by crop size)
            use_puzzle = (h % c_size != 0) or (w % c_size != 0)
            
            if use_puzzle:
                # Initialize accumulators for Puzzle Method
                full_density = torch.zeros((1, 1, h, w), device=device)
                overlap_count = torch.zeros((1, 1, h, w), device=device)
                pred_count = 0.0 # Placeholder
            else:
                pred_count = 0.0
            
            # Calculate number of patches
            h_stride = int(math.ceil(1.0 * h / c_size))
            w_stride = int(math.ceil(1.0 * w / c_size))
            
            # Iterate over patches
            for i in range(h_stride):
                for j in range(w_stride):
                    h_start = i * c_size
                    w_start = j * c_size
                    
                    # Boundary check: shift back if exceeding image size
                    if h_start + c_size > h:
                        h_start = h - c_size
                    if w_start + c_size > w:
                        w_start = w - c_size
                        
                    h_end = h_start + c_size
                    w_end = w_start + c_size
                    
                    # Extract patch
                    input_patch = inputs[:, :, h_start:h_end, w_start:w_end]
                    
                    # Forward pass
                    density_map, logits, uncertainty_map = model(input_patch)
                    
                    # Ensure non-negative density
                    density_map = torch.clamp(density_map, min=0)
                    
                    if use_puzzle:
                        # Upsample density map if downsampling occurred (usually 8x)
                        patch_h = h_end - h_start
                        patch_w = w_end - w_start
                        
                        density_map = F.interpolate(density_map, size=(patch_h, patch_w), mode='bilinear', align_corners=False)
                        
                        # Adjust for area scaling (divide by ratio^2)
                        downsample_ratio = 8
                        ratio_sq = downsample_ratio * downsample_ratio
                        density_map /= ratio_sq
                        
                        full_density[:, :, h_start:h_end, w_start:w_end] += density_map
                        overlap_count[:, :, h_start:h_end, w_start:w_end] += 1.0
                    else:
                        # Sum count for this patch
                        pred_count += torch.sum(density_map).item()
            
            if use_puzzle:
                # Average overlapping areas
                full_density /= overlap_count
                pred_count = torch.sum(full_density).item()
            
            # Get Ground Truth Count
            keypoints = keypoints_batch[0]
            gt_count = len(keypoints)
                
            # Calculate Error
            res = pred_count - gt_count
            epoch_res.append(abs(res))
            epoch_sq_res.append(res * res)
        
        mse = np.sqrt(np.mean(epoch_sq_res))
        mae = np.mean(epoch_res)
        
        log_str = 'model_name {}, mae {}, mse {} (Resized 512x512)'.format(os.path.basename(model_path), mae, mse)
        log_list.append(log_str)
        print(log_str)
    
    # Save results
    date_str = datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_path = os.path.join(args.save_dir, 'test_resize_results_{}.txt'.format(date_str))
    with open(save_path, 'w') as f:
        for log_str in log_list:
            f.write(log_str + '\n')
    print(f"Results saved to {save_path}")
