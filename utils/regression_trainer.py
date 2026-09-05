from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import time
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F
import logging
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.model_ppa import (
    UncertaintyVGG19_FPN_PPA_MultiBranch
)
from datasets.maize_semi import Crowd
from losses.post import Post_Prob
from losses.losses import Losses
from math import ceil

def train_collate(batch):
    transposed_batch = list(zip(*batch))
    raw_images = transposed_batch[0]
    
    img_weaks = []
    img_strongs = []
    
    for img_item in raw_images:
        if isinstance(img_item, (tuple, list)):
            img_weaks.append(img_item[0])
            img_strongs.append(img_item[1])
        else:
            img_weaks.append(img_item)
            img_strongs.append(img_item)
            
    images_weak = torch.stack(img_weaks, 0)
    images_strong = torch.stack(img_strongs, 0)
    
    points = transposed_batch[1]
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    label = transposed_batch[4]
    
    return (images_weak, images_strong), points, targets, st_sizes, label

class RegTrainer(Trainer):
    def setup(self):
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        self.downsample_ratio = args.downsample_ratio
        self.datasets = {x: Crowd(os.path.join(args.data_dir, x),
                                  args.crop_size,
                                  args.downsample_ratio,
                                  args.is_gray, x, args.info) for x in ['train', 'val']}
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(args.batch_size
                                          if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers*self.device_count,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val']}

        logging.info(f"Initializing model: {args.model_name}")
        
        from models.model_ppa import make_layers, cfg, model_zoo, model_urls
        features = make_layers(cfg['E'])
        
        if args.model_name == 'vgg19_fpn_ppa_base':
            logging.info("Using base multi-branch model: UncertaintyVGG19_FPN_PPA_MultiBranch")
            self.model = UncertaintyVGG19_FPN_PPA_MultiBranch(features,
                                                              use_ppa=args.use_ppa,
                                                              use_cls_head=args.use_cls_head,
                                                              use_unc_head=args.use_unc_head)
        else:
            raise ValueError(f"Unknown model name: {args.model_name}. Only 'vgg19_fpn_ppa_base' is supported.")

        if args.model_name == 'vgg19_fpn_ppa_base':
            vgg_state_dict = model_zoo.load_url(model_urls['vgg19'])
            self.model.features.load_state_dict(vgg_state_dict, strict=False)
        self.model.to(self.device)

        features_t = make_layers(cfg['E'])
        if args.model_name == 'vgg19_fpn_ppa_base':
             self.model_t = UncertaintyVGG19_FPN_PPA_MultiBranch(features_t,
                                                                 use_ppa=args.use_ppa,
                                                                 use_cls_head=args.use_cls_head,
                                                                 use_unc_head=args.use_unc_head)
        else:
            raise ValueError(f"Unknown model name: {args.model_name}. Only 'vgg19_fpn_ppa_base' is supported.")

        self.model_t.load_state_dict(self.model.state_dict())
        self.model_t.to(self.device)
        
        for param in self.model_t.parameters():
            param.detach_()

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = None 

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
                if 'model_t_state_dict' in checkpoint:
                    self.model_t.load_state_dict(checkpoint['model_t_state_dict'])
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device), strict=False)
                self.model_t.load_state_dict(self.model.state_dict())

        self.post_prob = Post_Prob(args.sigma, args.crop_size,
                                   args.downsample_ratio, self.device)

        self.criterion_mse = torch.nn.MSELoss(reduction='sum')
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_mae_t = np.inf
        self.best_mse_t = np.inf
        self.best_mae_s = np.inf
        self.best_mse_s = np.inf
        
        self.save_all = args.save_all
        self.best_count = 0
        
        idx_count = torch.tensor(
            [0.00000000, 0.00062337, 0.00528802, 0.01124929, 0.01798808, 0.02534289, 0.03324278, 0.04170712, 0.05114119, 0.06394629]
        )
        self.idx_count = idx_count.unsqueeze(1).to(self.device)
        
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        
        self.losses = Losses(self.device)

        self.w_sup_cls = 0.01
        self.w_sup_unc = 100.0
        
        self.w_cons_mse = 1.0
        self.w_cons_cls = 0.10
        self.w_cons_unc = 1.0

    def update_ema_variables(self, model, model_t, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(model_t.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
        for ema_buffer, buffer in zip(model_t.buffers(), model.buffers()):
            ema_buffer.data.copy_(buffer.data)

    def train(self):
        args = self.args
        
        logging.info("="*20 + " Loss Weights Configuration " + "="*20)
        logging.info(f"Supervised Loss Weights: Cls={self.w_sup_cls}, Unc={self.w_sup_unc}")
        logging.info(f"Unsupervised Loss Weights: MSE={self.w_cons_mse}, Cls={self.w_cons_cls}, Unc={self.w_cons_unc}")
        logging.info(f"Threshold Strategy: UHHM={args.uhhm_thresh_mode}, DRF={args.drf_thresh_mode}")
        logging.info(f"Uncertainty Components: UHHM_Error={args.uhhm_use_error}, DRF_Entropy={args.drf_use_cls_unc}")
        logging.info("="*66)
        
        for epoch in range(self.start_epoch, args.max_epoch):
            if hasattr(args, 'stop_epoch') and args.stop_epoch > 0 and epoch >= args.stop_epoch:
                logging.info(f"Reached stop_epoch {args.stop_epoch}, stopping training early.")
                break
                
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            
            current_weight = self.get_current_consistency_weight(epoch) * args.unsup_weight
            logging.info(f"Current Unsupervised Weight: {current_weight:.4f}")
            
            self.epoch = epoch
            self.train_eopch(epoch >= args.unlabel_start)
            
            if self.scheduler is not None:
                self.scheduler.step()
                
            current_lr = self.optimizer.param_groups[0]['lr']
            logging.info(f"Epoch {epoch} LR: {current_lr:.2e}")

            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

    def get_current_consistency_weight(self, epoch):
        rampup_length = self.args.max_epoch // 2
        if epoch < self.args.unlabel_start:
            return 0.0
        current = np.clip(epoch - self.args.unlabel_start, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

    def get_drf_growth(self, epoch):
        
        args = self.args
        growth_start_epoch = args.drf_growth_start
        
        if epoch < growth_start_epoch:
            return 0.0
            
        growth_epochs = args.max_epoch - growth_start_epoch
        current_growth_epoch = epoch - growth_start_epoch
        progress = float(current_growth_epoch) / float(max(1, growth_epochs))
        
        # Growth (k-factor): 0.0 -> 0.5
        growth = 0.0 + (0.5 - 0.0) * progress
        return growth

    def train_eopch(self, unlabel):
        args = self.args
        epoch_loss = AverageMeter()
        epoch_loss_sup = AverageMeter()
        epoch_loss_reg = AverageMeter()
        epoch_loss_cls = AverageMeter()
        epoch_loss_uncer = AverageMeter()
        epoch_loss_cons = AverageMeter()
        epoch_loss_cons_den = AverageMeter()
        epoch_loss_cons_cls = AverageMeter()
        epoch_loss_cons_unc = AverageMeter()
        
        epoch_consistency_weight = AverageMeter()
        epoch_uhhm_thresh = AverageMeter()
        epoch_drf_thresh = AverageMeter()
        epoch_uhhm_ratio = AverageMeter()
        epoch_drf_ratio = AverageMeter()
        current_weight = 0.0
        
        drf_growth = self.get_drf_growth(self.epoch)
        
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()
        self.model_t.train()

        for step, (inputs, points, targets, st_sizes, label) in enumerate(self.dataloaders['train']):
            inputs_weak, inputs_strong = inputs
            inputs_weak = inputs_weak.to(self.device)
            inputs_strong = inputs_strong.to(self.device)
            
            st_sizes = st_sizes.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            targets = [t.to(self.device) for t in targets]

            idx_l = [i for i, l in enumerate(label) if l]
            idx_u = [i for i, l in enumerate(label) if not l]

            if not unlabel and len(idx_l) == 0:
                continue

            with torch.set_grad_enabled(True):
                loss = 0.0
                N = inputs_weak.size(0)
                
                # Supervised Phase
                if len(idx_l) > 0:
                    inputs_l = inputs_weak[idx_l]
                    st_sizes_l = st_sizes[idx_l]
                    points_l = [points[i] for i in idx_l]
                    
                    gt_density = self.post_prob(points_l, st_sizes_l)
                    gt_density = gt_density.unsqueeze(1)
                    
                    s_density, s_logits, s_unc_logvar = self.model(inputs_l)
                    
                    centers = self.idx_count.view(1, 10, 1, 1)
                    dist = torch.abs(gt_density - centers)
                    gt_labels = torch.argmin(dist, dim=1)
                    
                    s_outputs = None
                    if s_logits is not None:
                        s_outputs = s_logits
                    
                    mask_uhhm = None
                    if self.epoch >= args.uhhm_start and s_unc_logvar is not None:
                        s_var = s_unc_logvar
                        
                        s_mean = torch.mean(s_var)
                        s_std = torch.std(s_var)
                        
                        if args.uhhm_thresh_mode == 'mean+0.5std':
                            tau_s = s_mean + 0.5 * s_std
                        elif args.uhhm_thresh_mode == 'mean-0.5std':
                            tau_s = s_mean - 0.5 * s_std
                        else:
                            tau_s = s_mean
                        
                        error_l = torch.abs(s_density.detach() - gt_density)
                        
                        e_mean = torch.mean(error_l)
                        e_std = torch.std(error_l)
                        
                        if args.uhhm_thresh_mode == 'mean+0.5std':
                            tau_e = e_mean + 0.5 * e_std
                        elif args.uhhm_thresh_mode == 'mean-0.5std':
                            tau_e = e_mean - 0.5 * e_std
                        else:
                            tau_e = e_mean
                        
                        s_var_min = torch.min(s_var)
                        s_var_max = torch.max(s_var)
                        
                        if (s_var_max - s_var_min) < 1e-5:
                             if args.uhhm_use_error:
                                 mask_uhhm = (error_l > tau_e).float()
                             else:
                                 mask_uhhm = torch.ones_like(s_density)
                        else:
                            if args.uhhm_use_error:
                                mask_uhhm = ((s_var > tau_s) | (error_l > tau_e)).float()
                            else:
                                mask_uhhm = (s_var > tau_s).float()
                        
                        epoch_uhhm_thresh.update(tau_s.item(), len(idx_l))
                        epoch_uhhm_ratio.update(torch.mean(mask_uhhm).item(), len(idx_l))
                    else:
                        mask_uhhm = torch.ones_like(s_density)
                        epoch_uhhm_thresh.update(0.0, len(idx_l))
                        epoch_uhhm_ratio.update(1.0, len(idx_l))
                    
                    loss_mse = (s_density - gt_density) ** 2
                    loss_mse = torch.sum(loss_mse * mask_uhhm) / (torch.sum(mask_uhhm) + 1e-7)
                    
                    loss_ssim = self.losses.masked_ssim_loss(s_density, gt_density, mask_uhhm)
                    
                    loss_reg = 5000.0 * loss_mse + 1.0 * loss_ssim
                    
                    loss_cls = 0.0
                    if s_logits is not None:
                        s_log_probs = torch.log(s_logits + 1e-7) 
                        pixel_cls_loss = F.nll_loss(s_log_probs, gt_labels, reduction='none')
                        loss_cls = torch.sum(pixel_cls_loss * mask_uhhm) / (torch.sum(mask_uhhm) + 1e-7)
                    
                    loss_uncer = 0.0
                    if s_unc_logvar is not None:
                        error_map = torch.abs(s_density.detach() - gt_density)
                        uncertainty_map = s_unc_logvar
                        
                        B, C, H, W = error_map.shape
                        N_samples = 2000
                        
                        flat_error = error_map.view(B, -1)
                        flat_unc = uncertainty_map.view(B, -1)
                        
                        loss_uncer_batch = 0
                        for b in range(B):
                            num_pixels = H * W
                            if num_pixels > N_samples:
                                indices = torch.randperm(num_pixels)[:N_samples].to(self.device)
                            else:
                                indices = torch.arange(num_pixels).to(self.device)
                                
                            a_sampled = flat_error[b, indices]
                            p_sampled = flat_unc[b, indices]
                            

                            
                            a_i = a_sampled.unsqueeze(1) # [N, 1]
                            a_j = a_sampled.unsqueeze(0) # [1, N]
                            p_i = p_sampled.unsqueeze(1) # [N, 1]
                            p_j = p_sampled.unsqueeze(0) # [1, N]
                      
                            # Calculate components
                            target = torch.sign(a_i - a_j)
                            logits_diff = p_i - p_j
                            
                            chunk_size = 500
                            weighted_loss_sum = 0
                            
                            for i in range(0, N_samples, chunk_size):
                                end_i = min(i + chunk_size, N_samples)
                                a_i_chunk = a_sampled[i:end_i].unsqueeze(1)
                                p_i_chunk = p_sampled[i:end_i].unsqueeze(1)
                                
                                # Broadcast against full set j
                                a_j_full = a_sampled.unsqueeze(0)
                                p_j_full = p_sampled.unsqueeze(0)
                                
                                target_chunk = torch.sign(a_i_chunk - a_j_full)
                                weight_chunk = torch.abs(a_i_chunk - a_j_full)
                                logits_diff_chunk = p_i_chunk - p_j_full
                                
                                pair_loss_chunk = F.softplus(-target_chunk * logits_diff_chunk)
                                weighted_loss_chunk = weight_chunk * pair_loss_chunk
                                
                                weighted_loss_sum += torch.sum(weighted_loss_chunk)
                                
                                # Free memory
                                del target_chunk, weight_chunk, logits_diff_chunk, pair_loss_chunk, weighted_loss_chunk
                            
                            loss_uncer_batch += weighted_loss_sum / (N_samples * N_samples)
                            
                        loss_rank = loss_uncer_batch / B
                        loss_uncer = loss_rank
                    
                    loss_s = loss_reg + self.w_sup_cls * loss_cls + self.w_sup_unc * loss_uncer
                    
                    loss += loss_s
                    epoch_loss_sup.update(loss_s.item(), len(idx_l)) 
                    epoch_loss_reg.update(loss_reg.item(), len(idx_l))
                    epoch_loss_cls.update(loss_cls.item() if isinstance(loss_cls, torch.Tensor) else loss_cls, len(idx_l))
                    epoch_loss_uncer.update(loss_uncer.item() if isinstance(loss_uncer, torch.Tensor) else loss_uncer, len(idx_l)) 

                # Unsupervised Phase
                if unlabel and len(idx_u) > 0:
                    inputs_u_weak = inputs_weak[idx_u]
                    inputs_u_strong = inputs_strong[idx_u]
                    
                    s_density_u, s_logits_u, s_unc_logvar_u = self.model(inputs_u_strong)

                    with torch.no_grad():
                        t_density_u, t_logits_u, t_unc_logvar_u = self.model_t(inputs_u_weak)
                        
                        mask_unc = None
                        tau_u_unc = 0.0 
                        
                        # Use the pre-calculated growth value from start of epoch
                        # drf_growth is calculated by self.get_drf_growth(self.epoch)
                        
                        # Determine Base K from settings
                        if args.drf_thresh_mode == 'mean+0.5std':
                            base_k = 0.5
                        elif args.drf_thresh_mode == 'mean-0.5std':
                            base_k = -0.5
                        else:
                            base_k = 0.0 # 'mean'
                            
                        # Add Growth Delta (drf_growth is calculated at start of epoch)
                        # If epoch < growth_start, drf_growth is 0.0
                        current_k = base_k + drf_growth
                        
                        if t_unc_logvar_u is not None:
                            t_var_u = t_unc_logvar_u
                            
                            t_mean = torch.mean(t_var_u)
                            t_std = torch.std(t_var_u)
                            
                            # Final Threshold = Mean + Current_K * Std
                            tau_u_unc = t_mean + current_k * t_std
                            
                            t_var_min = torch.min(t_var_u)
                            t_var_max = torch.max(t_var_u)
                            if (t_var_max - t_var_min) < 1e-5:
                                mask_unc = torch.ones_like(t_var_u)
                            else:
                                # Normal: Keep Low Uncertainty (Noise Filtering)
                                mask_unc = (t_var_u < tau_u_unc).float()
                        else:
                        
                            mask_unc = None 
                            
                        # 2. Compute Entropy-based Mask (Mask 2)
                        mask_ent = None
                        tau_u_ent = 0.0
                        
                        if args.drf_use_cls_unc and t_logits_u is not None:
                            # Entropy: H(p) = - sum(p * log(p))
                            t_probs_u = F.softmax(t_logits_u, dim=1)
                            t_entropy_u = -torch.sum(t_probs_u * torch.log(t_probs_u + 1e-7), dim=1, keepdim=True) # [B, 1, H, W]
                            
                            ent_mean = torch.mean(t_entropy_u)
                            ent_std = torch.std(t_entropy_u)
                            
                            # Apply same Step Change logic to Entropy Threshold
                            tau_u_ent = ent_mean + current_k * ent_std
                            
                            ent_min = torch.min(t_entropy_u)
                            ent_max = torch.max(t_entropy_u)
                            
                            if (ent_max - ent_min) < 1e-5:
                                mask_ent = torch.ones_like(t_entropy_u)
                            else:
                                # Normal: Keep Low Entropy (Noise Filtering)
                                mask_ent = (t_entropy_u < tau_u_ent).float()
                        else:
                            # If no classification branch, set mask_ent to None
                            mask_ent = None

                        # 3. Fuse Masks
                        if mask_unc is None:
                            # Case A: No uncertainty branch -> No masking (Keep all)
                            mask_drf = torch.ones_like(t_density_u)
                        elif mask_ent is None:
                            # Case B: Uncertainty branch only -> Single branch strategy
                            mask_drf = mask_unc
                        else:
                            # Case C: Both branches exist -> Dual filtering strategy
                            # Normal Phase (Noise Filtering): Union (OR)
                            mask_drf = ((mask_unc > 0.5) | (mask_ent > 0.5)).float()

                        if isinstance(tau_u_unc, float):
                             epoch_drf_thresh.update(tau_u_unc, len(idx_u))
                        else:
                             epoch_drf_thresh.update(tau_u_unc.item(), len(idx_u)) # Log primary uncertainty threshold
                        epoch_drf_ratio.update(torch.mean(mask_drf).item(), len(idx_u))

                    loss_cons_mse = self.losses.consistency_loss(s_density_u, t_density_u.detach(), mask_drf)
                    if t_logits_u is not None:
                        loss_cons_cls = self.losses.classification_consistency_loss(s_logits_u, t_logits_u.detach(), mask_drf)
                    else:
                        loss_cons_cls = 0.0

                    if t_unc_logvar_u is not None:
                        loss_cons_unc = self.losses.ranking_consistency_loss(s_unc_logvar_u, t_unc_logvar_u.detach(), mask_drf)
                    else:
                        loss_cons_unc = 0.0
                    
                    loss_cons = self.w_cons_mse * loss_cons_mse + self.w_cons_cls * loss_cons_cls + self.w_cons_unc * loss_cons_unc
                    
                    if args.use_dynamic_unsup_weight:
                        current_weight = self.get_current_consistency_weight(self.epoch) * args.unsup_weight
                    else:
                        current_weight = args.unsup_weight
                    loss += current_weight * loss_cons
                    
                    epoch_loss_cons.update(loss_cons.item(), len(idx_u))
                    epoch_loss_cons_den.update(loss_cons_mse.item(), len(idx_u))
                    epoch_loss_cons_cls.update(loss_cons_cls.item() if isinstance(loss_cons_cls, torch.Tensor) else loss_cons_cls, len(idx_u))
                    epoch_loss_cons_unc.update(loss_cons_unc.item() if isinstance(loss_cons_unc, torch.Tensor) else loss_cons_unc, len(idx_u))
                    
                    if isinstance(tau_u_unc, float):
                         epoch_consistency_weight.update(tau_u_unc, len(idx_u))
                    else:
                         epoch_consistency_weight.update(tau_u_unc.item(), len(idx_u))
                    
                epoch_loss.update(loss.item(), N)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                global_step = self.epoch * len(self.dataloaders['train']) + step
                
                # EMA Decay Strategy Update
                # Supervised Phase: 0.9 -> 0.99 (Linear Growth)
                # Semi-supervised Phase: 0.99 -> 0.999 (Linear Growth)
                if self.epoch < args.unlabel_start:
                    progress = float(self.epoch) / float(max(1, args.unlabel_start))
                    current_decay = 0.9 + (0.99 - 0.9) * progress
                else:
                    semi_epochs = args.max_epoch - args.unlabel_start
                    current_semi_epoch = self.epoch - args.unlabel_start
                    progress = float(current_semi_epoch) / float(max(1, semi_epochs))
                    current_decay = 0.99 + (0.999 - 0.99) * progress
                    
                self.update_ema_variables(self.model, self.model_t, current_decay, global_step)

                if len(idx_l) > 0:
                    pre_count = torch.sum(s_density.view(len(idx_l), -1), dim=1).detach().cpu().numpy()
                    gd_count_l = gd_count[idx_l]
                    res = pre_count - gd_count_l
                    epoch_mse.update(np.mean(res * res), len(idx_l))
                    epoch_mae.update(np.mean(abs(res)), len(idx_l))

        current_lr = self.optimizer.param_groups[0]['lr']
        logging.info('Epoch {} Train, Loss: {:.5f} (Sup: {:.5f} [Reg: {:.5f}, Cls: {:.5f}, Unc: {:.5f}], Cons: {:.5f} [MSE: {:.5f}, Cls: {:.5f}, Unc: {:.5f}]), UnsupW: {:.4f}, UHHM: {:.6f} ({:.2f}), DRF: {:.6f} ({:.2f}), MSE: {:.2f} MAE: {:.2f}, LR: {:.2e}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), 
                             epoch_loss_sup.get_avg(), 
                             epoch_loss_reg.get_avg(),
                             epoch_loss_cls.get_avg(),
                             epoch_loss_uncer.get_avg(),
                             epoch_loss_cons.get_avg(),
                             epoch_loss_cons_den.get_avg(),
                             epoch_loss_cons_cls.get_avg(),
                             epoch_loss_cons_unc.get_avg(),
                             current_weight, 
                             epoch_uhhm_thresh.get_avg(),
                             epoch_uhhm_ratio.get_avg(),
                             epoch_drf_thresh.get_avg(),
                             epoch_drf_ratio.get_avg(),
                             np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(), current_lr,
                             time.time()-epoch_start))
        
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic,
            'model_t_state_dict': self.model_t.state_dict()
        }, save_path)
        
        self.save_list.append(save_path)

    def val_epoch(self):
        args = self.args
        self.model.eval()
        self.model_t.eval()
        epoch_start = time.time()
        
        epoch_res_t = []
        epoch_sq_res_t = []
        epoch_res_s = []
        epoch_sq_res_s = []

        for inputs, keypoints_batch, name in self.dataloaders['val']:
            inputs = inputs.to(self.device)
            keypoints = keypoints_batch[0]
            
            b, c, h, w = inputs.shape
            assert b == 1
            
            c_size = args.crop_size
            
            use_puzzle = (h % c_size != 0) or (w % c_size != 0)
            
            if use_puzzle:

                full_density_t = torch.zeros((1, 1, h, w), device=self.device)
                full_density_s = torch.zeros((1, 1, h, w), device=self.device)
                overlap_count = torch.zeros((1, 1, h, w), device=self.device)
            else:
                pred_count_t = 0.0
                pred_count_s = 0.0
            
            h_stride = int(ceil(1.0 * h / c_size))
            w_stride = int(ceil(1.0 * w / c_size)) 
            
            for i in range(h_stride):
                for j in range(w_stride):
                    h_start = i * c_size
                    w_start = j * c_size
                    if h_start + c_size > h: h_start = h - c_size
                    if w_start + c_size > w: w_start = w - c_size
                    h_end = h_start + c_size
                    w_end = w_start + c_size
                    
                    input_patch = inputs[:, :, h_start:h_end, w_start:w_end]
                    
                    with torch.set_grad_enabled(False):
                        # Teacher Model Prediction
                        t_density_patch, _, _ = self.model_t(input_patch)
                        t_density_patch = torch.clamp(t_density_patch, min=0)
                        
                        # Student Model Prediction
                        s_density_patch, _, _ = self.model(input_patch)
                        s_density_patch = torch.clamp(s_density_patch, min=0)
                        
                        if use_puzzle:
                            
                            patch_h = h_end - h_start
                            patch_w = w_end - w_start
                            
                            t_density_patch = F.interpolate(t_density_patch, size=(patch_h, patch_w), mode='bilinear', align_corners=False)
                            s_density_patch = F.interpolate(s_density_patch, size=(patch_h, patch_w), mode='bilinear', align_corners=False)
                            
                            ratio_sq = self.downsample_ratio * self.downsample_ratio
                            t_density_patch /= ratio_sq
                            s_density_patch /= ratio_sq
                            
                            full_density_t[:, :, h_start:h_end, w_start:w_end] += t_density_patch
                            full_density_s[:, :, h_start:h_end, w_start:w_end] += s_density_patch
                            overlap_count[:, :, h_start:h_end, w_start:w_end] += 1.0
                        else:
                            # Direct accumulation (No overlap guaranteed)
                            pred_count_t += torch.sum(t_density_patch).item()
                            pred_count_s += torch.sum(s_density_patch).item()

            if use_puzzle:
                # Normalize by overlap count to get average density
                full_density_t /= overlap_count
                full_density_s /= overlap_count
                
                pred_count_t = torch.sum(full_density_t).item()
                pred_count_s = torch.sum(full_density_s).item()

            gt_count = len(keypoints)
            
            res_t = pred_count_t - gt_count
            epoch_res_t.append(abs(res_t))
            epoch_sq_res_t.append(res_t * res_t)
            
            res_s = pred_count_s - gt_count
            epoch_res_s.append(abs(res_s))
            epoch_sq_res_s.append(res_s * res_s)

        mse_t = np.sqrt(np.mean(epoch_sq_res_t))
        mae_t = np.mean(epoch_res_t)
        
        mse_s = np.sqrt(np.mean(epoch_sq_res_s))
        mae_s = np.mean(epoch_res_s)
        
        logging.info('Epoch {} Val (Whole Image 512x512) Cost {:.1f} sec'.format(self.epoch, time.time()-epoch_start))
        logging.info('Teacher: MSE: {:.2f} MAE: {:.2f}'.format(mse_t, mae_t))
        logging.info('Student: MSE: {:.2f} MAE: {:.2f}'.format(mse_s, mae_s))

        if (2.0 * mse_t + mae_t) < (2.0 * self.best_mse_t + self.best_mae_t):
            self.best_mse_t = mse_t
            self.best_mae_t = mae_t
            logging.info("SAVE BEST TEACHER: mse {:.2f} mae {:.2f} epoch {}".format(self.best_mse_t, self.best_mae_t, self.epoch))
            torch.save(self.model_t.state_dict(), os.path.join(self.save_dir, 'best_model_teacher.pth'))
            
        if (2.0 * mse_s + mae_s) < (2.0 * self.best_mse_s + self.best_mae_s):
            self.best_mse_s = mse_s
            self.best_mae_s = mae_s
            logging.info("SAVE BEST STUDENT: mse {:.2f} mae {:.2f} epoch {}".format(self.best_mse_s, self.best_mae_s, self.epoch))
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model_student.pth'))

        if (2.0 * mse_t + mae_t) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse_t
            self.best_mae = mae_t
