import os
import glob
import logging
import importlib
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from core.lr_scheduler import MultiStepRestartLR, CosineAnnealingRestartLR
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from core.loss import AdversarialLoss, VGGLoss
from core.dataset import UnpairFaceDataset, FaceRetouchingDataset
import wandb
import lpips
from torchvision import transforms
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
class Trainer:
    def __init__(self, config):
        self.config = config
        self.epoch = 0
        self.iteration = 0
        self.psnr = 0
        self.ssim = 0
        self.maxiteration = 0
        self.train_dataset = FaceRetouchingDataset(path = config['train_data_loader']['dataroot'],
                                                   resolution=config['train_data_loader']['size'],
                                                   data_type="train", data_percentage=config['train_data_loader']['percentage'])
        self.test_dataset = FaceRetouchingDataset(path = config['train_data_loader']['dataroot'],
                                                  resolution=config['train_data_loader']['size'],
                                                  data_type="test", data_percentage=1)
        self.unpair_dataset = UnpairFaceDataset(path = config['train_data_loader']['dataroot'], 
                                                    resolution=config['train_data_loader']['size'], 
                                                    data_type="train", return_gt=True, data_percentage=0)
        print("net:", self.config['model']['net'])
        if config['trainer']['use_wandb']==1:
            wandb.init(project="retouching", name=self.config['model']['net'] + "_bs1")
            self.wandb = True       
        else:
            self.wandb = False
        self.train_sampler = None
        self.train_args = config['trainer']
        if config['distributed']:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=config['world_size'],
                rank=config['global_rank'])

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_args['batch_size'] // config['world_size'],
            shuffle=(self.train_sampler is None),
            num_workers=self.train_args['num_workers'],
            sampler=self.train_sampler)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=4)
        self.unpair_loader = DataLoader(self.unpair_dataset,
            batch_size=self.train_args['batch_size'] // config['world_size'],
            shuffle=True,
            num_workers=self.train_args['num_workers'])
        self.eval_txt = config['eval_txt']
        # set loss functions
        self.adversarial_loss = AdversarialLoss(type=self.config['losses']['GAN_LOSS'])
        self.adversarial_loss = self.adversarial_loss.to(self.config['device'])
        self.l1_loss = nn.SmoothL1Loss().to(self.config['device'])
        self.lpips_loss = lpips.LPIPS(net='alex', lpips=False).to(self.config['device'])
        self.vgg_loss = VGGLoss(self.config['device'])
        # setup models including generator and discriminator
        net = importlib.import_module('model.' + config['model']['net'])
        self.netG = net.InpaintGenerator()
        self.netG = self.netG.to(self.config['device'])
        # print(self.netG)
    
        if not self.config['model']['no_dis']:
            self.netD = net.Discriminator(
                in_channels=3,
                use_sigmoid=config['losses']['GAN_LOSS'] != 'hinge')
            self.netD = self.netD.to(self.config['device'])

        # setup optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        self.load()

        if config['distributed']:
            self.netG = DDP(self.netG,
                            device_ids=[self.config['local_rank']],
                            output_device=self.config['local_rank'],
                            broadcast_buffers=True,
                            find_unused_parameters=True)
            if not self.config['model']['no_dis']:
                self.netD = DDP(self.netD,
                                device_ids=[self.config['local_rank']],
                                output_device=self.config['local_rank'],
                                broadcast_buffers=True,
                                find_unused_parameters=False)

        # set summary writer
        self.dis_writer = None
        self.gen_writer = None
        self.summary = {}
        if self.config['global_rank'] == 0 or (not config['distributed']):
            self.dis_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'dis'))
            self.gen_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'gen'))

    def setup_optimizers(self):
        """Set up optimizers."""
        backbone_params = []
        maskG_params = []
        for name, param in self.netG.named_parameters():
            if not param.requires_grad:
                continue
            elif 'mask_generator' in name:
                maskG_params.append(param)
            else:
                backbone_params.append(param)

        optim_params = [{'params': backbone_params,'lr': self.config['trainer']['lr']}]
        self.optimG = torch.optim.Adam(optim_params, betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))

        opt_maskG_params =  [{'params':maskG_params, 'lr': 4.5e-6}]
        self.optim_maskG = torch.optim.Adam(opt_maskG_params, betas=(0.5, 0.9))

        if not self.config['model']['no_dis']:
            self.optimD = torch.optim.Adam(self.netD.parameters(), lr=self.config['trainer']['lr'], betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))        

    def setup_schedulers(self):
        """Set up schedulers."""
        scheduler_opt = self.config['trainer']['scheduler']
        scheduler_type = scheduler_opt.pop('type')

        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            self.scheG = MultiStepRestartLR(self.optimG, milestones=scheduler_opt['milestones'],
                                            gamma=scheduler_opt['gamma'])
            self.scheD = MultiStepRestartLR(self.optimD, milestones=scheduler_opt['milestones'],
                                            gamma=scheduler_opt['gamma'])
        elif scheduler_type == 'CosineAnnealingRestartLR':
            self.scheG = CosineAnnealingRestartLR(
                self.optimG,
                periods=scheduler_opt['periods'],
                restart_weights=scheduler_opt['restart_weights'])
            self.scheD = CosineAnnealingRestartLR(
                self.optimD,
                periods=scheduler_opt['periods'],
                restart_weights=scheduler_opt['restart_weights'])
            self.sche_maskG = torch.optim.lr_scheduler.MultiStepLR(self.optim_maskG, milestones=[400000,800000], gamma=0.1, verbose=True)
        elif scheduler_type == "ExponentialLR":
            self.scheG = ExponentialLR(self.optimG, gamma=0.7)
            self.scheD = ExponentialLR(self.optimD, gamma=0.7)
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')

    def update_learning_rate(self):
        """Update learning rate."""
        self.scheG.step()
        self.scheD.step()

    def get_lr(self):
        """Get current learning rate."""
        # return self.optimG.param_groups[0]['lr']
        return self.scheG.get_lr()[0]

    def add_summary(self, writer, name, val):
        """Add tensorboard summary."""
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        if writer is not None and self.iteration % 100 == 0:
            writer.add_scalar(name, self.summary[name] / 100, self.iteration)
            self.summary[name] = 0

    def load(self):
        """Load netG (and netD)."""
        # get the latest checkpoint
        model_path = self.config['save_dir']
        if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
            latest_epoch = open(os.path.join(model_path, 'latest.ckpt'), 'r').read().splitlines()[-1]
        else:
            ckpts = [
                os.path.basename(i).split('.pth')[0]
                for i in glob.glob(os.path.join(model_path, '*.pth'))
            ]
            ckpts.sort()
            latest_epoch = ckpts[-1] if len(ckpts) > 0 else None

        if latest_epoch is not None:
            gen_path = os.path.join(model_path,
                                    f'gen_{int(latest_epoch):06d}.pth')
            dis_path = os.path.join(model_path,
                                    f'dis_{int(latest_epoch):06d}.pth')
            opt_path = os.path.join(model_path,
                                    f'opt_{int(latest_epoch):06d}.pth')

            if self.config['global_rank'] == 0:
                print(f'Loading model from {gen_path}...')
            dataG = torch.load(gen_path, map_location=self.config['device'])
            self.netG.load_state_dict(dataG)
            if not self.config['model']['no_dis']:
                dataD = torch.load(dis_path,
                                   map_location=self.config['device'])
                self.netD.load_state_dict(dataD)

            data_opt = torch.load(opt_path, map_location=self.config['device'])
            self.optimG.load_state_dict(data_opt['optimG'])
            self.scheG.load_state_dict(data_opt['scheG'])
            self.optim_maskG.load_state_dict(data_opt['optim_maskG'])
            self.sche_maskG.load_state_dict(data_opt['sche_maskG'])
            if not self.config['model']['no_dis']:
                self.optimD.load_state_dict(data_opt['optimD'])
                self.scheD.load_state_dict(data_opt['scheD'])
            self.epoch = data_opt['epoch']
            self.iteration = data_opt['iteration']
            self.eval_mask(iteration=self.iteration)
            # self.test(self.iteration, lr = self.get_lr())
        else:
            if self.config['global_rank'] == 0:
                print('Warnning: There is no trained model found. An initialized model will be used.')

    def save(self, it):
        """Save parameters every eval_epoch"""
        if self.config['global_rank'] == 0:
            # configure path
            gen_path = os.path.join(self.config['save_dir'],
                                    f'gen_{it:06d}.pth')
            dis_path = os.path.join(self.config['save_dir'],
                                    f'dis_{it:06d}.pth')
            opt_path = os.path.join(self.config['save_dir'],
                                    f'opt_{it:06d}.pth')
            print(f'\nsaving model to {gen_path} ...')

            # remove .module for saving
            if isinstance(self.netG, torch.nn.DataParallel) \
               or isinstance(self.netG, DDP):
                netG = self.netG.module
                if not self.config['model']['no_dis']:
                    netD = self.netD.module
            else:
                netG = self.netG
                if not self.config['model']['no_dis']:
                    netD = self.netD

            # save checkpoints
            torch.save(netG.state_dict(), gen_path)
            if not self.config['model']['no_dis']:
                torch.save(netD.state_dict(), dis_path)
                torch.save(
                    {
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.optimG.state_dict(),
                        'optim_maskG': self.optim_maskG.state_dict(),
                        'optimD': self.optimD.state_dict(),
                        'scheG': self.scheG.state_dict(),
                        'sche_maskG': self.sche_maskG.state_dict(),
                        'scheD': self.scheD.state_dict()
                    }, opt_path)
            else:
                torch.save(
                    {
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.optimG.state_dict(),
                        'scheG': self.scheG.state_dict()
                    }, opt_path)

            latest_path = os.path.join(self.config['save_dir'], 'latest.ckpt')
            os.system(f"echo {it:06d} > {latest_path}")
    
    def test(self, iteration, lr):
        self.netG.eval()
        cnt = 0
        PSNR = 0
        SSIM = 0
        LPIPS = 0
        device = self.config['device']
        loss_fn = lpips.LPIPS(net='alex').to(device)
        for name, source_tensor, target_tensor in tqdm(self.test_loader):
            with torch.no_grad(): 
                pred_img, _ = self.netG(source_tensor.to(device))
                lpips_loss = loss_fn(pred_img, target_tensor.to(device)).mean()
                s_img = pred_img[0].cpu().numpy()
                t_img = target_tensor[0].numpy()
                psnr = compare_psnr(t_img,s_img)
                ssim = compare_ssim(t_img,s_img, channel_axis=0)
                PSNR += psnr
                SSIM += ssim
                LPIPS+= lpips_loss
                cnt += 1
        PSNR/=cnt
        SSIM/=cnt
        LPIPS/=cnt
        print(iteration, ": PSNR:",PSNR, "SSIM:", SSIM, "LPIPS:", LPIPS)
        if self.wandb:
            wandb.log({"PSNR" :PSNR.item(),"SSIM" :SSIM.item(),"LPIPS" :LPIPS.item()})
        with open(self.eval_txt,'a') as f:
            f.writelines(f"lr: {lr}; {iteration}: PSNR: {PSNR}; SSIM: {SSIM}; LPIPS: {LPIPS}\n")  
        f.close()
        self.netG.train()

    def train(self):
        """training entry"""
        pbar = range(int(self.train_args['iterations']))
        if self.config['global_rank'] == 0:
            pbar = tqdm(pbar,
                        initial=self.iteration,
                        dynamic_ncols=True,
                        smoothing=0.01)

        os.makedirs('logs', exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(filename)s[line:%(lineno)d]"
            "%(levelname)s %(message)s",
            datefmt="%a, %d %b %Y %H:%M:%S",
            filename=f"logs/{self.config['save_dir'].split('/')[-1]}.log",
            filemode='w')
        
        self.optimG.zero_grad()
        self.optimG.step()
        self.optim_maskG.zero_grad()
        self.optim_maskG.step()
        self.optimD.zero_grad()
        self.optimD.step()
        
        while True:
            self.epoch += 1      
            if self.config['distributed']:
                self.train_sampler.set_epoch(self.epoch)
            self._train_epoch(pbar)    
            # self.update_learning_rate()
            if self.iteration > self.train_args['iterations']:
                break
        print('\nEnd training....')
    
    def attention_loss(self, model, lq_paired_imgs, hq_paired_imgs, lq_unpaired_imgs, hq_unpaired_imgs):
        # Loss
        # step 1 =====================================

        head1_pair_lq, head2_pair_lq, _ = model(lq_paired_imgs, stage='mix') # paried lq images

        # get blemish label ------------------
        diff = torch.abs(hq_paired_imgs - lq_paired_imgs).mean(dim=1, keepdim=True)
        if mask is not None:
            diff = diff * mask
        diff_normed = (diff - diff.amin(dim=[2,3],keepdim=True)) / (diff.amax(dim=[2,3],keepdim=True) - diff.amin(dim=[2,3],keepdim=True))
        for i in range(len(lq_paired_imgs)):
            if diff[i].sum() == 0:
                diff_normed[i].fill_(0)
        
        diff_normed = sigmoid(diff_normed)
        diff_normed[diff_normed == 0.5] = 0.0
        # ------------------------------------

        lq_atten_loss = torch.tensor(0)
        head2_loss = torch.tensor(0)

        # multi-scale blemish mask loss for paired LQ images
        for gen_mask in head1_pair_lq:
            label = F.interpolate(diff_normed, size=gen_mask.shape[-2:], mode='bilinear', align_corners=True)
            loss = F.l1_loss(gen_mask, label)
            lq_atten_loss = lq_atten_loss + loss

        # cross entropy loss for paired LQ images
        labels = torch.zeros_like(diff_normed)
        labels[diff_normed>0] = 1
        head2_loss = F.binary_cross_entropy(head2_pair_lq, labels)

        stage_1_loss = (1.0*lq_atten_loss + 0.1 * head2_loss) * self.config['losses']['mask_weight']

        model.zero_grad()
        stage_1_loss.backward()

        print(stage_1_loss)

        # step 2 ===============================================

        head1_unpair_hq, _, _ = model(hq_unpaired_imgs,stage='mix') # unpaired hq images
        hq_atten_loss = torch.tensor(0)
        # multi-scale blemish mask loss for unpaired HQ images
        for gen_mask in head1_unpair_hq:
            loss = F.relu(gen_mask-0.7).mean()
            hq_atten_loss = hq_atten_loss + loss
        stage_2_loss = 0.1*hq_atten_loss

        stage_2_loss.backward()

        print(stage_2_loss)

        # step 3 ======================================

        head1_pair_lqhq, _, _ = model([lq_paired_imgs, hq_paired_imgs], stage='mask') # paired data

        # get blemish label ------------------
        diff = torch.abs(hq_paired_imgs - lq_paired_imgs).mean(dim=1, keepdim=True)
        if mask is not None:
            diff = diff * mask
        diff_normed = (diff - diff.amin(dim=[2,3],keepdim=True)) / (diff.amax(dim=[2,3],keepdim=True) - diff.amin(dim=[2,3],keepdim=True))
        for i in range(len(lq_paired_imgs)):
            if diff[i].sum() == 0:
                diff_normed[i].fill_(0)
        diff_normed = sigmoid(diff_normed)
        diff_normed[diff_normed==0.5]=0.0
        # ------------------------------------
        lqhq_atten_loss = torch.tensor(0)

        for gen_mask in head1_pair_lqhq:
            label = F.interpolate(diff_normed, size=gen_mask.shape[-2:], mode='bilinear', align_corners=True)
            loss = F.l1_loss(gen_mask, label)
            lqhq_atten_loss =  lqhq_atten_loss + loss

        stage_3_loss = 0.8*lqhq_atten_loss * self.config['losses']['mask_weight'] #+ self.consi_loss_weight * consi_loss

        stage_3_loss.backward()

        print(stage_3_loss)

        # step 4 =====================================================

        entropy_loss = torch.tensor(0)
        _, head2_unpair_lq, _ = model(lq_unpaired_imgs,stage='mix') # # unpaired LQ images
        entropy_loss = cal_entropy(head2_unpair_lq).mean()
        stage_4_loss = 0.2*entropy_loss
        
        self.optim_maskG.zero_grad()
        stage_4_loss.backward()
        self.optim_maskG.step()
        print(stage_4_loss)
        return stage_1_loss, stage_2_loss, stage_3_loss, stage_4_loss


    def unpair(self, unpair_tensor):
        # unpair loss
        unpair_loss = 0
        unpair_pred = self.netD(unpair_tensor)
        _, attention = self.netG(unpair_tensor)
        atten_acc = torch.zeros_like(unpair_pred)
        for atten in attention:
            atten_acc += atten.view(unpair_pred.shape)
        unpair_loss += (F.softplus(unpair_pred * atten_acc)).mean()
        self.optimD.zero_grad()
        unpair_loss.backward()
        self.optimD.step()
        return unpair_loss


    def pair(self, source_tensor, target_tensor):
        b, c, h, w = source_tensor.size()
        pred_imgs, _ = self.netG(source_tensor)
        pred_imgs = pred_imgs.view(b, c, h, w)

        gen_loss = 0
        dis_loss = 0
        
        real_clip = self.netD(target_tensor)
        fake_clip = self.netD(pred_imgs.detach())
        dis_real_loss = self.adversarial_loss(real_clip, True, True)
        dis_fake_loss = self.adversarial_loss(fake_clip, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2
        self.add_summary(self.dis_writer, 'loss/dis_vid_fake', dis_fake_loss.item())
        self.add_summary(self.dis_writer, 'loss/dis_vid_real', dis_real_loss.item())
        self.optimD.zero_grad()
        dis_loss.backward()
        self.optimD.step()

        # generator adversarial loss
        gen_clip = self.netD(pred_imgs)
        gan_loss = self.adversarial_loss(gen_clip, True, False)
        gan_loss = gan_loss * self.config['losses']['adversarial_weight']
        gen_loss += gan_loss
        self.add_summary(self.gen_writer, 'loss/gan_loss', gan_loss.item())

        # generator l1 loss
        valid_loss = self.l1_loss(pred_imgs, target_tensor)
        valid_loss = valid_loss * self.config['losses']['valid_weight']
        gen_loss += valid_loss
        self.add_summary(self.gen_writer, 'loss/valid_loss', valid_loss.item())
            
        # VGG loss
        vgg_loss = self.vgg_loss(pred_imgs, target_tensor)
        vgg_loss = vgg_loss * self.config['losses']['vgg_weight']
        gen_loss += vgg_loss
        self.add_summary(self.gen_writer, 'loss/vgg_loss', vgg_loss.item())
        
        self.optimG.zero_grad()
        gen_loss.backward()
        self.optimG.step()            
        return dis_loss, valid_loss, vgg_loss

    def _train_epoch(self, pbar):
        """Process input and calculate loss every training epoch"""
        device = self.config['device']
        for (source_tensor, target_tensor), (unpair_tensor_lq, unpair_tensor_hq) in zip(self.train_loader, self.unpair_loader):
            self.iteration += 1
            source_tensor, target_tensor = source_tensor.to(device), target_tensor.to(device)
            unpair_tensor_lq, unpair_tensor_hq = unpair_tensor_lq.to(device), unpair_tensor_hq.to(device)
            stage_1_loss, stage_2_loss, stage_3_loss, stage_4_loss = self.attention_loss(self.netG.soft_mask, source_tensor, target_tensor, unpair_tensor_lq, unpair_tensor_hq)
            dis_loss, valid_loss, vgg_loss = self.pair(source_tensor, target_tensor)
            unpair_loss = self.unpair(unpair_tensor_lq)
            if self.iteration % 14e3 == 0:
                self.update_learning_rate()            
            # console logs
            if self.config['global_rank'] == 0:
                pbar.update(1)
                lr = self.get_lr()
                pbar.set_description((f"d: {dis_loss.item():.3f}; "
                                      f"valid: {valid_loss.item():.3f}; "
                                      f"vgg_loss: {vgg_loss.item():.3f}; "
                                      f"stage_1_loss: {stage_1_loss.item():.3f}; "
                                      f"stage_2_loss: {stage_2_loss.item():.3f}; "
                                      f"stage_3_loss: {stage_3_loss.item():.3f}; "
                                      f"stage_4_loss: {stage_4_loss.item():.3f}; "
                                      f"unpair_loss: {unpair_loss.item():.6f}; "
                                      f"lr: {lr:.6f}"
                                     ))
                
                if self.wandb:
                    wandb.log({
                    "dis" :dis_loss.item(),
                    "valid":valid_loss.item(),
                    "vgg_loss":vgg_loss.item(),
                    "unpair_loss": unpair_loss.item(),
                    "stage_1": stage_1_loss.item(),
                    "stage_2": stage_2_loss.item(),
                    "stage_3": stage_3_loss.item(),
                    "stage_4": stage_4_loss.item(),
                    "lr": lr
                })
            # saving models
            if self.iteration % self.train_args['save_freq'] == 0:
                self.save(int(self.iteration))
                self.eval_mask(self.iteration)
                if self.iteration == 2e4:
                    self.test(self.iteration, lr = self.get_lr())
                elif self.iteration == 6e4:
                    self.test(self.iteration, lr = self.get_lr())
                elif self.iteration == 8e4:
                    self.test(self.iteration, lr = self.get_lr())                
                elif self.iteration > 10e4-1:
                    self.test(self.iteration, lr = self.get_lr())
            if self.iteration > self.train_args['iterations']:
                break