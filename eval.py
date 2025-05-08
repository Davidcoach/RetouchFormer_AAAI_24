# -*- coding: utf-8 -*-
import importlib
import os
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from core.dataset import FaceRetouchingDataset
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description="RetouchFormer") #
parser.add_argument("-e", "--epoch", type=str, default="best")
parser.add_argument("-c", "--ckpt", type=str, default= "release_model")
parser.add_argument("--size",  type=int, default=512)
parser.add_argument("--model",  type=str, default='')
parser.add_argument("--input_path", type=str, default="datasets/retouching")  # wild image
args = parser.parse_args()

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    net = importlib.import_module('model.' + args.model)
    model = net.InpaintGenerator().to(device)
    print(args.model) 
    test_dataset = FaceRetouchingDataset(path = args.input_path, resolution=512, data_type="test", data_percentage=1)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    data = torch.load(path, map_location=device)
    model.load_state_dict(data)
    print('loading from: {}'.format(path))
    model.eval()

    cnt = 0
    PSNR = 0
    SSIM = 0
    LPIPS = 0
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    for source_tensor, target_tensor in tqdm(test_loader):
        with torch.no_grad():
            pred_img, atten = model(source_tensor)
            lpips_loss = loss_fn_alex(pred_img, target_tensor_list[-1].to(device)).mean()
            s_img = pred_img[0].cpu().numpy()
            t_img = target_tensor_list[-1][0].numpy()
            psnr = compare_psnr(t_img,s_img)
            ssim = compare_ssim(t_img,s_img, channel_axis=0)
            PSNR += psnr
            SSIM += ssim
            LPIPS+= lpips_loss
            cnt+=1
    PSNR /= cnt
    SSIM /= cnt
    LPIPS /= cnt
    print("PSNR:", PSNR)
    print("SSIM:", SSIM)
    print("LPIPS:", LPIPS)
    with open('results.txt', 'a') as f:
        f.writelines(f"{basename}: PSNR: {PSNR}; SSIM: {SSIM}; LPIPS: {LPIPS}\n")
    f.close()