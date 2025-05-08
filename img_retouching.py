# -*- coding: utf-8 -*-
import importlib
import os
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from core.dataset import wildDataset
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description="RetouchFormer") #
parser.add_argument("-e", "--epoch", type=str, default="best")
parser.add_argument("-c", "--ckpt", type=str, default= "release_model")
parser.add_argument("--size",  type=int, default=512)
parser.add_argument("--model", type=str, default='RetouchFormer')
parser.add_argument("--input_path", type=str, default="datasets/test")  # wild image
parser.add_argument("--save_path", type=str, default= "results")
args = parser.parse_args()

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = importlib.import_module('model.' + args.model)
    model = net.InpaintGenerator().to(device)
    print(args.model)
    data = torch.load("{0}/gen_{1}.pth".format(args.ckpt, args.epoch), map_location=device)
    model.load_state_dict(data)
    print('loading from: {}'.format("{0}/gen_{1}.pth".format(args.ckpt, args.epoch)))
    model.eval()
    test_dataset = wildDataset(args.input_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(save_path)
    for name, source_tensor in tqdm(test_loader):
        name = name[0]
        with torch.no_grad():
            pred_img, _ = model(source_tensor.to(device))
            path = os.path.join(save_path, f"{str(name)}_out.png")
            save_image(pred_img, path, normalize=True, value_range=(-1, 1))
