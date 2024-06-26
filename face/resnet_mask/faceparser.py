#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os.path as osp
from PIL import Image
import torch
import torchvision.transforms as transforms

from .model import BiSeNet


class FaceParser:
    def __init__(self, device="cpu"):
        # mapper = [0, 1, 2, 3, 4, 5, 0, 11, 12, 0, 6, 8, 7, 9, 13, 0, 0, 10, 0]
        mapper = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        self.device = device
        self.dic = torch.tensor(mapper, device=device, requires_grad=False).unsqueeze(1) # pylint: disable=E1101
        save_pth = osp.split(osp.realpath(__file__))[0] + '/resnet.pth'

        net = BiSeNet(n_classes=19)
        net.load_state_dict(torch.load(save_pth, map_location=device))
        self.net = net.to(device).eval()
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.normalize = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    def parse(self, image: Image):
        assert image.shape[:2] == (512, 512)

        with torch.no_grad():
            image = self.to_tensor(image).to(self.device)
            image = torch.unsqueeze(image, 0) # pylint: disable=E1101
            out = self.net(image)[0]
            parsing = out.squeeze(0).argmax(0)
            parsing = torch.nn.functional.embedding(parsing, self.dic)

        return parsing.detach().float().squeeze(2)
    

    def parse_torch(self, image: Image):
        assert image.shape[-2:] == (512, 512)

        image = self.normalize(image)
        out = self.net(image)[0]
        parsing = out.argmax(1)
        parsing = torch.nn.functional.embedding(parsing, self.dic)

        return parsing.detach().float().squeeze(3)

