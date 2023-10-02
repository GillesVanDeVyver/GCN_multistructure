import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from nets.encoders import image_encoder
from typing import List, Tuple
import os


class backbone_only_displacement(nn.Module):
    def __init__(self,num_kpts, num_distances,backbone):

        super(backbone_only_displacement, self).__init__()

        self.image_encoder = image_encoder(backbone=backbone)
        self.num_kpts = num_kpts
        self.num_distances= num_distances

        self.decoder_layer = nn.Linear(self.image_encoder.img_feature_size, num_kpts*2+num_distances)


    def forward(self, x):
        features = self.image_encoder(x)
        out = self.decoder_layer(features)
        kpts,distances = out[:,:self.num_kpts*2],out[:,self.num_kpts*2:]
        kpts_reshaped = torch.reshape(kpts,(kpts.size(dim=0),self.num_kpts,2))
        return kpts_reshaped,distances

class backbone_only(nn.Module):
    def __init__(self,num_kpts,backbone):

        super(backbone_only, self).__init__()

        self.image_encoder = image_encoder(backbone=backbone)
        self.num_kpts = num_kpts

        self.decoder_layer = nn.Linear(self.image_encoder.img_feature_size, num_kpts*2)


    def forward(self, x):
        features = self.image_encoder(x)
        out = self.decoder_layer(features)
        kpts_reshaped = torch.reshape(out,(out.size(dim=0),self.num_kpts,2))
        return kpts_reshaped


if __name__ == '__main__':
    from losses import load_loss
    batch_size = 4
    num_kpts = [43,21,43]
    kpt_channels = 2 # kpts dim
    img = torch.rand(batch_size, 3, 256, 256).cuda()
    #m = CNN_GCN(kpt_channels=kpt_channels, gcn_channels=[16, 32, 32, 48], backbone='mobilenet2_quantize', num_kpts=num_kpts)

    m = backbone_only(107,backbone='mobilenet2')
    loss = load_loss("L2", device='cuda', class_weights=None)
    img = torch.rand(batch_size, 3, 256, 256).cuda()
    m = m.cuda()
    o = m(img)
    kpts_output = o
    kpts_random = torch.rand(batch_size, (num_kpts[0]+num_kpts[1]+num_kpts[2]), kpt_channels).cuda()
    print(loss(kpts_output, kpts_random))

    optimizer = torch.optim.Adam(m.parameters(), lr=1e-4)
    optimizer.step()
    total_nb_params = sum(p.numel() for p in m.parameters())
    print(total_nb_params)
    '''
    m = backbone_only_displacement(64, 43,backbone='mobilenet2')
    loss = load_loss("L2_distances", device='cuda', class_weights=None)
    img = torch.rand(batch_size, 3, 256, 256).cuda()
    m = m.cuda()
    o = m(img)
    kpts_output,distance_output = o
    kpts_inner_random = torch.rand(batch_size, (num_kpts[0]+num_kpts[1]), kpt_channels).cuda() #num_kpts[2]
    target_distances_random = torch.rand(batch_size, (num_kpts[2])).cuda()
    print(loss(kpts_output, distance_output, kpts_inner_random, target_distances_random))

    optimizer = torch.optim.Adam(m.parameters(), lr=1e-4)
    optimizer.step()
    total_nb_params = sum(p.numel() for p in m.parameters())
    print(total_nb_params)
    '''