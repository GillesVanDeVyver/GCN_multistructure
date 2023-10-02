import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from nets.encoders import image_encoder
from typing import List, Tuple
import os

"""
Code for spiral convolutions adapted from 
https://github.com/gbouritsas/Neural3DMM
Bouritsas, G. et al.: Neural 3D Morphable Models: Spiral Convolutional Networks for 3D Shape Representation Learning and Generation, ICCV, 2019
"""
class SpiralConv_multistructure(nn.Module):
    def __init__(self, in_channels, out_channels, indices_inner,indices_outer, dim=1, tsteps=1,
                 is_gpu=True):
        super(SpiralConv_multistructure, self).__init__()
        self.dim = dim
        self.indices_inner = indices_inner
        self.indices_outer = indices_outer
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length_inner = indices_inner.size(1)
        if len(indices_outer)==0:
            self.seq_length_outer = 0
        else:
            self.seq_length_outer = indices_outer.size(1)

        if tsteps==2:
            self.cycle = 1
        elif tsteps==1:
            self.cycle = 0
        else:
            self.cycle = 2

        self.is_gpu = is_gpu

        self.inner_layer = nn.Linear(in_channels * (self.seq_length_inner+self.cycle), out_channels)
        self.outer_layer = nn.Linear(in_channels * (self.seq_length_outer+self.cycle), out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.inner_layer.weight)
        torch.nn.init.constant_(self.inner_layer.bias, 0)
        torch.nn.init.xavier_uniform_(self.outer_layer.weight)
        torch.nn.init.constant_(self.outer_layer.bias, 0)

    def forward(self, x):
        nb_nodes_inner = self.indices_inner.size()[0]
        nb_nodes_outer = self.indices_outer.size()[0]
        if x.dim() == 3:
            bs = x.size(0)
            inner_x = x[:,:nb_nodes_inner,:]
            outer_x = x[:, nb_nodes_inner:, :]
            zeros = torch.zeros(bs,nb_nodes_inner-nb_nodes_outer,self.in_channels)
            if self.is_gpu:
                zeros = zeros.cuda()
            else:
                zeros = zeros.cpu()
            outer_x_padded = torch.cat((outer_x,zeros),1)
            x_padded = torch.cat((inner_x,outer_x_padded),1)
            inner_x_output = torch.index_select(x_padded,self.dim,self.indices_inner.view(-1))
            inner_x_output = inner_x_output.view(bs, nb_nodes_inner, -1)
            inner_x_output = self.inner_layer(inner_x_output)
            if len(self.indices_outer)!=0:
                outer_x_output = torch.index_select(x_padded,self.dim,self.indices_outer.view(-1))
                outer_x_output = outer_x_output.view(bs, nb_nodes_outer, -1)
                outer_x_output = self.outer_layer(outer_x_output)
                x = torch.cat((inner_x_output,outer_x_output),1)
            else:
                x = inner_x_output
        else:
            raise RuntimeError(
                'x.dim() is expected to be 3 but received {}'.format(
                    x.dim()))
        return x

    def __repr__(self):
        return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.seq_length)


class kpts_decoder_multistructure(nn.Module):

    def __init__(self, features_size, kpt_channels, gcn_channels,range_inner_outer,
                 num_kpts=[43,21,43], tsteps=1, is_gpu=True,
                 ):

        super(kpts_decoder_multistructure, self).__init__()

        self.kpt_channels = kpt_channels
        self.gcn_channels = gcn_channels

        # construct nodes for graph CNN decoder:
        self.num_kpts = num_kpts
        self.num_nodes = np.sum(num_kpts)

        # construct edges for graph CNN decoder:
        adjacency_inner,adjacency_outer = self.create_graph(self.num_kpts,
                                                            range_inner_outer)

        self.is_gpu=is_gpu

        # init GCN:
        self.init_gcn(adjacency_inner,adjacency_outer, features_size, tsteps)


    def get_index(self,index,nb_kpts_ring,start_index_ring):
        stop_index_ring = start_index_ring+nb_kpts_ring-1
        if index>stop_index_ring:
            return index - nb_kpts_ring
        elif index<start_index_ring:
            return index + nb_kpts_ring
        else:
            return index





    def create_graph(self, num_kpts: [int],
                     range_inner_outer) -> np.ndarray:
        if num_kpts[0]!=num_kpts[2]:
            raise ValueError('ERROR number of keypoints of epicardium must be equal to number of points for lv')
        adjacency_inner = []
        adjacency_outer = []
        nb_kpts_lv,nb_kpts_la,nb_kpts_ep = num_kpts
        nb_kpts_inner_ring = nb_kpts_lv+nb_kpts_la # lv+la as inner ring, ep as outer ring
        # inner ring and outer ring are each fully connected
        # range_inner_outer connections between inner ring and corresponding outer ring
        # use zero padding to accommodate for size difference between inner and outer ring
        # the points are ordered counter clockwise starting from the endocardium right after the left base point
        # continuing to the contour of the left atrium, then the epicardium starting from the left side
        # then finally extra imaginary points or zero padding is appended at the end to get two rings of the same size
        adjacency_inner_row = list(range(nb_kpts_inner_ring))
        outer_index = nb_kpts_inner_ring
        if range_inner_outer%2==0:
            uneven_offset = 1
            offset_in_out = int((range_inner_outer-2)/2)
        else:
            uneven_offset = 0
            offset_in_out = int((range_inner_outer-1)/2)
        for ii in range(nb_kpts_inner_ring):
            adjacency_inner_row_to_add = adjacency_inner_row.copy()
            for outer_index_conn in range(outer_index-offset_in_out-uneven_offset,outer_index+offset_in_out+1):
                outer_index_corrected = self.get_index(outer_index_conn,nb_kpts_inner_ring,nb_kpts_inner_ring)
                adjacency_inner_row_to_add.append(outer_index_corrected)
            adjacency_inner.append(adjacency_inner_row_to_add)
            adjacency_inner_row = adjacency_inner_row[1:] + adjacency_inner_row[:1] # rotate by 1
            outer_index+=1
        adjacency_inner = np.array(adjacency_inner)

        adjacency_outer_row = list(range(nb_kpts_inner_ring,nb_kpts_inner_ring+nb_kpts_ep))
        inner_index = 0
        for ii in range(nb_kpts_inner_ring,nb_kpts_inner_ring+nb_kpts_ep):
            adjacency_outer_row_to_add = adjacency_outer_row.copy()
            for inner_index_conn in range(inner_index-offset_in_out-uneven_offset,inner_index+offset_in_out+1):
                inner_index_corrected = self.get_index(inner_index_conn,nb_kpts_inner_ring,0)
                adjacency_outer_row_to_add.append(inner_index_corrected)
            adjacency_outer.append(adjacency_outer_row_to_add)
            adjacency_outer_row = adjacency_outer_row[1:] + adjacency_outer_row[:1] # rotate by 1
            inner_index+=1
        adjacency_outer = np.array(adjacency_outer)
        return adjacency_inner,adjacency_outer

    def init_gcn(self, adjacency_inner: np.ndarray,adjacency_outer:np.ndarray, features_size: int, tsteps: int):

        self.spiral_indices_inner = torch.from_numpy(adjacency_inner)
        self.spiral_indices_outer = torch.from_numpy(adjacency_outer)

        if not self.is_gpu:
            self.spiral_indices_inner = self.spiral_indices_inner.cpu()
            self.spiral_indices_outer = self.spiral_indices_outer.cpu()
        else:
            self.spiral_indices_inner = self.spiral_indices_inner.cuda()
            self.spiral_indices_outer = self.spiral_indices_outer.cuda()

        # construct graph CNN layers:
        self.decoder_layers = nn.ModuleList()
        self.decoder_layers.append(nn.Linear(features_size, self.num_nodes * self.gcn_channels[-1]))
        for idx in range(len(self.gcn_channels)):
            if idx == 0:
                self.decoder_layers.append(
                    SpiralConv_multistructure(self.gcn_channels[-idx - 1],
                               self.gcn_channels[-idx - 1],
                               self.spiral_indices_inner,self.spiral_indices_outer, tsteps=tsteps,
                                              is_gpu=self.is_gpu))
            else:
                self.decoder_layers.append(
                    SpiralConv_multistructure(self.gcn_channels[-idx], self.gcn_channels[-idx - 1],
                               self.spiral_indices_inner,self.spiral_indices_outer, tsteps=tsteps,
                                              is_gpu=self.is_gpu))
        self.decoder_layers.append(
            SpiralConv_multistructure(self.gcn_channels[0], self.kpt_channels,
                                      self.spiral_indices_inner,self.spiral_indices_outer, tsteps=tsteps,
                                          is_gpu=self.is_gpu))

    def forward(self, x):
        num_layers = len(self.decoder_layers)
        for i, layer in enumerate(self.decoder_layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.num_nodes, self.gcn_channels[-1])
            elif i != num_layers - 1:
                x = F.elu(layer(x))
            else:
                x = layer(x)
        return x



class GCN_multi(nn.Module):
    def __init__(self, kpt_channels, gcn_channels,range_inner_outer, backbone=18, num_kpts=[43,21,43], is_gpu=True):

        super(GCN_multi, self).__init__()

        self.image_encoder = image_encoder(backbone=backbone)

        self.kpts_decoder = kpts_decoder_multistructure(features_size=self.image_encoder.img_feature_size,
                                         kpt_channels=kpt_channels,
                                         gcn_channels=gcn_channels,
                                         num_kpts=num_kpts,
                                         is_gpu=is_gpu,
                                         range_inner_outer=range_inner_outer)

    def forward(self, x):
        features = self.image_encoder(x)
        kpts = self.kpts_decoder(features)
        return kpts


if __name__ == '__main__':
    gpu = True if torch.cuda.is_available() else False
    print('gpu: ', gpu)
    batch_size = 4
    num_kpts = [43,21,43] #lv,la,ep
    kpt_channels = 2
    img = torch.rand(batch_size, 3, 256, 256)
    if gpu:
        img=img.cuda()
    m = GCN_multi(range_inner_outer=11,
        kpt_channels=2, gcn_channels=[4, 8, 8, 16, 16, 32, 32, 48], backbone='mobilenet2',
                                   num_kpts=num_kpts, is_gpu=gpu)
    img = torch.rand(batch_size, 3, 256, 256)
    if gpu:
        img=img.cuda()
    m = m
    if gpu:
        m = m.cuda()
    o = m(img)
    kpts = torch.rand(batch_size, (num_kpts[0]+num_kpts[1]+num_kpts[2]), kpt_channels)
    if gpu:
        kpts = kpts.cuda()
    l2loss = nn.L1Loss()
    print(l2loss(o, kpts))
    optimizer = torch.optim.Adam(m.parameters(), lr=1e-4)
    optimizer.step()
    total_nb_params = sum(p.numel() for p in m.parameters())
    print(total_nb_params)

