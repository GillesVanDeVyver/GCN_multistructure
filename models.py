import torch
from fvcore.common.config import CfgNode

# models:
from nets.CNN_GCN import CNN_GCN
from nets.EFNet import EFNet
from nets.EFKptsNet import EFKptsNet
from nets.EFKptsSDNet import EFKptsNetSDNet
from nets.EFGCN import EFGCN
from nets.EFGCNTmp import EFGCNTmp
from nets.EFGCNSD import EFGCNSD
from nets.GCN_multi import GCN_multi
from nets.GCN_multi_displacement import GCN_multi_displacement
from nets.backbone_only import backbone_only, backbone_only_displacement

import os

def load_freezed_model(weights_filename: str=None, is_gpu: bool = True,continue_train: bool = False):
    model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if weights_filename is not None and os.path.exists(weights_filename):
        checkpoint = torch.load(weights_filename,map_location=device)
        print('freezed model epoch is %d' % checkpoint['epoch'])
        if 'cfg' in checkpoint:
            print(checkpoint['cfg'])
            cfg = checkpoint['cfg']
            model = load_model(cfg, is_gpu=is_gpu)
            model.load_state_dict(checkpoint['model_state_dict'])

    if continue_train == False:
        for name, p in model.named_parameters():
            #if "kpts_decoder" in name: #freeze image encoder part only
            p.requires_grad = False
    return model

def load_model(cfg:CfgNode, is_gpu: bool = True, kpts_extractor_weights: str = None) -> torch.nn.Module:

    model_name = cfg.MODEL.NAME
    backbone = cfg.MODEL.BACKBONE

    try:
        num_kpts = cfg.TRAIN.NUM_KPTS_MULTISTRUCTURE
    except:
        num_kpts = None
    if num_kpts is None:
        num_kpts = cfg.TRAIN.NUM_KPTS


    kpts_extractor_weights = cfg.KPTS_EXTRACTOR_WEIGHTS

    print("loading network type: {}..".format(model_name))
    if model_name == 'CNNGCN_freezed':
        model = load_freezed_model(weights_filename=kpts_extractor_weights, is_gpu=is_gpu, continue_train= False)
        model.output_type = 'img2kpts'

    elif model_name == 'CNNGCN':
        model = CNN_GCN(kpt_channels=2, gcn_channels=[16, 32, 32, 48], backbone=backbone, num_kpts=num_kpts, is_gpu=is_gpu)
        model.output_type = 'img2kpts'

    elif model_name == 'CNNGCNV2':
        model = CNN_GCN(kpt_channels=2, gcn_channels=[4, 4, 8, 8, 8, 16, 16, 16], backbone=backbone, num_kpts=num_kpts, is_gpu=is_gpu)
        model.output_type = 'img2kpts'

    elif model_name == 'CNNGCNV3':
        model = CNN_GCN(kpt_channels=2, gcn_channels=[4, 8, 8, 16, 16, 32, 32, 48], backbone=backbone, num_kpts=num_kpts, is_gpu=is_gpu)
        model.output_type = 'img2kpts'

    elif model_name == "EFNet":
        model = EFNet(backbone='r3d_18', MLP_channels=[16, 32, 32, 48])
        model.output_type = 'seq2ef'

    elif model_name == "EFKptsNet":
        model = EFKptsNet(backbone='r3d_18', MLP_channels_ef=[16, 32, 32, 48], MLP_channels_kpts=[256, 256, 256])
        model.output_type = 'seq2ef&kpts'

    elif model_name == "EFGCN":
        #model = EFGCN(backbone='r3d_18', MLP_channels_ef=[16, 32, 32, 48], GCN_channels_kpts=[2, 2, 3, 3, 4, 4, 4], is_gpu=is_gpu)
        model = EFGCN(backbone='r3d_18', MLP_channels_ef=[16, 32, 32, 48], GCN_channels_kpts=[2, 3, 4], is_gpu=is_gpu)
        model.output_type = 'seq2ef&kpts'

    elif model_name == "EFGCNTmp":
        model = EFGCNTmp(backbone='r3d_18', MLP_channels_ef=[16, 32, 32, 48], GCN_channels_kpts=[2, 3, 4], is_gpu=is_gpu)
        model.output_type = 'seq2ef&kpts'

    elif model_name == "EFGCNV2":
        #model = EFGCN(backbone='r3d_18', MLP_channels_ef=[16, 32, 32, 48], GCN_channels_kpts=[2, 2, 3, 3, 4, 4, 4], is_gpu=is_gpu)
        model = EFGCN(backbone='r3d_18', MLP_channels_ef=[16, 32, 32, 48], GCN_channels_kpts=[16, 32, 32, 48], is_gpu=is_gpu)
        model.output_type = 'seq2ef&kpts'

    elif model_name == "EFGCNTmpV2":
        model = EFGCNTmp(backbone='r3d_18', MLP_channels_ef=[16, 32, 32, 48], GCN_channels_kpts=[16, 32, 32, 48], is_gpu=is_gpu)
        model.output_type = 'seq2ef&kpts'

    elif model_name == "EFKptsNetSD":
        model = EFKptsNetSDNet(backbone='r3d_18', MLP_channels_ef=[16, 32, 32, 48], MLP_channels_kpts=[256, 256, 256])
        model.output_type = 'seq2ef&kpts&sd'

    elif model_name == "EFGCNSD":
        model = EFGCNSD(backbone='r3d_18', MLP_channels_ef=[16, 32, 32, 48], GCN_channels_kpts=[16, 32, 32, 48], is_gpu=is_gpu)
        model.output_type = 'seq2ef&kpts&sd'

    elif model_name == 'GCN_multi_large':
        model = GCN_multi(range_inner_outer=11,
                                           kpt_channels=2, gcn_channels=[4, 8, 8, 16, 16, 32, 32, 48],
                                           backbone='mobilenet2',
                                           num_kpts=num_kpts, is_gpu=True)
        model.output_type = 'img2kpts'

    elif model_name == 'GCN_multi_medium':
        model = GCN_multi(range_inner_outer=5,
                                      kpt_channels=2, gcn_channels=[4, 8, 16, 32], backbone='mobilenet2',
                                      num_kpts=num_kpts, is_gpu=True)
        model.output_type = 'img2kpts'

    elif model_name == 'GCN_multi_small':
        model = GCN_multi(range_inner_outer=1,
                                      kpt_channels=2, gcn_channels=[4, 8], backbone='mobilenet2',
                                      num_kpts=num_kpts, is_gpu=True)
        model.output_type = 'img2kpts'

    elif model_name == 'no_decoder':
        model = backbone_only(sum(num_kpts),backbone = 'mobilenet2')
        model.output_type = 'img2kpts'

    elif model_name == 'GCN_multi_displacement_large':
        model = GCN_multi_displacement(range_inner_outer=11,
        kpt_channels=2,kpt_channels_ep=1, gcn_channels=[4, 8, 8, 16, 16, 32, 32, 48], backbone=backbone,
                                   num_kpts=num_kpts, is_gpu=True)
        model.output_type = 'img2kpts&distances'

    elif model_name == 'GCN_multi_displacement_medium':
        model = GCN_multi_displacement(range_inner_outer=5,
        kpt_channels=2,kpt_channels_ep=1, gcn_channels=[4, 8, 16,32], backbone=backbone,
                                   num_kpts=num_kpts, is_gpu=True)
        model.output_type = 'img2kpts&distances'

    elif model_name == 'GCN_multi_displacement_small':
        model = GCN_multi_displacement(range_inner_outer=1,
                               kpt_channels=2, kpt_channels_ep=1, gcn_channels=[4, 8], backbone=backbone,
                               num_kpts=num_kpts, is_gpu=True)
        model.output_type = 'img2kpts&distances'


    elif model_name == 'no_decoder_displacement':
        model = backbone_only_displacement(sum(num_kpts),backbone = 'mobilenet2')
        model.output_type = 'img2kpts&distances'
    else:
        raise NotImplementedError("Model name is not supported..")

    print('loading model {} of network type: {} and output_type {}..'.format(model_name, model.__class__.__name__, model.output_type))

    return model



