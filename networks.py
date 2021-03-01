import csv
from collections import Counter
import copy
import json
import functools
import gc
import logging
import math
import os
import pdb
import pickle
import random
import sys
import tables
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.autograd import Variable
from torch.nn import init, Parameter
from torch.utils.data import DataLoader
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchvision import datasets, transforms, models
import torch.optim.lr_scheduler as lr_scheduler

from sparsegnc_layer import sparsegcn
import adabound
from utils import *

def define_net(opt, k):
    net = None
    act_1 = define_act_layer(act_type=opt.act_type_1)
    act_2 = define_act_layer(act_type=opt.act_type_2)
    init_max = True if opt.init_type == "max" else False

    if opt.mode == "path_vgg":
        net = get_vgg(path_dim=opt.path_dim, act_1=act_1, act_2=act_2, label_dim_1=opt.label_dim_1, label_dim_2=opt.label_dim_2)
    elif opt.mode == "path_resnet":
        net = get_vgg(path_dim=opt.path_dim, act_1=act_1, act_2=act_2, label_dim_1=opt.label_dim_1, label_dim_2=opt.label_dim_2)
    elif opt.mode == "omic_gsnn":
        net = GSNN(input_dim=opt.input_size_omic, omic_dim=opt.omic_dim, dropout_rate=opt.dropout_rate, act_1=act_1, act_2=act_2, label_dim_1=opt.label_dim_1, label_dim_2=opt.label_dim_2, init_max=init_max)
    elif opt.mode == "omic_sgcn":
        net = SGCN(input_dim=opt.input_size_omic, omic_dim=opt.omic_dim, dropout_rate=opt.dropout_rate, act_1=act_1, act_2=act_2, label_dim_1=opt.label_dim_1, label_dim_2=opt.label_dim_2, init_max=init_max)
    elif opt.mode == "pathomic": # MultiCoFusion
        net = PathomicNet(opt=opt, act_1=act_1, act_2=act_2, k=k)
    elif opt.mode == "multimodalprognosis":
        net = MultimodalPrognosis()
    else:
        raise NotImplementedError('model [%s] is not implemented' % opt.model)
    return init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)


def define_optimizer(opt, model):
    optimizer = None
    if opt.optimizer_type == 'adabound':
        optimizer = adabound.AdaBound(model.parameters(), lr=opt.lr, final_lr=opt.final_lr)
    elif opt.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
    elif opt.optimizer_type == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, initial_accumulator_value=0.1)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % opt.optimizer)
    return optimizer

def define_reg(opt, model):
    loss_reg = None
    
    if opt.reg_type == 'none':
        loss_reg = 0
    elif opt.reg_type == 'path':
        loss_reg = regularize_path_weights(model=model)
    elif opt.reg_type == 'mm':
        loss_reg = regularize_MM_weights(model=model)
    elif opt.reg_type == 'all':
        loss_reg = regularize_weights(model=model)
    elif opt.reg_type == 'omic':
        loss_reg = regularize_MM_omic(model=model)
    else:
        raise NotImplementedError('reg method [%s] is not implemented' % opt.reg_type)
    return loss_reg

def define_scheduler(opt, optimizer):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)
    elif opt.lr_policy == 'step':
       scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
       scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
       scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
       return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def define_act_layer(act_type='Tanh'):
    if act_type == 'Tanh':
        act_layer = nn.Tanh()
    elif act_type == 'ReLU':
        act_layer = nn.ReLU()
    elif act_type == 'Sigmoid':
        act_layer = nn.Sigmoid()
    elif act_type == 'LSM':
        act_layer = nn.LogSoftmax(dim=1)
    elif act_type == "none":
        act_layer = None
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return act_layer

### Pathomic Fusion
def define_bifusion(fusion_type, skip=1, use_bilinear=1, gate1=1, gate2=1, dim1=32, dim2=32, scale_dim1=1, scale_dim2=1, mmhid=64, dropout_rate=0.25):
    fusion = None
    if fusion_type == 'pofusion':
        fusion = BilinearFusion(skip=skip, use_bilinear=use_bilinear, gate1=gate1, gate2=gate2, dim1=dim1, dim2=dim2, scale_dim1=scale_dim1, scale_dim2=scale_dim2, mmhid=mmhid, dropout_rate=dropout_rate)
    else:
        raise NotImplementedError('fusion type [%s] is not found' % fusion_type)
    return fusion

############
# SGCN
############
class SGCN(nn.Module):
    def __init__(self, input_dim=80, omic_dim=32, dropout_rate=0.25, act_1=None, act_2=None, label_dim_1=1, label_dim_2=1,  init_max=True):
        super(SGCN, self).__init__()
        hidden = [64, 48, 32, 32]
        self.act_1 = act_1
        self.act_2 = act_2

        self.encoder0_sgcn = sparsegcn(input_dim, input_dim)
        self.encoder0_selu = nn.SELU()
        self.encoder0_alphadropout = nn.AlphaDropout(p=dropout_rate, inplace=False)

        encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.SELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder2 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            nn.SELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder3 = nn.Sequential(
            nn.Linear(hidden[1], hidden[2]),
            nn.SELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder4 = nn.Sequential(
            nn.Linear(hidden[2], omic_dim),
            nn.SELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder1_2 = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.SELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder2_2 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            nn.SELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder3_2 = nn.Sequential(
            nn.Linear(hidden[1], hidden[2]),
            nn.SELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder4_2 = nn.Sequential(
            nn.Linear(hidden[2], omic_dim),
            nn.SELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        self.encoder_1 = nn.Sequential(encoder1, encoder2, encoder3, encoder4)
        self.encoder_2 = nn.Sequential(encoder1_2, encoder2_2, encoder3_2, encoder4_2)

        self.classifier_1 = nn.Sequential(nn.Linear(omic_dim, label_dim_1))
        self.classifier_2 = nn.Sequential(nn.Linear(omic_dim, label_dim_2))

        if init_max: init_max_weights(self)

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        x = kwargs['x_omic']
        gene_adj = kwargs['gene_adj']
        # clin = kwargs["clin"]
        x = self.encoder0_sgcn(x, gene_adj)
        x = self.encoder0_selu(x)
        x = self.encoder0_alphadropout(x)

        # features = self.encoder_1(x)
        # out_1 = self.classifier_1(features)
        # out_2 = self.classifier_2(features)
        features_1 = self.encoder_1(x)
        # # features_1 = torch.cat((features_1, clin), axis=1)
        out_1 = self.classifier_1(features_1)

        features_2 = self.encoder_2(x)
        # features_2 = torch.cat((features_2, clin), axis=1)
        out_2 = self.classifier_2(features_2)

        if self.act_1 is not None:
            out_1 = self.act_1(out_1)
            out_2 = self.act_2(out_2)

            if isinstance(self.act_1, nn.Sigmoid):
                out_1 = out_1 * self.output_range + self.output_shift

        return x, features_1, features_2, out_1, out_2


############
# Omic Model
############
class GSNN(nn.Module):
    def __init__(self, input_dim=80, omic_dim=32, dropout_rate=0.25, act_1=None, act_2=None, label_dim_1=1, label_dim_2=2, init_max=True):
        super(GSNN, self).__init__()
        hidden = [64, 48, 32, 32]
        self.act_1 = act_1
        self.act_2 = act_2

        self.encoder0 = nn.Linear(input_dim, input_dim)
        self.encoder0_elu = nn.ELU()
        self.encoder0_alphadropout = nn.AlphaDropout(p=dropout_rate, inplace=False)

        encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder2 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder3 = nn.Sequential(
            nn.Linear(hidden[1], hidden[2]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder4 = nn.Sequential(
            nn.Linear(hidden[2], omic_dim),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder1_2 = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder2_2 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder3_2 = nn.Sequential(
            nn.Linear(hidden[1], hidden[2]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder4_2 = nn.Sequential(
            nn.Linear(hidden[2], omic_dim),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        self.encoder_1 = nn.Sequential(encoder1, encoder2, encoder3, encoder4)
        self.encoder_2 = nn.Sequential(encoder1_2, encoder2_2, encoder3_2, encoder4_2)
        self.classifier_1 = nn.Sequential(nn.Linear(omic_dim, label_dim_1))
        self.classifier_2 = nn.Sequential(nn.Linear(omic_dim, label_dim_2))

        if init_max: init_max_weights(self)

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        x = kwargs['x_omic']
        ### for _2
        x = self.encoder0(x)
        x = self.encoder0_elu(x)
        x = self.encoder0_alphadropout(x)
        ### end _2
        # clin = kwargs["clin"]
        features_1 = self.encoder_1(x)
        features_2 = self.encoder_2(x)
        # features = torch.cat((features, clin), axis=1)
        out_1 = self.classifier_1(features_1)
        out_2 = self.classifier_2(features_2)
        if self.act_1 is not None:
            out_1 = self.act_1(out_1)
            out_2 = self.act_2(out_2)

            if isinstance(self.act_1, nn.Sigmoid):
                out_1 = out_1 * self.output_range + self.output_shift

        return features_1, features_2, out_1, out_2

############
# Path Model
############
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class PathNet_VGG(nn.Module):

    def __init__(self, features, path_dim=32, act_1=None, act_2=None, num_classes_1=1, num_classes_2=2):
        super(PathNet_VGG, self).__init__()
        self.features = features

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier_1 = nn.Sequential( ## for VGG
            nn.Linear(512 * 7 * 7, 1024), # (512 * 7 * 7, 1024)
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(1024, path_dim),
            nn.ReLU(True),
            nn.Dropout(0.05)
        )

        self.classifier_2 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(1024, path_dim),
            nn.ReLU(True),
            nn.Dropout(0.05)
        )
        self.linear_1 = nn.Linear(path_dim, num_classes_1)
        self.linear_2 = nn.Linear(path_dim, num_classes_2)
        self.act_1 = act_1
        self.act_2 = act_2

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

        dfs_freeze(self.features)

    def forward(self, **kwargs):
        x = kwargs['x_path']
        x = self.features(x)
        x = self.avgpool(x) ### for VGG
        x = x.view(x.size(0), -1) ### for VGG
        features_1 = self.classifier_1(x)
        features_2 = self.classifier_2(x)

        hazard_1 = self.linear_1(features_1)
        hazard_2 = self.linear_2(features_2)

        if self.act_1 is not None:
            hazard_1 = self.act_1(hazard_1)
            hazard_2 = self.act_2(hazard_2)

            if isinstance(self.act_1, nn.Sigmoid):
                hazard_1 = hazard_1 * self.output_range + self.output_shift

        return features_1, features_2, hazard_1, hazard_2

class PathNet_ResNet(nn.Module):

    def __init__(self, features, path_dim=32, act_1=None, act_2=None, num_classes_1=1, num_classes_2=2):
        super(PathNet_ResNet, self).__init__()
        self.features = features

        self.classifier_1 = nn.Sequential( ## for Resnet
            nn.Linear(1000, 512),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(256, path_dim),
            nn.ReLU(True),
            nn.Dropout(0.05)
        )

        self.classifier_2 = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(256, path_dim),
            nn.ReLU(True),
            nn.Dropout(0.05)
        )

        self.linear_1 = nn.Linear(path_dim, num_classes_1)
        self.linear_2 = nn.Linear(path_dim, num_classes_2)
        self.act_1 = act_1
        self.act_2 = act_2

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

        dfs_freeze(self.features)

    def forward(self, **kwargs):
        x = kwargs['x_path']
        x = self.features(x)
        features_1 = self.classifier_1(x)
        features_2 = self.classifier_2(x)

        hazard_1 = self.linear_1(features_1)
        hazard_2 = self.linear_2(features_2)

        if self.act_1 is not None:
            hazard_1 = self.act_1(hazard_1)
            hazard_2 = self.act_2(hazard_2)

            if isinstance(self.act_1, nn.Sigmoid):
                hazard_1 = hazard_1 * self.output_range + self.output_shift

        return x, features_1, features_2, hazard_1, hazard_2


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



def get_vgg(arch='vgg19_bn', cfg='E', act_1=None, act_2=None, batch_norm=True, label_dim_1=1, label_dim_2=1, pretrained=True, progress=True, **kwargs):
    model = PathNet_VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), act_1=act_1, act_2=act_2, num_classes_1=label_dim_1,num_classes_2=label_dim_2, **kwargs)

    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls[arch], progress=progress)

        for key in list(pretrained_dict.keys()):
            if 'classifier' in key: pretrained_dict.pop(key)

        model.load_state_dict(pretrained_dict, strict=False)
        print("Initializing Path Weights")

    return model

def get_resnet(act_1=None, act_2=None, label_dim_1=1, label_dim_2=1, pretrained=True, **kwargs):
    resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=pretrained)
    model = PathNet_ResNet(resnet, act_1=act_1, act_2=act_2, num_classes_1=label_dim_1,num_classes_2=label_dim_2, **kwargs)

    return model


##############################################################################
# Path + Omic （MultiCoFusion）
##############################################################################
class PathomicNet(nn.Module):
    def __init__(self, opt, act_1, act_2, k):
        super(PathomicNet, self).__init__()


        self.path_net = get_resnet(path_dim=opt.path_dim, act_1=act_1, act_2=act_2, label_dim_1=opt.label_dim_1, label_dim_2=opt.label_dim_2) # add by KT
        # self.path_net = get_vgg(path_dim=opt.path_dim, act_1=act_1, act_2=act_2, label_dim_1=opt.label_dim_1, label_dim_2=opt.label_dim_2) # add by KT
        self.omic_net = SGCN(input_dim=opt.input_size_omic, omic_dim=opt.omic_dim, dropout_rate=opt.dropout_rate, act_1=act_1, act_2=act_2, label_dim_1=opt.label_dim_1, label_dim_2=opt.label_dim_2, init_max=False)
        # self.omic_net = GSNN(input_dim=opt.input_size_omic, omic_dim=opt.omic_dim, dropout_rate=opt.dropout_rate, act_1=act_1,act_2=act_2, label_dim_1=opt.label_dim_1, label_dim_2=opt.label_dim_2, init_max=True)

        if k is not None:
            pt_fname = '_%d.pt' % k
            best_path_ckpt = torch.load(os.path.join(opt.checkpoints_dir, opt.exp_name, "Good/SM", 'path_resnet152_surv_grad_JT', 'path_resnet152_surv_grad_JT' + pt_fname),map_location=torch.device('cpu')) # add by KT
            self.path_net.load_state_dict(best_path_ckpt['model_state_dict']) #add by KT
            best_omic_ckpt = torch.load(os.path.join(opt.checkpoints_dir, opt.exp_name, "Good/SM", 'omic_sgcn_surv_grade_JT', 'omic_sgcn_surv_grade_JT'+pt_fname), map_location=torch.device('cpu'))
            self.omic_net.load_state_dict(best_omic_ckpt['model_state_dict'])

        self.fnn_omic = nn.Sequential(nn.Linear(10673, 1000), nn.ReLU())
        # self.fnn_path = nn.Sequential(nn.Linear(25088, 1000), nn.ReLU()) #for vgg

        self.fnn_omic_path = nn.Sequential(nn.Linear(2000, 512), nn.ReLU(),
                                 nn.Linear(512, 256), nn.ReLU(),
                                 nn.Linear(256, 32), nn.ReLU())
        self.classifier_1 = nn.Sequential(nn.Linear(32, 16), nn.ReLU(),
                                          nn.Linear(16, opt.label_dim_1))
        self.classifier_2 = nn.Sequential(nn.Linear(32, 16), nn.ReLU(),
                                          nn.Linear(16, opt.label_dim_2))

        self.act_1 = act_1
        self.act_2 = act_2

        dfs_freeze(self.omic_net)

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        path_x, path_vec_surv, path_vec_grad, _, _ = self.path_net(x_path = kwargs['x_path']) #add by KT
        omic_x, omic_vec_surv, omic_vec_grad, _, _ = self.omic_net(x_omic=kwargs['x_omic'], gene_adj=kwargs['gene_adj'])

        omic_vec = self.fnn_omic(omic_x)
        # path_vec = self.fnn_path(path_x) for vgg
        features = torch.cat((path_x, omic_vec), axis=1)
        features = self.fnn_omic_path(features)
        hazard_1 = self.classifier_1(features)
        hazard_2 = self.classifier_2(features)

        if self.act_1 is not None:
            hazard_1 = self.act_1(hazard_1)
            hazard_2 = self.act_2(hazard_2)

            if isinstance(self.act_1, nn.Sigmoid):
                hazard_1 = hazard_1 * self.output_range + self.output_shift

        return 0, 0, hazard_1, hazard_2

    def __hasattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return True
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return True
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return True
        return False


class Highway(nn.Module):

    def __init__(self, size, num_layers, f):
        super(Highway, self).__init__()

        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.f = f

    def forward(self, x):
        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear

        return x

class MultimodalPrognosis(nn.Module):

    def __init__(self):
        super(MultimodalPrognosis, self).__init__()

        self.fcg = nn.Linear(10673, 512) ### gene
        self.highway = Highway(512, 10, f=F.relu)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

        self.squeezenet = models.squeezenet1_0()
        self.fci = nn.Linear(1000, 512)

        self.fcd = nn.Linear(512, 1)


    def forward(self, **kwargs):

        z = kwargs['x_omic']
        z = self.fcg(z)
        z = self.highway(z)
        z = self.dropout(z)
        z = self.sigmoid(z)

        w = kwargs['x_path']
        w = self.squeezenet(w)
        w = self.fci(w)

        mean = (z + w)/2

        x = mean

        hazard = self.fcd(x)

        return z, w, hazard

### for HU and RHU
class model_loss_layer(nn.Module):
    def __init__(self):
        super(model_loss_layer, self).__init__()
        self.a1 = nn.init.uniform_(Parameter(torch.Tensor(1)), 0.2, 1)
        self.a2 = nn.init.uniform_(Parameter(torch.Tensor(1)), 0.2, 1)
        self.a3 = nn.init.uniform_(Parameter(torch.Tensor(1)), 0.2, 1)


    def forward(self, loss_cox, loss_nll, loss_reg):
        factor_1 = torch.div(1.0, torch.mul(2.0, self.a1))
        loss_1 = torch.add(torch.mul(factor_1, loss_cox), torch.log(1+self.a1))
        factor_2 = torch.div(1.0, torch.mul(2.0, self.a2))
        loss_2 = torch.add(torch.mul(factor_2, loss_nll), torch.log(1+self.a2))
        factor_3 = torch.div(1.0, torch.mul(2.0, self.a3))
        loss_3 = torch.add(torch.mul(factor_3, loss_reg), torch.log(1+self.a3))
        loss = torch.add(loss_1, torch.add(loss_2,loss_3))
        return loss
