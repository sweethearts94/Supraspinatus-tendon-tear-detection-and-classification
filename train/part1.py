import os
import numpy as np
from pkg_resources import DEVELOP_DIST
import torch
import torch.nn as nn
import torch.nn.functional as F
import statistic.common as s_common
from torch.utils.data.dataloader import DataLoader
from data.process.part1 import Loader, LoaderV2
from config.model import BATCH_SIZE, EPOCHS, MAX_LR, MIN_LR, SAVE_PATH, NPY_SAVE_PATH
from config.data import LABEL_PART_2_PATH as LABEL_PART
from net.part1 import EnhancedNet, Bottleneck
from utils.log import Logger
from tqdm import tqdm
from copy import deepcopy
from loss.part1 import Part1Loss, Part2Loss
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
MODELFILENAME = '{} EPOCH{}.pth'
NUMPYFILENAME = '{} EPOCH{} _{}.npy'
global_logger = Logger('part1')



def train_epoch(dt_loader: DataLoader, net: nn.Module, loss_list: list, optim, model_name: str='enhanced_model', epoch: int=0) -> float:
    process_bar = tqdm(dt_loader)
    net_list = list(net.named_parameters())
    loss_container = []
    loss_weight = [0.5, 0.2, 0.3, 1, 1]
    # loss_weight = [0, 0, 0, 0, 1]
    net.train()
    for _, (img_tensor, label_tensor, img_filepath) in enumerate(process_bar):
        label_tensor = torch.squeeze(label_tensor, dim=0)
        img_tensor = torch.squeeze(img_tensor, dim=0)
        label_tensor_cuda = label_tensor.to(DEVICE)
        img_tensor_cuda = img_tensor.to(DEVICE)
        net_output = net(img_tensor_cuda)
        net_loss = None
        for l_index in range(len(loss_list)):
            if net_loss is None:
                net_loss = loss_list[l_index](net_output[l_index], label_tensor_cuda[:, l_index].long()) * loss_weight[l_index]
            else:
                net_loss += loss_list[l_index](net_output[l_index], label_tensor_cuda[:, l_index].long()) * loss_weight[l_index]
        pre_56_grad = deepcopy(net_list[56][1].grad) if net_list[56][1].grad is not None else 0
        pre_213_grad = deepcopy(net_list[213][1].grad) if net_list[213][1].grad is not None else 0
        pre_215_grad = deepcopy(net_list[215][1].grad) if net_list[215][1].grad is not None else 0
        optim.zero_grad()
        net_loss.backward()
        process_bar.set_description('[Train] LOSS: {:.5f}, Grad1: {:.10f}, Grad2: {:.10f}, Grad3: {:.10f}, t1p: {:.2f}, t2p: {:.2f}'.format(
            net_loss,
            torch.max(net_list[56][1].grad - pre_56_grad).item() - torch.min(net_list[56][1].grad - pre_56_grad).item(),
            torch.max(net_list[213][1].grad - pre_213_grad).item() - torch.min(net_list[213][1].grad - pre_213_grad).item(),
            torch.max(net_list[215][1].grad - pre_215_grad).item() - torch.min(net_list[215][1].grad - pre_215_grad).item(),
            torch.mean(net_output[-2].max(dim=1)[0]).item(),
            torch.mean(net_output[-1].max(dim=1)[0]).item()
        ))
        optim.step()
        loss_container.append(net_loss.cpu().item())
    loss_numpy = np.array(loss_container)
    torch.save(net.state_dict(), '{}.pth'.format(model_name, epoch))
    return loss_numpy.mean()


def test(epoch_id: int, dt_loader: DataLoader, net: nn.Module, loss_list: list, loss_name: list) -> tuple:
    net.eval()
    result_map = {
        'rotator_cuff':{
            'loss':[],
            'pred': [],
            "netout": [],
            "youden": None,
            "ci95": None,
            'acc':None, 
            'recall':None, 
            'ppv':None, 
            'npv':None, 
            'cm':None, 
            'num_class':0
        }, 
        'right_left':{
            'loss':[],
            'pred': [],
            "netout": [],
            "youden": None,
            "ci95": None,
            'acc':None, 
            'recall':None, 
            'ppv':None, 
            'npv':None, 
            'cm':None, 
            'num_class':0
        }, 
        'sag_cor':{
            'loss':[],
            'pred': [],
            "netout": [],
            "youden": None,
            "ci95": None,
            'acc':None, 
            'recall':None, 
            'ppv':None, 
            'npv':None, 
            'cm':None, 
            'num_class':0
        }, 
        'tear_class1':{
            'loss':[],
            'pred': [],
            "netout": [],
            "youden": None,
            "ci95": None,
            'acc':None, 
            'recall':None, 
            'ppv':None, 
            'npv':None, 
            'cm':None, 
            'num_class':0
        }, 
        'tear_class2':{
            'loss':[],
            'pred': [],
            "netout": [],
            "youden": None,
            "ci95": None,
            'acc':None, 
            'recall':None, 
            'ppv':None, 
            'npv':None, 
            'cm':None, 
            'num_class':0
        }
    }
    num_classes = [2, 3, 2, 4, 6]
    loss_weight = [0.5, 0.2, 0.3, 1, 1.2]
    # loss_weight = [0, 0, 0, 0, 1]
    for img_tensor, label_tensor, img_filepath in tqdm(dt_loader):
        label_tensor_cuda = label_tensor.to(DEVICE)
        img_tensor_cuda = img_tensor.to(DEVICE)
        net_output = net(img_tensor_cuda)
        with torch.no_grad():
            for l_index in range(len(loss_list)):
                n_class = net_output[l_index].shape[-1]
                net_output_i = net_output[l_index]
                current_max, _ = net_output_i.max(dim=1)
                net_loss = loss_list[l_index](net_output[l_index], label_tensor_cuda[:, l_index].long())
                result_map[loss_name[l_index]]['loss'].append(net_loss.cpu().item() * loss_weight[l_index])
                result_map[loss_name[l_index]]['pred'] += current_max.cpu().numpy().tolist()
                result_map[loss_name[l_index]]['netout'].append(net_output_i.cpu().numpy())
                if result_map[loss_name[l_index]]['num_class'] == 0:
                    result_map[loss_name[l_index]]['num_class'] = n_class
                if result_map[loss_name[l_index]]['cm'] is None:
                    net_pred = net_output[l_index].argmax(dim=1)
                    result_map[loss_name[l_index]]['cm'] = s_common.confusion_matrix(label_tensor_cuda[:, l_index], net_pred, n_class)
                else:
                    net_pred = net_output[l_index].argmax(dim=1)
                    result_map[loss_name[l_index]]['cm'] += s_common.confusion_matrix(label_tensor_cuda[:, l_index], net_pred, n_class)
    statistics_func_map = {
        'acc': s_common.acc, 
        'recall': s_common.recall,
        'ppv': s_common.precision,
        'npv': s_common.npv,
        'youden': s_common.youden_index
    }
    for data_key in result_map.keys():
        result_map[data_key]["ci95"] = s_common.ci95(np.array(result_map[data_key]["pred"]))
        for s_name, s_func in statistics_func_map.items():
            for n_num_class in range(int(result_map[data_key]['num_class'])):
                if result_map[data_key][s_name] is None:
                    result_map[data_key][s_name] = [
                        s_func(result_map[data_key]['cm'], n_num_class)
                    ]
                else:
                    result_map[data_key][s_name].append(s_func(result_map[data_key]['cm'], n_num_class))
    s_npy_path_list = list(NPY_SAVE_PATH) + [NUMPYFILENAME.format('enchanced_cbam', epoch_id, 'statistics')]
    s_npy_path = (os.path.join)(*s_npy_path_list)
    np.save(s_npy_path, result_map)
    loss_c = [item['loss'] for item in result_map.values()]
    loss_c = np.array(loss_c)
    return loss_c.sum(0)


def train_resnet_cbam():
    backbone = EnhancedNet(Bottleneck, [3, 4, 3, 2], [2, 3, 2, 4, 6])
    num_class = [2, 3, 2, 4, 6]
    optim = torch.optim.AdamW(backbone.parameters(), lr=MAX_LR)
    optim_lr_s = torch.optim.lr_scheduler.CosineAnnealingLR(optim, EPOCHS, MIN_LR)
    name_list = ["rotator_cuff", "right_left", "sag_cor", "tear_class1", "tear_class2"]
    loss_list = []
    for i in range(len(name_list)):
        if name_list[i] == 'tear_class1':
            loss_list.append(nn.CrossEntropyLoss(weight=(torch.Tensor([0.1, 3, 0.3, 1]).to(DEVICE))))
        elif name_list[i] == 'tear_class2':
            loss_list.append(nn.CrossEntropyLoss())
        elif name_list[i] == "rotator_cuff":
            loss_list.append(nn.CrossEntropyLoss(weight=(torch.Tensor([1, 0.2]).to(DEVICE))))
        elif name_list[i] == "right_left":
            loss_list.append(nn.CrossEntropyLoss())
        else:
            loss_list.append(nn.CrossEntropyLoss())
    train_loader = Loader(False, 'train')
    val_loader = Loader(False, 'val')
    train_loaderv2 = LoaderV2(False, 'train', BATCH_SIZE, [14, 10, 10, 10, 10, 10])
    train_dt = DataLoader(train_loader, batch_size=BATCH_SIZE, shuffle=True, num_workers=20)
    val_dt = DataLoader(val_loader, batch_size=BATCH_SIZE, shuffle=False, num_workers=20)
    train_dtv2 = DataLoader(train_loaderv2, batch_size=1, shuffle=True, num_workers=20)
    backbone = backbone.to(DEVICE)
    for epoch in range(EPOCHS):
        epoch_train_loss = train_epoch(train_dtv2, backbone, loss_list, optim, epoch=epoch)
        print('[Train] Summary Loss: {}'.format(epoch_train_loss))
        epoch_val_loss = test(epoch, val_dt, backbone, loss_list, name_list)
        print('[Test] Summary Loss: {}'.format(epoch_val_loss.mean()))
        optim_lr_s.step()
        print('[LR CHANGE] Now LR: {}'.format(optim.state_dict()['param_groups'][0]['lr']))