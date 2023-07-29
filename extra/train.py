import torch
import torch.nn as nn
import torch.nn.functional
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from dataloader import Loader
from model import make_vgg16, make_resnet18
from pathlib import Path
import numpy as np
import common as s_common
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

LR_MAX = 1e-4
LR_MIN = 1e-7
EPOCHS = 80
BS = 64
MODELFILENAME = '{} EPOCH{}.pth'
NUMPYFILENAME = '{} EPOCH{} _{}.npy'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def train_epoch(epoch_id: int, net: nn.Module, dt_loader: DataLoader, loss_function: nn.Module, optim):
    net.train()
    process_bar = tqdm(dt_loader)
    loss_c = []
    log_softmax = nn.Softmax(dim=1)
    for image, label in process_bar:
        image_cuda = image.to(DEVICE)
        label_cuda = label.to(DEVICE)
        net_out = net(image_cuda)
        net_out = log_softmax(net_out * 10)
        loss = loss_function(net_out, label_cuda)
        process_bar.set_description("[Train] Epoch: {} Loss: {}".format(epoch_id, loss.cpu().item()))
        optim.zero_grad()
        loss.backward()
        optim.step()
        loss_c.append(loss.cpu().item())
    save_path = Path("/root/Graduation thesis/model")
    torch.save(net.state_dict(), save_path / '{}_{}.pth'.format("classification", epoch_id))
    return np.mean(np.array(loss_c)).item()


def test_epoch(epoch_id: int, net: nn.Module, dt_loader: DataLoader, loss_function: nn.Module):
    net.eval()
    result_map = {
        'loss':[],
        'pred': [],
        "youden": None,
        "ci95": None,
        'acc':None, 
        'recall':None, 
        'ppv':None, 
        'npv':None, 
        'cm':None, 
        'num_class':2
    }
    process_bar = tqdm(dt_loader)
    log_softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for image, label in process_bar:
            image_cuda = image.to(DEVICE)
            label_cuda = label.to(DEVICE)
            net_out = net(image_cuda)
            net_out = log_softmax(net_out * 10)
            current_max, current_max_index = net_out.max(dim=1)
            result_map['pred'] += current_max.cpu().numpy().tolist()
            loss = loss_function(net_out, label_cuda)
            process_bar.set_description("[Test] Epoch: {} Loss: {}".format(epoch_id, loss.cpu().item()))
            result_map["loss"].append(loss.cpu().item())
            if result_map["cm"] is None:
                result_map["cm"] = s_common.confusion_matrix(label_cuda, current_max_index, 2)
            else:
                result_map["cm"] += s_common.confusion_matrix(label_cuda, current_max_index, 2)
    statistics_func_map = {
        'acc': s_common.acc, 
        'recall': s_common.recall,
        'ppv': s_common.precision,
        'npv': s_common.npv,
        'youden': s_common.youden_index
    }
    n_num_class = 2
    result_map["ci95"] = s_common.ci95(np.array(result_map["pred"]))
    for s_name, s_func in statistics_func_map.items():
        for index in range(n_num_class):
            if result_map[s_name] is None:
                result_map[s_name] = [
                    s_func(result_map['cm'], index)
                ]
            else:
                result_map[s_name].append(s_func(result_map['cm'], index))
    npy_save_path = Path(r"/root/Graduation thesis/data/npy") / NUMPYFILENAME.format('classification', epoch_id, 'statistics')
    np.save(npy_save_path, result_map)
    loss_c = np.array(result_map["loss"])
    return loss_c.mean(0)


def train():
    backbone = make_resnet18()
    backbone.to(DEVICE)
    optim = torch.optim.Adam(backbone.parameters(), lr=LR_MAX)
    optim_s = torch.optim.lr_scheduler.CosineAnnealingLR(optim, LR_MIN, EPOCHS)
    loss_func = nn.CrossEntropyLoss().to(DEVICE)
    train_dt = Loader(r"/home/niming/Shoulder-RC/Raw data", r"/root/Graduation thesis/data/image/part1.5", True)
    test_dt = Loader(r"/home/niming/Shoulder-RC/Raw data", r"/root/Graduation thesis/data/image/part1.5", False)
    train_dataloader = DataLoader(train_dt, BS, True, num_workers=20)
    test_dataloader = DataLoader(test_dt, BS, True, num_workers=20)
    for epoch in range(EPOCHS):
        train_loss = train_epoch(epoch, backbone, train_dataloader, loss_func, optim)
        print("[Train] Total loss: {}".format(train_loss))
        test_loss = test_epoch(epoch, backbone, test_dataloader, loss_func)
        print("[Test] Total loss: {}".format(test_loss))
        optim_s.step()
        
    
if __name__ == "__main__":
    train()
