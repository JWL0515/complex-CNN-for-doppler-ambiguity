#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Creation date: July 08 2022
@author: Jiawei Li, Technische Universität Dresden
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import Conv2d, Linear, BatchNorm1d, BatchNorm2d, Conv3d, BatchNorm3d
from torch.nn.functional import relu, max_pool3d, max_pool2d
import torch.nn.functional as F
from torch.utils.data import Subset, Dataset, random_split, DataLoader
from torch.optim import lr_scheduler
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear, ComplexConv3d
from complexPyTorch.complexLayers import ComplexDropout2d, NaiveComplexBatchNorm2d, NaiveComplexBatchNorm3d
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d, complex_avg_pool3d, complex_max_pool3d
from torchinfo import summary
from tqdm import tqdm
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import requests
from pyunpack import Archive
from time import time


class CONFIG:
    seed = 515

    batch_size = 1
    num_epochs = 3

    # Loss functions
    loss_func = F.cross_entropy

    # Optimizer
    lr = 0.0001
    momentum = 0.9

    # scheduler
    scheduler = 'CosineAnnealingLR'

    T_max = 10
    min_lr = 1e-6
    T_0 = 10
    step_size = 50
    gamma = 0.5

    # Earlystopping
    patience = 7

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class PrepareDataset(Dataset):

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = pd.read_csv(os.path.join(dataset_path, 'label.csv'))
        # self.df = pd.read_csv(os.path.join(dataset_path, 'label_balanced.csv'))
        self.df = self.df.dropna().reset_index(drop=True)
        # self.labels = self.df.factor
        self.labels = self.df.t1_factor
        self.labels = torch.tensor(self.labels)
        self.labels = self.labels - torch.min(self.labels)
        self.labels = self.labels.to(torch.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        npy_path = os.path.join(self.dataset_path, self.df.record_name[index], self.df.frame_name[index])
        feature = np.load(npy_path)
        feature = torch.tensor(feature, dtype=torch.complex64)
        feature = torch.unsqueeze(feature, 0)
        label = self.labels[index]
        return feature, label


def seed_everything(seed=515):
    """Set the seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # For running on the CuDNN backend, these two options must be set
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def download_smalldataset(dataset_name: str, username: str, password: str):
    if os.path.exists(f'./{dataset_name}'):
        print('Small dataset exists! No need to download!')
    else:
        os.mkdir(f'./{dataset_name}')

        names_urls = {'Dataset Norm': 'https://cloudstore.zih.tu-dresden.de/remote.php/webdav/Shared'
                                      '/dataset_dnn_disambiguation/small%20dataset/Dataset%20Norm.rar',
                      'Dataset Quan16b': 'https://cloudstore.zih.tu-dresden.de/remote.php/webdav/Shared'
                                         '/dataset_dnn_disambiguation/small%20dataset/Dataset%20Quan16b.rar',
                      'Element Norm': 'https://cloudstore.zih.tu-dresden.de/remote.php/webdav/Shared'
                                      '/dataset_dnn_disambiguation/small%20dataset/Element%20Norm.rar',
                      'Frame Norm': 'https://cloudstore.zih.tu-dresden.de/remote.php/webdav/Shared'
                                    '/dataset_dnn_disambiguation/small%20dataset/Frame%20Norm.rar',
                      'Frame Quan16b': 'https://cloudstore.zih.tu-dresden.de/remote.php/webdav/Shared'
                                       '/dataset_dnn_disambiguation/small%20dataset/Frame%20Quan16b.rar',
                      'Lite Norm': 'https://cloudstore.zih.tu-dresden.de/remote.php/webdav/Shared'
                                   '/dataset_dnn_disambiguation/small%20dataset/Lite%20Norm.rar',
                      'No Norm': 'https://cloudstore.zih.tu-dresden.de/remote.php/webdav/Shared/dataset_dnn_disambiguation'
                                 '/small%20dataset/No%20Norm.rar'}

        username = username
        password = password
        url = names_urls[dataset_name]
        response = requests.get(url, auth=(username, password))

        with open(f'{dataset_name}.rar', 'wb') as f:
            f.write(response.content)
        f.close()

        Archive(f'{dataset_name}.rar').extractall(f'./{dataset_name}')


def fetch_scheduler(optimizer):
    if CONFIG.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG.T_max, eta_min=CONFIG.min_lr)
    elif CONFIG.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CONFIG.T_0, T_mult=1, eta_min=CONFIG.min_lr)
    elif CONFIG.scheduler == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif CONFIG.scheduler == None:
        return None

    return scheduler


def train(model, device, train_loader, optimizer, loss_func, epoch, loss_dict, acc_dict):

    model.train()
    bar = tqdm(train_loader)
    train_losses = []
    for data, target in bar:
        bar.set_description(f'Epoch {epoch + 1}')

        data, target = data.to(device).type(torch.complex64), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        bar.set_postfix(loss=loss.item())


def complete_train(model, train_loader, valid_loader):
    optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG.lr, momentum=CONFIG.momentum)
    scheduler = fetch_scheduler(optimizer)
    loss_dict = {'Epoch': [], 'train_loss': [], 'valid_loss': []}
    acc_dict = {'Epoch': [], 'train_accuracy': [], 'valid_accuracy': []}

    early_stopping = EarlyStopping(patience=CONFIG.patience, verbose=True)

    for epoch in range(CONFIG.num_epochs):
        train(model, CONFIG.device, train_loader, optimizer, CONFIG.loss_func, epoch, loss_dict, acc_dict)
        valid_loss = valid(model, CONFIG.device, valid_loader, CONFIG.loss_func, loss_dict, acc_dict)
        scheduler.step()
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    df_loss = pd.DataFrame(loss_dict)
    df_loss.to_csv(f'loss.csv', index=False)
    df_acc = pd.DataFrame(acc_dict)
    df_acc.to_csv(f'accuracy.csv', index=False)

    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(loss_dict['train_loss']) + 1), loss_dict['train_loss'], label='Training Loss')
    plt.plot(range(1, len(loss_dict['valid_loss']) + 1), loss_dict['valid_loss'], label='Validation Loss')
    plt.plot(range(1, len(acc_dict['train_accuracy']) + 1), acc_dict['train_accuracy'], label='Training Accuracy')
    plt.plot(range(1, len(acc_dict['valid_accuracy']) + 1), acc_dict['valid_accuracy'], label='Validation Accuracy')

    # plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('loss_acc_plot.png', bbox_inches='tight')


def valid(model, device, valid_loader, loss_func, loss_dict, acc_dict):
    model.eval()
    valid_losses = []
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device).type(torch.complex64), target.to(device)
            output = model(data)
            loss = loss_func(output, target)
            valid_losses.append(loss.item())

    valid_loss = np.average(valid_losses)

    return valid_loss


def test(model, device, test_loader, model_pth=None):
    predict = {'targets': [], 'predictions': [], 'probability': []}
    score = {'accuracy': [], 'precision': [], 'recall': [], 'F1_score': []}

    model.eval()

    if model_pth is None:
        pass
    else:
        model.load_state_dict(torch.load(model_pth))

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).type(torch.complex64), target.to(device)
            output = model(data)
            predict['probability'] += output.detach().cpu().squeeze().tolist()
            true_value = target.detach().cpu().squeeze().tolist()
            _, pred_value = torch.max(output.data, 1)
            pred_value = pred_value.detach().cpu().squeeze().tolist()
            predict['targets'] += true_value
            predict['predictions'] += pred_value

    df = pd.DataFrame(predict)
    df.to_csv(f'predictions.csv', index=False)

    score['accuracy'].append(accuracy_score(predict['targets'], predict['predictions']))
    score['precision'].append(precision_score(predict['targets'], predict['predictions'], average=None))
    score['recall'].append(recall_score(predict['targets'], predict['predictions'], average=None))
    score['F1_score'].append(f1_score(predict['targets'], predict['predictions'], average=None))
    df_score = pd.DataFrame(score)
    df_score.to_csv(f'score.csv', index=False)
    print(score)


def complete_test(model, test_path, model_path, save_path):
    test_data = PrepareDataset(test_path)
    test_loader = DataLoader(test_data, batch_size=CONFIG.batch_size, shuffle=False)

    model.eval()
    model.load_state_dict(torch.load(model_path))
    predict = {'targets': [], 'predictions': [], 'KCNN_times': []}
    KCNN_time_l = []
    KCNN_time = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(CONFIG.device).type(torch.complex64), target.to(CONFIG.device)
            # KCNN_time = 0
            t_ = time()
            output = model(data)
            KCNN_time += time() - t_
            predict['KCNN_times'].append(KCNN_time)

            true_value = target.detach().cpu().squeeze().tolist()
            _, pred_value = torch.max(output.data, 1)
            pred_value = pred_value.detach().cpu().squeeze().tolist()
            predict['targets'].append(true_value)
            predict['predictions'].append(pred_value)

    predict = pd.DataFrame(predict)
    predictions = pd.concat(
        [test_data.df.record_name, test_data.df.frame_name, test_data.df.t1_true_velocity, test_data.df.t1_classID,
         predict], axis=1)
    predictions.to_csv(os.path.join(save_path, 'predictions.csv'), index=False)
    print(accuracy_score(predict['targets'], predict['predictions']))


class ComplexNet(nn.Module):

    def __init__(self):
        super(ComplexNet, self).__init__()
        self.conv1 = ComplexConv2d(1, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = NaiveComplexBatchNorm2d(12, track_running_stats=False)
        self.conv2 = Conv2d(12, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = BatchNorm2d(24, track_running_stats=False)

        self.conv3 = Conv2d(24, 12, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.bn3 = BatchNorm2d(12, track_running_stats=False)
        self.conv4 = Conv2d(12, 4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.bn4 = BatchNorm2d(4, track_running_stats=False)
        self.conv5 = Conv2d(96, 4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = complex_relu(x)
        x = complex_max_pool2d(x, kernel_size=(2, 2), stride=(1, 1))

        x_igm_copy = x.imag.clone()
        x_igm_copy[x_igm_copy == 0] = 1
        x = torch.mul(abs(x), x_igm_copy / abs(x_igm_copy))

        x = self.conv2(x)
        x = self.bn2(x)
        x = relu(x)
        x = max_pool2d(x, kernel_size=(3, 3), stride=(1, 1))

        x = self.conv3(x)
        x = self.bn3(x)
        x = relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = relu(x)
        #         print(x.shape[1]*x.shape[2]*x.shape[3])
        x = x.view(x.shape[0], 96, 1, 1)
        x = self.conv5(x)
        x = x.view(x.shape[0], 4)

        return x


if __name__ == "__main__":
    seed_everything(CONFIG.seed)
    model = ComplexNet().to(CONFIG.device)
    # ten_in = torch.randn(4, 1, 9, 7, dtype=torch.cfloat)
    # summary(model, input_data=ten_in)

    # for small dataset
    # Small dataset is one of : 'Dataset Norm', 'Dataset Quan16b', 'Element Norm', 'Frame Norm', 'Frame Quan16b',
    # 'Lite Norm', 'No Norm'
    # dataset_name = 'Lite Norm'
    # username = ''
    # password = ''
    # download_smalldataset(dataset_name, username, password)
    # main_path = f'./{dataset_name}'

    # fro training
    main_path = 'HPC_dataset_compressed_cfar_rect_9x7_compressed_not_cleaned_quan_normalized'
    dataset = PrepareDataset(main_path)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG.batch_size, shuffle=False)
    complete_train(model, train_loader, valid_loader)

    # for test
    # test_path = 'HPC_dataset_compressed_cfar_rect_9x7_compressed_not_cleaned_quan_normalized'
    # model_path = '1_target_models/9x7/2D-KCNN-f1/best_model_test_acc.pt'
    # save_path = ''
    # complete_test(model, test_path, model_path, save_path)

    print('finished')
