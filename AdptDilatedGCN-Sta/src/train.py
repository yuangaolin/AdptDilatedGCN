#!/user/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import dgl
from tqdm.auto import tqdm
from IPython.display import clear_output
import time
from dataset import SepDataset, collate_fn
from model import ModelNew
import metrics
from torch.backends import cudnn

seed = 2024

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(seed)
np.random.seed(seed)


def timeSince(since):
    now = time.time()
    s = now - since
    return now, s


def train(model, train_loader_compound, criterion, optimizer, epoch, device):
    model.train()
    tbar = tqdm(train_loader_compound, total=len(train_loader_compound))
    losses = []

    for i, data in enumerate(tbar):
        data0 = [i.to(device) for i in data[0]]
        ga, gr, gi, aff = data0

        vina = data[1]

        y_pred = model(ga, gr, gi, vina).squeeze()
        y_true = aff.float().squeeze()
        assert y_pred.shape == y_true.shape

        loss = criterion(y_pred, y_true).cuda()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    m_losses = np.mean(losses)

    return m_losses


def valid(model, valid_loader_compound, criterion, device):
    model.eval()
    losses = []
    outputs = []
    targets = []
    tbar = tqdm(valid_loader_compound, total=len(valid_loader_compound))
    for i, data in enumerate(tbar):
        data0 = [i.to(device) for i in data[0]]
        ga, gr, gi, aff = data0
        vina = data[1]
        with torch.no_grad():
            y_pred = model(ga, gr, gi, vina).squeeze()
        y_true = aff.float().squeeze()
        assert y_pred.shape == y_true.shape
        loss = criterion(y_pred, y_true).cuda()
        losses.append(loss.item())
        outputs.append(y_pred.cpu().detach().numpy().reshape(-1))
        targets.append(y_true.cpu().detach().numpy().reshape(-1))
    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)

    evaluation = {
        'ci': metrics.c_index(targets, outputs),
        'RMSE': metrics.RMSE(targets, outputs),
        'MAE': metrics.MAE(targets, outputs),
        'SD': metrics.SD(targets, outputs),
        'R': metrics.CORR(targets, outputs), }
    ml = np.mean(losses)

    return ml, evaluation


def main():
    SW = open(r'../data/train_Smith-Waterman.pkl', 'rb')
    content = pickle.load(SW)
    vina_list = []
    graphs = dgl.load_graphs('../data/train.bin')[0]
    labels = pd.read_csv('../data/labels_train.csv')
    vina_weights = open(r'../data/Vina_train.pkl', 'rb')
    vina = pickle.load(vina_weights)
    for i in range(len(graphs)):
        if labels.id[i] in vina.keys():
            vina_list.append(vina[labels.id[i]])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compound_train = content[0] + content[1]
    compound_valid = content[2]
    train_dataset_compound = SepDataset([graphs[i] for i in compound_train], [vina_list[i] for i in compound_train],
                                        [labels.id[i] for i in compound_train],
                                        [labels.affinity[i] for i in compound_train], ['a_conn', 'r_conn', 'int_l'])

    valid_dataset_compound = SepDataset([graphs[i] for i in compound_valid], [vina_list[i] for i in compound_valid],
                                       [labels.id[i] for i in compound_valid],
                                       [labels.affinity[i] for i in compound_valid], ['a_conn', 'r_conn', 'int_l'])

    train_loader_compound = DataLoader(train_dataset_compound, batch_size=5, shuffle=True, num_workers=0,
                                       collate_fn=collate_fn, pin_memory=False, drop_last=False, )

    valid_loader_compound = DataLoader(valid_dataset_compound, batch_size=5, shuffle=False, num_workers=0,
                                      collate_fn=collate_fn)

    model = ModelNew()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), 1.2e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=45, eta_min=1e-6)
    criterion = torch.nn.MSELoss()

    n_epoch = 90
    best_R1 = 0.0
    train_losses = []
    valid_losses = []
    for epoch in range(n_epoch):
        ll = train(model, train_loader_compound, criterion, optimizer, epoch, device)
        train_losses.append(ll)
        if epoch % 1 == 0:
            l, evaluation_ = valid(model, valid_loader_compound, criterion, device)
            valid_losses.append(l)
            print(f'epoch {epoch + 1} train_loss {ll:.5f} valid_loss {l:.5f}')
            clear_output()
            if evaluation_['R'] > best_R1:
                best_R1 = evaluation_['R']
                torch.save({'model': model.state_dict()}, '../model/Max_.pth')
        scheduler.step()


if __name__ == "__main__":
    main()
