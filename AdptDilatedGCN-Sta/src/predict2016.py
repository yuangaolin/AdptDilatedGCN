#!/user/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pickle
import dgl
from tqdm.auto import tqdm

from dataset import SepDataset, collate_fn
from model import ModelNew
import metrics


def test(model, valid_loader_compound, criterion, device):
    model.eval()
    losses = []
    outputs = []
    targets = []
    name_list = []
    tbar = tqdm(valid_loader_compound, total=len(valid_loader_compound))
    for i, data in enumerate(tbar):
        data0 = [i.to(device) for i in data[0]]
        ga, gr, gi, aff = data0
        vina = data[1]
        idnames = data[2]
        name_l = []
        for name in idnames:
            name_l.append(name)
        name_list.append(name_l)
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
    name_list = np.concatenate(name_list).reshape(-1)
    evaluation = {
        'ci': metrics.c_index(targets, outputs),
        'RMSE': metrics.RMSE(targets, outputs),
        'MAE': metrics.MAE(targets, outputs),
        'SD': metrics.SD(targets, outputs),
        'R': metrics.CORR(targets, outputs), }

    return evaluation, targets, outputs, name_list


def main1():
    flag = 'Predict'
    model_path = '../model/ModelWeight.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vina_list16 = []
    graphs16 = dgl.load_graphs('../data/test_CASF2016.bin')[0]
    labels16 = pd.read_csv('../data/labels_CASF2016.csv')
    vina_terms16 = open(r'../data/Vina_CASF2016.pkl', 'rb')
    vina16 = pickle.load(vina_terms16)
    for i in range(len(graphs16)):
        if labels16.id[i] in vina16.keys():
            vina_list16.append(vina16[labels16.id[i]])

    test16_dataset = SepDataset([graphs16[i] for i in range(len(graphs16))],
                                [vina_list16[i] for i in range(len(vina_list16))],
                                [labels16.id[i] for i in range(len(labels16))],
                                [labels16.affinity[i] for i in range(len(labels16))], ['a_conn', 'r_conn', 'int_l'])
    test2016_loader = DataLoader(test16_dataset, batch_size=36, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = ModelNew()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    criterion = torch.nn.MSELoss()
    p = test2016_loader
    p_f = 'CASF2016'
    print(f'{flag}_{p_f}.csv')
    evaluation, targets, outputs, names = test(model, p, criterion, device)
    a = pd.DataFrame()
    a = a.assign(pdbid=names, predicted=outputs, real=targets, set=p_f)
    a.to_csv(f'../result/{flag}_{p_f}.csv')
    print(evaluation)
    return evaluation


if __name__ == "__main__":
    main1()
