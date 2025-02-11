# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# !/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import argparse
import os
import yaml
import random
import shutil
from collections.abc import Iterable
from multiprocessing import Pool
import pickle

from tqdm import tqdm
import torch
# torch.autograd.set_detect_anomaly(True)
import matplotlib.pyplot as plt
import numpy as np

from ML4S_Tools.Dataset import ContinousDataset, DiscreteDataset
from model.orientation_model import OriModel, WholeModel, WholeNodewiseModel
from utils.criterions import skeleton_learning_loss, get_scores, graph_learning_loss, vstruc_learning_loss, \
    vstruc_get_scores
from utils.tools import shuffle, dag_to_vstrucs, batched_dag_to_vstrucs, set_seed
import data_generator

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="config file",
                    default=f"configs/orientation.yml")
args = parser.parse_args()
with open(args.config, 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
for key, value in config.items():
    print(key, ':', value)

if 'seed' in config:
    set_seed(config['seed'])

metrics = ["vstruc_f1s", "vstruc_aucs", "vstruc_auprcs", "vstruc_hammings"]

test_metrics = {}
for metric in metrics:
    test_metrics[metric] = {}
train_losses = []


def train(model, datasets, optimizer, device, config=None):
    model.train()
    f1s = {i: [] for i in np.arange(0.0, 1.0, 0.1)}
    train_loss = 0
    random.shuffle(datasets)
    batch_size = 15
    # for batch_idx, (inputs, targets) in tqdm(enumerate(datasets)):
    for batch_idx in tqdm(range(0, len(datasets), batch_size)):
        inputs = [torch.tensor(datasets[i][0].IndexedDataT).transpose(1, 0) for i in
                  range(batch_idx, batch_idx + batch_size)]
        targets = [torch.tensor(datasets[i][1].IndexedDataT).transpose(-1, -2) for i in
                   range(batch_idx, batch_idx + batch_size)]

        inputs = torch.stack(inputs)
        targets = torch.stack(targets)
        inputs, targets = inputs.to(device), targets.to(device)

        cube_tforks, cube_vstrucs = batched_dag_to_vstrucs(targets.cpu().numpy())
        cube_vstrucs_torch = torch.tensor(cube_vstrucs).to(device)
        cube_tforks_torch = torch.tensor(cube_tforks).to(device)

        optimizer.zero_grad()
        outputs = model(inputs, freeze=config['freeze'])

        loss = vstruc_learning_loss(outputs['vstruc'], cube_vstrucs_torch, mask=cube_tforks_torch)

        for i in range(batch_size):
            if (cube_vstrucs[i]).max() != 0:
                vstruc_scores = vstruc_get_scores(cube_vstrucs[i], outputs['vstruc'][i].detach(), cube_tforks[i], check_all_threshold=True)
                for thres in np.arange(0.0, 1.0, 0.1):
                    f1s[thres].append(vstruc_scores['f1' + str(thres)])

        total_loss = loss

        train_losses.append(loss.item())
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()
        if config['smoke_test']:
            break
        # print(loss)
    train_loss = train_loss / len(datasets) * batch_size
    avg_f1s = {thre: np.mean(f1s[thre]) for thre in f1s}
    best_threshold = max(avg_f1s, key=avg_f1s.get)
    print(best_threshold)
    model.best_threshold.data.fill_(best_threshold)
    print('train_loss:', train_loss)
    return train_loss


def test(model, name, dataset, device, epoch, config, batch_size=1000, ensembles=1):
    model.eval()
    for metric in metrics:
        test_metrics[metric][name].append([])

    if isinstance(dataset, Iterable):
        for data, label in dataset:
            data = torch.tensor(data.IndexedDataT).transpose(1, 0).to(device)
            label = torch.tensor(label.IndexedDataT.transpose(-1, -2)).to(device)
            data_size = len(data)
            index = 0
            vstrucs_sum = 0
            for j in range(ensembles):
                shuffled_data = torch.randperm(data.size(0))
                data = data[shuffled_data]
                for i in range(0, max(data_size, batch_size), batch_size):
                    batch_data = data[i: i + batch_size] if i + batch_size <= data_size else data[i:]
                    if len(batch_data) > 0:  # Ensures there is data to process
                        inputs, targets = data[i: i + batch_size], label
                        outputs = model(inputs.unsqueeze(0))
                        vstrucs_sum += outputs['vstruc'].detach()
                        index += 1
                    else:
                        raise NameError("Error no batch data")

            vstrucs = vstrucs_sum / index
            cube_tforks, cube_vstrucs = batched_dag_to_vstrucs(targets.cpu().unsqueeze(0).numpy())
            if np.sum(cube_tforks) == 0:
                continue
            vstruc_scores = vstruc_get_scores(cube_vstrucs, vstrucs, cube_tforks)
            vstruc_f1, vstruc_auc, vstruc_auprc = vstruc_scores['f1'], vstruc_scores['auc'], vstruc_scores['auprc']
            vstruc_hamming = vstruc_scores['hamming_distance']
            for mname, value in zip(test_metrics, [vstruc_f1, vstruc_auc, vstruc_auprc, vstruc_hamming]):
                if value:
                    test_metrics[mname][name][epoch].append(value)


if __name__ == '__main__':
    prefix = ""
    currentpath = f"results/{prefix}{config['title']}_test{''.join(config['testset_path'].keys())}"
    if config['freeze']:
        assert config['feature_extrator_path']

    if not os.path.isdir(currentpath):
        print("project path: ", currentpath)
        os.mkdir(currentpath)
    else:
        print("current path: ", currentpath)
        print("WARNING: rootpath exists")

    shutil.copyfile("main_orientation.py", currentpath + "/main_bak.py")
    shutil.copyfile("criterions.py", currentpath + "/criterions_bak.py")
    shutil.copyfile("model/model.py", currentpath + "/model_bak.py")
    shutil.copyfile("model/orientation_model.py", currentpath + "/orientation_model_bak.py")
    shutil.copyfile("utils/tools.py", currentpath + "/tools_bak.py")
    shutil.copyfile("data_generator.py", currentpath + "/data_generator_bak.py")
    shutil.copyfile(config['train_config_path'], currentpath + "/train_config.yaml")
    shutil.copyfile(args.config, currentpath + "/config_bak.yml")

    if config['continuous']:
        Dataset = ContinousDataset
    else:
        Dataset = DiscreteDataset

    test_datasets = config['testset_path']
    for key, value in test_datasets.items():
        with open(value + '/config.yml', 'r') as file:
            dataset_config = yaml.load(file, Loader=yaml.FullLoader)
        datasets = []
        for i in range(dataset_config['num']):
            data = Dataset(f"{value}/struc_{i}.npy")
            label = Dataset(f"{value}/struc_{i}.txt")
            datasets.append((data, label))
        test_datasets[key] = datasets

        for metric in metrics:
            test_metrics[metric][key] = []

    num_of_nodes = test_datasets[key][0][0].VarCount
    if not config['continuous']:
        num_of_classes = test_datasets[key][0][0].IndexedDataT.max(axis=1) + 1
    else:
        num_of_classes = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    if config['pairwise']:
        if config['continuous']:
            model = WholeModel().to(device)
        else:
            model = WholeModel(continuous_data=False,
                               num_of_nodes=num_of_nodes,
                               num_of_classes=num_of_classes,
                               input_embedding_dim=None).to(device)
    else:
        if config['continuous']:
            model = WholeNodewiseModel().to(device)
        else:
            raise NotImplementedError

    start_epoch = 0
    if config['load_path'] is not None:
        model.load(config['load_path'] + '/model.pt')
        print(f"Loaded from {config['load_path']}, testing")
        for name, test_dataset in test_datasets.items():
            test(model, name, test_dataset, device, 0, config=config)
        start_epoch += 1
    if config['feature_extrator_path']:
        model.load_feature_extrator(config['feature_extrator_path'])

    if config['freeze']:
        parameters = model.orientation_model.parameters()
    else:
        parameters = model.parameters()

    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=config['learning_rate'])
    elif config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=config['learning_rate'])
    else:
        raise NotImplementedError

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, verbose=True)
    warmup_schedular = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                         lr_lambda=lambda epoch: epoch / config['warmup_epoch'])
    min_trainloss = 1000
    for epoch in range(start_epoch, config['total_epoch']):
        print(f"=======epoch {epoch}=======")

        if config['smoke_test']:
            batch_size = 100
        else:
            batch_size = 6000
        if not config['continuous']:
            raise NotImplementedError
        else:
            def to_dataset(dl):
                return Dataset(dl[0]), Dataset(dl[1])


            def get_dataset(i):
                return to_dataset(
                    data_generator.avici_continuous_generator(d=config['nodes'], datasize=200,
                                                              path=config['train_config_path'], forcetforks=True))


            # with Pool(50) as p:
            #     train_datasets = list(p.map(get_dataset, list(range(batch_size))))
            train_datasets = []
            with Pool(24) as p:
                results = p.imap_unordered(get_dataset, list(range(batch_size)))
                for result in tqdm(results, total=batch_size):
                    train_datasets.append(result)
            print("generate training data finished")

        train_loss = train(model, train_datasets, optimizer, device=device, config=config)
        for name, test_dataset in test_datasets.items():
            test(model, name, test_dataset, device, epoch, config=config)

        scheduler.step(train_loss)
        if epoch < config['warmup_epoch']:
            warmup_schedular.step()

        model.save(f'{currentpath}/model.pt')
        if epoch % 100 == 0:
            model.save(f'{currentpath}/model{epoch}.pt')

        if train_loss < min_trainloss:
            model.save(f'{currentpath}/model_min_trainloss.pt')

        torch.save(train_losses, f"{currentpath}/train_losses.pt")

        intervals = [100, 1000, 10000]
        for interval in intervals:
            plt.clf()
            plt.plot([loss for i, loss in enumerate(train_losses) if i % interval == 0], alpha=0.9, linewidth=0.5)
            plt.xlabel(f"batch / {interval}")
            plt.ylabel("loss")
            plt.title(f"{config['title']}")
            plt.legend(["loss"])
            plt.savefig(f"{currentpath}/train_losses_{interval}.png")

        for metric in metrics:
            torch.save(test_metrics[metric], f"{currentpath}/{metric}.pt")

        for name in test_datasets:
            plt.clf()
            mean_metrics = []
            for metric in metrics:
                mean_metrics.append([np.mean(i) for i in test_metrics[metric][name]])
            for mm in mean_metrics:
                plt.plot(mm)
            plt.legend([f"{name}: {np.argmax(mm)} {max(mm):.4f}" for name, mm in zip(metrics, mean_metrics)])
            plt.title(f"{name}: {config['title']}")
            plt.savefig(f"{currentpath}/vstruc_metrics_{name}.png")
