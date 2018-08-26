import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import patches, patheffects
import json
from collections import defaultdict
import argparse

from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models

from ignite.metrics import MeanSquaredError, BinaryAccuracy
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers.checkpoint import ModelCheckpoint

import utils
from data import datasets

"""
python models/train_largest.py \
--train_json /Users/shendrickson/Development/ssd/data/processed/PascalVOC/train/pascal_train2007.json \
--valid_json /Users/shendrickson/Development/ssd/data/processed/PascalVOC/train/pascal_val2007.json \
--image_dir /Users/shendrickson/Development/ssd/data/processed/PascalVOC/train/VOCdevkit/VOC2007/JPEGImages/ \
--checkpoint_dir /tmp/ssd_checkpoints/ \
--restore /tmp/ssd_checkpoints \
--max_samples 64
"""


def get_img_id(file_path):
    return int(file_path.name.split(".")[0])


class MyModel(nn.Module):

    def __init__(self, num_classes=20):
        super(MyModel, self).__init__()
        resnet = models.resnet34(pretrained=True)
        res_layers = list(resnet.children())[:-1]
        self.model = nn.Sequential(*res_layers)
        for param in self.model.parameters():
            param.requires_grad = False
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.model.forward(x)
        out = self.head(x.view(x.shape[0], -1))
        return out


def choose_restore_path(path):
    time, file_path = max((f.stat().st_mtime, f) for f in path.iterdir())
    return file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_json", type=str)
    parser.add_argument("--valid_json", type=str)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--restore", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_gpu", type=int, default=None)
    args = parser.parse_args()

    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        devices = [torch.device("cpu")]
    elif args.num_gpu is None:
        devices = [torch.device(f"cuda:{i}") for i in range(num_devices)]
    else:
        devices = [torch.device(f"cuda:{i}") for i in range(args.num_gpu)]

    ############################################
    # Loading data
    ############################################
    train_json = json.load(Path(args.train_json).open())
    valid_json = json.load(Path(args.valid_json).open())
    jpg_path = Path(args.image_dir)

    categories = [x['name'] for x in train_json['categories']]
    num_categories = len(categories)
    train_paths = [jpg_path / im['file_name'] for im in train_json['images']]
    valid_paths = [jpg_path / im['file_name'] for im in valid_json['images']]
    if args.max_samples is not None:
        train_paths = np.random.choice(train_paths, args.max_samples, replace=False)
        valid_paths = np.random.choice(valid_paths, args.max_samples, replace=False)

    annos = {}
    for json in [train_json, valid_json]:
        d = defaultdict(list)
        for anno in json['annotations']:
            d[anno['image_id']].append(anno)
        annos.update(d)

    largest_annos = {}
    for img_id, anno_list in annos.items():
        largest_idx = np.argmax([anno['area'] for anno in anno_list])
        largest_annos[img_id] = anno_list[largest_idx]

    train_labels = [largest_annos[get_img_id(p)]['category_id'] - 1 for p in train_paths]
    valid_labels = [largest_annos[get_img_id(p)]['category_id'] - 1 for p in valid_paths]

    normalize = transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
    img_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = datasets.ImageDataset(train_paths, train_labels, transform=img_transforms, device=devices[0])
    valid_ds = datasets.ImageDataset(valid_paths, valid_labels, transform=img_transforms, device=devices[0])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=args.batch_size)


    ############################################
    # Define model
    ############################################
    if args.restore is not None:
        path = choose_restore_path(Path(args.restore))
        print(f"Restoring model from {path.name}")
        model = torch.load(path)
    else:
        model = MyModel(num_categories)
    model_par = nn.DataParallel(model, device_ids=[d.index for d in devices])

    ############################################
    # Optimizer/Loss
    ############################################
    criterion = nn.CrossEntropyLoss().to(devices[0])
    model_opt = torch.optim.Adam(model_par.parameters(), lr=0.0001)

    best_checkpointer = ModelCheckpoint(args.checkpoint_dir, 'my_model', create_dir=True,
                                   score_function=lambda eng: -eng.state.output,
                                   require_empty=False, n_saved=1)

    ############################################
    # Training
    ############################################
    results = {'best_loss': 1000.}
    def training_update_function(engine, batch):
        model_par.train()
        inp, targ = batch
        out = model_par.forward(inp)
        loss = criterion(out, targ)
        loss.backward()
        model_opt.step()
        model_opt.zero_grad()

        return loss.item() / targ.shape[0]


    def inference(engine, batch):
        model_par.eval()
        inp, targ = batch
        out = model_par.forward(inp)
        return out, targ


    trainer = Engine(training_update_function)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, best_checkpointer, {'best_model': model})

    evaluator = Engine(inference)

    @trainer.on(Events.ITERATION_COMPLETED)
    def track_results(trainer):
        results['best_loss'] = min(results['best_loss'], trainer.state.output)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        if (trainer.state.iteration + 1) % 1 == 0:
            print("Epoch[{}] Iteration[{}] Loss: {:.8f}".format(trainer.state.epoch,
                                                                trainer.state.iteration + 1,
                                                                trainer.state.output))
    trainer.run(train_loader, max_epochs=5)

    # metric = MeanSquaredError()
    # metric.attach(evaluator, 'mse')



