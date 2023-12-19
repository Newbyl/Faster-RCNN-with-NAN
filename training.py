import numpy as np
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
import random
import matplotlib.patches as patches
from utils import *
from model import *
import os

import torch
import torchvision
from torchvision import ops
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from dataset import ObjectDetectionDataset


img_width = 640
img_height = 480
annotation_path = "data/annotations.xml"
image_dir = os.path.join("data", "images")
name2idx = {'pad': -1, 'camel': 0, 'bird': 1}
idx2name = {v:k for k, v in name2idx.items()}

batch_size = 2

od_dataset = ObjectDetectionDataset(annotation_path, image_dir, (img_height, img_width), name2idx)

od_dataloader = DataLoader(od_dataset, batch_size=2)

for img_batch, gt_bboxes_batch, gt_classes_batch in od_dataloader:
    img_data_all = img_batch
    gt_bboxes_all = gt_bboxes_batch
    gt_classes_all = gt_classes_batch
    break

img_data_all = img_data_all[:batch_size]
gt_bboxes_all = gt_bboxes_all[:batch_size]
gt_classes_all = gt_classes_all[:batch_size]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_batch, gt_bboxes_batch, gt_classes_batch = img_batch.to(device, dtype=torch.float32), gt_bboxes_batch, gt_classes_batch

model = torchvision.models.resnet50(pretrained=True).cuda()

req_layers = list(model.children())[:8]
backbone = nn.Sequential(*req_layers)

out = backbone(img_data_all)

# unfreeze all the parameters
for param in backbone.named_parameters():
    param[1].requires_grad = True
    

# run the image through the backbone
out = backbone(img_data_all)

out_c, out_h, out_w = out.size(dim=1), out.size(dim=2), out.size(dim=3)




img_size = (img_height, img_width)
out_size = (out_h, out_w)
n_classes = len(name2idx) - 1 # exclude pad idx
roi_size = (7, 7)

detector = TwoStageDetector(img_size, out_size, out_c, n_classes, roi_size).to(device)

detector.eval()
total_loss = detector(img_batch, gt_bboxes_batch, gt_classes_batch)
proposals_final, conf_scores_final, classes_final = detector.inference(img_batch)





def training_loop(model, learning_rate, train_dataloader, n_epochs):

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    loss_list = []

    for i in tqdm(range(n_epochs)):
        total_loss = 0
        for img_batch, gt_bboxes_batch, gt_classes_batch in train_dataloader:

            # forward pass
            loss = model(img_batch, gt_bboxes_batch, gt_classes_batch)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        loss_list.append(total_loss)

    return loss_list

learning_rate = 1e-3
n_epochs = 1000


if __name__ == "__main__" :
    loss_list = training_loop(detector, learning_rate, od_dataloader, n_epochs)
    torch.save(detector.state_dict(), "model.pt")