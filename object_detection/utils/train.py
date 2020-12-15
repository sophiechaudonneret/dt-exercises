#!/usr/bin/env python3

import os
import sys
import numpy as np
import torch
from PIL import Image
import transforms as T
from engine import train_one_epoch, evaluate
import utils # ???

MODEL_PATH="../exercise_ws/src/object_detection/include/object_detection"
sys.path.insert(1,MODEL_PATH)
from model import Model

class Dataset(object):
    def __init__(self, transforms):
        # # self.root = root
        self.transforms = transforms
        self.DATASET_PATH = "../dataset"
        # load all image files, sorting them to
        # ensure that they are aligned
        # # self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        # # self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
        self.nbr_items = 0
        while os.path.exists(f"{self.DATASET_PATH}/{self.nbr_items}.npz"):
            self.nbr_items += 1

    def __getitem__(self, idx):
        # load images ad masks
        filename = f"{self.DATASET_PATH}/{idx}.npz"
        # # mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        data = np.load(filename)
        img = data[f"arr_{0}"]
        boxes = data[f"arr_{1}"]
        labels = data[f"arr_{2}"]

        img = Image.fromarray(img)
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return self.nbr_items

def main():

    # TODO train loop here!
    # TODO don't forget to save the model's weights inside of f"{MODEL_PATH}/weights`!

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 5
    # use our dataset and defined transformations
    # TODO : get right dataset
    dataset = Dataset(get_transform(train=True))
    dataset_test = Dataset(get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = Model()

    # move model to the right device
    model.model.to(device)

    # construct an optimizer
    params = [p for p in model.model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 100 epochs
    num_epochs = 100

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model.model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        print(f"Epoch {epoch}")

    print("That's it!")

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

if __name__ == "__main__":
    main()