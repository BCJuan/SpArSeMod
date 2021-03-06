# -*- coding: utf-8 -*-
"""
Model function to train and evaluate the models
It is called from the metric accuracy
"""
from __future__ import division
from os import path
import copy
from typing import Dict
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch import (
    device,
    cuda as torchcuda,
    save,
    float as floatp,
    set_grad_enabled,
    no_grad,
    max as maxim,
    sum as summ,
)
from abc import ABC, abstractmethod
from .heir import copy_weights
from .quant_n_prune import prune_net


class AbstractTrainer(ABC):
    """ 
    ABstract class for a trainer. Best to build up 
    from simpleTrainer or a based implementation. In simple
    Trainer you can look up to all the required elements to return
    """
    def __init__(self, ):
        pass

    @property
    @abstractmethod
    def load_dataloaders(self):
        """For loading the dataloaders with the inputted collate"""
        pass

    @abstractmethod
    def train(self):
        """ Train general loop"""
        pass

    @abstractmethod
    def evaluate(self):
        """ Evaluation loop """
        pass


class SimpleTrainer(AbstractTrainer):
    """
    Class for trainning and evaluating models.
    It includes dataloaders, datasets, training
    routine, pruning and evaluation
    """

    def __init__(
        self,
        pruning=False,
        ddtype=floatp,
        datasets=None,
        models_path=None,
        cuda="cuda:0",
    ):
        super().__init__()
        self.datasets = datasets
        self.dtype = ddtype
        # TODO: choose GPU with less memory
        self.devicy = device(cuda if torchcuda.is_available() else "cpu")
        self.datasizes = {
            i: len(sett) for i, sett in zip(["train", "val", "test"], self.datasets)
        }
        self.pruning = pruning
        self.models_path = models_path
        self.dataloader = None
        self.criterion = nn.CrossEntropyLoss()


    def load_dataloaders(self, batch_size, collate_fn):
        """
        Defines data loaders as a call to be able to define
        collates from outside
        """
        self.dataloader = {
            i: DataLoader(
                sett,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                drop_last=True,
                collate_fn=collate_fn,
            )
            for i, sett in zip(["train", "val", "test"], self.datasets)
        }

    def train(
        self,
        net: nn.Module,
        parameters: Dict[str, float],
        name: str,
        epochs: int,
        reload: bool,
        old_net: nn.Module.state_dict,
    ) -> nn.Module:
        """
        Train CNN on provided data set.
        Args
        ----
            net:
                initialized neural ntwork
            train_loader:
                DataLoader containing training set
            parameters:
                dictionary containing parameters to be passed to
                the optimizer.
            dtype:
                torch dtype
            device:
                torch device
        Returns
        ------
            nn.Module: trained CNN.
        """

        # Initialize network
        if reload:
            net = copy_weights(old_net, net)
        net.to(dtype=self.dtype, device=self.devicy)  # pyre-ignore [28]
        # Define loss and optimizer
        optimizer = Adam(net.parameters(), lr=parameters.get("learning_rate"))
        # TODO: change to reduce on plateau, is for cifar change 1000
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=parameters.get("learning_step") * 1000,
            gamma=parameters.get("learning_gamma"),
        )

        # Train Network
        net = self.train_loop(
            net,
            optimizer,
            exp_lr_scheduler,
            name,
            epochs,
            parameters.get("prune_threshold"),
        )
        return net

    def train_loop(self, model, optimizer, scheduler, name, epochs, threshold):
        """
        Training loop
        """
        best_acc = 0.0
        best_model_wts = copy.deepcopy(model.state_dict())
        # pruning with steps
        cnt = 0
        steps = epochs * len(self.dataloader["train"])
        init_threshold = 0.01
        thres_step = (threshold - init_threshold) / steps
        #
        for _ in range(epochs):

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                # for inputs, labels in tqdm(self.dataloader[phase],
                #                             total=len(
                #                                 self.dataloader[phase])):
                for _, (inputs, labels) in enumerate(self.dataloader[phase], start=1):

                    inputs = inputs.to(self.devicy)
                    labels = labels.to(self.devicy)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = maxim(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * labels.size(0)
                    running_corrects += summ(preds == labels.data)

                    if phase == "train":
                        scheduler.step()

                    if phase == "train" and self.pruning:
                        model = prune_net(model, init_threshold + thres_step * cnt)
                        cnt += 1

                epoch_acc = running_corrects.double() / self.datasizes[phase]

                # deep copy the model
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        # load best model weights
        model.load_state_dict(best_model_wts)
        # TODO: be aware of problems with multibatch due to arm names
        save(model.state_dict(), path.join(self.models_path, str(name) + ".pth"))
        return model

    def evaluate(self, net: nn.Module, quant_mode: bool) -> float:
        """
        Compute classification accuracy on provided dataset.
        Args
        ----
            net: trained model
            data_loader: DataLoader containing the evaluation set
            dtype: torch dtype
            device: torch device
        Returns
        -------
            float: classification accuracy
        """
        correct = 0
        total = 0
        if quant_mode:
            data_loader = self.dataloader["train"]
        else:
            data_loader = self.dataloader["test"]
        # TODO: add pruning before evaluation
        cnt = 0
        net.eval()
        with no_grad():
            # for inputs, labels in tqdm(data_loader, total=len(data_loader)):
            for inputs, labels in data_loader:
                # move data to proper dtype and device
                inputs = inputs.to(device="cpu")
                labels = labels.to(device="cpu")
                outputs = net(inputs)
                _, predicted = maxim(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # print("predicted", predicted, "labels", labels)
                cnt += 1
                if quant_mode and cnt > 2000:
                    break

        return correct / total, net
