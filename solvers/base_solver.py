import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from utils.general import LOCALIZATION, LOCALIZATION_abbrev, tensorboard_confusion_matrix


class BaseSolver():
    def __init__(self, model, args, optim=torch.optim.Adam, loss_func=torch.nn.MSELoss()):
        self.optim = optim(list(model.parameters()), args.lrate)
        self.loss_func = loss_func
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.writer = SummaryWriter(
            'runs/{}_{}'.format(args.experiment_name, datetime.now().strftime('%d-%m_%H-%M-%S')))

    def train(self, train_loader, val_loader):
        args = self.args
        t_iters = len(train_loader)  # number of iterations in training loop
        v_iters = (len(val_loader) or not len(val_loader))  # number of iterations in validation loop or 1 if it's empty

        for epoch in range(args.num_epochs):  # loop over the dataset multiple times
            self.model.train()
            train_results = []  # prediction and corresponding label
            train_loss = 0
            for i, batch in enumerate(train_loader):
                embedding, label = batch
                embedding, label = embedding.to(self.device), label.to(self.device)

                prediction = self.model(embedding)

                self.optim.zero_grad()
                loss = self.loss_func(prediction, label)
                loss.backward()
                self.optim.step()

                prediction = torch.max(prediction, dim=1)[1]  # get indices of the highest value in the prediction
                train_results.append(torch.stack((prediction, label), dim=1).detach().cpu().numpy())
                loss_item = loss.item()
                train_loss += loss_item
                if i % args.log_iterations == args.log_iterations - 1:  # log every log_iterations
                    print('[Epoch %d, Iteration %5d/%5d] TRAIN loss: %.7f' % (
                        epoch + 1, i + 1, t_iters, loss_item))
                    print('[Epoch %d, Iteration %5d/%5d] TRAIN accuracy: %.4f%%' % (
                        epoch + 1, i + 1, t_iters, 100 * (prediction == label).sum().item() / args.batch_size))

            self.model.eval()
            val_results = []  # prediction and corresponding label
            val_loss = 0
            for i, batch in enumerate(val_loader):
                embedding, label = batch
                embedding, label = embedding.to(self.device), label.to(self.device)

                prediction = self.model(embedding)

                loss = self.loss_func(prediction, label)
                prediction = torch.max(prediction, dim=1)[1]  # get indices of the highest value in the prediction
                val_results.append(torch.stack((prediction, label), dim=1).detach().cpu().numpy())
                val_loss += loss.item()

            train_results = np.concatenate(train_results)  # [number_train_proteins, 2] prediction and label
            val_results = np.concatenate(val_results)  # [number_val_proteins, 2] prediction and label

            train_acc = 100 * np.equal(train_results[:, 0], train_results[:, 1]).sum() / len(train_results)
            val_acc = 100 * np.equal(val_results[:, 0], val_results[:, 1]).sum() / len(val_results)
            print('[Epoch %d] VAL accuracy: %.4f%% train accuracy: %.4f%%' % (epoch + 1, val_acc, train_acc))

            # write to tensorboard
            tensorboard_confusion_matrix(train_results, val_results, self.writer, epoch + 1)
            self.writer.add_scalars('Epoch Accuracy', {'train acc': train_acc, 'val acc': val_acc}, epoch + 1)
            self.writer.add_scalars('Epoch Loss', {'train loss': train_loss / t_iters, 'val loss': val_loss / v_iters},
                                    epoch + 1)

            # TODO: implement saving of model
            # save_run(self.writer.log_dir, [self.model], parser)
