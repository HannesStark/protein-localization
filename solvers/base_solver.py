import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


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
            train_preds = []  # predictions during training
            train_labels = []
            train_loss = 0
            for i, batch in enumerate(train_loader):
                embedding, label = batch
                embedding, label = embedding.to(self.device), label.to(self.device)

                prediction = self.model(embedding)

                self.optim.zero_grad()
                loss = self.loss_func(prediction, label)
                loss.backward()
                self.optim.step()

                train_preds.append(prediction.detach().cpu())
                train_labels.append(label.detach().cpu())
                loss_item = loss.item()
                train_loss += loss_item
                if i % args.log_iterations == args.log_iterations - 1:  # log every log_iterations
                    print('[Epoch %d, Iteration %5d/%5d] TRAIN loss: %.7f' % (
                        epoch + 1, i + 1, t_iters, loss_item))
                    prediction = torch.max(prediction, dim=1)[1]  # get indices of the highest value in the prediction
                    print('[Epoch %d, Iteration %5d/%5d] TRAIN accuracy: %.4f%%' % (
                        epoch + 1, i + 1, t_iters, 100 * (prediction == label).sum().item() / args.batch_size))

            self.model.eval()
            val_preds = []  # predictions during validation
            val_labels = []
            val_loss = 0
            for i, batch in enumerate(val_loader):
                embedding, label = batch
                embedding, label = embedding.to(self.device), label.to(self.device)

                prediction = self.model(embedding)

                loss = self.loss_func(prediction, label)
                val_preds.append(prediction.detach().cpu())
                val_labels.append(label.detach().cpu())
                val_loss += loss.item()

            train_preds = torch.cat(train_preds)
            train_labels = torch.cat(train_labels)
            val_preds = torch.cat(val_preds)
            val_labels = torch.cat(val_labels)

            val_confusion = torch.zeros((val_preds.shape[-1], val_preds.shape[-1]), dtype=torch.int64)
            train_confusion = torch.zeros((train_preds.shape[-1], train_preds.shape[-1]), dtype=torch.int64)
            train_preds = torch.max(train_preds, dim=1)[1]  # get indices of the highest value in the prediction
            val_preds = torch.max(val_preds, dim=1)[1]  # get indices of the highest value in the prediction
            for pred_index, true_index in zip(train_preds, train_labels):  # + 1 for each predicted row and true column
                train_confusion[pred_index, true_index] += 1
            for pred_index, true_index in zip(val_preds, val_labels):
                val_confusion[pred_index, true_index] += 1

            train_acc = 100 * (train_preds == train_labels).sum().item() / len(train_preds)
            val_acc = 100 * (val_preds == val_labels).sum().item() / len(val_preds)
            print('[Epoch %d] VAL accuracy: %.4f%% train accuracy: %.4f%%' % (epoch + 1, val_acc, train_acc))
            # write to tensorboard
            self.writer.add_scalars('Epoch Accuracy', {'train acc': train_acc, 'val acc': val_acc}, epoch)
            self.writer.add_scalars('Epoch Loss', {'train loss': train_loss / t_iters, 'val loss': val_loss / v_iters},
                                    epoch)

            # TODO: implement saving of model
            # save_run(self.writer.log_dir, [self.model], parser)
