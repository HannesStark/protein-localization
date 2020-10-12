import os

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from models.loss_functions import cross_entropy_joint
from utils.general import tensorboard_confusion_matrix, experiment_checkpoint


class BaseSolver():
    def __init__(self, model, args, optim=torch.optim.Adam, loss_func=cross_entropy_joint):
        self.optim = optim(list(model.parameters()), **args.optimizer_parameters)
        self.loss_func = loss_func
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        if args.checkpoint:
            checkpoint = torch.load(os.path.join(args.checkpoint, 'checkpoint.pt'), map_location=self.device)
            self.writer = SummaryWriter(args.checkpoint)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
        else:
            self.start_epoch = 0
            self.writer = SummaryWriter(
                'runs/{}_{}_{}'.format(args.model_type, args.experiment_name,
                                       datetime.now().strftime('%d-%m_%H-%M-%S')))

    def train(self, train_loader, val_loader):
        args = self.args
        t_iters = len(train_loader)  # number of iterations in training loop
        v_iters = (len(val_loader) or not len(val_loader))  # number of iterations in validation loop or 1 if it's empty

        for epoch in range(self.start_epoch, args.num_epochs):  # loop over the dataset multiple times
            self.model.train()
            train_results = []  # prediction and corresponding localization
            train_loc_loss = 0
            train_sol_loss = 0
            for i, batch in enumerate(train_loader):
                embedding, loc, sol, metadata = batch  # get localization and solubility label
                embedding, loc, sol, sol_known = embedding.to(self.device), loc.to(self.device), sol.to(self.device), \
                                                 metadata['solubility_known'].to(self.device)
                prediction = self.model(embedding)
                loss, loc_loss, sol_loss = self.loss_func(prediction, loc, sol, sol_known, args)
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()

                prediction = torch.max(prediction[..., :10], dim=1)[1]  # get indices of the highest value for loc
                train_results.append(torch.stack((prediction, loc), dim=1).detach().cpu().numpy())
                loc_loss_item = loc_loss.item()
                train_loc_loss += loc_loss_item
                train_sol_loss += sol_loss.item()
                if i % args.log_iterations == args.log_iterations - 1:  # log every log_iterations
                    print('[Epoch %d, Iter %5d/%5d] TRAIN localization loss: %.7f,TRAIN accuracy: %.4f%%' % (
                        epoch + 1, i + 1, t_iters, loc_loss_item,
                        100 * (prediction == loc).sum().item() / args.batch_size))

            self.model.eval()
            val_results = []  # prediction and corresponding loc
            val_loc_loss = 0
            val_sol_loss = 0
            for i, batch in enumerate(val_loader):
                embedding, loc, sol, metadata = batch  # get localization and solubility label
                embedding, loc, sol, sol_known = embedding.to(self.device), loc.to(self.device), sol.to(self.device), \
                                                 metadata['solubility_known'].to(self.device)
                with torch.no_grad():
                    prediction = self.model(embedding)

                    loss, loc_loss, sol_loss = self.loss_func(prediction, loc, sol, sol_known, args)
                    prediction = torch.max(prediction[..., :10], dim=1)[1]  # get indices of the highest value for loc
                    val_results.append(torch.stack((prediction, loc), dim=1).data.detach().cpu().numpy())
                    val_loc_loss += loc_loss.item()
                    val_sol_loss += sol_loss.item()

            train_results = np.concatenate(train_results)  # [number_train_proteins, 2] prediction and loc
            val_results = np.concatenate(val_results)  # [number_val_proteins, 2] prediction and loc

            train_acc = 100 * np.equal(train_results[:, 0], train_results[:, 1]).sum() / len(train_results)
            val_acc = 100 * np.equal(val_results[:, 0], val_results[:, 1]).sum() / len(val_results)
            print('[Epoch %d] VAL accuracy: %.4f%% train accuracy: %.4f%%' % (epoch + 1, val_acc, train_acc))

            # write to tensorboard
            tensorboard_confusion_matrix(train_results, val_results, self.writer, epoch + 1)
            self.writer.add_scalars('Loc_Acc', {'train': train_acc, 'val': val_acc}, epoch + 1)
            self.writer.add_scalars('Loc_Loss', {'train': train_loc_loss / t_iters, 'val': val_loc_loss / v_iters},
                                    epoch + 1)
            if args.solubility_loss != 0:
                self.writer.add_scalars('Sol_Loss', {'train': train_sol_loss, 'val': val_sol_loss}, epoch + 1)

            experiment_checkpoint(self.writer.log_dir, self.model, self.optim, epoch + 1, args.config.name)
