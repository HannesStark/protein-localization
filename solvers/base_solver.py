import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class BaseSolver():
    def __init__(self, model, args, optim=torch.optim.Adam, loss_func=torch.nn.MSELoss()):
        self.optim = optim(list(model.parameters()), args.lrate)
        self.loss_func = loss_func
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.writer = SummaryWriter('runs/' + args.experiment_name + datetime.now().strftime('%d-%m_%H:%M'))

    def train(self, train_loader, val_loader):
        args = self.args

        for epoch in range(args.num_epochs):  # loop over the dataset multiple times
            self.model.train()
            train_loss = 0
            for i, batch in enumerate(train_loader):
                embedding, label = batch

                classification = self.model(embedding)

                self.optim.zero_grad()
                loss = self.loss_func(classification, label)
                loss.backward()
                self.optim.step()

                loss_item = loss.item()
                train_loss += loss_item
                if i % args.log_iterations == args.log_iterations - 1:
                    print('[Epoch %d, Iteration %5d/%5d] TRAIN loss: %.7f' % (
                        epoch + 1, i + 1, len(train_loader), loss_item))
            print('[Epoch %d] Train loss of Epoch: %.7f' % (epoch + 1, train_loss / len(train_loader)))

            self.model.eval()
            val_loss = 0
            for i, batch in enumerate(val_loader):
                embedding, label = batch

                classification = self.model(embedding)

                loss = self.loss_func(classification, label)
                val_loss += loss.item()

            print(
                '[Epoch %d] VAL loss of Epoch: %.7f' % (epoch + 1, val_loss / (len(val_loader) or not len(val_loader))))
            self.writer.add_scalars('Epoch Loss', {'train loss': train_loss / len(train_loader),
                                                   'val loss': val_loss / (len(val_loader) or not len(val_loader))},
                                    epoch)

            # TODO: implement saving of model
            # save_run(self.writer.log_dir, [self.model], parser)
