import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class BaseSolver():
    def __init__(self, model, args, optim=torch.optim.Adam, loss_func=torch.nn.MSELoss()):
        self.optim = optim(list(model.parameters()), args.lrate)
        self.loss_func = loss_func
        self.writer = SummaryWriter()
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def train(self, train_loader, val_loader, h: int, w: int,  parser):

        args = self.args

        print('START TRAIN.')
        for epoch in range(args.num_epochs):  # loop over the dataset multiple times
            self.model.train()
            train_loss = 0
            for i, data in enumerate(train_loader):
                

                self.optim.zero_grad()

                loss = self.loss(rgb, rgb_fine, rgb_truth)
                loss.backward()
                self.optim.step()

                loss_item = loss.item()
                if i % args.log_iterations == args.log_iterations - 1:
                    print('[Epoch %d, Iteration %5d/%5d] TRAIN loss: %.7f' %
                          (epoch + 1, i + 1, iter_per_epoch, loss_item))
                    if args.early_validation:
                        self.model_coarse.eval()
                        self.model_fine.eval()
                        val_loss = 0
                        for j, data in enumerate(val_loader):
                            for j, element in enumerate(data):
                                data[j] = element.to(self.device)
                            rgb_truth = data[-1]

                            rgb, rgb_fine, _, _ = self.pipeline(data)

                            loss = self.nerf_loss(rgb, rgb_fine, rgb_truth)
                            val_loss += loss.item()
                        self.writer.add_scalars('Loss curve every nth iteration', {'train loss': loss_item,
                                                                                   'val loss': val_loss / len(
                                                                                       val_loader)},
                                                i // args.log_iterations + epoch * (
                                                        iter_per_epoch // args.log_iterations))
                train_loss += loss_item
            print('[Epoch %d] Average loss of Epoch: %.7f' %
                  (epoch + 1, train_loss / iter_per_epoch))

            self.model_coarse.eval()
            self.model_fine.eval()
            val_loss = 0
            rerender_images = []
            samples = []
            ground_truth_images = []
            densities_list = []
            image_counter = 0
            for i, data in enumerate(val_loader):
                for j, element in enumerate(data):
                    data[j] = element.to(self.device)
                rgb_truth = data[-1]

                rgb, rgb_fine, ray_samples, densities = self.pipeline(data)

                loss = self.nerf_loss(rgb, rgb_fine, rgb_truth)
                val_loss += loss.item()

                ground_truth_images.append(rgb_truth.detach().cpu().numpy())
                rerender_images.append(rgb_fine.detach().cpu().numpy())
                samples.append(ray_samples.detach().cpu().numpy())
                densities_list.append(densities.detach().cpu().numpy())
                if np.concatenate(densities_list).shape[0]>=(h*w):
                    while np.concatenate(densities_list).shape[0]>=(h*w):
                        densities_list = np.concatenate(densities_list)
                        image_densities = densities_list[:h*w].reshape(-1)
                        densities_list = [densities_list[h*w:]]
                        samples = np.concatenate(samples)
                        image_samples = samples[:h*w].reshape(-1, 3)
                        samples = [samples[h*w:]]
                        vedo_data(self.writer, image_densities, image_samples,
                                  image_warps=None, epoch=epoch + 1,
                                  image_idx=image_counter)
                        image_counter += 1
            if len(val_loader) != 0:
                rerender_images = np.concatenate(rerender_images, 0).reshape((-1, h, w, 3))
                ground_truth_images = np.concatenate(ground_truth_images).reshape((-1, h, w, 3))

            tensorboard_rerenders(self.writer, args.number_validation_images, rerender_images, ground_truth_images,
                                  step=epoch, ray_warps=None)

            print('[Epoch %d] VAL loss: %.7f' % (epoch + 1, val_loss / (len(val_loader) or not len(val_loader))))
            self.writer.add_scalars('Loss Curve', {'train loss': train_loss / iter_per_epoch,
                                                   'val loss': val_loss / (len(val_loader) or not len(val_loader))},
                                    epoch)

            save_run(self.writer.log_dir, [self.model_coarse, self.model_fine],
                         ['model_coarse.pt', 'model_fine.pt'], parser)
        print('FINISH.')