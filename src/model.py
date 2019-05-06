import torch
from torch import nn

import modules


class Pix2Pix:
    def __init__(self, lr, lambda_l1, lambda_d, dataset):
        if dataset == 'places':
            self.net_g = modules.UNet(in_channels=1, out_channels=3)
            self.net_d = modules.PatchGAN(in_channels=4, out_channels=1)
        else:
            self.net_g = modules.UNet()
            self.net_d = modules.PatchGAN()

        self.net_g.apply(self.init)
        self.net_d.apply(self.init)

        self.optimizer_g = torch.optim.Adam(self.net_g.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.Adam(self.net_d.parameters(), lr=lr, betas=(0.5, 0.999))

        self.lr_lambda = lambda epoch: 1.0
        self.scheduler_g = torch.optim.lr_scheduler.LambdaLR(self.optimizer_g, lr_lambda=self.lr_lambda)
        self.scheduler_d = torch.optim.lr_scheduler.LambdaLR(self.optimizer_d, lr_lambda=self.lr_lambda)

        self.loss_fn = nn.BCELoss()
        self.gan_loss_fn = lambda y_hat, y: self.loss_fn(y_hat, torch.ones_like(y_hat) * y)
        self.l1_loss_fn = nn.L1Loss()

        self.lr = lr
        self.lambda_l1 = lambda_l1
        self.lambda_d = lambda_d

    def init(self, net):
        for m in net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def scheduler_step(self):
        self.scheduler_g.step()
        self.scheduler_d.step()

    def _apply(self, fn):
        fn(self.net_g)
        fn(self.net_d)

        self.optimizer_g = torch.optim.Adam(self.net_g.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.Adam(self.net_d.parameters(), lr=self.lr, betas=(0.5, 0.999))
        # TODO: Restore optimizer's state dict on restarting training...
        # See https://github.com/pytorch/pytorch/issues/2830#issuecomment-336194949
        
        for optimizer in (self.optimizer_g, self.optimizer_d):
            for state in self.optimizer_g.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = fn(value)

    def cpu(self):
        self._apply(lambda t: t.cpu())
        return self

    def to(self, device=None):
        self._apply(lambda t: t.to(device))
        return self

    def train(self, input):
        x, y = input
        output_g = self.net_g(x)

        self.optimizer_d.zero_grad()
        loss_d_real = self.gan_loss_fn(self.net_d(y, x), 1) * self.lambda_d
        loss_d_fake = self.gan_loss_fn(self.net_d(output_g.detach(), x), 0) * self.lambda_d
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        self.optimizer_d.step()

        self.optimizer_g.zero_grad()
        loss_g_gan = self.gan_loss_fn(self.net_d(output_g, x), 1)
        loss_g_l1 = self.l1_loss_fn(output_g, y) * self.lambda_l1
        loss_g = loss_g_gan + loss_g_l1
        loss_g.backward()
        self.optimizer_g.step()

        return {
            'g': loss_g.item(), 'g_gan': loss_g_gan.item(), 'g_l1': loss_g_l1.item(),
            'd': loss_d.item(), 'd_real': loss_d_real.item(), 'd_fake': loss_d_fake.item()
        }, output_g

    def eval(self, input):
        with torch.no_grad():
            x, y = input
            output_g = self.net_g(x)

            loss_d_real = self.gan_loss_fn(self.net_d(y, x), 1) * self.lambda_d
            loss_d_fake = self.gan_loss_fn(self.net_d(output_g, x), 0) * self.lambda_d
            loss_d = loss_d_real + loss_d_fake

            loss_g_gan = self.gan_loss_fn(self.net_d(output_g, x), 1)
            loss_g_l1 = self.l1_loss_fn(output_g, y) * self.lambda_l1
            loss_g = loss_g_gan + loss_g_l1

        return {
            'g': loss_g.item(), 'g_gan': loss_g_gan.item(), 'g_l1': loss_g_l1.item(),
            'd': loss_d.item(), 'd_real': loss_d_real.item(), 'd_fake': loss_d_fake.item()
        }, output_g

    def named_components(self):
        yield 'generator', self.net_g
        yield 'discriminator', self.net_d
        yield 'generator_optimizer', self.optimizer_g
        yield 'discriminator_optimizer', self.optimizer_d
        yield 'generator_scheduler', self.scheduler_g
        yield 'discriminator_scheduler', self.scheduler_d

    def state_dict(self):
        return {name: component.state_dict() for name, component in self.named_components()}

    def load_state_dict(self, state_dict):
        for name, component in self.named_components():
            component.load_state_dict(state_dict[name])
