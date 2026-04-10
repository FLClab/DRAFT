import torch 
from torch import nn 
import networks 

class Pix2Pix(nn.Module):
    def __init__(
        self,
        num_epochs: int = 100,
        in_channels: int = 1,
        out_channels: int = 1,
        is_train: bool = True,
    ):
        super(Pix2Pix, self).__init__() 
        self.num_epochs = num_epochs
        self.is_train = is_train
        self.loss_names = ["G_GAN", "G_GEN", "D_real", "D_fake"]
        self.visual_names = ["real_conf", "fake_STED", "real_STED"]
        if self.is_train:
            self.model_names = ["G", "D"] 
        else:
            self.model_names = ["G"] 

        self.netG = networks.define_G(
            input_nc=in_channels,
            output_nc=out_channels,
            ngf=64,
            netG="resnet_9blocks", 
            norm="batch",
        )
        if self.is_train:
            self.netD = networks.define_D(
                input_nc=in_channels + out_channels,
                ndf=64,
                netD="basic",
            )
            self.criterionGAN = networks.GANLoss(gan_mode="vanilla")
            self.criterionL1 = nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
            self.optimizers = [self.optimizer_G, self.optimizer_D] 
            self.schedulers = [
                networks.get_scheduler(optimizer, "cosine", T_max=self.num_epochs) for optimizer in self.optimizers
            ]
            
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def set_input(self, batch: torch.Tensor):
        device = next(self.parameters()).device
        self.real_conf = batch[0].to(device)
        self.real_sted = batch[1].to(device)

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            scheduler.step()


    def forward(self):
        self.fake_sted = self.netG(self.real_conf) 

    def backward_D(self):
        fake_cat = torch.cat((self.real_conf, self.fake_sted), 1)
        pred_fake = self.netD(fake_cat.detach())
        self.loss_d_fake = self.criterionGAN(pred_fake, False)
        real_cat = torch.cat((self.real_conf, self.real_sted), 1) 
        pred_real = self.netD(real_cat)
        self.loss_D_real = self.criterionGAN(pred_real, True) 
        self.loss_D = (self.loss_D_real + self.loss_d_fake) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_cat = torch.cat((self.real_conf, self.fake_sted), 1) 
        pred_fake = self.netD(fake_cat)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) 
        self.loss_G_GEN = self.criterionL1(self.fake_sted, self.real_sted) * 100.0
        self.loss_G = self.loss_G_GAN + self.loss_G_GEN 
        self.loss_G.backward()

    def backprop(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step() 

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step() 




