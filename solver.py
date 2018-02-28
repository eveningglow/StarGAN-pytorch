import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision

import os

from dataloader import data_loader
import model
import util

class Solver():
    def __init__(self, img_dir='data/img', ann_path='data/list_attr_celeba.txt', 
                 result_dir='result', weight_dir='weight', 
                 batch_size=16, lr=0.001, beta1=0.5, beta2=0.999, lambda_gp=10, lambda_recon=10,
                 n_critic=5, class_num=5, num_epoch=20, save_every=500, load_weight=False):
        
        if torch.cuda.is_available() is True:
            self.dtype = torch.cuda.FloatTensor
            self.itype = torch.cuda.LongTensor
        else:
            self.dtype = torch.FloatTensor
            self.itype = torch.LongTensor
        
        self.dloader, self.dlen = data_loader(img_dir, ann_path, batch_size=batch_size, 
                                              shuffle=True, mode='train')
        
        self.D = model.Discriminator(class_num=class_num).type(self.dtype)
        self.G = model.Generator().type(self.dtype)
        
        self.optim_D = optim.Adam(self.D.parameters(), lr=lr, betas=(beta1, beta2))
        self.optim_G = optim.Adam(self.G.parameters(), lr=lr, betas=(beta1, beta2))

        self.BCE_loss = nn.BCEWithLogitsLoss(size_average=False)
        self.L1_loss = nn.L1Loss(size_average=True)
        
        self.start_epoch = 0
        self.num_epoch = num_epoch
        self.n_critic = n_critic
        self.lr = lr
        self.lambda_gp = lambda_gp
        self.lambda_recon = lambda_recon
        
        self.save_every = save_every
        self.load_weight = load_weight
        self.class_num = class_num
        self.result_dir = result_dir
        self.weight_dir = weight_dir
        
    def lr_scheduler(self, optim, epoch, init_lr):
        if epoch >= 10:
            lr = init_lr - init_lr / 10 * (epoch - 10)
        else:
            lr = init_lr

        print(str(optim.__class__.__name__), end='')
        print(' lr is set to %f' % lr)

        for param_group in optim.param_groups:
            param_group['lr'] = lr

        return optim
    
    def load_pretrained(self):
        self.D.load_state_dict(torch.load(os.path.join(self.weight_dir, 'D.pkl')))
        self.G.load_state_dict(torch.load(os.path.join(self.weight_dir, 'G.pkl')))
        
        log_file = open('log.txt', 'r')
        line = log_file.readline()
        self.start_epoch = int(line)
        
    def train(self):
        if self.load_weight is True:
            self.load_pretrained()
            
        for epoch in range(self.start_epoch, self.num_epoch):
            self.optim_D = self.lr_scheduler(self.optim_D, epoch, self.lr)
            self.optim_G = self.lr_scheduler(self.optim_G, epoch, self.lr)

            for iters, (real_img, real_label) in enumerate(self.dloader):
                N, C, H, W = real_img.size()   

                real_img = util.var(real_img, requires_grad=False)    
                real_label = util.var(real_label, requires_grad=False)

                # Source and classification loss with real img and real label
                real_src_score, real_cls_score = self.D(real_img)

                real_src_loss = -torch.mean(real_src_score)
                real_cls_loss = self.BCE_loss(real_cls_score, real_label) / N

                # Make random target label and c using real label and concat with real img
                target_label = real_label[torch.randperm(real_label.size(0)).type(self.itype)]

                # Source and classification loss with fake img and target label
                fake_img = self.G(real_img, target_label)
                fake_src_score, fake_cls_score = self.D(fake_img)
                fake_src_loss = torch.mean(fake_src_score)

                # Gradient Penalty
                alpha = torch.rand(N, 1, 1, 1).type(self.dtype)
                x_hat = util.var((alpha * real_img.data + (1 - alpha) * fake_img.data), requires_grad=True)
                x_hat_score, _ = self.D(x_hat)

                grad = torch.autograd.grad(outputs=x_hat_score,
                                           inputs=x_hat,
                                           grad_outputs=torch.ones(x_hat_score.size()).type(self.dtype),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]                  

                grad = grad.view(grad.size()[0], -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))            
                gp_loss = self.lambda_gp * torch.mean((grad_l2norm - 1)**2)

                # Total Loss and update
                D_loss = real_src_loss + fake_src_loss + real_cls_loss + gp_loss

                self.optim_D.zero_grad()
                self.optim_G.zero_grad()
                D_loss.backward()
                self.optim_D.step()

                if iters % self.n_critic == 0:
                    # Source and classification loss with fake img and target label
                    fake_img = self.G(real_img, target_label)
                    recon_img = self.G(fake_img, real_label)

                    fake_src_score, fake_cls_score = self.D(fake_img)
                    fake_src_loss = -torch.mean(fake_src_score)
                    fake_cls_loss = self.BCE_loss(fake_cls_score, target_label)
                    recon_loss = self.lambda_recon * self.L1_loss(recon_img, real_img)

                    # Total loss and update
                    G_loss = fake_src_loss + fake_cls_loss + recon_loss

                    self.optim_D.zero_grad()
                    self.optim_G.zero_grad()
                    G_loss.backward()
                    self.optim_G.step()

                # Write log
                log_file = open('log.txt', 'w')
                log_file.write(str(epoch))
                
                # Print loss, save result images and weights
                if iters % self.save_every == 0:
                    if os.path.exists(self.result_dir) is False:
                        os.makedirs(self.result_dir)

                    if os.path.exists(self.weight_dir) is False:
                        os.makedirs(self.weight_dir)

                    # Print loss
                    print('[Epoch : %d / Iter : %d] : D_loss : %f, G_loss : %f, real_cls : %f, fake_cls : %f'\
                              %(epoch, iters, D_loss.data[0], G_loss.data[0], \
                                real_cls_loss.data[0], fake_cls_loss.data[0]))
                    # Save image
                    img_name = str(epoch) + '_' + str(iters) + '.png'
                    img_path = os.path.join(self.result_dir, img_name)

                    real_img = util.denorm(real_img)
                    fake_img = util.denorm(fake_img)
                    util.save_img(real_img, fake_img, img_path)

                    # Save weight
                    torch.save(self.D.state_dict(), os.path.join(self.weight_dir, 'D.pkl'))
                    torch.save(self.G.state_dict(), os.path.join(self.weight_dir, 'G.pkl'))

            # Save weight at the end of every epoch
            torch.save(self.D.state_dict(), os.path.join(self.weight_dir, 'D.pkl'))
            torch.save(self.G.state_dict(), os.path.join(self.weight_dir, 'G.pkl'))

        # Save weight at the end of training
        torch.save(self.D.state_dict(), os.path.join(self.weight_dir, 'D.pkl'))
        torch.save(self.G.state_dict(), os.path.join(self.weight_dir, 'G.pkl'))