import torch
import torch.nn
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import os
from torch.autograd import Variable
from os.path import expanduser
from PYTORCH.loader.anime_loader import get_loader_folder, get_pure_loader_folder
from PYTORCH.packageList import *
from CV.ImageIO import *
from PYTORCH.loader.my_transform import ToTensor, Normalize, CenterCrop, Compose, ToPILImage

def save_checkpoint_here(state, filename='checkpoint'):
    torch.save(state, filename+".pth.tar")

home = expanduser("~")


cnt = 0
lr = 1e-3


# class afterprocess(torch.nn.Module):
#     def __init__(self, ):
#         super().__init__()

#     def forward(self, x):
#         x = x.view(3,100,100)
#         # x = torch.squeeze(x)

# after = afterprocess().cuda()

# preprocess
Precoder = torch.nn.Sequential(
    torch.nn.AdaptiveMaxPool2d((100,100))
)


# Encoder

class Encoder(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.Qe = self.Q()
    def forward(self, x):
        x = self.Qe(x)
        return x
    def Q(self):
        return torch.nn.Sequential(
            torch.nn.Conv2d(1,10,(3,3),padding=1),
            torch.nn.MaxPool2d((2,2),stride=1),
            torch.nn.BatchNorm2d(10),
            torch.nn.ReLU(),
            torch.nn.Conv2d(10,20,(3,3),padding=1),
            torch.nn.MaxPool2d((2,2),stride=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool2d((100,100)),
            # torch.nn.Linear(h_dim, z_dim)
        )



class Decoder(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.Qe = self.Q()
    def forward(self, x):
        x = self.Qe(x)
        # x = x.view(x.size(0), -1)

        return x
    def Q(self):
        return torch.nn.Sequential(
            torch.nn.Conv2d(20,10,(3,3),padding=1),
            torch.nn.MaxPool2d((2,2),stride=1),
            torch.nn.BatchNorm2d(10),
            torch.nn.ReLU(),
            torch.nn.Conv2d(10,1,(3,3),padding=1),
            torch.nn.MaxPool2d((2,2),stride=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool2d((100,100)),
        )


class autoencoder(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.forwardlayer = self.layer()

    def forward(self,x):
        x = self.forwardlayer(x)
        return x

    def layer(self):
        layers =[Encoder(), Decoder()]
        return torch.nn.Sequential(*layers)


# autoencoder = torch.nn.Sequential(*[Encoder, Decoder])


gpu_id=[0, 1]



minibatch=1
path = home+"/CODE/Kaggle/eye_washed_binary/"
train_loader = get_pure_loader_folder(path, batch_size=minibatch, binary=True)

# Q = Encoder().cuda()
# P = Decoder().cuda()

A = autoencoder().cuda()

# Q = DataParallel(Q, device_ids=gpu_id).cuda()
# P = DataParallel(P, device_ids=gpu_id).cuda()

A = DataParallel(A, device_ids=gpu_id).cuda()


def reset_grad():
    A.zero_grad()
    # Q.zero_grad()
    # P.zero_grad()

# Q_solver = optim.Adam(Q.parameters(), lr=lr)
# P_solver = optim.Adam(P.parameters(), lr=lr)

A_solver = optim.Adam(A.parameters(), lr=lr)

# criterion = torch.nn.binary_cross_entropy()
# criterion = F.binary_cross_entropy()

# criterion.cuda()

for it in range(100000):
    for data in train_loader:
        inputs, labels, path = data
        inputs, labels  = Variable(inputs.cuda()), Variable(labels)
        inputs = Precoder(inputs)
        # print(inputs.size())

        #     """ Reconstruction phase """
        outputs = A(inputs)
        # outputs = P(outputs)
        recon_loss = F.binary_cross_entropy(inputs, outputs)
        # kl_loss = 0.5 * torch.sum(torch.exp())

        recon_loss.backward()
        A_solver.step()
        # Q_solver.step()

        Load_Name = "./VAE_RESULT"

        save_checkpoint_here({'state_dict':A.state_dict()},filename=Load_Name)

        # Print and plot every now and then
        if it % 1000 == 0:
            print('Iter-{}; recon_loss: {:.4}'
                  .format(it, recon_loss.data[0]))

            samples = A(inputs)
            # print(samples.size())
            samples = torch.squeeze(samples)
            samples = torch.unsqueeze(samples,0)
            # print(samples.size())
            samples = samples.cpu().data
            _transform = ToPILImage()
            img = _transform(samples)

            if not os.path.exists(home+"/CODE/Kaggle/eye_result"):
                os.makedirs(home+"/CODE/Kaggle/eye_result")
            _name = home+"/CODE/Kaggle/eye_result/" + str(cnt).zfill(3)+".jpeg"
            img.save(_name)
            cnt += 1

