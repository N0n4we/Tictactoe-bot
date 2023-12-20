_exp_name='sample'

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from PIL import Image


test_tfm = transforms.Compose([
    transforms.ToTensor(),
])


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),  # [16, 3, 3]
            nn.BatchNorm2d(16),
            nn.ReLU(),

        )

        #self.attention = nn.MultiheadAttention(embed_dim=9, num_heads=3)

        self.fc = nn.Sequential(
            nn.Linear(16 * 3 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 9)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

    '''
    Attention mechanism
    
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], out.size()[1], -1)
        attention_out, _ = self.attention(out, out, out)
        fc_input = attention_out.view(out.size()[0], -1)
        return self.fc(fc_input)
    '''


def reset(fname):
    image = cv2.imread(fname, cv2.IMREAD_COLOR)
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
    for n in range(3):
        for m in range(3):
            image[n, m] = [255, 0, 0]
    cv2.cvtColor(image, cv2.COLOR_RGB2BGR, image)
    cv2.imwrite(fname, image)
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
    return image

def save(fname, image):
    cv2.cvtColor(image, cv2.COLOR_RGB2BGR, image)
    cv2.imwrite(fname, image)
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)

def read(fname):
    image = cv2.imread(fname, cv2.IMREAD_COLOR)
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
    return image

def tic(player,place,image):
    m = place // 3
    n = place % 3
    if player==1:
        image[m, n] = [0, 255, 0]
    elif player==2:
        image[m, n] = [0, 0, 255]

def predict(fname,image):
    test_set = Image.open(fname).convert('RGB')
    data = test_tfm(test_set).unsqueeze(0)
    with torch.no_grad():
        test_pred = model_best(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        prediction = test_label  # .squeeze(1)
    tic(2,prediction,image)
    save(fname, image)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

#载入模型
version = '22(cnn+fc)'

model_best = Classifier().to(device)
model_best.load_state_dict(torch.load(f"{_exp_name}_best"+version+".ckpt"))
model_best.eval()

#创建棋盘文件
fname = 'board.png'
image = Image.new('RGB', (3, 3), (0, 0, 0))
image.save(fname)

#初始化棋盘
image = reset(fname)

print('输入-1停止，输入-2重置并AI先手，输入-3重置并玩家先手')

while True:
    player = int(input('->'))

    if player == -1:
        break
    elif player == -2:
        image = reset(fname)
    elif player == -3:
        image = reset(fname)
        stop = input('输入任意键继续')
        continue
    else:
        tic(1,player,image)
        save(fname, image)
    stop = input('输入任意键继续')
    predict(fname,image)



