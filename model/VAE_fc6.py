import torch
import torch.nn as nn
from torch import from_numpy
import pickle as pkl


class VAE_fc6(nn.Module):
    def __init__(self):
        # C64
        super(VAE_fc6, self).__init__()
        # define layer
        self.defc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.fc7_relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.defc6 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.fc6_relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)        
        self.defc5 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.fc5_relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)        
        self.deconv5 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.deconv5_relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.conv5_1 = nn.ConvTranspose2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_1_relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.deconv4 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.deconv4_relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.conv4_1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4_1_relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.deconv3_relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.conv3_1 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_1_relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.deconv2_relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.deconv1_relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.deconv0 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        x = self.defc7(x)
        x = self.fc7_relu(x)
        x = self.defc6(x)
        x = self.fc6_relu(x)
        x = self.defc5(x)
        x = self.fc5_relu(x)
        x = x.view(64, 256, 4, 4)
        x = self.deconv5(x)
        x = self.deconv5_relu(x)
        x = self.conv5_1(x)
        x = self.conv5_1_relu(x)
        x = self.deconv4(x)
        x = self.deconv4_relu(x)
        x = self.conv4_1(x)
        x = self.conv4_1_relu(x)
        x = self.deconv3(x)
        x = self.deconv3_relu(x)
        x = self.conv3_1(x)
        x = self.conv3_1_relu(x)
        x = self.deconv2(x)
        x = self.deconv2_relu(x)
        x = self.deconv1(x)
        x = self.deconv1_relu(x)
        x = self.deconv0(x)
        return x
        
    def load_weights_from_pkl(self, model_pkl):
        with open(model_pkl, 'rb') as wp:
            try:
                # for python3
                name_weights = pkl.load(wp, encoding='latin1')
            except TypeError as e:
                # for python2
                name_weights = pkl.load(wp)
            state_dict = {}
        
            def _set(layer, key):
                state_dict[layer + '.weight'] = from_numpy(name_weights[key]['weight'])
                state_dict[layer + '.bias'] = from_numpy(name_weights[key]['bias'])
        
            _set('defc7', 'defc7')
            _set('defc6', 'defc6')
            _set('defc5', 'defc5')
            _set('deconv5', 'deconv5')
            _set('conv5_1', 'conv5_1')
            _set('deconv4', 'deconv4')
            _set('conv4_1', 'conv4_1')
            _set('deconv3', 'deconv3')
            _set('conv3_1', 'conv3_1')
            _set('deconv2', 'deconv2')
            _set('deconv1', 'deconv1')
            _set('deconv0', 'deconv0')    
            
            self.load_state_dict(state_dict)


#start_layer = 'fc6'
#model_pkl = f'VAE_{start_layer}.pickle'
#
## 载入提取到的权重
#f = open(model_pkl, 'rb')
#name_weights = pkl.load(f)
#f.close()
## 新建网络实例
#net = VAE_fc6()
## 载入参数
#net.load_weights_from_pkl(model_pkl)
## 设置为测试模式
#net.eval()
#params = dict(net.named_parameters())
#
## 检查defc7的参数是否载入正确
#par_torch = params['deconv2.weight'].detach().numpy()
#par_caffe = name_weights['deconv2']['weight']
#print((par_torch==par_caffe).all())
#
#torch.save(net.state_dict(), f'VAE_{start_layer}_state_dict.pth')
