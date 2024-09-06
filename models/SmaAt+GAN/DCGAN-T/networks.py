
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# class Discriminator(nn.Module):
#     def __init__(self, in_channels=20, img_size=256):
#         super(Discriminator, self).__init__()
#         self.conv_layers = nn.Sequential(
#             # 输入大小: [1, 20, 256, 256]
#             nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2, inplace=True),
#             # 大小: [1, 32, 128, 128]
#             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),
#             # 大小: [1, 64, 64, 64]
#             nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2, inplace=True),
#             # 大小: [1, 64, 32, 32]
#             nn.Conv2d(32, 8, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(8),
#             nn.LeakyReLU(0.2, inplace=True)
#             # 大小: [1, 64, 16, 16]
#         )
#         self.fc_layers = nn.Sequential(
#             nn.Linear(8 * 16 * 16, 128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(128, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc_layers(x)
#         return x


class Discriminator(nn.Module):
    def __init__(self, in_c=5, kernel_num=4):
        super(Discriminator, self).__init__()
        self.kernel_num = kernel_num
        
        self.conv1 = spectral_norm(nn.Conv2d(in_c, kernel_num, 4, 2, 1))
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = spectral_norm(nn.Conv2d(kernel_num, kernel_num * 2, 4, 2, 1))
        self.norm2 = nn.InstanceNorm2d(kernel_num * 2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = spectral_norm(nn.Conv2d(kernel_num * 2, kernel_num * 4, 4, 2, 1))
        self.norm3 = nn.InstanceNorm2d(kernel_num * 4)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = spectral_norm(nn.Conv2d(kernel_num * 4, kernel_num * 8, 4, 2, 1))
        self.norm4 = nn.InstanceNorm2d(kernel_num * 8)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = spectral_norm(nn.Conv2d(kernel_num * 8, kernel_num * 4, 4, 2, 1))
        self.norm5 = nn.InstanceNorm2d(kernel_num * 4)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)

        self.conv6 = spectral_norm(nn.Conv2d(kernel_num * 4, kernel_num * 2, 4, 1, 0))
        
        self.fc1 = nn.Linear(8 * 5 * 5, 1024)
        self.fc_relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.fc3 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.norm4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.norm5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        x = self.fc1(x)
        x = self.fc_relu1(x)
        x = self.fc2(x)
        x = self.fc_relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        return x

# # Example usage
# in_c = 20
# kernel_num = 4
# model = Discriminator(in_c, kernel_num)

# input_tensor = torch.randn(1, 20, 256, 256)
# output = model(input_tensor)
# print(output.shape)  # Should be [1, 1]

