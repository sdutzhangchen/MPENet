import torch
import torch.nn as nn
import torch.nn.functional as F



class DepthwiseSeparableConv(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, padding=1, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=inp, bias=False)
        self.pointwise = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
class Multiscale(nn.Module): 
    def __init__(self,inp,dilation=[1,2,3,4],squeeze_radio=2,group_kernel_size=3,group_size=2):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(inp, inp // 4, kernel_size=3, dilation=dilation[0], padding=dilation[0])
        self.conv2 = DepthwiseSeparableConv(inp, inp // 4, kernel_size=3, dilation=dilation[1], padding=dilation[1])
        self.conv3 = DepthwiseSeparableConv(inp, inp // 4, kernel_size=3, dilation=dilation[2], padding=dilation[2])
        self.conv4 = DepthwiseSeparableConv(inp, inp // 4, kernel_size=3, dilation=dilation[3], padding=dilation[3])
        
    def forward(self, x):
    #多尺度提取部分
        x1= self.conv1(x)   
        x2= self.conv2(x)    
        x3= self.conv3(x)   
        x4= self.conv4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1) 
        return x 




class FCA(nn.Module):
    def __init__(self, channels, reduction=16):
        super(FCA, self).__init__()
        self.channels = channels
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        std_out = torch.std(x, dim=[2, 3], keepdim=True)
        
        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)
        std_out = self.fc(std_out)
        
        x0 = x * avg_out
        x1 = x * max_out
        x2 = x * std_out
        x = x0 + x1 + x2
        
        return x
class FSA(nn.Module):
    def __init__(self, channels):
        super(FSA, self).__init__()
        self.channels = channels
        # 使用分组卷积减少参数量
        self.conv7x7 = nn.Conv2d(channels * 3, channels, kernel_size=7, padding=3, groups=channels // 8, bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        std_out = torch.std(x, dim=[2, 3], keepdim=True)

        # 在通道维度拼接
        out = torch.cat([avg_out, max_out, std_out], dim=1)
        
        out = self.conv7x7(out)
        out = self.sigmoid(out)

        return x * out

class HSP(nn.Module):
    def __init__(self, channels, reduction=16):
        super(HSP, self).__init__()
        self.multi_scale = Multiscale(channels)
        self.fca = FCA(channels, reduction)
        self.fsa = FSA(channels)
          # 使用替代模块

    def forward(self, x):
        ms_out = self.multi_scale(x)
        fca_out = self.fca(ms_out)
        x = ms_out + fca_out
        fsa_out = self.fsa(x)
        x = x + fsa_out
        
        return x



