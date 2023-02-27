import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation(name='relu'):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'silu':
        return nn.SiLU()
    elif name == 'prelu':
        return nn.PReLU()
    else:
        raise NotImplementedError

def get_norm(channels,group_norm_channels,name='batch'):
    if name == 'batch':
        return nn.BatchNorm2d(channels)

    elif name == 'group':
        return nn.GroupNorm(num_groups=group_norm_channels,num_channels=channels)

    elif name == 'layer':
        #dont use this
        return nn.LayerNorm()
    else:
        raise NotImplementedError

class ConvNormAct(nn.Sequential):

    def __init__(
        self,
        in_channels,  
        out_channels,
        kernel_size, 
        stride=1, 
        padding=0, 
        dilation=1, 
        groups=1,
        activation='relu',
        norm_type= 'batch',
        group_norm_channels=4,
        ):
        super().__init__()
        conv = nn.Conv2d(
                in_channels = in_channels, 
                out_channels = out_channels, 
                kernel_size = kernel_size, 
                stride=stride, 
                padding=padding, 
                dilation=dilation, 
                groups=groups, 
                bias=False,
                padding_mode='zeros')
        act = get_activation(activation)

        norm = get_norm(out_channels,group_norm_channels,norm_type)
        super().__init__(conv,norm,act)



class RefineNetv0(nn.Module):

    def __init__(
        self,
        in_channels = 6,
        out_channels = 1,
        inter_channels = [16,32],
        aspp = False,
        aspp_dilations = [2,4,8],
        aspp_channels = 16,
        activation='relu',
        norm_type= 'batch',
        group_norm_channels=4,
        aspp_dropout = 0.2,
        
        ):

        super().__init__()

        all_channels = [in_channels] + inter_channels #+ [out_channels]
        
        self.conv_blocks = nn.ModuleList(
            [
                ConvNormAct(
                    in_channels=all_channels[i],
                    out_channels=all_channels[i+1],
                    kernel_size=3,
                    padding=1,
                    norm_type=norm_type,
                    activation=activation,
                    group_norm_channels=group_norm_channels
                    ) for i in range(len(all_channels)-1)
            ]
        )
        self.out_conv = nn.Conv2d(
                    in_channels=all_channels[-1],
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1
                ) 
        
        self.aspp = aspp
        if aspp:
            self.aspp_convs = nn.ModuleList(
                [
                ConvNormAct(
                    in_channels=all_channels[-1],
                    out_channels=aspp_channels,
                    kernel_size=3,
                    padding=dil,
                    dilation=dil,
                    norm_type=norm_type,
                    activation=activation,
                    group_norm_channels=group_norm_channels
                    ) for dil in aspp_dilations
                ]
                )
            
            self.aspp_project = ConvNormAct(
                                    in_channels=aspp_channels*len(aspp_dilations)+all_channels[-1],
                                    out_channels=all_channels[-1],
                                    kernel_size=1,
                                    padding=0,
                                    dilation=1,
                                    norm_type=norm_type,
                                    activation=activation,
                                    group_norm_channels=group_norm_channels
                                    )
            self.aspp_drop = nn.Identity() if aspp_dropout == 0 else nn.Dropout2d(aspp_dropout)

    def forward(self,x):
        for conv in self.conv_blocks:
            x = conv(x)
        
        if self.aspp:
            x_cat = [x]
            for aspp_conv in self.aspp_convs:
                x_aspp = aspp_conv(x)
                x_cat.append(x_aspp)
            
            x_cat = torch.cat(x_cat,dim=1)
            x = self.aspp_drop(x_cat)
            x = self.aspp_project(x_cat)

        out = self.out_conv(x)

        return out
        
            
        

if __name__ == '__main__':
    conv = RefineNetv0(
        in_channels=3,
        out_channels=1,
        aspp_channels=32,
        inter_channels=[16,32],
        aspp=True,
        aspp_dilations=[4,8,12],
        activation='relu',
        norm_type='group',
        group_norm_channels=4,
        )
    print(conv)
    
    x = torch.zeros((1,3,100,100))
    with torch.no_grad():
        y = conv(x)
    print(y.shape)


