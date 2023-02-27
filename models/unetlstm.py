import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch import Unet,UnetPlusPlus
from .conv_lstm import ConvLSTM

class Conv2dNormRelu(nn.Sequential):

    def __init__(
        self,
        in_channels,  
        out_channels,
        kernel_size, 
        stride=1, 
        padding=0, 
        dilation=1, 
        groups=1,
        use_relu = True,
        group_norm = False,
        group_norm_channels=4
    ):
        
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
        
        norm = nn.BatchNorm2d(num_features=out_channels) if not group_norm \
            else nn.GroupNorm(num_channels=out_channels,num_groups=group_norm_channels)
        relu = nn.ReLU() if use_relu else nn.SiLU()
        
        super().__init__(conv,norm,relu)

class ConvLSTMSeg(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        out_dim,
        kernel_size=(3,3),
        num_layers=1,
        group_norm = False,
        group_norm_channels=4
        ):
        super().__init__()
        self.lstm_encoder = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            return_all_layers=False,
            bias=False,
            batch_first=True,
            num_layers=num_layers
        )
        self.out_conv = Conv2dNormRelu(
            in_channels=hidden_dim,
            out_channels=out_dim,
            kernel_size=3,
            padding=1,
            use_relu=True,
            group_norm=group_norm,
            group_norm_channels=group_norm_channels
        )
    
    def forward(self,x):
        _,states = self.lstm_encoder(x)
        out = states[0][1]
        out = self.out_conv(out)
        return out



class UnetLstm(Unet):

    def __init__(
        self,
        tsteps = 6,
        kernel_3d = 1,
        dropout=0.0,
        decoder_channels = (256, 128, 64, 32, 16),
        replace_all_norms=False,
        group_norm = False,
        group_norm_channels = 4,
        sa_att = False,
        **kwargs):
        super().__init__(decoder_channels = decoder_channels,**kwargs)
        
        #print(self.encoder.model.bn1)
        if group_norm and replace_all_norms:
            self.batch_norm_to_group_norm(self,group_norm_channels)

        self.encoder_channels = self.encoder.out_channels[1:]
        self.decoder_channels = decoder_channels
        self.kernel_3d = kernel_3d
        self.dropout = dropout
        self.tsteps = tsteps
        self.group_norm_mode = group_norm
        self.sa_att = sa_att
        self.out_3d_channels = [ch for ch in self.encoder.out_channels]
        self.gnc = group_norm_channels


        self.out_3d_channels = self.out_3d_channels[1:]
        self._get_lstm_bottlenecks()
        if self.sa_att:
            self._get_attention_blocks()


    def batch_norm_to_group_norm(self,layer,gnc=4):
        """Iterates over a whole model (or layer of a model) and replaces every batch norm 2D with a group norm

        Args:
            layer: model or one layer of a model like resnet34.layer1 or Sequential(), ...
        """
        for name, module in layer.named_modules():
            if name:
                
                try:
                    if isinstance(module, torch.nn.BatchNorm2d):
                        bn = getattr(layer, name)
                        num_channels = bn.num_features
                        #print(f'Swapping {name} with group_norm')
                        # first level of current layer or model contains a batch norm --> replacing.
                        layer._modules[name] = torch.nn.GroupNorm(gnc, num_channels)

                except AttributeError:
                    # go deeper: set name to layer1, getattr will return layer1 --> call this func again
                    name = name.split('.')[0]
                    sub_layer = getattr(layer, name)
                    sub_layer = self.batch_norm_to_group_norm(sub_layer,gnc)
                    layer.__setattr__(name=name, value=sub_layer)
        return layer

    def _get_attention_blocks(self):
        from .squeeze_att import SqueezeAttentionBlock
        att_blocks = []
        
        for i in range(len(self.encoder_channels)):
            att_blocks.append(
                SqueezeAttentionBlock(
                    ch_in=self.out_3d_channels[i],
                    ch_out=self.out_3d_channels[i]
                )
            )
        self.att_blocks = nn.ModuleList(att_blocks)

        
    def _get_lstm_bottlenecks(self):
        botts_3d = []
        
        for i in range(len(self.encoder_channels)):
            ker = self.kernel_3d if i < 3 else 1
            botts_3d.append(
                ConvLSTMSeg(
                    input_dim=self.encoder_channels[i],
                    hidden_dim=self.encoder_channels[i],
                    out_dim=self.encoder_channels[i],
                    kernel_size=(3,3),
                    num_layers=1,
                    group_norm=self.group_norm_mode,
                    group_norm_channels=self.gnc
                )

            )
        self.botts_3d = nn.ModuleList(botts_3d)


    def forward(self,x):
        B,T,C,H,W = x.shape
        x = x.view(B*T,C,H,W)

        # Separately pass each image to backbone
        features = self.encoder(x)
        
        #3D conv to get temporal features
        for i,bot_conv in enumerate(self.botts_3d):
            feat = features[i+1]
            _,C,H,W = feat.shape 
            feat = feat.view(B,T,C,H,W)

            feat = bot_conv(feat)

            
            if self.sa_att:
                feat = self.att_blocks[i](feat)

            features[i+1] = feat

        decoder_output = self.decoder(*features)



        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks 

        


if __name__ == '__main__':
    #"""
    model = UnetLstm(
        tsteps = 6,
        encoder_name = "tu-regnety_004",
        encoder_weights =True,
        kernel_3d = 3,
        dropout=0.0,
        decoder_channels = (256, 128, 64, 32, 16),
        replace_all_norms=False,
        group_norm = False,
        group_norm_channels = 4,
        sa_att = True,
        decoder_use_batchnorm = True,
        decoder_attention_type= None,
        in_channels = 3,
        classes = 1,
        activation = None,
        aux_params = None)
    #"""


    #print(model)

    #mod = PCConv3dNormRelu(tsteps=6,in_channels=256,out_channels=256,kernel_size=1,padding=0,dropout=0.5)
    x = torch.ones((4,6,3,320,320)) #B,T,C,H,W
    print(x.shape)
    print('-------')

    y = model(x)
    print(model.encoder_channels)
    #print(model.att_blocks)
    print(y.shape)