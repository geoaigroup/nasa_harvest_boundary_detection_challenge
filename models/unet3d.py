import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch import Unet,UnetPlusPlus
#if sa_att:


class Conv3dNormRelu(nn.Module):

    def __init__(
        self,
        in_channels,  
        out_channels,
        kernel_size, 
        stride=1, 
        padding=0, 
        dilation=1, 
        groups=1,
        dropout=0.5,
        use_relu = True,
        group_norm = False
    ):
        super().__init__()
        self.conv = nn.Conv3d(
                in_channels = in_channels, 
                out_channels = 1, 
                kernel_size = kernel_size, 
                stride=stride, 
                padding=padding, 
                dilation=dilation, 
                groups=groups, 
                bias=False,
                padding_mode='zeros')
        
        self.norm = nn.BatchNorm2d(num_features=out_channels) if not group_norm \
            else nn.GroupNorm(num_channels=out_channels,num_groups=4)
        self.relu = nn.ReLU() if use_relu else nn.SiLU()
        
        if dropout > 0:
            self.drop = nn.Dropout2d(dropout)
        self.dropout =dropout
    
    def forward(self,x):
        x = self.conv(x) #B,T,C,H,W -- > B,1,C,H,W
        x = x[:,0,:,:,:] #B,1,C,H,W -- > B,C,H,W
        x = self.norm(x)
        x = self.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        return x

class PCConv3dNormRelu(nn.Module):

    def __init__(
        self,
        tsteps,
        in_channels,  
        out_channels,
        kernel_size, 
        padding=0, 
        dropout=0.5,
        use_relu = True,
        group_norm = False
    ):
        super().__init__()
        self.conv = nn.Conv3d(
                in_channels = in_channels, 
                out_channels = out_channels, 
                kernel_size = (tsteps,kernel_size,kernel_size),
                stride=1, 
                padding=(0,padding,padding),
                dilation=1,
                groups=in_channels, 
                bias=False,
                padding_mode='zeros')
        
        self.norm = nn.BatchNorm2d(num_features=out_channels) if not group_norm \
            else nn.GroupNorm(num_channels=out_channels,num_groups=4)
        self.relu = nn.ReLU() if use_relu else nn.SiLU()
        
        if dropout > 0:
            self.drop = nn.Dropout2d(dropout)
        self.dropout =dropout
    
    def forward(self,x):
        x = x.permute(0,2,1,3,4) #B,T,C,H,W --> B,C,T,H,W
        x = self.conv(x) #B,C,T,H,W -- > B,C,1,H,W
        x = x[:,:,0,:,:] #B,C,1,H,W -- > B,C,H,W
        x = self.norm(x)
        x = self.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        return x

class Unet3D(Unet):

    def __init__(
        self,
        tsteps = 6,
        kernel_3d = 1,
        dropout=0.0,
        decoder_channels = (256, 128, 64, 32, 16),
        ch_mul=1,
        conv3d_mode = 'conv3d',
        group_norm = False,
        use_relu = True,
        sa_att = False,
        **kwargs):
        super().__init__(decoder_channels = decoder_channels,**kwargs)

        self.encoder_channels = self.encoder.out_channels[1:]
        self.decoder_channels = decoder_channels
        self.kernel_3d = kernel_3d
        self.dropout = dropout
        self.tsteps = tsteps
        self.group_norm_mode = group_norm
        self.use_relu = use_relu
        self.sa_att = sa_att
        self.ch_mul = ch_mul
        #print(self.encoder_channels)
        self.out_3d_channels = [ch * ch_mul for ch in self.encoder.out_channels]
        #print(self.out_3d_channels)
        
        if ch_mul > 1:
            del self.decoder
            from segmentation_models_pytorch.unet.decoder import UnetDecoder

            self.decoder = UnetDecoder(
                encoder_channels=self.out_3d_channels,
                decoder_channels=decoder_channels,
                n_blocks=kwargs['encoder_depth'],
                use_batchnorm=kwargs['decoder_use_batchnorm'],
                center=True if kwargs['encoder_name'].startswith("vgg") else False,
                attention_type=kwargs['decoder_attention_type'],
            )

        self.out_3d_channels = self.out_3d_channels[1:]
        if self.sa_att:
            self._get_attention_blocks()

        if conv3d_mode == 'conv3d':
            self._get_3d_bottlenecks()
        elif conv3d_mode == 'conv3d_pc':
            self._get_pc_3d_bottlenecks()
        else:
            raise ValueError(f'Conv3d mode : {conv3d_mode} is not implemented!')


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

        
    def _get_3d_bottlenecks(self):
        botts_3d = []
        
        for i in range(len(self.encoder_channels)):
            ker = self.kernel_3d if i < 3 else 1
            botts_3d.append(
                Conv3dNormRelu(
                    in_channels=self.tsteps,
                    out_channels=self.out_3d_channels[i],
                    kernel_size=ker,
                    dropout=self.dropout,
                    group_norm=self.group_norm_mode,
                    use_relu=self.use_relu,
                    stride=1,padding=ker//2,dilation=1,groups=1)
            )
        self.botts_3d = nn.ModuleList(botts_3d)

    def _get_pc_3d_bottlenecks(self):
        botts_3d = []
        for i in range(len(self.encoder_channels)):
            ker = self.kernel_3d if i < 3 else 1
            botts_3d.append(
                PCConv3dNormRelu(
                    in_channels=self.encoder_channels[i],
                    out_channels=self.out_3d_channels[i],
                    tsteps=self.tsteps,
                    kernel_size=ker,
                    group_norm=self.group_norm_mode,
                    dropout=self.dropout,
                    use_relu=self.use_relu,
                    padding=ker//2)
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
            #print(feat.shape)
            feat = feat.view(B,T,C,H,W)
            #print(feat.shape)
            feat = bot_conv(feat)
            #print(feat.shape)
            if self.sa_att:
                feat = self.att_blocks[i](feat)
                #print(feat.shape)
            features[i+1] = feat
            #print('----------------')
        #Unet Decoder
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks 

        


if __name__ == '__main__':
    #"""
    model = Unet3D(
        tsteps = 6,
        kernel_3d = 3,
        conv3d_mode = 'conv3d_pc',
        encoder_name = "tu-tf_efficientnet_b1",
        encoder_depth = 5,
        sa_att=True,
        ch_mul=4,
        encoder_weights = "imagenet",
        decoder_use_batchnorm = True,
        decoder_channels = (256, 128, 64, 32, 16),
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
    #print(model.att_blocks)
    print(y.shape)