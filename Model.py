import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


# Down sample
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2, padding_mode='reflect')
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2, padding_mode='reflect')
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.10, inplace=True)
        
    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x1 = self.lrelu(self.conv2(x))
        x2 = self.lrelu(self.conv3(x1))
        return x,x1,x2

   
#  Deformable convolution
class deformableconv(nn.Module):
  def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1,padding_mode = 'reflect',bias=False):
    
    super(deformableconv, self).__init__()
      
    self.stride = stride if type(stride) == tuple else (stride, stride)
    self.padding = padding

    self.out_ch = 2 * kernel_size * kernel_size
    #  offset
    self.offset = nn.Conv2d(in_channels, out_channels = self.out_ch , kernel_size=kernel_size,
                                  stride=stride, padding=self.padding)
      
    self.out_chal = kernel_size * kernel_size
    self.modulator = nn.Conv2d(in_channels,out_channels = self.out_chal , kernel_size=kernel_size,
                                     stride=stride, padding=self.padding)
    
    self.regular_conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
                                  stride=stride,padding=self.padding,bias=bias)

  def forward(self, x):
    offset = self.offset(x)
    modulator = 2. * torch.sigmoid(self.modulator(x))
    
    x = torchvision.ops.deform_conv2d(input=x, offset=offset, weight=self.regular_conv.weight, 
                                      bias=self.regular_conv.bias, padding=self.padding,mask=modulator,stride=self.stride)
    return x
  

# Deformable Residual Block
class deformableResidualblock(nn.Module):
    def __init__(self,in_channels,out_channels, padding_mode):
        super(deformableResidualblock, self).__init__()
        
       
        #batch norm
        self.norm = nn.BatchNorm2d(num_features = in_channels)

        # activation fun
        self.lrelu = nn.LeakyReLU(negative_slope=0.10, inplace=True)

        # drop out
        self.drop_out = nn.Dropout()#default p =0.5
        
        self.conv_block = self.build_block(in_channels,out_channels, padding_mode)
        
    def build_block(self,in_channels,out_channels, padding_mode):
        block = []
        # when ever we are using the padding_type as reflect anything other then zero, padding is 0(default)
        # pad = 0
        # if padding_mode != 'reflect':
        #     pad =1
        pad =1
            
        # Baic resenet block
        # first deformableconvolution in deformableresnetblock
        # kernel_size = 3
        block +=[ deformableconv(in_channels,in_channels, padding = pad, padding_mode = padding_mode),
                 self.norm, self.lrelu]
        # if drop out is required add it to the list block about
        
        # second deformableconvolution in deformableresnetblock
        block +=[deformableconv(in_channels,in_channels, padding = pad, padding_mode = padding_mode),
                 self.norm]
        # if needed add lrelu to the last
        return nn.Sequential(*block)

    def forward(self, x):
        # adding skip connections
        x = x+self.conv_block(x)
        return x

 
# Up Sample
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Upsample layers
        self.upsample1 = nn.Conv2d(256, 128, kernel_size=3, padding=1, padding_mode ='reflect')
        self.upsample2 = nn.Conv2d(128, 64, kernel_size=3, padding=1,padding_mode ='reflect')
        self.upsample3 = nn.Conv2d(64, 3, kernel_size=3, padding=1,padding_mode ='reflect')

        self.lrelu = nn.LeakyReLU(negative_slope=0.10, inplace=True)
    
    def forward(self, x, x1, x2):
        # Upsample steps
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        tmp=self.upsample1(x)
        x = self.lrelu(torch.cat([x2,tmp],dim=1))
        # print(x.shape,tmp.shape)
        tmp = F.interpolate(tmp, scale_factor=2, mode='bilinear')
        tmp1=self.upsample2(tmp)
        x = self.lrelu(torch.cat([x1,tmp1],dim=1))
        # print(x.shape,tmp1.shape)
        x = F.interpolate(tmp1, scale_factor=1, mode='bilinear')
        x = self.lrelu(self.upsample3(x))
        return x
    

# calls all the unet parts
class Rnet(nn.Module):
    def __init__(self,in_channels=3, out_channels=256,nof_blocks = 9):
        super(Rnet,self).__init__()
        # downsample
        self.encoder = Encoder()
        # defarmable resnetblcok
        self.deformableresbl = deformableResidualblock(in_channels=256, out_channels=256, padding_mode='reflect')
        # upsample
        self.decoder = Decoder()
        
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.flat = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x,shadowfree):
        # downsample
        x, x1, x2 = self.encoder(x)
        # defarmable resnetblcok
        x3 = self.deformableresbl(x2)
        # Vectorization
        vector = self.flat(self.pool(x3))
        # upsample

        if shadowfree == True:
            output  = self.decoder(x3, x, x1)
        else:
            output = None
            
        return output, vector
    

class ShadowRemoval(nn.Module):
    def __init__(self):
        super(ShadowRemoval, self).__init__()
        # 
        self.Rnet1 = Rnet()
    def forward(self, x):
        # shared weight encoder
        # input shadow free image and output is shadow free and vector
        out_img, feature = self.Rnet1(x,True)
        # input is ground truth image and output is vector
        _, feature1 = self.Rnet1(x,False)

        return out_img, feature,feature1