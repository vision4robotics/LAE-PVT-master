
import torch.nn as nn
from Deform import DeformConv2d as deform2
import torch.nn.functional as F
import torch as t


class grader(nn.Module):
    
    def __init__(self):
        super(grader, self).__init__()
        channels=384

#        self.conv_shape = nn.Sequential(
#                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
#                nn.BatchNorm2d(channels),
#                nn.ReLU(inplace=True),
#                nn.Conv2d(channels, 4, kernel_size=1),
#                )
        self.conv_shape = nn.Sequential(
                nn.Conv2d(channels, channels,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, 4,  kernel_size=3, stride=1,padding=1),
                )
       

        for modules in [self.conv_shape]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    t.nn.init.normal_(l.weight, std=0.01)
                    t.nn.init.constant_(l.bias, 0)
#        nn.init.normal_(self.conv_kernel[0].weight, mean=0,std=0.1)
#        nn.init.normal_(self.conv_search[0].weight, mean=0,std=0.1)
#        self.sig=nn.Sigmoid()
#        self.relu = nn.ReLU(inplace=True)
#        self.tanh = nn.Tanh() 

    def xcorr_depthwise(self,x, kernel):
        """depthwise cross correlation
        """
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch*channel, x.size(2), x.size(3))
        kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch*channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out
    
    def forward(self,x,z):
        
#        z=self.conv_kernel(z)
#        x=self.conv_search(x)
        res=self.xcorr_depthwise(x,z)
        shape_pred=self.conv_shape(res)
#        shape_pred2=self.new(shape_pred)
        
#        f=z.view(96,29,29).cpu().detach().numpy()
#        d=zz.view(96,27,27).cpu().detach().numpy()
#        g=x.view(96,69,69).cpu().detach().numpy()
#        e=xx.view(96,67,67).cpu().detach().numpy()
#        
#        c=res.view(96,41,41).cpu().detach().numpy()
#        a=shape_pred.view(96,287,287).cpu().detach().numpy()
#        b=shape_pred2.view(2,287,287).cpu().detach().numpy()
#        x = self.feature_adaption(x,shape_pred)
        
        return shape_pred,res
#class grader(nn.Module):
#    
#    def __init__(self,channels):
#        super(grader, self).__init__()
##
#        self.conv_kernel = nn.Sequential(
#                nn.Conv2d(channels, channels, kernel_size=3, bias=False),
#                nn.BatchNorm2d(channels),
#                nn.ReLU(inplace=True)
#                )
#        
#        self.conv_search = nn.Sequential(
#                nn.Conv2d(channels, channels, kernel_size=3, bias=False),
#                nn.BatchNorm2d(channels),
#                nn.ReLU(inplace=True)
#                )
#        
#        self.conv_loc = nn.Conv2d(channels,1, 1)
#
#        self.conv_shape = nn.Conv2d(channels,2, 1)
#
#        
##        self.new=nn.Sequential(
##                nn.ConvTranspose2d(channels,channels,7,7),
##                nn.BatchNorm2d(channels),
##                nn.Tanh(),
##                nn.Conv2d(channels,2,1),
##                )
#        
#
##        self.feature_adaption = FeatureAdaption(channels)
#
#        nn.init.normal_(self.conv_loc.weight, std=0.01)
#        nn.init.normal_(self.conv_shape.weight, std=0.01)
#        self.sig=nn.Sigmoid()
#        self.relu = nn.ReLU(inplace=True)
#        self.tanh = nn.Tanh() 
#        
##        for i in self.conv_loc.parameters():
##            i.requires_grad=False
##        for i in self.conv_shape.parameters():
##            i.requires_grad=False
##        for i in self.lin1.parameters():
##            i.requires_grad=False
##        for i in self.lin2.parameters():
##            i.requires_grad=False
##        for i in self.feature_adaption.parameters():
##            i.requires_grad=False
#
#    def xcorr_depthwise(self,x, kernel):
#        """depthwise cross correlation
#        """
#        batch = kernel.size(0)
#        channel = kernel.size(1)
#        x = x.view(1, batch*channel, x.size(2), x.size(3))
#        kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
#        out = F.conv2d(x, kernel, groups=batch*channel)
#        out = out.view(batch, channel, out.size(2), out.size(3))
#        return out
#    
#    def forward(self,x,z):
#        
#        z=self.conv_kernel(z)
#        x=self.conv_search(x)
#        res=self.xcorr_depthwise(x,z)
#        
#        loc_pre=self.conv_loc(res)
#        shape_pred=self.tanh(self.conv_shape(res))
##        shape_pred2=self.new(shape_pred)
#        
##        f=z.view(96,29,29).cpu().detach().numpy()
##        d=zz.view(96,27,27).cpu().detach().numpy()
##        g=x.view(96,69,69).cpu().detach().numpy()
##        e=xx.view(96,67,67).cpu().detach().numpy()
##        
##        c=res.view(96,41,41).cpu().detach().numpy()
##        a=shape_pred.view(96,287,287).cpu().detach().numpy()
##        b=shape_pred2.view(2,287,287).cpu().detach().numpy()
##        x = self.feature_adaption(x,shape_pred)
#        
#        return loc_pre,shape_pred
class FeatureAdaption(nn.Module):

    def __init__(self):
        super(FeatureAdaption, self).__init__()
        channels=18
        in_channels=256
        
        self.conv_offset = nn.Sequential(
                nn.ConvTranspose2d(4, channels, kernel_size=4),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channels, channels, kernel_size=3),
                nn.ReLU(inplace=True),
                )

#        nn.init.xavier_uniform_(self.conv_offset[2].weight, gain=1)

        nn.init.constant_(self.conv_offset[2].weight, 0) 
        
        self.conv_adaption = deform2(in_channels,in_channels,3,0,modulation=False)

        
    def forward(self, x, shape):

        offset = self.conv_offset(shape.detach())
    
        x = self.conv_adaption(x, offset)
  
        return x    
    
class new(nn.Module):

    def __init__(self):
        super(new, self).__init__()
    
#        self.convloc = nn.Sequential(
#                nn.Conv2d(256, 256, kernel_size=3,padding=1,bias=False),
#                nn.BatchNorm2d(256),
#                nn.ReLU(inplace=True),
#                nn.Conv2d(256, 256, kernel_size=3,padding=1,bias=False),
#                nn.BatchNorm2d(256),
#                nn.ReLU(inplace=True),
#                nn.Conv2d(256, 4, kernel_size=1)
#                )
#        self.convcls = nn.Sequential(
#                nn.Conv2d(256, 256, kernel_size=3,padding=1,bias=False),
#                nn.BatchNorm2d(256),
#                nn.ReLU(inplace=True),
#                nn.Conv2d(256, 256, kernel_size=3,padding=1,bias=False),
#                nn.BatchNorm2d(256),
#                nn.ReLU(inplace=True),
#                nn.Conv2d(256, 2, kernel_size=1)
#                )
#        
#        self.conv_offset = nn.Sequential(
#                nn.ConvTranspose2d(4, 256, kernel_size=4),
#                nn.ReLU(inplace=True),
#                nn.ConvTranspose2d(256, 256, kernel_size=3),
#                nn.ReLU(inplace=True),
#                )
        #new 2
#        self.convloc = nn.Sequential(
#                nn.Conv2d(256, 256, kernel_size=3,bias=False),
#                nn.BatchNorm2d(256),
#                nn.ReLU(inplace=True),
#                nn.Conv2d(256, 256, kernel_size=3,bias=False),
#                nn.BatchNorm2d(256),
#                nn.ReLU(inplace=True),
#                nn.Conv2d(256, 4, kernel_size=1)
#                )
#        self.convcls = nn.Sequential(
#                nn.Conv2d(256, 256, kernel_size=3,bias=False),
#                nn.BatchNorm2d(256),
#                nn.ReLU(inplace=True),
#                nn.Conv2d(256, 256, kernel_size=3,bias=False),
#                nn.BatchNorm2d(256),
#                nn.ReLU(inplace=True),
#                nn.Conv2d(256, 2, kernel_size=1)
#                )
#        
#        self.conv_offset = nn.Sequential(
#                nn.Conv2d(384, 256, kernel_size=1,bias=False),
#                nn.BatchNorm2d(256),
#                nn.ReLU(inplace=True),
#                nn.Conv2d(256, 256, kernel_size=1)
#                )
#        
#        self.changex = nn.Sequential(
#                nn.Conv2d(256, 256, kernel_size=3,bias=False),
#                nn.BatchNorm2d(256),
#                nn.ReLU(inplace=True),
#                )
#        self.changez = nn.Sequential(
#                nn.Conv2d(256, 256, kernel_size=3,bias=False),
#                nn.BatchNorm2d(256),
#                nn.ReLU(inplace=True),
#                )
#        new3
#        self.convloc = nn.Sequential(
#                nn.Conv2d(256, 256, kernel_size=3,bias=False),
#                nn.GroupNorm(32,256),
#                nn.ReLU(inplace=True),
#                nn.Conv2d(256, 256, kernel_size=3,bias=False),
#                nn.GroupNorm(32,256),
#                nn.ReLU(inplace=True),
#                nn.Conv2d(256, 4, kernel_size=1)
#                )
#        self.convcls = nn.Sequential(
#                nn.Conv2d(256, 256, kernel_size=3,bias=False),
#                nn.GroupNorm(32,256),
#                nn.ReLU(inplace=True),
#                nn.Conv2d(256, 256, kernel_size=3,bias=False),
#                nn.GroupNorm(32,256),
#                nn.ReLU(inplace=True),
#                )
#
#        self.conv_offset = nn.Sequential(
#                nn.Conv2d(384, 256, kernel_size=1,bias=False),
#                nn.GroupNorm(32,256),
#                nn.ReLU(inplace=True),
#                nn.Conv2d(256, 256, kernel_size=1)
#                )
#        
#        self.relu=nn.ReLU(inplace=True)
#        self.add=nn.Conv2d(256*2,256,1)
#        self.resize=nn.Conv2d(256,256,kernel_size=1)
#        self.cls1=nn.Conv2d(256, 2, kernel_size=1)
#        self.cls2=nn.Conv2d(256, 2, kernel_size=1)
#        self.cls3=nn.Conv2d(256, 1, kernel_size=1)
        
#        #new333
#        self.convloc = nn.Sequential(
#                nn.Conv2d(256, 256, kernel_size=3),
#                nn.GroupNorm(32,256),
#                nn.ReLU(inplace=True),
#                nn.Conv2d(256, 256, kernel_size=3),
#                nn.GroupNorm(32,256),
#                nn.ReLU(inplace=True),
#                nn.Conv2d(256, 4, kernel_size=1)
#                )
#        self.convcls = nn.Sequential(
#                nn.Conv2d(256, 256, kernel_size=3),
#                nn.GroupNorm(32,256),
#                nn.ReLU(inplace=True),
#                nn.Conv2d(256, 256, kernel_size=3),
#                nn.GroupNorm(32,256),
#                nn.ReLU(inplace=True),
#                )
#
#        self.conv_offset = nn.Sequential(
#                nn.Conv2d(384, 256, kernel_size=1),
#                nn.GroupNorm(32,256),
#                nn.ReLU(inplace=True),
#                nn.Conv2d(256, 256, kernel_size=1)
#                )
#        
#        self.relu=nn.ReLU(inplace=True)
#        self.add=nn.ConvTranspose2d(256 * 2, 256, 1, 1)
#        
#        self.resize=nn.Conv2d(256,256,kernel_size=1)
#        self.cls1=nn.Conv2d(256, 2,  kernel_size=3, stride=1,padding=1)
#        self.cls2=nn.Conv2d(256, 2,  kernel_size=3, stride=1,padding=1)
#        self.cls3=nn.Conv2d(256, 1,  kernel_size=3, stride=1,padding=1)
#        
#        for modules in [self.convloc, self.convcls,
#                        self.resize, self.cls1,
#                        self.cls2,self.cls3]:
#            for l in modules.modules():
#                if isinstance(l, nn.Conv2d):
#                    t.nn.init.normal_(l.weight, std=0.01)
#                    t.nn.init.constant_(l.bias, 0)
        
        #new3333
        self.convloc = nn.Sequential(
                nn.Conv2d(256, 256,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(32,256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(32,256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 4,  kernel_size=3, stride=1,padding=1),
                )
        self.convcls = nn.Sequential(
                nn.Conv2d(256, 256,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(32,256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(32,256),
                nn.ReLU(inplace=True),
                )

        self.conv_offset = nn.Sequential(
                nn.Conv2d(384, 256,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(32,256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256,  kernel_size=3, stride=1,padding=1),
                )
        
        self.relu=nn.ReLU(inplace=True)
        self.add=nn.ConvTranspose2d(256 * 2, 256, 1, 1)
        
        self.resize=nn.Conv2d(256,256,kernel_size=1)
        self.cls1=nn.Conv2d(256, 2,  kernel_size=3, stride=1,padding=1)
        self.cls2=nn.Conv2d(256, 2,  kernel_size=3, stride=1,padding=1)
        self.cls3=nn.Conv2d(256, 1,  kernel_size=3, stride=1,padding=1)
        
        for modules in [self.convloc, self.convcls,
                        self.resize, self.cls1,
                        self.cls2,self.cls3]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    t.nn.init.normal_(l.weight, std=0.01)
                    t.nn.init.constant_(l.bias, 0)
#
#        nn.init.xavier_normal_(self.convloc[0].weight,0.5)
#        nn.init.xavier_normal_(self.convloc[3].weight,0.5)
#        nn.init.xavier_normal_(self.convloc[6].weight,0.5)
#
#        nn.init.xavier_normal_(self.convcls[0].weight,0.5)
#        nn.init.xavier_normal_(self.convcls[3].weight,0.5)
##        nn.init.xavier_normal_(self.convcls[6].weight,0.5)
#        nn.init.xavier_normal_(self.add.weight,0.5)
##        nn.init.xavier_normal_(self.add.bias,0.5)
#        
#        nn.init.xavier_normal_(self.conv_offset[0].weight,1)
#        nn.init.xavier_normal_(self.conv_offset[3].weight,1)
#        
#        nn.init.normal_(self.resize.weight,0,0.01)
##        nn.init.xavier_normal_(self.resize[1].weight,1)
#        nn.init.normal_(self.resize.bias,0,0.01)
#        
##        nn.init.xavier_normal_(self.changex[0].weight,0.5)
##        nn.init.xavier_normal_(self.changez[0].weight,0.5)
#
#
##        nn.init.xavier_normal_(self.cls1.weight,1)
##        nn.init.xavier_normal_(self.cls2.weight,0.1)
##        nn.init.xavier_normal_(self.cls3.weight,0.1)
#        nn.init.normal_(self.cls1.weight,0,0.01)
#        nn.init.xavier_normal_(self.cls2.weight,0.1)
#        nn.init.normal_(self.cls3.weight,0,0.01)
#        

    def xcorr_depthwise(self,x, kernel):
        """depthwise cross correlation
        """
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch*channel, x.size(2), x.size(3))
        kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch*channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out

    def forward(self, x,z,xff):
#       new1
#        x=self.add(self.relu(t.cat((x,self.conv_offset(xff)),1)))  #融合shape和x  是否融合z？
#        
#        res=self.xcorr_depthwise(x,z)
#
#        cls=self.convcls(res)
#
#        loc=self.convloc(res)
        
        #new2
#        x=self.changex(x)
#        z=self.changez(z)   #new222
        
#        res=self.xcorr_depthwise(x,z)
#        res=self.resize(res)  #new22
#        res=self.add(self.relu(t.cat((res,self.conv_offset(xff)),1)))  #融合shape和x  是否融合z？
#        
#        cls=self.convcls(res)
##        cls1=self.cls1(cls)
##        cls2=self.cls2(cls)
##        cls3=self.cls3(cls)
#
#        loc=self.convloc(res)
#        return cls,loc

        #new3
        res=self.xcorr_depthwise(x,z)
        res=self.resize(res)
        res=self.add(self.relu(t.cat((res,self.conv_offset(xff)),1)))  #融合shape和x  是否融合z？
        
        cls=self.convcls(res)
        cls1=self.cls1(cls)
        cls2=self.cls2(cls)
        cls3=self.cls3(cls)

        loc=self.convloc(res)

        return cls1,cls2,cls3,loc

        
#class FeatureAdaption(nn.Module):
#
#    def __init__(self,in_channels):
#        super(FeatureAdaption, self).__init__()
#        
#        self.conv_offset = nn.ConvTranspose2d(2, 2, 6, bias=False)
#        
##        nn.init.xavier_uniform_(self.conv_offset.weight, gain=1)
#
#        
#        self.conv_adaption = deform2(in_channels,in_channels,1,0,modulation=False)
#        self.relu = nn.ReLU(inplace=True)
#
#
#    def forward(self, x, shape):
#        offset = self.conv_offset(shape.detach())
##        offset = self.conv_offset(shape)
#        x = self.conv_adaption(x, offset)
##        x=self.relu(x)
#        return x
# 
#import torch as t      
#
#
## 
a=grader().cuda()
#
b=new().cuda()
#b=t.rand(1,256,41,41).cuda()
#c=t.rand(1,2,41,41).cuda()
#a(b,c).size()
#
#a(b,[b,c])
#x=t.rand(1,96,69,69).cuda()
#z=t.rand(1,96,29,29).cuda()
#b=a(x,z)    
#import torch as t    
#a=grader(256)
#shape = t.Tensor([[[[ 2.6350,  1.5847, -0.0599,  0.0155,  0.8626,  3.0632],
#          [ 3.4695,  1.8928,  0.7853,  1.4270,  1.1427,  3.3063],
#          [ 4.2469,  3.8537,  2.0365,  2.8323,  2.9546,  3.4344],
#          [ 3.7309,  3.8352,  1.9084,  2.9748,  2.6733,  3.6679],
#          [ 2.4514,  2.7778,  1.9359,  3.1869,  2.3975,  3.4775],
#          [ 2.3193,  3.1924,  2.5530,  3.7451,  2.8680,  2.8440]],
#
#         [[-1.8433, -0.4709, -0.5363,  1.0430,  1.1971,  0.8239],
#          [ 0.1081, -0.1575, -0.5913, -1.9359, -3.3214, -2.7639],
#          [-0.0695, -0.0186, -0.1189, -1.8462, -3.9267, -2.8503],
#          [-0.3966, -1.2125, -0.4971, -1.6049, -2.5719, -1.5402],
#          [-0.1171, -1.3174, -1.6415, -2.3986, -2.2195, -0.7554],
#          [-0.4449, -1.2681, -2.2561, -2.4181, -2.1331, -1.3224]]]])

#b=t.randn(1,256,25,25)
#c=a(b)
#print(c[2])


#class grader(nn.Module):
#    
#    def __init__(self,channels):
#        super(grader, self).__init__()
#
#        self.conv_kernel = nn.Sequential(
#                nn.Conv2d(channels, channels, kernel_size=3, bias=False),
#                nn.BatchNorm2d(channels),
#                nn.ReLU(inplace=True)
#                )
#        
#        self.conv_search = nn.Sequential(
#                nn.Conv2d(channels, channels, kernel_size=3, bias=False),
#                nn.BatchNorm2d(channels),
#                nn.ReLU(inplace=True)
#                )
#
#        self.conv_loc = nn.Sequential(
#                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
#                nn.BatchNorm2d(channels),
#                nn.Tanh(),
#                nn.Conv2d(channels,1, 1)
#                )
#        self.conv_shape = nn.Sequential(
#                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
#                nn.BatchNorm2d(channels),
#                nn.Sigmoid(),
#                nn.Conv2d(channels,2, 1)
#                )
#        self.new=nn.Sequential(
#                nn.ConvTranspose2d(2,channels,7,7),
#                nn.BatchNorm2d(channels),
#                nn.Sigmoid(),
#                nn.Conv2d(channels,2,1),
#
#                
#                )
#        
#        
##        self.feature_adaption = FeatureAdaption(channels)
#
#        self.sig=nn.Sigmoid()
#        self.relu = nn.ReLU(inplace=True)
##        nn.init.xavier_uniform_(self.lin2.weight, gain=1)
##        nn.init.constant_(self.lin2.bias,0)
#        
##        for i in self.conv_loc.parameters():
##            i.requires_grad=False
##        for i in self.conv_shape.parameters():
##            i.requires_grad=False
##        for i in self.lin1.parameters():
##            i.requires_grad=False
##        for i in self.lin2.parameters():
##            i.requires_grad=False
##        for i in self.feature_adaption.parameters():
##            i.requires_grad=False
#
#    def xcorr_depthwise(self,x, kernel):
#        """depthwise cross correlation
#        """
#        batch = kernel.size(0)
#        channel = kernel.size(1)
#        x = x.view(1, batch*channel, x.size(2), x.size(3))
#        kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
#        out = F.conv2d(x, kernel, groups=batch*channel)
#        out = out.view(batch, channel, out.size(2), out.size(3))
#        return out
#    
#    def forward(self,x,z):
#        
##        zz=self.conv_kernel(z)
##        xx=self.conv_search(x)
#        res=self.xcorr_depthwise(x,z)
#        
#        loc_pre=self.conv_loc(res)
##        loc_pre=(loc_pre-loc_pre.min())/(loc_pre.max()-loc_pre.min()+1e-6)
#        shape_pred=self.conv_shape(res)
#
#        shape_pred2=self.new(shape_pred)
#        
##        f=z.view(96,29,29).cpu().detach().numpy()
##        d=zz.view(96,27,27).cpu().detach().numpy()
##        g=x.view(96,69,69).cpu().detach().numpy()
##        e=xx.view(96,67,67).cpu().detach().numpy()
##        
##        c=res.view(96,41,41).cpu().detach().numpy()
#        a=shape_pred.view(2,41,41).cpu().detach().numpy()
#        b=shape_pred2.view(2,287,287).cpu().detach().numpy()
##        x = self.feature_adaption(x,shape_pred)
#        
#        return loc_pre,shape_pred2

#class grader(nn.Module):
#    
#    def __init__(self,channels):
#        super(grader, self).__init__()
#
#        self.conv_kernel = nn.Sequential(
#                nn.Conv2d(channels, channels, kernel_size=3,padding=1, bias=False),
#                nn.BatchNorm2d(channels),
#                nn.ReLU(inplace=True)
#                )
#        
#        self.conv_search = nn.Sequential(
#                nn.Conv2d(channels, channels, kernel_size=3,padding=1, bias=False),
#                nn.BatchNorm2d(channels),
#                nn.ReLU(inplace=True)
#                )
#
#        self.conv_loc = nn.Sequential(
#                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
#                nn.BatchNorm2d(channels),
#                nn.Tanh(),
#                nn.Conv2d(channels,1, 1)
#                )
#        self.conv_shape = nn.Sequential(
#                nn.Conv2d(channels, channels, kernel_size=3, bias=False),
#                nn.BatchNorm2d(channels),
#                nn.ReLU(inplace=True),
#                nn.Conv2d(channels,2,1),
#
#                )
#        
#        self.new=nn.Sequential(
#                nn.ConvTranspose2d(channels,channels,7,7),
#                nn.BatchNorm2d(channels),
#                nn.ReLU(inplace=True),
#                nn.ConvTranspose2d(channels,channels,3),
#                nn.BatchNorm2d(channels),
#                nn.ReLU(inplace=True)
#    
#                )
#        
#        
##        self.feature_adaption = FeatureAdaption(channels)
#
#        self.sig=nn.Sigmoid()
#        self.relu = nn.ReLU(inplace=True)
#        
##        nn.init.xavier_uniform_(self.conv_kernel[0].weight, gain=1)
##        nn.init.xavier_uniform_(self.conv_search[0].weight, gain=1)
##        nn.init.xavier_uniform_(self.conv_loc[0].weight, gain=1)
##        nn.init.xavier_uniform_(self.conv_loc[3].weight, gain=1)  
##        nn.init.constant_(self.conv_loc[3].bias,0.5)
##
##        nn.init.xavier_uniform_(self.conv_shape[3].weight, gain=1)
##        nn.init.constant_(self.conv_shape[3].bias,0.5)
#        
##        for i in self.conv_loc.parameters():
##            i.requires_grad=False
##        for i in self.conv_shape.parameters():
##            i.requires_grad=False
##        for i in self.lin1.parameters():
##            i.requires_grad=False
##        for i in self.lin2.parameters():
##            i.requires_grad=False
##        for i in self.feature_adaption.parameters():
##            i.requires_grad=False
#
#    def xcorr_depthwise(self,x, kernel):
#        """depthwise cross correlation
#        """
#        batch = kernel.size(0)
#        channel = kernel.size(1)
#        x = x.view(1, batch*channel, x.size(2), x.size(3))
#        kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
#        out = F.conv2d(x, kernel, groups=batch*channel)
#        out = out.view(batch, channel, out.size(2), out.size(3))
#        return out
#    
#    def forward(self,x,z):
#        
#        zz=self.conv_kernel(z)+z
#        xx=self.conv_search(x)+x
#        res=self.xcorr_depthwise(xx,zz)
#        
#        loc_pre=self.conv_loc(res)
#        
#        shape_pred=self.new(res)
#
#        shape_pred2=self.conv_shape(shape_pred)
#        
##        f=z.view(96,29,29).cpu().detach().numpy()
##        d=zz.view(96,27,27).cpu().detach().numpy()
##        g=x.view(96,69,69).cpu().detach().numpy()
##        e=xx.view(96,67,67).cpu().detach().numpy()
###        
##        c=res.view(96,41,41).cpu().detach().numpy()
##        a=shape_pred.view(96,289,289).cpu().detach().numpy()
##        b=shape_pred2.view(2,287,287).cpu().detach().numpy()
##        x = self.feature_adaption(x,shape_pred)
#        
#        return loc_pre,shape_pred2
#    