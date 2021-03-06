# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from pysot.core.config import cfg
from loss import select_cross_entropy_loss, weight_l1_loss


from pysot.datasets.augmentation import Augmentation
from newalexnet import AlexNet
from pysot.utils.utile import new
import numpy as np
from loss import myloss,newloss,weightloss,l1loss,smooth_l1_loss,clsloss,locloss,BoundedIoULoss,IoULoss
from pysot.datasets.anchortarget import AnchorTarget


class ModelB(nn.Module):
    def __init__(self,path,name):
        super(ModelB, self).__init__()

        self.backbone = AlexNet().cuda()
        self.new=new().cuda()
 
        self.loss1=BoundedIoULoss()
        self.loss2=smooth_l1_loss

        self.shapeloss=IoULoss()
        self.cls2loss=cls2loss
        self.tanh=nn.Tanh() 
        
        self.template_aug = Augmentation(
                cfg.DATASET.TEMPLATE.SHIFT,
                cfg.DATASET.TEMPLATE.SCALE,
                cfg.DATASET.TEMPLATE.BLUR,
                cfg.DATASET.TEMPLATE.FLIP,
                cfg.DATASET.TEMPLATE.COLOR
            )
        self.search_aug = Augmentation(
                cfg.DATASET.SEARCH.SHIFT,
                cfg.DATASET.SEARCH.SCALE,
                cfg.DATASET.SEARCH.BLUR,
                cfg.DATASET.SEARCH.FLIP,
                cfg.DATASET.SEARCH.COLOR
            )
     
    def con(self, x):
        return  x*143      
        
    def template(self, z):

        zf,zf1 = self.backbone(z)

        self.zf=zf
        
        self.zf1=zf1

    
    def track(self, x):
        
        xf,xf1 = self.backbone(x)  
        xff,ress=self.grader(xf1,self.zf1)    

        self.ranchors=xff
              

        
        cls1,cls2,cls3,loc =self.new(xf,self.zf,ress)
           
        return {
                'cls1': cls1,
                'cls2': cls2,
                'cls3': cls3,
                'loc': loc
               }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)

        return cls
 
 




   
 
    def generatedata(self,anchors,bbox,se):
        

        fin=AnchorTarget()
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()

        label_cls,label_loc,label_loc_weight,over,labelcls2=fin.get(anchors,bbox,se,neg)
       
        return{
                'label_cls':label_cls,
                'label_loc':label_loc,
                'label_loc_weight':label_loc_weight,
                'over':over,
                'label_cls2':labelcls2
                }


        
    def getcenter(self,mapp,gt):

        def con(self, x):
           return  x*143
       
        size=mapp.size()[3]
        #location 
        x=np.tile((8*(np.linspace(0,size-1,size))+63)-287//2,size).reshape(-1)
        y=np.tile((8*(np.linspace(0,size-1,size))+63).reshape(-1,1)-287//2,size).reshape(-1)
        shap=self.con(mapp[0]).cpu().detach().numpy()
        xx=np.int16(np.tile(np.linspace(0,size-1,size),size).reshape(-1))
        yy=np.int16(np.tile(np.linspace(0,size-1,size).reshape(-1,1),size).reshape(-1))

        w=shap[0,yy,xx]+shap[1,yy,xx]
        h=shap[2,yy,xx]+shap[3,yy,xx]
        x=x-shap[0,yy,xx]+w/2
        y=y-shap[2,yy,xx]+h/2


        
        anchor=np.zeros((size**2,4))
        

        
        anchor[:,0]=x+287//2
        anchor[:,1]=y+287//2
        anchor[:,2]=w
        anchor[:,3]=h


        return anchor
    








    def forward(self,data):
    # def forward(self,gt,idx, img,tracker,bbox,se):
        """ only used in training
        """
                
        template = data['template'].cuda().unsqueeze(0)
        search =data['search'].cuda().unsqueeze(0)
        bbox=data['bbox']

        
        zf,zf1 = self.backbone(template)
        xf,xf1 = self.backbone(search)
        xff,ress=self.grader(xf1,zf1)
        
        anchors=self.getcenter(xff,gt)   #gt != bbox
        size=xff.size()[3]  
        data2=self.generatedata(anchors,bbox,size)
        
        label_cls = t.Tensor(data2['label_cls']).cuda().unsqueeze(0)
        label_loc = t.Tensor(data2['label_loc']).cuda().unsqueeze(0)
        label_loc_weight = t.Tensor(data2['label_loc_weight']).cuda().unsqueeze(0)
        labelcls2=data2['label_cls2']
        
        
        
      #new3
        cls1,cls2,cls3,loc=self.new(xf,zf,ress)
        shape=cls1.size()[3]
        label_cls=label_cls[:,:,size//2-shape//2:size//2+shape//2+1,size//2-shape//2:size//2+shape//2+1].contiguous()
        label_loc=label_loc[:,:,:,size//2-shape//2:size//2+shape//2+1,size//2-shape//2:size//2+shape//2+1].contiguous()
        label_loc_weight=label_loc_weight[:,:,size//2-shape//2:size//2+shape//2+1,size//2-shape//2:size//2+shape//2+1].contiguous()
        
        pre=(8*(np.linspace(0,size-1,size))+63).reshape(-1,1)-287//2
        pr=t.zeros(size**2,4).cuda()
        pr[:,0]=t.Tensor(np.maximum(0,np.tile(pre,(size)).T.reshape(-1)+287//2))
        pr[:,1]=t.Tensor(np.maximum(0,np.tile(pre,(size)).reshape(-1)+287//2))
    
        labelxff=t.zeros_like(xff).cuda()
        labelxff[0,0,:,:]=(pr[:,0]-bbox[0]).reshape(21,21)
        labelxff[0,1,:,:]=(bbox[2]-pr[:,0]).reshape(21,21)
        labelxff[0,2,:,:]=(pr[:,1]-bbox[1]).reshape(21,21)
        labelxff[0,3,:,:]=(bbox[3]-pr[:,1]).reshape(21,21)
        labelxff=labelxff/143

        pr[:,2]=self.con(xff[0,0,:,:]).view(-1)+self.con(xff[0,1,:,:]).view(-1)
        pr[:,3]=self.con(xff[0,2,:,:]).view(-1)+self.con(xff[0,3,:,:]).view(-1)
        pr[:,0]=pr[:,0]-self.con(xff[0,0,:,:]).view(-1)+pr[:,2]/2
        pr[:,1]=pr[:,1]-self.con(xff[0,2,:,:]).view(-1)+pr[:,3]/2
        
    

        
        def transform(center):
            x, y, w, h = center[:,0], center[:,1], center[:,2], center[:,3]
            x1 = x - w * 0.5
            y1 = y - h * 0.5
            x2 = x + w * 0.5
            y2 = y + h * 0.5
            return  t.cat((x1.view(-1,1),y1.view(-1,1),x2.view(-1,1),y2.view(-1,1)),1)
        pr=transform(pr)
        
 
        index=np.minimum(shape-1,np.maximum(0,np.int32((bbox-63-(size-shape)*4)/8)))
        w=int(index[2]-index[0])
        h=int(index[3]-index[1])

        weightcls3=t.zeros(1,1,shape,shape).cuda()
        weightcls3[0,0,index[1]:index[3]+1,index[0]:index[2]+1]=1
        weightcls33=t.zeros(1,1,shape,shape).cuda()
        for ii in np.arange(index[1],index[3]+1):
            for jj in np.arange(index[0],index[2]+1):
                 l1=t.min(t.Tensor([ii-index[1]]),t.Tensor([(index[3]-ii)]))/(t.max(t.Tensor([ii-index[1]]),t.Tensor([(index[3]-ii)]))+1e-4)
                 l2=t.min(t.Tensor([jj-index[0]]),t.Tensor([(index[2]-jj)]))/(t.max(t.Tensor([jj-index[0]]),t.Tensor([(index[2]-jj)]))+1e-4)
                 weightcls33[0,0,ii,jj]=weightcls3[0,0,ii,jj]*t.sqrt(l1*l2)
        

        
        cls1 = self.log_softmax(cls1)  
        cls2 = self.log_softmax(cls2) 

        
        cls_loss1 = select_cross_entropy_loss(cls1, label_cls,0.5)
        cls_loss2 = select_cross_entropy_loss(cls2, labelcls2,0.5)
        cls_loss3 = l1loss(cls3, weightcls33,weightcls3)  

        cls_loss= cls_loss3 + cls_loss1 + cls_loss2
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)  
        
        weightxff=t.zeros(1,1,size,size).cuda()
        index2=np.int32((bbox-63)/8)      #特征图上的位置
        w=int(index2[2]-index2[0])
        h=int(index2[3]-index2[1])
        weightxff[0,0,np.maximum(0,index2[1]-h//2):np.minimum(size,index2[3]+1+h//2),np.maximum(0,index2[0]-w//2):np.minimum(size,index2[2]+1+w//2)]=1

        
        shapeloss=l1loss(xff,labelxff,weightxff) 
        
        

      
        outputs=loc_loss+cls_loss+shapeloss    #2 4 1  都用loss2

        return outputs
    
    

    



    

    
     


  
    

