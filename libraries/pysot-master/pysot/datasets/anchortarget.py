

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch as t
from pysot.core.config import cfg
from pysot.utils.bbox import IoU, corner2center
from pysot.utils.anchor import Anchors


class AnchorTarget():
    def __init__(self):

        return
    
    def get(self, anchor,target, size, neg=False):
       
            
        anchor_num=1
        
        # -1 ignore 0 negative 1 positive
        cls = -1 * np.ones((anchor_num, size, size), dtype=np.int64)
        delta = np.zeros((4, anchor_num, size, size), dtype=np.float32)
        delta_weight = np.zeros((anchor_num, size, size), dtype=np.float32)

        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num

        
        tcx = (target[0]+target[2])/2
        tcy= (target[1]+target[3])/2
        tw=target[2]-target[0]
        th=target[3]-target[1]
        
        shape=21
        
        labelcls2=t.zeros(1,1,size,size).cuda()-1
        index=np.minimum(shape-1,np.maximum(0,np.int32((target-63-(size-shape)*4)/8)))
        ww=int(index[2]-index[0])
        hh=int(index[3]-index[1])
        
        negg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()
        
        if negg :
#        if anchor[:,2].min()<0 or anchor[:,3].min()<0:

            # l = size // 2 - 3
            # r = size // 2 + 3 + 1
            # cls[:, l:r, l:r] = 0
            
            cx = size // 2
            cy = size // 2
            cx += int(np.ceil((tcx - cfg.TRAIN.SEARCH_SIZE // 2) /
                      cfg.ANCHOR.STRIDE + 0.5))
            cy += int(np.ceil((tcy - cfg.TRAIN.SEARCH_SIZE // 2) /
                      cfg.ANCHOR.STRIDE + 0.5))
            l = max(0, cx - 3)
            r = min(size, cx + 4)
            u = max(0, cy - 3)
            d = min(size, cy + 4)
            cls[:, u:d,l:r ] = 0

            neg, neg_num = select(np.where(cls == 0), cfg.TRAIN.NEG_NUM)
            cls[:] = -1
            cls[neg] = 0

            overlap = np.zeros((anchor_num, size, size), dtype=np.float32)
            
            labelcls2[0,0, u:d,l:r ] = -2

            neg=np.where(labelcls2.view(size,size).cpu().numpy()==-2)
            neg = select(neg, cfg.TRAIN.NEG_NUM)
            labelcls2[:,:,neg[0][0],neg[0][1]] = 0
            
            return cls, delta, delta_weight, overlap,labelcls2
        
         
        

        
        
            



#        anchor=anchor.reshape(delta.shape)
#        anchor[2]=np.maximum(anchor[2],10)
#        anchor[3]=np.maximum(anchor[2],10)
        cx, cy, w, h = anchor[:,0].reshape(1,size,size),anchor[:,1].reshape(1,size,size),anchor[:,2].reshape(1,size,size),anchor[:,3].reshape(1,size,size)
        x1 = cx - w * 0.5
        y1 = cy - h * 0.5
        x2 = cx + w * 0.5
        y2 = cy + h * 0.5
        
#        tcx=tcx-gt[0]
#        tcy=tcy-gt[1]
#        cx=cx-gt[0]
#        cy=cy-gt[1]

        
        overlap = IoU([x1, y1, x2, y2], target)

#        pos = np.where(overlap > cfg.TRAIN.THR_HIGH)
#        neg = np.where(overlap < cfg.TRAIN.THR_LOW)
        
#        
#        delta_weight[pos] = 1. / (pos_num + 1e-6)
#        delta_weight[neg] =0 #0
        
        pos1 = np.where((overlap > 0.6) )               
        neg1 = np.where((overlap <= 0.3) )
        pos1, pos_num1 = select(pos1, cfg.TRAIN.POS_NUM)
        neg1, neg_num1 = select(neg1, cfg.TRAIN.TOTAL_NUM - cfg.TRAIN.POS_NUM)
        cls[pos1] = 1
        cls[neg1] = 0
      
        labelcls2[0,0,index[1]:index[3]+1,index[0]:index[2]+1]=-2
        labelcls2[0,0,index[1]+hh//3:index[3]+1-hh//3,index[0]+ww//3:index[2]+1-ww//3]=1
     
        pos = np.where((overlap > 0.6) | (labelcls2.view(1,size,size).cpu().numpy()==1))                 ################### 0.6

        neg = np.where((overlap <= 0.3)&(labelcls2.view(1,size,size).cpu().numpy()==-1) )
        
        pos, pos_num = select(pos, cfg.TRAIN.POS_NUM)
        neg, neg_num = select(neg, cfg.TRAIN.TOTAL_NUM - cfg.TRAIN.POS_NUM)

        neg2=np.where(labelcls2.view(size,size).cpu().numpy()==-1)
        neg2 = select(neg2, cfg.TRAIN.TOTAL_NUM - cfg.TRAIN.POS_NUM)
        labelcls2[:,:,neg2[0][0],neg2[0][1]] = 0
        
        if anchor[:,2].min()>0 and anchor[:,3].min()>0:
            
            delta[0] = (tcx - cx) / (w+1e-6)
            delta[1] = (tcy - cy) / (h+1e-6)
                
            delta[2] = np.log(tw / (w+1e-6) + 1e-6)
            delta[3] = np.log(th / (h+1e-6) + 1e-6)
            
            delta_weight[pos] = 1. / (pos_num + 1e-6)
            
            delta_weight[neg] =0 #0
        
        #newloc
#        delta1 = np.zeros((4, anchor_num, size, size), dtype=np.float32)
#        delta1[0] = (tcx - x1) 
#        delta1[1] = (x2 - tcy) 
#        delta1[2] = (tcy - y1) 
#        delta1[3] = (y2 - tcy)

        
        return cls, delta, delta_weight, overlap,labelcls2
