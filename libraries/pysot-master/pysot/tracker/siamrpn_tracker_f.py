# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import torch
import pycocotools.mask as maskUtils

from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker


class SiamRPNTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamRPNTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])
        self.forecaster = self.KF(np.array([bbox]) , img)

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.model.template(z_crop)

    def track(self, img, box_f):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        # use KF to correct current center to realize search region forecast
        self.size[0] = box_f[2]
        self.size[1] = box_f[3]
        cx_f = box_f[0] + box_f[2]/2
        cy_f = box_f[1] + box_f[3]/2
        self.center_pos = np.array([cx_f,cy_f])
        
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
                'bbox': bbox,
                'best_score': best_score
               }
    
    class KF:
        def __init__(self,bbox_init,img):
            self.kf_x = self.bbox2x(bbox_init)
            self.kf_P = torch.empty((0, 8, 8))
            self.kf_F = torch.eye(8)
            self.kf_Q = torch.eye(8)
            self.kf_R = 10*torch.eye(4)
            self.kf_P_init = 100*torch.eye(8).unsqueeze(0)
            self.kf_P = self.kf_P_init.expand(len(bbox_init), -1, -1)
            self.tidx = 0
            self.updated=False
            self.tracks = np.arange(self.tidx, self.tidx+1, dtype=np.uint32)
            self.tidx+=1
            self.t=0
            self.w_img, self.h_img = img.shape[1], img.shape[0]
            
        def forecast(self,fidx_t,fidx_curr,box_curr):
            #new result forecast first
            dt = fidx_curr-self.t
            dt=int(dt)
            self.kf_F = self.make_F(self.kf_F, dt)
            self.kf_Q = self.make_Q(self.kf_Q, dt)
            self.kf_x, self.kf_P = self.batch_kf_predict(self.kf_F, self.kf_x, self.kf_P, self.kf_Q)
            self.bboxes_f = self.x2bbox(self.kf_x) #bboxes_f is the forecast result
            #update KF time
            self.t = fidx_curr
            self.bboxes_t2 = box_curr
            self.updated=False
            #association based on IoU match
            order1, order2, n_matched12, self.tracks, self.tkidx = self.iou_assoc(
                                    self.bboxes_f, self.tracks, self.tidx,
                                    self.bboxes_t2, 0.3,
                                    no_unmatched1=True,
                                )
            # If match, update x, P in KF with box_curr
            if n_matched12:
                self.kf_x = self.kf_x[order1]
                self.kf_P = self.kf_P[order1]
                self.kf_x, kf_P = self.batch_kf_update(
                    self.bbox2z(self.bboxes_t2[order2[:n_matched12]]),
                    self.kf_x,
                    self.kf_P,
                    self.kf_R,
                )
        
                kf_x_new = self.bbox2x(self.bboxes_t2[order2[n_matched12:]])
                n_unmatched2 = len(self.bboxes_t2) - n_matched12
                kf_P_new = self.kf_P_init.expand(n_unmatched2, -1, -1)
                self.kf_x = torch.cat((self.kf_x, kf_x_new))
                self.kf_P = torch.cat((self.kf_P, kf_P_new))
                self.updated = True
            if not self.updated:
                self.kf_x = self.bbox2x(self.bboxes_t2)
                self.kf_P = self.kf_P_init.expand(len(self.bboxes_t2), -1, -1)
                self.tracks = np.arange(self.tidx, self.tidx + 1, dtype=np.uint32)
                self.tkidx += 1
            # Forecast to fidx_t
            dt = fidx_t - fidx_curr
            # PyTorch small matrix multiplication is slow
            # use numpy instead
            kf_x_np = self.kf_x[:, :, 0].numpy()
            bboxes_t3 = kf_x_np[:n_matched12, :4] + dt*kf_x_np[:n_matched12, 4:]
            if n_matched12 < len(self.kf_x):
                bboxes_t3 = np.concatenate((bboxes_t3, kf_x_np[n_matched12:, :4]))
            
            bboxes_t3, keep = self.extrap_clean_up(bboxes_t3, self.w_img, self.h_img, lt=True)
            if len(bboxes_t3):
                return bboxes_t3
            else :
                return box_curr
            
        
        def bbox2z(self,bboxes):
            return torch.from_numpy(bboxes).unsqueeze_(2)

        def bbox2x(self,bboxes):
            x = torch.cat((torch.from_numpy(bboxes), torch.zeros(bboxes.shape)), dim=1)
            return x.unsqueeze_(2)
        
        def x2bbox(self,x):
            return x[:, :4, 0].numpy()
        
        def make_F(self,F, dt):
            F[[0, 1, 2, 3], [4, 5, 6, 7]] = dt
            return F.double()
        
        def make_Q(self,Q, dt):
            # assume the base Q is identity
            Q[[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]] = dt*dt
            return Q.double()
        
        def batch_kf_predict_only(self,F, x):
            return F @ x
        
        def batch_kf_predict(self,F, x, P, Q):
            x = F @ x
            P = F @ P.double() @ F.t() + Q
            return x.double(), P.double()
        
        def batch_kf_update(self,z, x, P, R):
            # assume H is just slicing operation
            # y = z - Hx
            y = z - x[:, :4]
        
            # S = HPH' + R
            S = P[:, :4, :4] + R
        
            # K = PH'S^(-1)
            K = P[:, :, :4] @ S.inverse()
        
            # x = x + Ky
            x += K @ y
        
            # P = (I - KH)P
            P -= K @ P[:, :4]
            return x.double(), P.double()
        
        def iou_assoc(self,bboxes1, tracks1, tkidx, bboxes2, match_iou_th, no_unmatched1=False):
            # iou-based association
            # shuffle all elements so that matched stays in the front
            # bboxes are in the form of a list of [l, t, w, h]
            m, n = len(bboxes1), len(bboxes2)
                
            _ = n*[0]
            ious = maskUtils.iou(bboxes1, bboxes2, _)
        
            match_fwd = m*[None]
            matched1 = []
            matched2 = []
            unmatched2 = []
        
            for j in range(n):
                best_iou = match_iou_th
                match_i = None
                for i in range(m):
                    if match_fwd[i] is not None or ious[i, j] < best_iou:
                        # or labels1[i] != labels2[j] \
                        continue
                    best_iou = ious[i, j]
                    match_i = i
                if match_i is None:
                    unmatched2.append(j)
                else:
                    matched1.append(match_i)
                    matched2.append(j)
                    match_fwd[match_i] = j
        
            if no_unmatched1:
                order1 = matched1
            else:
                unmatched1 = list(set(range(m)) - set(matched1))
                order1 = matched1 + unmatched1
            order2 = matched2 + unmatched2
        
            n_matched = len(matched2)
            n_unmatched2 = len(unmatched2)
            tracks2 = np.concatenate((tracks1[order1][:n_matched],
                np.arange(tkidx, tkidx + n_unmatched2, dtype=tracks1.dtype)))
            tkidx += n_unmatched2
        
            return order1, order2, n_matched, tracks2, tkidx
        
        def extrap_clean_up(self,bboxes, w_img, h_img, min_size=75, lt=False):
            # bboxes in the format of [cx or l, cy or t, w, h]
            wh_nz = bboxes[:, 2:] > 0
            keep = np.logical_and(wh_nz[:, 0], wh_nz[:, 1])
        
            if lt:
                # convert [l, t, w, h] to [l, t, r, b]
                bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:]
            else:
                # convert [cx, cy, w, h] to [l, t, r, b]
                bboxes[:, :2] = bboxes[:, :2] - bboxes[:, 2:]/2
                bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:]
        
            # clip to the image
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(0, w_img)
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(0, h_img)
        
            # convert [l, t, r, b] to [l, t, w, h]
            bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]
        
            # int conversion is neccessary, otherwise, there are very small w, h that round up to 0
            keep = np.logical_and(keep, bboxes[:, 2].astype(np.int)*bboxes[:, 3].astype(np.int) >= min_size)
            bboxes = bboxes[keep]
            return bboxes, keep
