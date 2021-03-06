# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import torch
import pycocotools.mask as maskUtils
from pysot.core.config import cfg
from pysot.utils.bbox import cxy_wh_2_rect
from pysot.tracker.siamrpn_tracker_f import SiamRPNTracker


class SiamMaskTracker(SiamRPNTracker):
    def __init__(self, model):
        super(SiamMaskTracker, self).__init__(model)
        assert hasattr(self.model, 'mask_head'), \
            "SiamMaskTracker must have mask_head"
        assert hasattr(self.model, 'refine_head'), \
            "SiamMaskTracker must have refine_head"

    def _crop_back(self, image, bbox, out_sz, padding=0):
        a = (out_sz[0] - 1) / bbox[2]
        b = (out_sz[1] - 1) / bbox[3]
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz[0], out_sz[1]),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=padding)
        return crop

    def _mask_post_processing(self, mask):
        target_mask = (mask > cfg.TRACK.MASK_THERSHOLD)
        target_mask = target_mask.astype(np.uint8)
        if cv2.__version__[-5] == '4':
            contours, _ = cv2.findContours(target_mask,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(target_mask,
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(contours) != 0 and np.max(cnt_area) > 100:
            contour = contours[np.argmax(cnt_area)]
            polygon = contour.reshape(-1, 2)
            prbox = cv2.boxPoints(cv2.minAreaRect(polygon))
            rbox_in_img = prbox
        else:  # empty mask
            location = cxy_wh_2_rect(self.center_pos, self.size)
            rbox_in_img = np.array([[location[0], location[1]],
                                    [location[0] + location[2], location[1]],
                                    [location[0] + location[2], location[1] + location[3]],
                                    [location[0], location[1] + location[3]]])
        return rbox_in_img

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
        s_x = round(s_x)

        x_crop = self.get_subwindow(img,
                                    self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    s_x,
                                    self.channel_average)
        crop_box = [self.center_pos[0] - s_x / 2,
                    self.center_pos[1] - s_x / 2,
                    s_x,
                    s_x]

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
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
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
        cx, cy, width, height = self._bbox_clip(cx, cy,
                                                width, height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]

        # processing mask
        pos = np.unravel_index(best_idx, (5, self.score_size, self.score_size))
        delta_x, delta_y = pos[2], pos[1]

        mask = self.model.mask_refine((delta_y, delta_x)).sigmoid().squeeze()
        out_size = cfg.TRACK.MASK_OUTPUT_SIZE
        mask = mask.view(out_size, out_size).cpu().data.numpy()

        s = crop_box[2] / cfg.TRACK.INSTANCE_SIZE
        base_size = cfg.TRACK.BASE_SIZE
        stride = cfg.ANCHOR.STRIDE
        sub_box = [crop_box[0] + (delta_x - base_size/2) * stride * s,
                   crop_box[1] + (delta_y - base_size/2) * stride * s,
                   s * cfg.TRACK.EXEMPLAR_SIZE,
                   s * cfg.TRACK.EXEMPLAR_SIZE]
        s = out_size / sub_box[2]

        im_h, im_w = img.shape[:2]
        back_box = [-sub_box[0] * s, -sub_box[1] * s, im_w*s, im_h*s]
        mask_in_img = self._crop_back(mask, back_box, (im_w, im_h))
        polygon = self._mask_post_processing(mask_in_img)
        polygon = polygon.flatten().tolist()
        return {
                'bbox': bbox,
                'best_score': best_score,
                'mask': mask_in_img,
                'polygon': polygon,
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