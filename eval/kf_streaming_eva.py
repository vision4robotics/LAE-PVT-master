''' 
IoU-based greedy association + batched Kalman Filter
implemented as post-processing (zero runtime assumption)
batching is based on pytorch's batched matrix operations
using notations from Wikipedia
'''


import argparse, json, pickle
from os.path import join, isfile
import os
from time import perf_counter
import numpy as np
import pycocotools.mask as maskUtils
import sys
sys.path.append('/home/li/sAP-master')
import torch
import cv2

# the line below is for running in both the current directory 
# and the repo's root directory
import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2, print_stats
# from track import iou_assoc
from forecast import extrap_clean_up


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_root', type=str, default='/home/li/sAP-master/pysot-master/tools/results_rt_raw_f/VISDRONE/siamrpn_mobilev2_l234_dwxcorr')
    parser.add_argument('--data_root', type=str, default='/media/li/无人机组-暗黑数据/VisDrone2018-SOT-test')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--eta', type=float, default=0, help='eta >= -1')
    parser.add_argument('--assoc', type=str, default='iou')
    parser.add_argument('--match-iou-th', type=float, default=0.3)
    parser.add_argument('--forecast-rt-ub', type=float, default=0)
    parser.add_argument('--forecast-before-assoc', action='store_true', default=True)
    parser.add_argument('--out-dir', type=str, default='/home/li/sAP-master/pysot-master/tools/results_f_rt_f/VISDRONE/siamrpn_mobilev2_l234_dwxcorr')
    parser.add_argument('--overwrite', action='store_true', default=False)

    opts = parser.parse_args()
    return opts


def bbox2z(bboxes):
    return torch.from_numpy(bboxes).unsqueeze_(2)

def bbox2x(bboxes):
    x = torch.cat((torch.from_numpy(bboxes), torch.zeros(bboxes.shape)), dim=1)
    return x.unsqueeze_(2)

def x2bbox(x):
    return x[:, :4, 0].numpy()

def make_F(F, dt):
    F[[0, 1, 2, 3], [4, 5, 6, 7]] = dt
    return F.double()

def make_Q(Q, dt):
    # assume the base Q is identity
    Q[[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]] = dt*dt
    return Q.double()

def batch_kf_predict_only(F, x):
    return F @ x

def batch_kf_predict(F, x, P, Q):
    x = F @ x
    P = F @ P.double() @ F.t() + Q
    return x.double(), P.double()

def batch_kf_update(z, x, P, R):
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

def iou_assoc(bboxes1, tracks1, tkidx, bboxes2, match_iou_th, no_unmatched1=False):
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

def main():
    opts = parse_args()
    assert opts.forecast_before_assoc, "Not implemented"

    mkdir2(opts.out_dir)
    data_root=opts.data_root
    if 'DTB70' not in data_root:
        seqs=os.listdir(os.path.join(data_root,'data_seq'))
    else:
        seqs=os.listdir(data_root)
    # db = COCO(opts.annot_path)
    # class_names = [c['name'] for c in db.dataset['categories']]
    # n_class = len(class_names)
    # coco_mapping = db.dataset.get('coco_mapping', None)
    # if coco_mapping is not None:
    #     coco_mapping = np.asarray(coco_mapping)
    # seqs = db.dataset['sequences']
    # seq_dirs = db.dataset['seq_dirs']

    given_tracks = opts.assoc == 'given'
    assert not given_tracks, "Not implemented"
            
    for sid, seq in enumerate(seqs):
        ou_path=os.path.join(opts.out_dir)
        if not os.path.isdir(ou_path):
            os.makedirs(ou_path)
        result_path = os.path.join(ou_path, '{}.txt'.format(seq))
        if isfile(result_path):
            print("Evaluated with forcasting seq: {} ({}/{}), done!".format(seq, sid+1,len(seqs)))
            continue
        with torch.no_grad():
            kf_F = torch.eye(8)
            kf_Q = torch.eye(8)
            kf_R = 10*torch.eye(4)
            kf_P_init = 100*torch.eye(8).unsqueeze(0)
            results_ccf = []
            in_time = 0
            miss = 0
            shifts = 0
            t_assoc = []
            t_forecast = []
            if 'DTB70' in data_root:
                frame_path = os.path.join(data_root, seq, 'img')
            else:
                frame_path = os.path.join(data_root,'data_seq', seq)
            frames = os.listdir(frame_path)
            frame_list=[]
            for frame in frames:
                frame_list.append(os.path.join(frame_path, frame))
    
            results = pickle.load(open(join(opts.result_root, seq + '.pkl'), 'rb'))
            # use raw results when possible in case we change class subset during evaluation
            results_raw = results['results_raw']
            timestamps = results['timestamps']
            timestamps[0] = 0
            input_fidx = results['input_fidx']
    
            # t1 -> det1, t2 -> det2, interpolate at t3 (t3 is the current time)
            det_latest_p1 = 0           # latest detection index + 1
            det_t2 = None               # detection index at t2
            kf_x = torch.empty((0, 8, 1))
            kf_P = torch.empty((0, 8, 8))
            kf_F = torch.eye(8)
            kf_Q = torch.eye(8)
            kf_R = 10*torch.eye(4)
            kf_P_init = 100*torch.eye(8).unsqueeze(0)
            n_matched12 = 0
    
            if not given_tracks:
                tkidx = 0               # track starting index
    
            for ii, img_path in enumerate(frame_list):
                # pred, gt association by time
                if ii==0:
                    img=cv2.imread(img_path)
                t = (ii - opts.eta)/opts.fps
                while det_latest_p1 < len(timestamps) and timestamps[det_latest_p1] <= t:
                    det_latest_p1 += 1
                if det_latest_p1 == 0:
                    continue
                    # no output
                else:
                    det_latest = det_latest_p1 - 1
                    ifidx = input_fidx[det_latest]
                    in_time += int(ii == ifidx)
                    shifts += ii - ifidx
    
                    if det_latest != det_t2:
                        # new detection
                        # we can now throw away old result (t1)
                        # the old one is kept for forecasting purpose
    
                        if len(kf_x) and opts.forecast_before_assoc:
                            dt = ifidx - input_fidx[det_t2]
                            dt = int(dt) # convert from numpy to basic python format
                            w_img, h_img = img.shape[1], img.shape[0]
    
                            kf_F = make_F(kf_F, dt)
                            kf_Q = make_Q(kf_Q, dt)
    
                            kf_x, kf_P = batch_kf_predict(kf_F, kf_x, kf_P, kf_Q)
                            bboxes_f = x2bbox(kf_x)
                            
                        det_t2 = det_latest
                        bboxes_t2 = np.array([results_raw[det_t2]])
    
                        t1 = perf_counter()
                        n = len(bboxes_t2)
                        if n:
                            updated = False
                            if len(kf_x):
                                order1, order2, n_matched12, tracks, tkidx = iou_assoc(
                                    bboxes_f, tracks, tkidx,
                                    bboxes_t2, opts.match_iou_th,
                                    no_unmatched1=True,
                                )
    
                                if n_matched12:
                                    kf_x = kf_x[order1]
                                    kf_P = kf_P[order1]
                                    kf_x, kf_P = batch_kf_update(
                                        bbox2z(bboxes_t2[order2[:n_matched12]]),
                                        kf_x,
                                        kf_P,
                                        kf_R,
                                    )
                            
                                    kf_x_new = bbox2x(bboxes_t2[order2[n_matched12:]])
                                    n_unmatched2 = len(bboxes_t2) - n_matched12
                                    kf_P_new = kf_P_init.expand(n_unmatched2, -1, -1)
                                    kf_x = torch.cat((kf_x, kf_x_new))
                                    kf_P = torch.cat((kf_P, kf_P_new))
                                    updated = True
    
                            if not updated:
                                # start from scratch
                                kf_x = bbox2x(bboxes_t2)
                                kf_P = kf_P_init.expand(len(bboxes_t2), -1, -1)
                                if not given_tracks:
                                    tracks = np.arange(tkidx, tkidx + n, dtype=np.uint32)
                                    tkidx += n
    
                            t2 = perf_counter()
                            t_assoc.append(t2 - t1)
    
                    t3 = perf_counter()
                    if len(kf_x):
                        dt = ii - ifidx
                        w_img, h_img = img.shape[1], img.shape[0]
    
                        # PyTorch small matrix multiplication is slow
                        # use numpy instead
                        kf_x_np = kf_x[:, :, 0].numpy()
                        bboxes_t3 = kf_x_np[:n_matched12, :4] + dt*kf_x_np[:n_matched12, 4:]
                        if n_matched12 < len(kf_x):
                            bboxes_t3 = np.concatenate((bboxes_t3, kf_x_np[n_matched12:, :4]))
                        
                        bboxes_t3, keep = extrap_clean_up(bboxes_t3, w_img, h_img, lt=True)
    
    
                    t4 = perf_counter()
                    t_forecast.append(t4 - t3)
    
                if len(bboxes_t3):
                    results_ccf.append(list(bboxes_t3[0]))
                else:
                    results_ccf.append(list(bboxes_t2[0]))
                    miss+=1
        
        # ou_path=os.path.join(opts.out_dir)
        # if not os.path.isdir(ou_path):
        #     os.makedirs(ou_path)
        # result_path = os.path.join(ou_path, '{}.txt'.format(seq))
        with open(result_path, 'w') as f:
            for x in results_ccf:
                f.write(','.join([str(i) for i in x])+'\n')
        print("Evaluate with forcasting seq: {} ({}/{}), done!,miss {}".format(seq, sid+1,len(seqs),miss))
    # s2ms = lambda x: 1e3*x
    # if len(t_assoc):
    #     print_stats(t_assoc, "RT association (ms)", cvt=s2ms)
    # if len(t_forecast):
    #     print_stats(t_forecast, "RT forecasting (ms)", cvt=s2ms)    

    # out_path = join(opts.out_dir, 'results_ccf.pkl')
    # if opts.overwrite or not isfile(out_path):
    #     pickle.dump(results_ccf, open(out_path, 'wb'))

    # if not opts.no_eval:
    #     eval_summary = eval_ccf(db, results_ccf)
    #     out_path = join(opts.out_dir, 'eval_summary.pkl')
    #     if opts.overwrite or not isfile(out_path):
    #         pickle.dump(eval_summary, open(out_path, 'wb'))

    # if vis_out:
    #     print(f'python vis/make_videos.py "{opts.vis_dir}" --fps {opts.fps}')

if __name__ == '__main__':
    main()