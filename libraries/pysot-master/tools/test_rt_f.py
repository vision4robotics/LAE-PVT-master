# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse, pickle
import os
import sys
sys.path.append('/home/v4r/LBW/streaming_object_tracking/pysot-master/')
import cv2
import torch
import numpy as np
from time import perf_counter

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder_f import build_tracker_f
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
#from toolkit.utils.region import vot_overlap, vot_float2str


parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', default='DTB70',type=str,
        help='datasets')
parser.add_argument('--datasetroot', default='/media/li/CA5CF8AE5CF89683/research/DTB70/',type=str,
        help='datasetsroot')
parser.add_argument('--fps', default=30,type=int,
        help='input frame rate')
parser.add_argument('--config', default='/home/li/sAP-master/pysot-master/experiments/siammask_r50_l3/config1.yaml', type=str,
        help='config file')
parser.add_argument('--snapshot', default='/home/li/sAP-master/pysot-master/experiments/siammask_r50_l3/model.pth', type=str,
        help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', default=False,action='store_true',
        help='whether visualzie result')
parser.add_argument('--overwrite', default=True,action='store_true',
        help='whether to overwrite existing results')
args = parser.parse_args()

torch.set_num_threads(1)

def main():
    # load config
    cfg.merge_from_file(args.config)

    # cur_dir = os.path.dirname(os.path.realpath(__file__))
    # dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)
    # UAVDTdataset = '/media/li/CA5CF8AE5CF89683/research/UAVDT/'
    # UAVDTdataset = '/media/li/DATA/VisDrone2019-SOT/'
    UAVDTdataset = args.datasetroot
    dataset_root = os.path.join(UAVDTdataset)
    
    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = build_tracker_f(model)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.snapshot.split('/')[-2].split('.')[0]
    torch.cuda.synchronize()

    # OPE tracking
    for v_idx, video in enumerate(dataset):
        #if v_idx>40:
        o_path=os.path.join('results_rt_raw_f', args.dataset, model_name)
        if not os.path.isdir(o_path):
            os.makedirs(o_path)
        out_path = os.path.join('results_rt_raw_f', args.dataset, model_name, video.name + '.pkl')
        if os.path.isfile(out_path):
            print('({:3d}) Video: {:12s} already done!'.format(
            v_idx+1, video.name))
            continue
        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []
        input_fidx = []
        runtime = []
        timestamps = []
        last_fidx = None
        n_frame=len(video)
        t_total = n_frame/args.fps
        t_start = perf_counter()
        while 1:
            t1 = perf_counter()
            t_elapsed=t1-t_start
            if t_elapsed>t_total:
                break
            # identify latest available frame
            fidx_continous = t_elapsed*args.fps
            fidx = int(np.floor(fidx_continous))
            #if the tracker finishes current frame before next frame comes, continue
            if fidx == last_fidx:
                continue
            last_fidx=fidx
            tic = cv2.getTickCount()
            (img,gt_bbox)=video[fidx]
            if fidx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                tracker.init(img, gt_bbox_)
                torch.cuda.synchronize()
                t2 = perf_counter()
                t_elapsed=t2-t_start
                timestamps.append(t_elapsed)
                runtime.append(t2-t1)
                pred_bbox = gt_bbox_
                scores.append(None)
                pred_bboxes.append(pred_bbox)
                input_fidx.append(fidx)
            else:
                box_f = tracker.forecaster.forecast(fidx, input_fidx[-1], np.array([pred_bboxes[-1]]))
                outputs = tracker.track(img,box_f[0])
                torch.cuda.synchronize()
                t2 = perf_counter()
                t_elapsed=t2-t_start
                timestamps.append(t_elapsed)
                runtime.append(t2-t1)
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
                scores.append(outputs['best_score'])
                input_fidx.append(fidx)
            if t_elapsed>t_total:
                break
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())

        #save results and run time
        if args.overwrite or not os.path.isfile(out_path):
            pickle.dump({
                'results_raw': pred_bboxes,
                'timestamps': timestamps,
                'input_fidx': input_fidx,
                'runtime': runtime,
            }, open(out_path, 'wb'))
        toc /= cv2.getTickFrequency()
        # save results
        # model_path = os.path.join('results', args.dataset, model_name)
        # if not os.path.isdir(model_path):
        #     os.makedirs(model_path)
        # result_path = os.path.join(model_path, '{}.txt'.format(video.name))
        # with open(result_path, 'w') as f:
        #     for x in pred_bboxes:
        #         f.write(','.join([str(i) for i in x])+'\n')
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx+1, video.name, toc, fidx / toc))


if __name__ == '__main__':
    main()
