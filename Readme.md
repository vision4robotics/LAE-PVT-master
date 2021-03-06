# Predictive Visual Tracking: A New Benchmark and Baseline Approach

### Bowen Li *, Yiming Li *, Junjie Ye, Changhong Fu, and Hang Zhao

Python implementation of proposed *latency-aware evaluation* (**LAE**) benchmark and *predictive visual tracking* (**PVT**) baseline. LAE benchmark can currently be conducted on the following tracking libraries:

- [x] [Pysot](https://github.com/STVIR/pysot)
- [x] [Pytracking](https://github.com/visionml/pytracking)

More libraries are under construction...

- [ ] [DaSiamRPN](https://github.com/foolwood/DaSiamRPN)
- [ ] [SiamFC++](https://github.com/foolwood/DaSiamRPN)

## Abstract

As a crucial robotic perception capability, visual tracking has been intensively studied recently. In the real-world scenarios, the onboard processing time of the image streams inevitably leads to a discrepancy between the tracking results and the real-world states. However, existing visual tracking benchmarks commonly run the trackers offline and ignore such latency in the evaluation. In this work, we aim to deal with a more realistic problem of latency-aware tracking. The state-of-the-art trackers are evaluated in the aerial scenarios with new metrics jointly assessing the tracking accuracy and efficiency. Moreover, a new predictive visual tracking baseline is developed to compensate for the latency stemming from the onboard computation. Our latency-aware benchmark can provide a more realistic evaluation of the trackers for the robotic applications. Besides, exhaustive experiments have proven the effectiveness of our predictive visual tracking baseline approach.



![](.\fig\exhibition.png)

Here offers a vivid explanation of real-world tracking with latency. When the tracker finishes processing the input frame, the world state has already changed.



## Contact

Bowen Li

Email: [1854152@tongji.edu.cn]()

Yiming Li

Email: [yimingli@nyu.edu]()

Changhong Fu

Email: [changhongfu@tongji.edu.cn]()



## Requirements

This code has been tested on Ubuntu 18.04, Python 3.8.3, Pytorch 0.7.0/1.6.0, CUDA 10.2. Please install related libraries before running this code:

```
pip install -r requirements.txt
```



## Running Tutorial

### 1. Preparation

Follow the instructions in [Pysot](./libraries/pysot-master) and [Pytracking](./libraries/pytracking) to firstly track offline with no latency. 

Tips:

For [Pysot](./libraries/pysot-master) , you are expected to update the dataset paths in the dataset .py files ./pysot-master/toolkit/datasets/ and in tools folder, run:

```
python test.py --dataset dataset name --datasetroot path_to_dataset --config path_to_config --snapshot path_to_model
```

For [Pytracking](./libraries/pytracking) , you are expected to update paths in ./pytracking/pytracking/evaluation/local.py and in pytracking folder run:

```
python run_tracker.py --tracker_name dimp --tracker_param dimp18
```



### 2. Tracking with Latency

For [Pysot](./libraries/pysot-master):

In tools folder, run:

```
python test_rt.py --dataset dataset_name --datasetroot path_to_dataset --config path_to_config --snapshot path_to_model
```

You'll get results in .pkl files in results_rt_raw/dataset_name/tracker_name

For [Pytracking](./libraries/pytracking):

In pytracking folder, run:

```
python run_tracker.py --tracker_name dimp --tracker_param dimp18 --rt_running 1
```

You'll get results in .pkl files in tracking_results_rt_raw/tracker_name/parameter_name



### 3. Evaluation with latency

In folder eval with the .pkl result files obtained in tracking:

run

```
python streaming_eval_fast.py --raw_root path_to_rawresults --data_root --tar_root path_to_output
```

You'll get the .txt files similar to offline benchmark that pairs each ground-truths with output results.



### 4. Predictive Tracking Baseline

Our PVT tracker (with pre-forecaster and post-forecaster) is implemented in [Pysot](./libraries/pysot-master).

Specifically, in tools folder run:

```
python test_rt_f.py --dataset dataset_name --datasetroot path_to_dataset --config path_to_config --snapshot path_to_model
```

You'll get raw results with pre-forecaster.

Then in eval folder run

```
python kf_streaming_eva.py --result_root path_to_rawresults --data_root path_to_dataset --out-dir path_to_output
```

You'll get results with both pre-forecaster and post-forecaster.

*Note:*

In this way, the post-forecaster is implemented off-line. However, the forecasters are fast enough (>500FPS on single CPU) and cause very little latency, which can be left out.





## Overall Results Comparison

![](.\fig\overall.png)

Performance of state-of-the-art trackers on the proposed LAE benchmark. The curves in solid colors report the performance of the 8 benchmarked trackers on LAE, whereas the dotted curves overlaid in semi-transparent colors outline the performance obtained by the same trackers on the traditional offline benchmark. In brackets, we report the distance precision (DP) and area under curve (AUC) on LAE (in black) and on offline benchmark (in gray). Clearly, many offline promising trackers fail to maintain their robustness and accuracy in LAE benchmark.

## Effect of PVT baseline

![](.\fig\PVT.png)

Performance of the state-of-the-art trackers with offline latency-free and online latency-aware benchmarks on DTB70. The distance precision (DP) is employed for evaluation. The same shape indicates the same tracker, *e.g.*, star for SiamRPN++ with ResNet50 as backbone. Blue denotes the results on the offline benchmark. Red means the results on the latency-aware benchmark. Our predictive tracking baseline is marked out by red circles, where the original performance is improved by a considerable margin, denoted by green arrows and percentages.



## Qualitative Evaluation

![](.\fig\visulization.png)

More latency-aware tracking sequences can be found at [Video](https://youtu.be/n8i8bREIFeM).



# Acknowledgements

We thank the contribute of Qiang Wang and Martin Danelljan for their brilliant tracking libraries  [Pysot](https://github.com/STVIR/pysot) and [Pytracking](https://github.com/visionml/pytracking) sincerely.

