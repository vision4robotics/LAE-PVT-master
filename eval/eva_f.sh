echo "eva with forecaster UAV123@10fps"
echo "mobile net"
python kf_streaming_eva.py --result_root /home/li/sAP-master/pysot-master/tools/results_rt_raw_f/UAV10/siamrpn_mobilev2_l234_dwxcorr --data_root /media/li/My_Passport/dataset/UAV123_10fps --fps 10 --out-dir /home/li/sAP-master/pysot-master/tools/results_f_rt_f/UAV10/siamrpn_mobilev2_l234_dwxcorr

echo "Alex net"
python kf_streaming_eva.py --result_root /home/li/sAP-master/pysot-master/tools/results_rt_raw_f/UAV10/siamrpn_alex_dwxcorr --data_root /media/li/My_Passport/dataset/UAV123_10fps --fps 10 --out-dir /home/li/sAP-master/pysot-master/tools/results_f_rt_f/UAV10/siamrpn_alex_dwxcorr

echo "RES net"
python kf_streaming_eva.py --result_root /home/li/sAP-master/pysot-master/tools/results_rt_raw_f/UAV10/siamrpn_r50_l234_dwxcorr --data_root /media/li/My_Passport/dataset/UAV123_10fps --fps 10 --out-dir /home/li/sAP-master/pysot-master/tools/results_f_rt_f/UAV10/siamrpn_r50_l234_dwxcorr

echo "Mask"
python kf_streaming_eva.py --result_root /home/li/sAP-master/pysot-master/tools/results_rt_raw_f/UAV10/siammask_r50_l3 --data_root /media/li/My_Passport/dataset/UAV123_10fps --fps 10 --out-dir /home/li/sAP-master/pysot-master/tools/results_f_rt_f/UAV10/siammask_r50_l3

echo "eva with UAVDark"
echo "mobile net"
python kf_streaming_eva.py --result_root /home/li/sAP-master/pysot-master/tools/results_rt_raw_f/UAVDARK/siamrpn_mobilev2_l234_dwxcorr --data_root /media/li/My_Passport/dataset/UAVDark135_TSP_out --fps 30 --out-dir /home/li/sAP-master/pysot-master/tools/results_f_rt_f/UAVDARK/siamrpn_mobilev2_l234_dwxcorr

echo "Alex net"
python kf_streaming_eva.py --result_root /home/li/sAP-master/pysot-master/tools/results_rt_raw_f/UAVDARK/siamrpn_alex_dwxcorr --data_root /media/li/My_Passport/dataset/UAVDark135_TSP_out --fps 30 --out-dir /home/li/sAP-master/pysot-master/tools/results_f_rt_f/UAVDARK/siamrpn_alex_dwxcorr

echo "RES net"
python kf_streaming_eva.py --result_root /home/li/sAP-master/pysot-master/tools/results_rt_raw_f/UAVDARK/siamrpn_r50_l234_dwxcorr --data_root /media/li/My_Passport/dataset/UAVDark135_TSP_out --fps 30 --out-dir /home/li/sAP-master/pysot-master/tools/results_f_rt_f/UAVDARK/siamrpn_r50_l234_dwxcorr

echo "Mask"
python kf_streaming_eva.py --result_root /home/li/sAP-master/pysot-master/tools/results_rt_raw_f/UAVDARK/siammask_r50_l3 --data_root /media/li/My_Passport/dataset/UAVDark135_TSP_out --fps 30 --out-dir /home/li/sAP-master/pysot-master/tools/results_f_rt_f/UAVDARK/siammask_r50_l3
