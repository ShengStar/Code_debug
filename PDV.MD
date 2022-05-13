# PDV
## 论文题目
Point Density-Aware Voxels for LiDAR 3D Object Detection
## 调试过程
```
# 查看显卡
nvidia-smi
# 部署
(base) lixusheng@cqu100:~$ conda create -n openpcdet5.0_PDV --clone openpcdet5.0_votr
conda activate openpcdet5.0_PDV
cd /data/lixusheng_data/code/OpenPCDet
# 部署
python setup.py develop
# 运行代码
(openpcdet5.0_PDV) lixusheng@cqu100:/data/lixusheng_data/code/OpenPCDet/tools$ CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file cfgs/kitti_models/pointpillar.yaml 
```
## 结果1
```
2022-05-13 08:05:31,254   INFO  *************** Performance of EPOCH 80 *****************
2022-05-13 08:05:31,255   INFO  Generate label finished(sec_per_example: 0.0358 second).
2022-05-13 08:05:31,255   INFO  recall_roi_0.3: 0.969530
2022-05-13 08:05:31,255   INFO  recall_rcnn_0.3: 0.969757
2022-05-13 08:05:31,255   INFO  recall_roi_0.5: 0.930117
2022-05-13 08:05:31,255   INFO  recall_rcnn_0.5: 0.937806
2022-05-13 08:05:31,255   INFO  recall_roi_0.7: 0.707939
2022-05-13 08:05:31,255   INFO  recall_rcnn_0.7: 0.758799
2022-05-13 08:05:31,259   INFO  Average predicted number of objects(3769 samples): 8.310
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (20) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (25) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (30) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (35) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (24) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (16) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (80) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (20) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/core/typed_passes.py:330: NumbaPerformanceWarning:
The keyword argument 'parallel=True' was specified but no transformation for parallel execution was possible.

To find out why, try turning on parallel diagnostics, see https://numba.readthedocs.io/en/stable/user/parallel.html#diagnostics for help.

File "../pcdet/datasets/kitti/kitti_object_eval_python/eval.py", line 122:
@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
^

  state.func_ir.loc))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (25) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/core/typed_passes.py:330: NumbaPerformanceWarning:
The keyword argument 'parallel=True' was specified but no transformation for parallel execution was possible.

To find out why, try turning on parallel diagnostics, see https://numba.readthedocs.io/en/stable/user/parallel.html#diagnostics for help.

File "../pcdet/datasets/kitti/kitti_object_eval_python/eval.py", line 122:
@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
^

  state.func_ir.loc))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (25) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (20) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (30) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (35) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (24) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (16) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (80) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
2022-05-13 08:05:56,104   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:97.3074, 89.7138, 89.3109
bev  AP:90.4282, 88.5310, 87.9945
3d   AP:89.6815, 84.5552, 79.0893
aos  AP:97.27, 89.63, 89.16
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:98.7611, 94.8800, 94.4788
bev  AP:95.5400, 91.4296, 89.1861
3d   AP:92.7439, 85.4642, 83.0565
aos  AP:98.73, 94.77, 94.27
Car AP@0.70, 0.50, 0.50:
bbox AP:97.3074, 89.7138, 89.3109
bev  AP:97.3999, 95.2609, 89.3280
3d   AP:97.3411, 89.6672, 89.2939
aos  AP:97.27, 89.63, 89.16
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:98.7611, 94.8800, 94.4788
bev  AP:98.7690, 96.7981, 94.6999
3d   AP:98.7058, 94.8990, 94.6187
aos  AP:98.73, 94.77, 94.27
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:73.2722, 68.8875, 65.3323
bev  AP:64.9643, 61.0535, 57.5193
3d   AP:63.7190, 58.3660, 53.9782
aos  AP:69.66, 64.92, 61.11
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:73.5897, 69.0867, 66.2584
bev  AP:66.0267, 60.5177, 56.1549
3d   AP:64.4485, 57.6583, 52.9356
aos  AP:69.60, 64.66, 61.36
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:73.2722, 68.8875, 65.3323
bev  AP:76.6307, 73.2143, 70.9283
3d   AP:76.5134, 72.9244, 70.5518
aos  AP:69.66, 64.92, 61.11
Pedestrian AP_R40@0.50, 0.25, 0.25:
bbox AP:73.5897, 69.0867, 66.2584
bev  AP:78.8879, 74.9609, 71.8595
3d   AP:78.7056, 74.5389, 71.4911
aos  AP:69.60, 64.66, 61.36
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:94.7722, 81.5301, 76.3866
bev  AP:91.5281, 74.0378, 71.1525
3d   AP:90.3256, 72.1872, 69.1229
aos  AP:94.54, 81.04, 75.97
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:95.8197, 82.2335, 79.1187
bev  AP:92.8801, 76.0579, 71.5401
3d   AP:91.3610, 73.0230, 69.5844
aos  AP:95.60, 81.73, 78.62
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:94.7722, 81.5301, 76.3866
bev  AP:92.2661, 77.9833, 72.5815
3d   AP:92.2661, 77.9833, 72.5815
aos  AP:94.54, 81.04, 75.97
Cyclist AP_R40@0.50, 0.25, 0.25:
bbox AP:95.8197, 82.2335, 79.1187
bev  AP:93.5963, 78.1133, 74.7827
3d   AP:93.5963, 78.1133, 74.7827
aos  AP:95.60, 81.73, 78.62

2022-05-13 08:05:56,111   INFO  Result is save to /data/lixusheng_data/code/OpenPCDet/output/kitti_models/pdv/default/eval/eval_with_train/epoch_80/val
```