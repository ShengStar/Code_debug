# M3DETR: Multi-representation, Multi-scale, Mutual-relation 3D Object Detection with Transformers
## 运行代码
```
(M3DETR) lixusheng@cqu100:~/M3DETR/tools$ CUDA_VISIBLE_DEVICES=0 python train.py --cfg_file cfgs/m3detr_models/m3detr_kitti.yaml
```
## 结果
```
2022-05-09 11:24:25,970   INFO  **********************Start evaluation m3detr_models/m3detr_kitti(default)**********************
2022-05-09 11:24:25,972   INFO  Loading KITTI dataset
2022-05-09 11:24:26,112   INFO  Total samples for KITTI dataset: 3769
2022-05-09 11:24:26,115   INFO  ==> Loading parameters from checkpoint /home/lixusheng/M3DETR/output/m3detr_models/m3detr_kitti/default/ckpt/checkpoint_epoch_80.pth to GPU
2022-05-09 11:24:26,948   INFO  ==> Checkpoint trained from version: pcdet+0.5.2+cb76890
2022-05-09 11:24:26,973   INFO  ==> Done (loaded 467/467)
2022-05-09 11:24:26,978   INFO  *************** EPOCH 80 EVALUATION *****************
eval: 100%|???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????| 1885/1885 [14:11<00:00,  2.21it/s, recall_0.3=(16925, 16934) / 17558]
2022-05-09 11:38:38,266   INFO  *************** Performance of EPOCH 80 *****************
2022-05-09 11:38:38,266   INFO  Generate label finished(sec_per_example: 0.2259 second).
2022-05-09 11:38:38,266   INFO  recall_roi_0.3: 0.963948
2022-05-09 11:38:38,267   INFO  recall_rcnn_0.3: 0.964461
2022-05-09 11:38:38,267   INFO  recall_roi_0.5: 0.919011
2022-05-09 11:38:38,267   INFO  recall_rcnn_0.5: 0.923909
2022-05-09 11:38:38,267   INFO  recall_roi_0.7: 0.686866
2022-05-09 11:38:38,267   INFO  recall_rcnn_0.7: 0.731177
2022-05-09 11:38:38,276   INFO  Average predicted number of objects(3769 samples): 10.429
/home/lixusheng/anaconda3/envs/M3DETR/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (24) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/M3DETR/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (30) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/M3DETR/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (35) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/M3DETR/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (40) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/M3DETR/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (28) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/M3DETR/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (20) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/M3DETR/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (96) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/M3DETR/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (24) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/M3DETR/lib/python3.7/site-packages/numba/core/typed_passes.py:330: NumbaPerformanceWarning: 
The keyword argument 'parallel=True' was specified but no transformation for parallel execution was possible.

To find out why, try turning on parallel diagnostics, see https://numba.readthedocs.io/en/stable/user/parallel.html#diagnostics for help.

File "../pcdet/datasets/kitti/kitti_object_eval_python/eval.py", line 122:
@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
^

  state.func_ir.loc))
/home/lixusheng/anaconda3/envs/M3DETR/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (30) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/M3DETR/lib/python3.7/site-packages/numba/core/typed_passes.py:330: NumbaPerformanceWarning: 
The keyword argument 'parallel=True' was specified but no transformation for parallel execution was possible.

To find out why, try turning on parallel diagnostics, see https://numba.readthedocs.io/en/stable/user/parallel.html#diagnostics for help.

File "../pcdet/datasets/kitti/kitti_object_eval_python/eval.py", line 122:
@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
^

  state.func_ir.loc))
/home/lixusheng/anaconda3/envs/M3DETR/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (30) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/M3DETR/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (24) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/M3DETR/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (35) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/M3DETR/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (40) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/M3DETR/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (28) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/M3DETR/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (20) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/M3DETR/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (96) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
2022-05-09 11:39:04,660   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:96.6369, 89.4990, 89.1097
bev  AP:90.3411, 87.9629, 87.3776
3d   AP:89.4548, 79.2053, 78.5602
aos  AP:96.59, 89.38, 88.92
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:98.4649, 94.1111, 92.0862
bev  AP:95.1303, 88.8511, 88.4685
3d   AP:92.2960, 82.7666, 80.3554
aos  AP:98.43, 93.96, 91.87
Car AP@0.70, 0.50, 0.50:
bbox AP:96.6369, 89.4990, 89.1097
bev  AP:96.7647, 89.5055, 89.2055
3d   AP:96.7092, 89.4816, 89.1603
aos  AP:96.59, 89.38, 88.92
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:98.4649, 94.1111, 92.0862
bev  AP:98.5672, 94.3904, 94.1998
3d   AP:98.5429, 94.3697, 94.0639
aos  AP:98.43, 93.96, 91.87
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:73.0342, 68.1377, 64.2195
bev  AP:63.9842, 57.5389, 53.3977
3d   AP:61.3836, 53.9164, 49.3874
aos  AP:68.95, 63.73, 59.72
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:73.6382, 68.1438, 64.2452
bev  AP:64.3892, 57.0051, 52.2681
3d   AP:60.5789, 53.0158, 47.5268
aos  AP:69.44, 63.48, 59.39
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:73.0342, 68.1377, 64.2195
bev  AP:76.8250, 72.4712, 69.8558
3d   AP:76.6481, 72.2093, 69.6020
aos  AP:68.95, 63.73, 59.72
Pedestrian AP_R40@0.50, 0.25, 0.25:
bbox AP:73.6382, 68.1438, 64.2452
bev  AP:78.7600, 74.0312, 70.0152
3d   AP:78.5692, 73.2103, 69.7206
aos  AP:69.44, 63.48, 59.39
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:89.1658, 82.0904, 77.1801
bev  AP:86.6656, 74.8201, 71.7223
3d   AP:86.1444, 72.4500, 66.1977
aos  AP:88.94, 81.58, 76.66
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:94.5072, 83.5121, 78.8940
bev  AP:91.2616, 75.8342, 72.3113
3d   AP:88.7194, 72.8701, 68.1134
aos  AP:94.20, 82.98, 78.31
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:89.1658, 82.0904, 77.1801
bev  AP:87.3917, 78.9489, 74.0816
3d   AP:87.3917, 78.9469, 74.0816
aos  AP:88.94, 81.58, 76.66
Cyclist AP_R40@0.50, 0.25, 0.25:
bbox AP:94.5072, 83.5121, 78.8940
bev  AP:92.5023, 80.0692, 75.4621
3d   AP:92.4988, 80.0673, 75.4571
aos  AP:94.20, 82.98, 78.31

2022-05-09 11:39:04,667   INFO  Result is save to /home/lixusheng/M3DETR/output/m3detr_models/m3detr_kitti/default/eval/eval_with_train/epoch_80/val
2022-05-09 11:39:04,667   INFO  ****************Evaluation done.*****************
2022-05-09 11:39:04,694   INFO  Epoch 80 has been evaluated
2022-05-09 11:39:34,726   INFO  **********************End evaluation m3detr_models/m3detr_kitti(default)**********************kpt 
(M3DETR) lixusheng@cqu100:~/M3DETR/tools$ CUDA_VISIBLE_DEVICES=0 python train.py --cfg_file cfgs/m3detr_models/m3detr_kitti.yaml
```