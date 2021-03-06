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
(openpcdet5.0_PDV) lixusheng@cqu100:/data/lixusheng_data/code/OpenPCDet/tools$ CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file cfgs/kitti_models/pvd.yaml 
```
## 网络结果
```
2022-05-13 15:11:42,735   INFO  PDV(
  (vfe): MeanVFE()
  (backbone_3d): VoxelBackBone8x(
    (conv_input): SparseSequential(
      (0): SubMConv3d(4, 16, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
      (1): BatchNorm1d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (conv1): SparseSequential(
      (0): SparseSequential(
        (0): SubMConv3d(16, 16, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
        (1): BatchNorm1d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
      )
    )
    (conv2): SparseSequential(
      (0): SparseSequential(
        (0): SparseConv3d(16, 32, kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[1, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
        (1): BatchNorm1d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (1): SparseSequential(
        (0): SubMConv3d(32, 32, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
        (1): BatchNorm1d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (2): SparseSequential(
        (0): SubMConv3d(32, 32, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
        (1): BatchNorm1d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
      )
    )
    (conv3): SparseSequential(
      (0): SparseSequential(
        (0): SparseConv3d(32, 64, kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[1, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
        (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (1): SparseSequential(
        (0): SubMConv3d(64, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
        (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (2): SparseSequential(
        (0): SubMConv3d(64, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
        (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
      )
    )
    (conv4): SparseSequential(
      (0): SparseSequential(
        (0): SparseConv3d(64, 64, kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[0, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
        (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (1): SparseSequential(
        (0): SubMConv3d(64, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
        (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (2): SparseSequential(
        (0): SubMConv3d(64, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
        (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
      )
    )
    (conv_out): SparseSequential(
      (0): SparseConv3d(64, 128, kernel_size=[3, 1, 1], stride=[2, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
      (1): BatchNorm1d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
      (2): ReLU()
    )
  )
  (map_to_bev_module): HeightCompression()
  (pfe): None
  (backbone_2d): BaseBEVBackbone(
    (blocks): ModuleList(
      (0): Sequential(
        (0): ZeroPad2d(padding=(1, 1, 1, 1), value=0.0)
        (1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), bias=False)
        (2): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (3): ReLU()
        (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (5): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (6): ReLU()
        (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (8): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (9): ReLU()
        (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (11): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (12): ReLU()
        (13): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (14): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (15): ReLU()
        (16): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (17): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (18): ReLU()
      )
      (1): Sequential(
        (0): ZeroPad2d(padding=(1, 1, 1, 1), value=0.0)
        (1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), bias=False)
        (2): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (3): ReLU()
        (4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (5): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (6): ReLU()
        (7): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (8): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (9): ReLU()
        (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (11): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (12): ReLU()
        (13): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (14): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (15): ReLU()
        (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (17): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (18): ReLU()
      )
    )
    (deblocks): ModuleList(
      (0): Sequential(
        (0): ConvTranspose2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (1): Sequential(
        (0): ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
      )
    )
  )
  (dense_head): AnchorHeadSingle(
    (cls_loss_func): SigmoidFocalClassificationLoss()
    (reg_loss_func): WeightedSmoothL1Loss()
    (dir_loss_func): WeightedCrossEntropyLoss()
    (conv_cls): Conv2d(512, 18, kernel_size=(1, 1), stride=(1, 1))
    (conv_box): Conv2d(512, 42, kernel_size=(1, 1), stride=(1, 1))
    (conv_dir_cls): Conv2d(512, 12, kernel_size=(1, 1), stride=(1, 1))
  )
  (point_head): None
  (roi_head): PDVHead(
    (proposal_target_layer): ProposalTargetLayer()
    (reg_loss_func): WeightedSmoothL1Loss()
    (roi_grid_pool_layers): ModuleList(
      (0): StackSAModuleMSGAttention(
        (groupers): ModuleList(
          (0): QueryAndGroup()
          (1): QueryAndGroup()
        )
        (mlps): ModuleList(
          (0): Sequential(
            (0): Conv2d(67, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): ReLU()
          )
          (1): Sequential(
            (0): Conv2d(67, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): ReLU()
          )
        )
      )
      (1): StackSAModuleMSGAttention(
        (groupers): ModuleList(
          (0): QueryAndGroup()
          (1): QueryAndGroup()
        )
        (mlps): ModuleList(
          (0): Sequential(
            (0): Conv2d(67, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): ReLU()
          )
          (1): Sequential(
            (0): Conv2d(67, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): ReLU()
          )
        )
      )
    )
    (attention_head): TransformerEncoder(
      (pos_encoder): FeedForwardPositionalEncoding(
        (ffn): Sequential(
          (0): Conv1d(4, 64, kernel_size=(1,), stride=(1,))
          (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
        )
      )
      (transformer_encoder): TransformerEncoder(
        (layers): ModuleList(
          (0): TransformerEncoderLayer(
            (self_attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=128, out_features=128, bias=True)
            )
            (linear1): Linear(in_features=128, out_features=128, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
            (linear2): Linear(in_features=128, out_features=128, bias=True)
            (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            (dropout1): Dropout(p=0.1, inplace=False)
            (dropout2): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (shared_fc_layer): Sequential(
      (0): Conv1d(27648, 256, kernel_size=(1,), stride=(1,), bias=False)
      (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Dropout(p=0.3, inplace=False)
      (4): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
      (5): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (6): ReLU()
    )
    (reg_layers): Sequential(
      (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
      (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Dropout(p=0.3, inplace=False)
      (4): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
      (5): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (6): ReLU()
      (7): Conv1d(256, 7, kernel_size=(1,), stride=(1,))
    )
    (cls_layers): Sequential(
      (0): Conv1d(260, 256, kernel_size=(1,), stride=(1,), bias=False)
      (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Dropout(p=0.3, inplace=False)
      (4): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
      (5): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (6): ReLU()
      (7): Conv1d(256, 1, kernel_size=(1,), stride=(1,))
    )
  )
)
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
## 测试2
```
2022-05-14 13:20:36,402   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:96.7304, 89.4421, 89.1318
bev  AP:90.2677, 88.2294, 87.7435
3d   AP:89.4383, 84.1664, 78.8734
aos  AP:96.70, 89.35, 88.98
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:98.2857, 94.5353, 94.1318
bev  AP:95.3039, 91.1201, 88.8481
3d   AP:92.3738, 85.0654, 82.7358
aos  AP:98.26, 94.41, 93.93
Car AP@0.70, 0.50, 0.50:
bbox AP:96.7304, 89.4421, 89.1318
bev  AP:96.8486, 94.7730, 89.1864
3d   AP:96.7937, 94.6991, 89.1693
aos  AP:96.70, 89.35, 88.98
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:98.2857, 94.5353, 94.1318
bev  AP:98.3586, 96.4425, 94.4312
3d   AP:98.3364, 96.2662, 94.3755
aos  AP:98.26, 94.41, 93.93
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:73.2836, 69.1904, 65.2962
bev  AP:65.0850, 61.1961, 55.9766
3d   AP:62.9159, 57.7832, 53.3633
aos  AP:69.74, 65.60, 61.39
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:73.6035, 69.5114, 66.1823
bev  AP:66.1065, 60.2629, 55.6971
3d   AP:62.9055, 56.6971, 52.0048
aos  AP:69.70, 65.26, 61.54
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:73.2836, 69.1904, 65.2962
bev  AP:76.1332, 72.4370, 69.6141
3d   AP:76.1056, 72.3136, 69.3804
aos  AP:69.74, 65.60, 61.39
Pedestrian AP_R40@0.50, 0.25, 0.25:
bbox AP:73.6035, 69.5114, 66.1823
bev  AP:78.3697, 73.9876, 70.6371
3d   AP:78.3335, 73.2419, 69.8264
aos  AP:69.70, 65.26, 61.54
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:89.2387, 81.4472, 76.8593
bev  AP:86.8299, 74.0810, 71.1032
3d   AP:85.3121, 72.1950, 68.2177
aos  AP:88.86, 80.64, 76.09
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:94.8371, 82.6669, 78.0563
bev  AP:91.9951, 76.3828, 71.7110
3d   AP:90.1708, 73.2363, 68.6531
aos  AP:94.35, 81.81, 77.24
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:89.2387, 81.4472, 76.8593
bev  AP:94.0226, 77.4707, 72.9264
3d   AP:94.0226, 77.4690, 72.9264
aos  AP:88.86, 80.64, 76.09
Cyclist AP_R40@0.50, 0.25, 0.25:
bbox AP:94.8371, 82.6669, 78.0563
bev  AP:94.6474, 78.2960, 74.5813
3d   AP:94.6474, 78.2955, 74.5755
aos  AP:94.35, 81.81, 77.24

2022-05-14 13:20:36,409   INFO  Result is save to /data/lixusheng_data/code/OpenPCDet/output/kitti_models/pdv/default/eval/eval_with_train/epoch_80/val
2022-05-14 13:20:36,409   INFO  ****************Evaluation done.*****************
2022-05-14 13:20:36,439   INFO  Epoch 80 has been evaluated
2022-05-14 13:21:06,471   INFO  **********************End evaluation kitti_models/pdv(default)**********************pdv/default/ckpt 
(openpcdet5.0_PDV) lixusheng@cqu100:/data/lixusheng_data/code/OpenPCDet/tools$ CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file cfgs/kitti_models/pdv.yaml

```

(openpcdet5.0_votr) lixusheng@cqu100:~/openpcdet5.0_votr/OpenPCDet/tools$ CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file cfgs/kitti_models/pointrcnn.yaml
2022-05-18 15:54:06,161   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:90.6717, 89.3833, 88.4240
bev  AP:89.9991, 87.0576, 86.4686
3d   AP:88.5642, 78.3217, 76.9797
aos  AP:90.65, 89.26, 88.24
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:96.4133, 92.6085, 90.1417
bev  AP:93.2129, 88.9557, 86.6039
3d   AP:89.5141, 80.2759, 77.7003
aos  AP:96.38, 92.47, 89.95
Car AP@0.70, 0.50, 0.50:
bbox AP:90.6717, 89.3833, 88.4240
bev  AP:90.6347, 89.6764, 89.4309
3d   AP:90.6313, 89.6496, 89.3721
aos  AP:90.65, 89.26, 88.24
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:96.4133, 92.6085, 90.1417
bev  AP:96.4490, 94.9103, 92.6680
3d   AP:96.4260, 92.9338, 90.5961
aos  AP:96.38, 92.47, 89.95
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:74.2874, 66.4382, 62.4665
bev  AP:63.2133, 58.1533, 52.2576
3d   AP:61.1516, 53.3261, 49.0594
aos  AP:71.35, 63.43, 59.19
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:75.2913, 68.4328, 62.5538
bev  AP:64.5904, 57.1917, 50.3103
3d   AP:61.0417, 53.5803, 46.7310
aos  AP:71.92, 64.82, 58.86
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:74.2874, 66.4382, 62.4665
bev  AP:80.5983, 74.0432, 66.0714
3d   AP:80.4966, 73.8927, 65.9192
aos  AP:71.35, 63.43, 59.19
Pedestrian AP_R40@0.50, 0.25, 0.25:
bbox AP:75.2913, 68.4328, 62.5538
bev  AP:81.6379, 75.3202, 67.9296
3d   AP:81.4835, 75.0982, 67.6788
aos  AP:71.92, 64.82, 58.86
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:89.5010, 77.7783, 75.2158
bev  AP:87.5530, 74.3291, 71.7942
3d   AP:86.2862, 71.7099, 66.4056
aos  AP:89.41, 76.99, 74.38
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:94.8953, 79.5890, 75.0800
bev  AP:92.5773, 75.8636, 71.2651
3d   AP:91.0775, 72.5100, 68.2431
aos  AP:94.78, 78.66, 74.22
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:89.5010, 77.7783, 75.2158
bev  AP:93.9529, 75.7393, 73.0027
3d   AP:93.9529, 75.7393, 73.0027
aos  AP:89.41, 76.99, 74.38
Cyclist AP_R40@0.50, 0.25, 0.25:
bbox AP:94.8953, 79.5890, 75.0800
bev  AP:95.7304, 77.1977, 72.6598
3d   AP:95.7304, 77.1977, 72.6598
aos  AP:94.78, 78.66, 74.22

2022-05-18 15:54:06,169   INFO  Result is save to /home/lixusheng/openpcdet5.0_votr/OpenPCDet/output/kitti_models/pointrcnn/default/eval/eval_with_train/epoch_80/val
2022-05-18 15:54:06,169   INFO  ****************Evaluation done.*****************
2022-05-18 15:54:06,199   INFO  Epoch 80 has been evaluated
2022-05-18 15:54:36,230   INFO  **********************End evaluation kitti_models/pointrcnn(default)**********************s/pointrcnn/default/ckpt 
(openpcdet5.0_votr) lixusheng@cqu100:~/openpcdet5.0_votr/OpenPCDet/tools$ CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file cfgs/kitti_models/pointrcnn.yaml
