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