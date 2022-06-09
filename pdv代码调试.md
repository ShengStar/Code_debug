# pvd
## 论文题目
Point Density-Aware Voxels for LiDAR 3D Object Detection
## 运行指令
```
(openpcdet5.0_PDV) lixusheng@cqu100:/data/lixusheng_data/code/my_openpcdet_pdv$ git add .
(openpcdet5.0_PDV) lixusheng@cqu100:/data/lixusheng_data/code/my_openpcdet_pdv$ git commit -m 'first'
(openpcdet5.0_PDV) lixusheng@cqu100:/data/lixusheng_data/code/my_openpcdet_pdv$ git push
(openpcdet5.0_PDV) lixusheng@cqu100:/data/lixusheng_data/code/my_openpcdet_pdv/tools$ CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file cfgs/kitti_models/pdv.yaml
```
## 结果
```
INFO  PDV(
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

2022-05-25 16:49:51,345   INFO  **********************Start evaluation kitti_models/pdv(default)**********************
2022-05-25 16:49:51,347   INFO  Loading KITTI dataset
2022-05-25 16:49:51,459   INFO  Total samples for KITTI dataset: 3769
2022-05-25 16:49:51,463   INFO  ==> Loading parameters from checkpoint /data/lixusheng_data/code/OpenPCDet/output/kitti_models/pdv/default/ckpt/checkpoint_epoch_80.pth to GPU
2022-05-25 16:49:51,566   INFO  ==> Checkpoint trained from version: pcdet+0.5.2+0000000
2022-05-25 16:49:51,578   INFO  ==> Done (loaded 272/272)
2022-05-25 16:49:51,583   INFO  *************** EPOCH 80 EVALUATION *****************
[07:42<00:00,  4.08it/s, recall_0.3=(17008, 17022) / 17558]
2022-05-25 16:57:33,982   INFO  *************** Performance of EPOCH 80 *****************
2022-05-25 16:57:33,982   INFO  Generate label finished(sec_per_example: 0.1227 second).
2022-05-25 16:57:33,983   INFO  recall_roi_0.3: 0.968675
2022-05-25 16:57:33,983   INFO  recall_rcnn_0.3: 0.969473
2022-05-25 16:57:33,983   INFO  recall_roi_0.5: 0.930345
2022-05-25 16:57:33,983   INFO  recall_rcnn_0.5: 0.936041
2022-05-25 16:57:33,983   INFO  recall_roi_0.7: 0.707313
2022-05-25 16:57:33,983   INFO  recall_rcnn_0.7: 0.755211
2022-05-25 16:57:33,993   INFO  Average predicted number of objects(3769 samples): 8.735
2022-05-25 16:58:00,163   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:96.8272, 89.4992, 89.1533
bev  AP:90.2729, 88.2753, 87.9204
3d   AP:89.3587, 83.9602, 78.8998
aos  AP:96.78, 89.38, 88.99
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:98.6342, 94.5931, 94.2871
bev  AP:95.3405, 91.1329, 90.7555
3d   AP:92.3043, 85.0354, 82.8172
aos  AP:98.59, 94.44, 94.06
Car AP@0.70, 0.50, 0.50:
bbox AP:96.8272, 89.4992, 89.1533
bev  AP:96.9475, 94.9201, 89.1938
3d   AP:96.8841, 94.8416, 89.1561
aos  AP:96.78, 89.38, 88.99
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:98.6342, 94.5931, 94.2871
bev  AP:98.6727, 96.5809, 94.5598
3d   AP:98.6469, 96.3990, 94.4879
aos  AP:98.59, 94.44, 94.06
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:72.1774, 67.6548, 63.7831
bev  AP:67.8096, 61.1898, 57.4909
3d   AP:64.9783, 58.8648, 54.0819
aos  AP:68.86, 64.34, 60.23
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:72.5472, 67.9253, 64.7578
bev  AP:67.7511, 60.8056, 56.3375
3d   AP:65.7917, 58.5709, 53.6483
aos  AP:68.89, 64.14, 60.65
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:72.1774, 67.6548, 63.7831
bev  AP:75.9336, 71.4719, 68.9710
3d   AP:75.8079, 71.3072, 68.7138
aos  AP:68.86, 64.34, 60.23
Pedestrian AP_R40@0.50, 0.25, 0.25:
bbox AP:72.5472, 67.9253, 64.7578
bev  AP:78.0423, 73.2318, 69.7440
3d   AP:77.9437, 73.0181, 69.4717
aos  AP:68.89, 64.14, 60.65
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:93.4621, 81.2432, 75.9322
bev  AP:91.1878, 74.6631, 71.9751
3d   AP:90.2364, 73.0873, 70.2006
aos  AP:93.29, 80.83, 75.54
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:96.0017, 81.8585, 77.5488
bev  AP:93.7790, 77.0450, 72.4699
3d   AP:92.4321, 74.3451, 70.7631
aos  AP:95.82, 81.43, 77.08
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:93.4621, 81.2432, 75.9322
bev  AP:92.0366, 78.2381, 73.0338
3d   AP:92.0366, 78.2381, 73.0338
aos  AP:93.29, 80.83, 75.54
Cyclist AP_R40@0.50, 0.25, 0.25:
bbox AP:96.0017, 81.8585, 77.5488
bev  AP:94.6752, 79.0186, 75.4593
3d   AP:94.6752, 79.0186, 75.4593
aos  AP:95.82, 81.43, 77.08
```
### 代码版本 1ac1bba
```
2022-05-27 19:14:29,791   INFO  Database filter by min points Car: 14357 => 13532
2022-05-27 19:14:29,791   INFO  Database filter by min points Pedestrian: 2207 => 2168
2022-05-27 19:14:29,792   INFO  Database filter by min points Cyclist: 734 => 705
2022-05-27 19:14:29,808   INFO  Database filter by difficulty Car: 13532 => 10759
2022-05-27 19:14:29,810   INFO  Database filter by difficulty Pedestrian: 2168 => 2075
2022-05-27 19:14:29,811   INFO  Database filter by difficulty Cyclist: 705 => 581
2022-05-27 19:14:29,816   INFO  Loading KITTI dataset
2022-05-27 19:14:29,906   INFO  Total samples for KITTI dataset: 3712
2022-05-27 19:14:35,396   INFO  PDV(
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
    (mhead_attention): MultiheadAttention(
      (out_proj): NonDynamicallyQuantizableLinear(in_features=16, out_features=16, bias=True)
    )
    (mhead_attention_1): MultiheadAttention(
      (out_proj): NonDynamicallyQuantizableLinear(in_features=16, out_features=16, bias=True)
    )
    (mhead_attention_2): MultiheadAttention(
      (out_proj): NonDynamicallyQuantizableLinear(in_features=32, out_features=32, bias=True)
    )
    (mhead_attention_3): MultiheadAttention(
      (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
    )
    (mhead_attention_4): MultiheadAttention(
      (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
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
2022-05-28 15:04:21,117   INFO  *************** Performance of EPOCH 80 *****************
2022-05-28 15:04:21,118   INFO  Generate label finished(sec_per_example: 0.0461 second).
2022-05-28 15:04:21,118   INFO  recall_roi_0.3: 0.971181
2022-05-28 15:04:21,118   INFO  recall_rcnn_0.3: 0.971010
2022-05-28 15:04:21,118   INFO  recall_roi_0.5: 0.930915
2022-05-28 15:04:21,118   INFO  recall_rcnn_0.5: 0.937237
2022-05-28 15:04:21,118   INFO  recall_roi_0.7: 0.705718
2022-05-28 15:04:21,118   INFO  recall_rcnn_0.7: 0.756578
2022-05-28 15:04:21,122   INFO  Average predicted number of objects(3769 samples): 9.018
2022-05-28 15:04:46,474   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:95.6473, 89.4248, 89.1445
bev  AP:90.2837, 88.0228, 87.6013
3d   AP:89.3022, 83.8229, 78.6633
aos  AP:95.62, 89.35, 89.01
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:97.9410, 94.4362, 94.0425
bev  AP:93.0686, 90.7746, 88.6851
3d   AP:92.1170, 84.6471, 82.4796
aos  AP:97.92, 94.33, 93.85
Car AP@0.70, 0.50, 0.50:
bbox AP:95.6473, 89.4248, 89.1445
bev  AP:95.7209, 94.5095, 89.1575
3d   AP:95.6633, 89.3927, 89.1220
aos  AP:95.62, 89.35, 89.01
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:97.9410, 94.4362, 94.0425
bev  AP:98.0256, 96.2187, 94.3542
3d   AP:98.0003, 94.4328, 94.2589
aos  AP:97.92, 94.33, 93.85
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:73.1677, 67.9692, 64.6311
bev  AP:64.6643, 59.4457, 56.6050
3d   AP:63.0729, 56.6303, 52.6939
aos  AP:69.04, 64.02, 60.30
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:73.3783, 67.9806, 65.2028
bev  AP:65.4125, 58.5621, 55.0566
3d   AP:63.4306, 55.8352, 51.4405
aos  AP:68.97, 63.60, 60.34
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:73.1677, 67.9692, 64.6311
bev  AP:75.5497, 71.3589, 69.2866
3d   AP:75.4522, 71.1014, 69.0224
aos  AP:69.04, 64.02, 60.30
Pedestrian AP_R40@0.50, 0.25, 0.25:
bbox AP:73.3783, 67.9806, 65.2028
bev  AP:77.6927, 72.8872, 70.0431
3d   AP:77.5650, 72.5544, 69.7687
aos  AP:68.97, 63.60, 60.34
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:88.9479, 81.6623, 76.8897
bev  AP:87.8420, 78.0168, 72.9893
3d   AP:85.7969, 72.9674, 70.1174
aos  AP:88.86, 81.39, 76.61
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:94.4342, 82.7778, 79.6350
bev  AP:92.9725, 78.6229, 73.9851
3d   AP:90.6334, 75.0230, 70.4760
aos  AP:94.32, 82.48, 79.31
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:88.9479, 81.6623, 76.8897
bev  AP:88.1833, 78.5574, 73.4536
3d   AP:88.1833, 78.5574, 73.4536
aos  AP:88.86, 81.39, 76.61
Cyclist AP_R40@0.50, 0.25, 0.25:
bbox AP:94.4342, 82.7778, 79.6350
bev  AP:93.3704, 80.0999, 75.7741
3d   AP:93.3704, 80.0999, 75.7741
aos  AP:94.32, 82.48, 79.31

2022-05-28 15:04:46,481   INFO  Result is save to /data/lixusheng_data/code/my_openpcdet_pdv/output/kitti_models/pdv/default/eval/eval_with_train/epoch_80/val
2022-05-28 15:04:46,481   INFO  ****************Evaluation done.*****************
2022-05-28 15:04:46,516   INFO  Epoch 80 has been evaluated
2022-05-28 15:05:16,548   INFO  **********************End evaluation kitti_models/pdv(default)**********************models/pdv/default/ckpt 
```

### 实验 9fe90c0
```
2022-05-30 12:43:41,973   INFO  ==> Loading parameters from checkpoint /data/lixusheng_data/code/my_openpcdet_pdv/output/kitti_models/pdv/default/ckpt/checkpoint_epoch_69.pth to GPU
2022-05-30 12:43:42,749   INFO  ==> Loading optimizer parameters from checkpoint /data/lixusheng_data/code/my_openpcdet_pdv/output/kitti_models/pdv/default/ckpt/checkpoint_epoch_69.pth to GPU
==> Checkpoint trained from version: pcdet+0.5.2+4d515fc+py9fe90c0
2022-05-30 12:43:42,763   INFO  ==> Done
2022-05-30 12:43:42,764   INFO  PDV(
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
    (mhead_attention): MultiheadAttention(
      (out_proj): NonDynamicallyQuantizableLinear(in_features=16, out_features=16, bias=True)
    )
    (mhead_attention_1): MultiheadAttention(
      (out_proj): NonDynamicallyQuantizableLinear(in_features=16, out_features=16, bias=True)
    )
    (mhead_attention_2): MultiheadAttention(
      (out_proj): NonDynamicallyQuantizableLinear(in_features=32, out_features=32, bias=True)
    )
    (mhead_attention_3): MultiheadAttention(
      (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
    )
    (mhead_attention_4): MultiheadAttention(
      (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
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
2022-05-30 12:43:42,767   INFO  **********************Start training kitti_models/pdv(default)**********************
epochs:   0%|                                                                                                                                                                                                             | 0/11 [00:00<?, ?it/s../pcdet/utils/voxel_aggregation_utils.py:160: UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').
  voxel_idxs[:, 1:] = centroid_voxel_idxs_first[:, 1:] // grid_scaling
epochs:  18%|????????????????????                                                                                          | 2/11 [1:23:30<4:52:33, 1950.35s/it, loss=0.75, lr=0.000742, d_time=0.00(0.00), f_time=0.63(0.61), b_time=1.04(1.05)]
train:  57%|?????????????????????????????????????????????????????????????????????????????????????????????????????                                                                           | 1057/1856 [18:29<13:56,  1.05s/it, total_it=132832]epochs:  18%|????????????????????                                                                                         | 2/11 [1:23:31<4:52:33, 1950.35s/it, loss=0.956, lr=0.000742, d_time=0.00(0.00), f_time=0.6epochs:  18%|????????????????????                                                                                         | 2/11 [1:23:31<4:52:33, 1950.35s/it, loss=0.956, lr=0.000742, d_time=0.00(0.00), f_time=0.62(0.61), b_time=1.06(1.05)]
train:  57%|????????????????????????                  | 1058/1856 [18:30<13:53,  1.04s/it, total_it=132833]epochs:  18%|?| 2/11 [1:23:32<4:52:33, 1950.35s/it, loss=0.904, lr=0.000742, d_time=0.00(0.00), f_time=0.62                                                                                                           train: 100%|????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????| 1856/1856 [32:34<00:00,  1.05s/it, total_it=148480]
epochs: 100%|??????????????????????????????????????????????????????????????????????????????????????????????????????????????????| 11/11 [5:57:54<00:00, 1952.27s/it, loss=0.778, lr=1e-7, d_time=0.00(0.00), f_time=0.63(0.62), b_time=1.08(1.05)]
2022-05-30 18:41:37,862   INFO  **********************End training kitti_models/pdv(default)**********************



2022-05-30 18:41:37,862   INFO  **********************Start evaluation kitti_models/pdv(default)**********************
2022-05-30 18:41:37,864   INFO  Loading KITTI dataset
2022-05-30 18:41:37,980   INFO  Total samples for KITTI dataset: 3769
2022-05-30 18:41:37,982   INFO  ==> Loading parameters from checkpoint /data/lixusheng_data/code/my_openpcdet_pdv/output/kitti_models/pdv/default/ckpt/checkpoint_epoch_80.pth to GPU
2022-05-30 18:41:38,648   INFO  ==> Checkpoint trained from version: pcdet+0.5.2+4d515fc+py9fe90c0
2022-05-30 18:41:38,667   INFO  ==> Done (loaded 292/292)
2022-05-30 18:41:38,672   INFO  *************** EPOCH 80 EVALUATION *****************
eval: 100%|???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????| 1885/1885 [08:46<00:00,  3.58it/s, recall_0.3=(17064, 17071) / 17558]
2022-05-30 18:50:24,718   INFO  *************** Performance of EPOCH 80 *****************
2022-05-30 18:50:24,718   INFO  Generate label finished(sec_per_example: 0.1396 second).
2022-05-30 18:50:24,718   INFO  recall_roi_0.3: 0.971865
2022-05-30 18:50:24,718   INFO  recall_rcnn_0.3: 0.972263
2022-05-30 18:50:24,718   INFO  recall_roi_0.5: 0.932395
2022-05-30 18:50:24,718   INFO  recall_rcnn_0.5: 0.939572
2022-05-30 18:50:24,718   INFO  recall_roi_0.7: 0.705832
2022-05-30 18:50:24,718   INFO  recall_rcnn_0.7: 0.759711
2022-05-30 18:50:24,724   INFO  Average predicted number of objects(3769 samples): 8.884
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (20) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (25) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (30) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (24) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (35) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (80) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (20) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/core/typed_passes.py:330: NumbaPerformanceWarning: 
The keyword argument 'parallel=True' was specified but no transformation for parallel execution was possible.

To find out why, try turning on parallel diagnostics, see https://numba.readthedocs.io/en/stable/user/parallel.html#diagnostics for help.

File "../pcdet/datasets/kitti/kitti_object_eval_python/eval.py", line 122:
@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
^

  state.func_ir.loc))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (25) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (30) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (20) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (24) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (35) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/core/typed_passes.py:330: NumbaPerformanceWarning: 
The keyword argument 'parallel=True' was specified but no transformation for parallel execution was possible.

To find out why, try turning on parallel diagnostics, see https://numba.readthedocs.io/en/stable/user/parallel.html#diagnostics for help.

File "../pcdet/datasets/kitti/kitti_object_eval_python/eval.py", line 122:
@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
^

  state.func_ir.loc))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (20) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (30) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (25) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (24) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (35) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (80) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
2022-05-30 18:50:56,930   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:96.5165, 89.4871, 89.2011
bev  AP:90.2632, 88.1777, 87.7644
3d   AP:89.4386, 83.7297, 78.8167
aos  AP:96.48, 89.42, 89.07
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:98.4554, 94.4993, 94.2269
bev  AP:95.2913, 90.9875, 90.4830
3d   AP:92.3293, 84.8155, 82.6887
aos  AP:98.43, 94.41, 94.05
Car AP@0.70, 0.50, 0.50:
bbox AP:96.5165, 89.4871, 89.2011
bev  AP:96.7273, 94.6095, 89.2244
3d   AP:96.6866, 89.4436, 89.1873
aos  AP:96.48, 89.42, 89.07
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:98.4554, 94.4993, 94.2269
bev  AP:98.4749, 96.3786, 94.4780
3d   AP:98.4582, 94.5434, 94.3900
aos  AP:98.43, 94.41, 94.05
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:72.8012, 68.3039, 63.9320
bev  AP:65.5870, 60.3041, 56.9415
3d   AP:64.6688, 58.1422, 53.7247
aos  AP:69.07, 64.37, 59.93
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:73.3623, 67.9864, 64.0062
bev  AP:66.5688, 59.5265, 55.4225
3d   AP:65.1383, 57.4090, 52.4930
aos  AP:69.09, 63.55, 59.38
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:72.8012, 68.3039, 63.9320
bev  AP:76.0312, 71.8945, 68.6141
3d   AP:76.0287, 71.8037, 68.4498
aos  AP:69.07, 64.37, 59.93
Pedestrian AP_R40@0.50, 0.25, 0.25:
bbox AP:73.3623, 67.9864, 64.0062
bev  AP:78.1523, 72.5711, 68.7673
3d   AP:78.1487, 72.4743, 68.6617
aos  AP:69.09, 63.55, 59.38
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:88.6680, 81.4276, 76.1801
bev  AP:87.0362, 74.5650, 71.7635
3d   AP:86.3168, 73.1022, 70.2063
aos  AP:88.21, 80.83, 75.66
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:93.8286, 82.1699, 78.9474
bev  AP:91.8610, 76.6100, 73.1540
3d   AP:91.0288, 74.0348, 70.6241
aos  AP:93.30, 81.56, 78.30
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:88.6680, 81.4276, 76.1801
bev  AP:87.2451, 77.8930, 72.6143
3d   AP:87.2451, 77.8930, 72.6143
aos  AP:88.21, 80.83, 75.66
Cyclist AP_R40@0.50, 0.25, 0.25:
bbox AP:93.8286, 82.1699, 78.9474
bev  AP:92.1869, 78.3698, 75.0324
3d   AP:92.1869, 78.3698, 75.0324
aos  AP:93.30, 81.56, 78.30

2022-05-30 18:50:56,938   INFO  Result is save to /data/lixusheng_data/code/my_openpcdet_pdv/output/kitti_models/pdv/default/eval/eval_with_train/epoch_80/val
2022-05-30 18:50:56,938   INFO  ****************Evaluation done.*****************
2022-05-30 18:50:56,966   INFO  Epoch 80 has been evaluated
2022-05-30 18:51:26,997   INFO  **********************End evaluation kitti_models/pdv(default)**********************models/pdv/default/ckpt 

```
### 实验 76e5ec7+norm
```
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/torch/functional.py:445: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1639180594101/work/aten/src/ATen/native/TensorShape.cpp:2157.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
2022-05-28 18:36:15,417   INFO  PDV(
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
    (mhead_attention_1): MultiheadAttention(
      (out_proj): NonDynamicallyQuantizableLinear(in_features=16, out_features=16, bias=True)
    )
    (mhead_attention_2): MultiheadAttention(
      (out_proj): NonDynamicallyQuantizableLinear(in_features=32, out_features=32, bias=True)
    )
    (mhead_attention_3): MultiheadAttention(
      (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
    )
    (mhead_attention_4): MultiheadAttention(
      (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
    )
    (norm1): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (norm2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (norm3): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (norm4): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
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
2022-05-28 18:36:15,420   INFO  **********************Start training kitti_models/pdv_1(default)**********************
epochs:   0%|                                                                                                                                                                                                             | 0/80 [00:00<?, ?it/s../pcdet/utils/voxel_aggregation_utils.py:160: UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').
  voxel_idxs[:, 1:] = centroid_voxel_idxs_first[:, 1:] // grid_scaling
train: 100%|????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????| 1856/1856 [15:05<00:00,  2.05it/s, total_it=148480]
epochs: 100%|?????????????????????????????????????????????????????????????????????????????????????????????????????????????????| 80/80 [49:39:47<00:00, 2234.85s/it, loss=0.775, lr=1e-7, d_time=0.00(0.00), f_time=0.18(0.19), b_time=0.48(0.49)]
2022-05-30 20:16:03,340   INFO  **********************End training kitti_models/pdv_1(default)**********************



2022-05-30 20:16:03,353   INFO  **********************Start evaluation kitti_models/pdv_1(default)**********************
2022-05-30 20:16:03,354   INFO  Loading KITTI dataset
2022-05-30 20:16:03,459   INFO  Total samples for KITTI dataset: 3769
2022-05-30 20:16:03,461   INFO  ==> Loading parameters from checkpoint /data/lixusheng_data/code/my_openpcdet_pdv/output/kitti_models/pdv_1/default/ckpt/checkpoint_epoch_80.pth to GPU
2022-05-30 20:16:03,576   INFO  ==> Checkpoint trained from version: pcdet+0.5.2+4d515fc+py76e5ec7
2022-05-30 20:16:03,589   INFO  ==> Done (loaded 308/308)
2022-05-30 20:16:03,592   INFO  *************** EPOCH 80 EVALUATION *****************
eval: 100%|???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????| 1885/1885 [02:54<00:00, 10.79it/s, recall_0.3=(17016, 17023) / 17558]
2022-05-30 20:18:58,359   INFO  *************** Performance of EPOCH 80 *****************
2022-05-30 20:18:58,360   INFO  Generate label finished(sec_per_example: 0.0464 second).
2022-05-30 20:18:58,360   INFO  recall_roi_0.3: 0.969131
2022-05-30 20:18:58,360   INFO  recall_rcnn_0.3: 0.969530
2022-05-30 20:18:58,360   INFO  recall_roi_0.5: 0.931256
2022-05-30 20:18:58,360   INFO  recall_rcnn_0.5: 0.935414
2022-05-30 20:18:58,360   INFO  recall_roi_0.7: 0.704807
2022-05-30 20:18:58,360   INFO  recall_rcnn_0.7: 0.757831
2022-05-30 20:18:58,366   INFO  Average predicted number of objects(3769 samples): 8.795
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (20) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (25) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (30) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (35) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (24) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (80) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (20) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/core/typed_passes.py:330: NumbaPerformanceWarning: 
The keyword argument 'parallel=True' was specified but no transformation for parallel execution was possible.

To find out why, try turning on parallel diagnostics, see https://numba.readthedocs.io/en/stable/user/parallel.html#diagnostics for help.

File "../pcdet/datasets/kitti/kitti_object_eval_python/eval.py", line 122:
@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
^

  state.func_ir.loc))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (25) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/core/typed_passes.py:330: NumbaPerformanceWarning: 
The keyword argument 'parallel=True' was specified but no transformation for parallel execution was possible.

To find out why, try turning on parallel diagnostics, see https://numba.readthedocs.io/en/stable/user/parallel.html#diagnostics for help.

File "../pcdet/datasets/kitti/kitti_object_eval_python/eval.py", line 122:
@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
^

  state.func_ir.loc))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (30) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (20) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (25) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (35) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (24) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_PDV/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (80) < 2 * SM count (112) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
2022-05-30 20:19:23,579   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:95.8993, 89.4571, 89.1571
bev  AP:90.1724, 87.9500, 87.5444
3d   AP:89.3819, 83.7675, 78.7863
aos  AP:95.88, 89.38, 89.02
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:98.1584, 94.5190, 94.1023
bev  AP:93.0703, 90.7407, 88.6773
3d   AP:92.0263, 84.7986, 82.6515
aos  AP:98.14, 94.41, 93.91
Car AP@0.70, 0.50, 0.50:
bbox AP:95.8993, 89.4571, 89.1571
bev  AP:95.9292, 94.5274, 89.1797
3d   AP:95.8689, 89.4383, 89.1575
aos  AP:95.88, 89.38, 89.02
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:98.1584, 94.5190, 94.1023
bev  AP:98.1543, 96.1743, 94.3885
3d   AP:98.1299, 94.5454, 94.3450
aos  AP:98.14, 94.41, 93.91
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:74.3676, 69.3915, 65.7445
bev  AP:65.6072, 60.5234, 55.7278
3d   AP:64.1036, 57.8504, 53.4424
aos  AP:70.35, 65.21, 61.57
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:74.9136, 69.6329, 66.7613
bev  AP:66.4927, 59.8443, 55.7624
3d   AP:63.7926, 56.8959, 52.4356
aos  AP:70.55, 65.07, 61.97
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:74.3676, 69.3915, 65.7445
bev  AP:77.1227, 73.1057, 70.9240
3d   AP:77.0966, 72.9549, 70.7947
aos  AP:70.35, 65.21, 61.57
Pedestrian AP_R40@0.50, 0.25, 0.25:
bbox AP:74.9136, 69.6329, 66.7613
bev  AP:79.4051, 74.6398, 71.7834
3d   AP:79.3703, 73.9182, 71.0240
aos  AP:70.55, 65.07, 61.97
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:92.6389, 82.0525, 76.5671
bev  AP:89.4619, 73.6579, 71.1361
3d   AP:91.0623, 71.7930, 68.0179
aos  AP:92.51, 81.59, 76.19
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:95.5714, 83.9342, 79.4228
bev  AP:92.1251, 75.9227, 72.0158
3d   AP:91.5107, 72.5864, 68.0922
aos  AP:95.45, 83.47, 78.98
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:92.6389, 82.0525, 76.5671
bev  AP:90.8385, 78.8203, 75.9479
3d   AP:90.8334, 78.8163, 75.9420
aos  AP:92.51, 81.59, 76.19
Cyclist AP_R40@0.50, 0.25, 0.25:
bbox AP:95.5714, 83.9342, 79.4228
bev  AP:93.6711, 79.7740, 76.0957
3d   AP:93.6696, 79.7723, 76.0713
aos  AP:95.45, 83.47, 78.98

2022-05-30 20:19:23,587   INFO  Result is save to /data/lixusheng_data/code/my_openpcdet_pdv/output/kitti_models/pdv_1/default/eval/eval_with_train/epoch_80/val
2022-05-30 20:19:23,587   INFO  ****************Evaluation done.*****************
2022-05-30 20:19:23,617   INFO  Epoch 80 has been evaluated
2022-05-30 20:19:53,648   INFO  **********************End evaluation kitti_models/pdv_1(default)**********************dels/pdv_1/default/ckpt 
(openpcdet5.0_PDV) lixusheng@cqu100:/data/lixusheng_data/code/my_openpcdet_pdv/tools$ 
```