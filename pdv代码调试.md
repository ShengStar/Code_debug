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