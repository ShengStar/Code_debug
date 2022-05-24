# 代码修改历史
## 修改一
```
(openpcdet5.0_votr) lixusheng@cqu100:~/openpcdet5.0_votr/OpenPCDet/tools$ CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file cfgs/kitti_models/votr_tsd.yaml
```
### 网络结构
```
2022-05-24 10:27:48,679   INFO  VoTrRCNN(
  (vfe): MeanVFE()
  (backbone_3d): VoxelTransformerV3(
    (input_transform): Sequential(
      (0): Linear(in_features=4, out_features=16, bias=True)
      (1): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (backbone): ModuleList(
      (0): AttentionResBlock(
        (sp_attention): SparseAttention3d(
          (mhead_attention): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=16, out_features=16, bias=True)
          )
          (drop_out): Dropout(p=0, inplace=False)
          (linear1): Linear(in_features=16, out_features=32, bias=True)
          (linear2): Linear(in_features=32, out_features=16, bias=True)
          (dropout1): Dropout(p=0, inplace=False)
          (dropout2): Dropout(p=0, inplace=False)
          (activation): ReLU()
          (output_layer): Sequential(
            (0): Linear(in_features=16, out_features=32, bias=True)
            (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
          )
          (norm): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (k_pos_proj): Sequential(
            (0): Conv1d(3, 16, kernel_size=(1,), stride=(1,))
            (1): ReLU()
          )
        )
        (subm_attention_modules): ModuleList(
          (0): SubMAttention3d(
            (mhead_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=32, out_features=32, bias=True)
            )
            (drop_out): Dropout(p=0, inplace=False)
            (linear1): Linear(in_features=32, out_features=32, bias=True)
            (linear2): Linear(in_features=32, out_features=32, bias=True)
            (dropout1): Dropout(p=0, inplace=False)
            (dropout2): Dropout(p=0, inplace=False)
            (activation): ReLU()
            (output_layer): Sequential(
              (0): Linear(in_features=32, out_features=32, bias=True)
              (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU()
            )
            (norm1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (norm2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (k_pos_proj): Sequential(
              (0): Conv1d(3, 32, kernel_size=(1,), stride=(1,))
              (1): ReLU()
            )
          )
          (1): SubMAttention3d(
            (mhead_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=32, out_features=32, bias=True)
            )
            (drop_out): Dropout(p=0, inplace=False)
            (linear1): Linear(in_features=32, out_features=32, bias=True)
            (linear2): Linear(in_features=32, out_features=32, bias=True)
            (dropout1): Dropout(p=0, inplace=False)
            (dropout2): Dropout(p=0, inplace=False)
            (activation): ReLU()
            (output_layer): Sequential(
              (0): Linear(in_features=32, out_features=32, bias=True)
              (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU()
            )
            (norm1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (norm2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (k_pos_proj): Sequential(
              (0): Conv1d(3, 32, kernel_size=(1,), stride=(1,))
              (1): ReLU()
            )
          )
        )
      )
      (1): AttentionResBlock(
        (sp_attention): SparseAttention3d(
          (mhead_attention): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=32, out_features=32, bias=True)
          )
          (drop_out): Dropout(p=0, inplace=False)
          (linear1): Linear(in_features=32, out_features=64, bias=True)
          (linear2): Linear(in_features=64, out_features=32, bias=True)
          (dropout1): Dropout(p=0, inplace=False)
          (dropout2): Dropout(p=0, inplace=False)
          (activation): ReLU()
          (output_layer): Sequential(
            (0): Linear(in_features=32, out_features=64, bias=True)
            (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
          )
          (norm): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (k_pos_proj): Sequential(
            (0): Conv1d(3, 32, kernel_size=(1,), stride=(1,))
            (1): ReLU()
          )
        )
        (subm_attention_modules): ModuleList(
          (0): SubMAttention3d(
            (mhead_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
            )
            (drop_out): Dropout(p=0, inplace=False)
            (linear1): Linear(in_features=64, out_features=64, bias=True)
            (linear2): Linear(in_features=64, out_features=64, bias=True)
            (dropout1): Dropout(p=0, inplace=False)
            (dropout2): Dropout(p=0, inplace=False)
            (activation): ReLU()
            (output_layer): Sequential(
              (0): Linear(in_features=64, out_features=64, bias=True)
              (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU()
            )
            (norm1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (norm2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (k_pos_proj): Sequential(
              (0): Conv1d(3, 64, kernel_size=(1,), stride=(1,))
              (1): ReLU()
            )
          )
          (1): SubMAttention3d(
            (mhead_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
            )
            (drop_out): Dropout(p=0, inplace=False)
            (linear1): Linear(in_features=64, out_features=64, bias=True)
            (linear2): Linear(in_features=64, out_features=64, bias=True)
            (dropout1): Dropout(p=0, inplace=False)
            (dropout2): Dropout(p=0, inplace=False)
            (activation): ReLU()
            (output_layer): Sequential(
              (0): Linear(in_features=64, out_features=64, bias=True)
              (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU()
            )
            (norm1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (norm2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (k_pos_proj): Sequential(
              (0): Conv1d(3, 64, kernel_size=(1,), stride=(1,))
              (1): ReLU()
            )
          )
        )
      )
      (2): AttentionResBlock(
        (sp_attention): SparseAttention3d(
          (mhead_attention): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
          )
          (drop_out): Dropout(p=0, inplace=False)
          (linear1): Linear(in_features=64, out_features=64, bias=True)
          (linear2): Linear(in_features=64, out_features=64, bias=True)
          (dropout1): Dropout(p=0, inplace=False)
          (dropout2): Dropout(p=0, inplace=False)
          (activation): ReLU()
          (output_layer): Sequential(
            (0): Linear(in_features=64, out_features=64, bias=True)
            (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
          )
          (norm): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (k_pos_proj): Sequential(
            (0): Conv1d(3, 64, kernel_size=(1,), stride=(1,))
            (1): ReLU()
          )
        )
        (subm_attention_modules): ModuleList(
          (0): SubMAttention3d(
            (mhead_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
            )
            (drop_out): Dropout(p=0, inplace=False)
            (linear1): Linear(in_features=64, out_features=64, bias=True)
            (linear2): Linear(in_features=64, out_features=64, bias=True)
            (dropout1): Dropout(p=0, inplace=False)
            (dropout2): Dropout(p=0, inplace=False)
            (activation): ReLU()
            (output_layer): Sequential(
              (0): Linear(in_features=64, out_features=64, bias=True)
              (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU()
            )
            (norm1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (norm2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (k_pos_proj): Sequential(
              (0): Conv1d(3, 64, kernel_size=(1,), stride=(1,))
              (1): ReLU()
            )
          )
          (1): SubMAttention3d(
            (mhead_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
            )
            (drop_out): Dropout(p=0, inplace=False)
            (linear1): Linear(in_features=64, out_features=64, bias=True)
            (linear2): Linear(in_features=64, out_features=64, bias=True)
            (dropout1): Dropout(p=0, inplace=False)
            (dropout2): Dropout(p=0, inplace=False)
            (activation): ReLU()
            (output_layer): Sequential(
              (0): Linear(in_features=64, out_features=64, bias=True)
              (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU()
            )
            (norm1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (norm2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (k_pos_proj): Sequential(
              (0): Conv1d(3, 64, kernel_size=(1,), stride=(1,))
              (1): ReLU()
            )
          )
        )
      )
    )
  )
  (map_to_bev_module): HeightCompression()
  (pfe): VoxelSetAbstraction(
    (SA_layers): ModuleList(
      (0): StackSAModuleMSG(
        (groupers): ModuleList(
          (0): QueryAndGroup()
          (1): QueryAndGroup()
        )
        (mlps): ModuleList(
          (0): Sequential(
            (0): Conv2d(19, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): ReLU()
          )
          (1): Sequential(
            (0): Conv2d(19, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): ReLU()
          )
        )
      )
      (1): StackSAModuleMSG(
        (groupers): ModuleList(
          (0): QueryAndGroup()
          (1): QueryAndGroup()
        )
        (mlps): ModuleList(
          (0): Sequential(
            (0): Conv2d(35, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): ReLU()
          )
          (1): Sequential(
            (0): Conv2d(35, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): ReLU()
          )
        )
      )
      (2): StackSAModuleMSG(
        (groupers): ModuleList(
          (0): QueryAndGroup()
          (1): QueryAndGroup()
        )
        (mlps): ModuleList(
          (0): Sequential(
            (0): Conv2d(67, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): ReLU()
          )
          (1): Sequential(
            (0): Conv2d(67, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): ReLU()
          )
        )
      )
      (3): StackSAModuleMSG(
        (groupers): ModuleList(
          (0): QueryAndGroup()
          (1): QueryAndGroup()
        )
        (mlps): ModuleList(
          (0): Sequential(
            (0): Conv2d(67, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): ReLU()
          )
          (1): Sequential(
            (0): Conv2d(67, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): ReLU()
          )
        )
      )
    )
    (SA_rawpoints): StackSAModuleMSG(
      (groupers): ModuleList(
        (0): QueryAndGroup()
        (1): QueryAndGroup()
      )
      (mlps): ModuleList(
        (0): Sequential(
          (0): Conv2d(4, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
          (3): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU()
        )
        (1): Sequential(
          (0): Conv2d(4, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
          (3): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU()
        )
      )
    )
    (vsa_point_feature_fusion): Sequential(
      (0): Linear(in_features=704, out_features=128, bias=False)
      (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
  )
  (backbone_2d): BaseBEVBackbone(
    (blocks): ModuleList(
      (0): Sequential(
        (0): ZeroPad2d(padding=(1, 1, 1, 1), value=0.0)
        (1): Conv2d(320, 128, kernel_size=(3, 3), stride=(1, 1), bias=False)
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
    (conv_cls): Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
    (conv_box): Conv2d(512, 14, kernel_size=(1, 1), stride=(1, 1))
    (conv_dir_cls): Conv2d(512, 4, kernel_size=(1, 1), stride=(1, 1))
  )
  (point_head): PointHeadSimple(
    (cls_loss_func): SigmoidFocalClassificationLoss()
    (cls_layers): Sequential(
      (0): Linear(in_features=704, out_features=256, bias=False)
      (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Linear(in_features=256, out_features=256, bias=False)
      (4): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU()
      (6): Linear(in_features=256, out_features=1, bias=True)
    )
  )
  (roi_head): PVRCNNHead(
    (proposal_target_layer): ProposalTargetLayer()
    (reg_loss_func): WeightedSmoothL1Loss()
    (roi_grid_pool_layer): StackSAModuleMSG(
      (groupers): ModuleList(
        (0): QueryAndGroup()
        (1): QueryAndGroup()
      )
      (mlps): ModuleList(
        (0): Sequential(
          (0): Conv2d(131, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
          (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU()
        )
        (1): Sequential(
          (0): Conv2d(131, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
          (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU()
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
    (cls_layers): Sequential(
      (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
      (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Dropout(p=0.3, inplace=False)
      (4): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
      (5): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (6): ReLU()
      (7): Conv1d(256, 1, kernel_size=(1,), stride=(1,))
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
  )
)
```
### 执行流程
```
BACKBONE_3D:
NAME: VoxelTransformerV3
voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
batch_size = batch_dict['batch_size']
voxel_features = self.input_transform(voxel_features)
torch.Size([31963, 4])
torch.Size([31963, 4])
2
torch.Size([31963, 16])
```