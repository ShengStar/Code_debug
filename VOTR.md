# VOTR
## 启动环境
```
(base) lixusheng@cqu100:~/VOTR$ conda activate openpcdet5.0_votr
```
## 安装部署
```
fatal error: THC/THC.h: No such file or directory
百度，谷歌全找了一遍，最后发现时pytorch在最新的版（1.11）本中将THC/THC.h文件删除了。降低pytorch版本即可。我将pytorch版本将为1.5运行成功。
# CUDA 11.3
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3
```
## 处理数据集
```
(openpcdet5.0_votr) lixusheng@cqu100:~/openpcdet5.0_votr/OpenPCDet$ python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
如果报错，一定要检查文件夹里边的文件是否全
```
## 运行代码
```
(openpcdet5.0_votr) lixusheng@cqu100:~/openpcdet5.0_votr/OpenPCDet/tools$ CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file cfgs/kitti_models/votr_tsd.yaml
```
## 基本数据
```
Database Pedestrian: 2207
Database Car: 14357
Database Cyclist: 734
Database Van: 1297
Database Truck: 488
Database Tram: 224
Database Misc: 337
Database Person_sitting: 56
```
## 运行代码
```
(openpcdet5.0_votr) lixusheng@cqu100:~/openpcdet5.0_votr/OpenPCDet/tools$ CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file cfgs/kitti_models/votr_ssd.yaml   0efe07d

```
## 运行日志
```
  (vfe): MeanVFE()
  (backbone_3d): VoxelTransformer(
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
  (pfe): None
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
  (point_head): None
  (roi_head): None
)

```
### 实验结果
```
2022-04-27 11:15:34,323   INFO  *************** Performance of EPOCH 100 *****************
2022-04-27 11:15:34,324   INFO  Generate label finished(sec_per_example: 0.0627 second).
2022-04-27 11:15:34,324   INFO  recall_roi_0.3: 0.000000
2022-04-27 11:15:34,324   INFO  recall_rcnn_0.3: 0.953076
2022-04-27 11:15:34,324   INFO  recall_roi_0.5: 0.000000
2022-04-27 11:15:34,324   INFO  recall_rcnn_0.5: 0.927911
2022-04-27 11:15:34,324   INFO  recall_roi_0.7: 0.000000
2022-04-27 11:15:34,324   INFO  recall_rcnn_0.7: 0.748418
2022-04-27 11:15:34,333   INFO  Average predicted number of objects(3769 samples): 7.677
2022-04-27 11:15:53,859   INFO  
Car AP@0.70, 0.70, 0.70:
bbox AP:90.6214, 89.4960, 88.4682
bev  AP:89.6319, 86.7863, 84.0589
3d   AP:86.8845, 77.4727, 75.5055
aos  AP:90.61, 89.21, 88.04
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:95.4887, 92.1514, 91.1640
bev  AP:92.1091, 87.9535, 85.3875
3d   AP:88.0891, 78.8381, 75.8965
aos  AP:95.47, 91.83, 90.69
Car AP@0.70, 0.50, 0.50:
bbox AP:90.6214, 89.4960, 88.4682
bev  AP:90.7136, 89.8863, 89.2171
3d   AP:90.6952, 89.7681, 88.9345
aos  AP:90.61, 89.21, 88.04
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:95.4887, 92.1514, 91.1640
bev  AP:95.6989, 94.5852, 93.8186
3d   AP:95.6415, 94.2288, 91.7972
aos  AP:95.47, 91.83, 90.69
2022-04-27 11:15:53,867   INFO  Result is save to /home/lixusheng/openpcdet5.0_votr/OpenPCDet/output/kitti_models/votr_ssd/default/eval/eval_with_train/epoch_100/val
2022-04-27 11:15:53,867   INFO  ****************Evaluation done.*****************
2022-04-27 11:15:53,890   INFO  Epoch 100 has been evaluated
2022-04-27 11:16:23,899   INFO  **********************End evaluation kitti_models/votr_ssd(default)**********************ls/votr_ssd/default/ckpt 
```
### git 代码
```
git add .
 2084  git commit -m 'message'
 2085  git push
 2086  git init
 2087  git add .
 2088  git commit -m 'message'
 2089  git remote add origin https://github.com/ShengStar/my_votr.git
 2090  git pull origin master --allow-unrelated-histories
 2091  git push
 2092  git init
 2093  vim .git
 2094  git add .
 2095  git commit -m'描述'
 2096  git remote add origin https://github.com/ShengStar/my_votr.git
 2097  git push -f origin master
 2098  ping www.baidu.com
 2099  git remote rm origin
 2100  git remote add origin https://github.com/ShengStar/my_votr.git
 2101  git push -f origin master
 2102  cat .gitignore
 2103  vim .gitignore
 2104  git push -f origin master
 2105  git add .
 2106  git commit -m'描述'
 2107  git push -f origin master
 2108  git push -f origin main
 2109  history
 2110  git push -f origin main
 2111  history

```

### 测试2
#### 原始代码
```
(openpcdet5.0_votr) lixusheng@cqu100:~/openpcdet5.0_votr/OpenPCDet/tools$ CUDA_VISIBLE_DEVICES=0 python train.py --cfg_file cfgs/kitti_models/mask_pointpillar.yaml
```
#### 结果
```
2022-04-28 02:51:06,295   INFO  **********************Start evaluation kitti_models/mask_pointpillar(default)**********************
2022-04-28 02:51:06,297   INFO  Loading KITTI dataset
2022-04-28 02:51:06,466   INFO  Total samples for KITTI dataset: 3769
2022-04-28 02:51:06,469   INFO  ==> Loading parameters from checkpoint /home/lixusheng/openpcdet5.0_votr/OpenPCDet/output/kitti_models/mask_pointpillar/default/ckpt/checkpoint_epoch_80.pth to GPU
2022-04-28 02:51:06,524   INFO  ==> Checkpoint trained from version: pcdet+0.5.2+846cf3e+py61de6db
2022-04-28 02:51:06,528   INFO  ==> Done (loaded 127/127)
2022-04-28 02:51:06,530   INFO  *************** EPOCH 80 EVALUATION *****************
eval: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 943/943 [00:39<00:00, 24.05it/s, recall_0.3=(0, 16374) / 17558]
2022-04-28 02:51:45,743   INFO  *************** Performance of EPOCH 80 *****************
2022-04-28 02:51:45,743   INFO  Generate label finished(sec_per_example: 0.0104 second).
2022-04-28 02:51:45,743   INFO  recall_roi_0.3: 0.000000
2022-04-28 02:51:45,743   INFO  recall_rcnn_0.3: 0.932566
2022-04-28 02:51:45,743   INFO  recall_roi_0.5: 0.000000
2022-04-28 02:51:45,744   INFO  recall_rcnn_0.5: 0.867753
2022-04-28 02:51:45,744   INFO  recall_roi_0.7: 0.000000
2022-04-28 02:51:45,744   INFO  recall_rcnn_0.7: 0.616585
2022-04-28 02:51:45,756   INFO  Average predicted number of objects(3769 samples): 18.297
2022-04-28 02:52:10,902   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:90.6569, 89.1328, 87.9948
bev  AP:89.7049, 87.0411, 83.7317
3d   AP:84.5946, 76.0407, 72.4577
aos  AP:90.64, 88.94, 87.64
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:95.3628, 91.6365, 89.0150
bev  AP:92.1866, 88.0770, 85.4601
3d   AP:86.6398, 75.9795, 73.0390
aos  AP:95.34, 91.43, 88.67
Car AP@0.70, 0.50, 0.50:
bbox AP:90.6569, 89.1328, 87.9948
bev  AP:90.7278, 89.8890, 89.2293
3d   AP:90.7278, 89.8256, 89.0485
aos  AP:90.64, 88.94, 87.64
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:95.3628, 91.6365, 89.0150
bev  AP:95.5664, 94.4655, 93.6747
3d   AP:95.5347, 94.2072, 91.7322
aos  AP:95.34, 91.43, 88.67
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:65.0047, 59.7678, 57.1662
bev  AP:56.7106, 51.5372, 47.8051
3d   AP:51.2427, 45.0895, 41.2583
aos  AP:50.51, 45.65, 43.81
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:64.7896, 59.7504, 56.4814
bev  AP:56.1155, 50.0656, 45.9583
3d   AP:49.3142, 43.2492, 39.0502
aos  AP:48.41, 43.69, 41.17
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:65.0047, 59.7678, 57.1662
bev  AP:70.2537, 66.8132, 63.5060
3d   AP:70.1823, 66.6492, 63.2011
aos  AP:50.51, 45.65, 43.81
Pedestrian AP_R40@0.50, 0.25, 0.25:
bbox AP:64.7896, 59.7504, 56.4814
bev  AP:71.4962, 67.3086, 63.7103
3d   AP:71.4117, 66.9084, 63.2948
aos  AP:48.41, 43.69, 41.17
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:85.7747, 74.4651, 71.5266
bev  AP:83.6195, 67.8130, 63.6958
3d   AP:81.6423, 63.7883, 61.3687
aos  AP:83.45, 70.96, 68.11
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:89.3499, 76.3290, 72.4395
bev  AP:86.9945, 68.4133, 63.9502
3d   AP:83.3719, 64.3932, 60.0505
aos  AP:86.70, 72.44, 68.59
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:85.7747, 74.4651, 71.5266
bev  AP:85.3936, 73.4066, 70.1935
3d   AP:85.3936, 73.4066, 70.1935
aos  AP:83.45, 70.96, 68.11
Cyclist AP_R40@0.50, 0.25, 0.25:
bbox AP:89.3499, 76.3290, 72.4395
bev  AP:89.1751, 75.0226, 70.8173
3d   AP:89.1751, 75.0226, 70.8173
aos  AP:86.70, 72.44, 68.59

```
### 三分支
#### 运行代码
```
(openpcdet5.0_votr) lixusheng@cqu100:~/openpcdet5.0_votr/OpenPCDet/tools$ CUDA_VISIBLE_DEVICES=0 python train.py --cfg_file cfgs/kitti_models/pointpillar_three.yaml
```
#### 结果
```
2022-04-28 20:35:43,770   INFO  *************** Performance of EPOCH 80 *****************
2022-04-28 20:35:43,771   INFO  Generate label finished(sec_per_example: 0.0120 second).
2022-04-28 20:35:43,771   INFO  recall_roi_0.3: 0.000000
2022-04-28 20:35:43,771   INFO  recall_rcnn_0.3: 0.926643
2022-04-28 20:35:43,771   INFO  recall_roi_0.5: 0.000000
2022-04-28 20:35:43,771   INFO  recall_rcnn_0.5: 0.853115
2022-04-28 20:35:43,771   INFO  recall_roi_0.7: 0.000000
2022-04-28 20:35:43,771   INFO  recall_rcnn_0.7: 0.596879
2022-04-28 20:35:43,778   INFO  Average predicted number of objects(3769 samples): 21.225
2022-04-28 20:36:08,898   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:90.1931, 88.3487, 86.2573
bev  AP:89.1323, 84.3367, 82.2726
3d   AP:81.6355, 71.4624, 68.2469
aos  AP:90.12, 88.03, 85.80
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:94.6379, 90.0023, 87.7817
bev  AP:91.3934, 86.3900, 84.1920
3d   AP:83.9750, 73.6891, 70.2766
aos  AP:94.54, 89.66, 87.31
Car AP@0.70, 0.50, 0.50:
bbox AP:90.1931, 88.3487, 86.2573
bev  AP:90.4144, 89.4183, 88.7166
3d   AP:90.3780, 89.2766, 88.3639
aos  AP:90.12, 88.03, 85.80
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:94.6379, 90.0023, 87.7817
bev  AP:95.0714, 93.4406, 92.1751
3d   AP:95.0027, 92.6950, 90.6930
aos  AP:94.54, 89.66, 87.31
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:64.2048, 59.0890, 56.4691
bev  AP:56.4367, 50.9502, 47.1829
3d   AP:51.4375, 45.1073, 40.5270
aos  AP:48.90, 44.27, 41.98
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:64.2726, 58.7705, 55.7802
bev  AP:56.3199, 49.5896, 45.5046
3d   AP:49.7960, 42.9468, 38.5935
aos  AP:46.84, 41.94, 39.40
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:64.2048, 59.0890, 56.4691
bev  AP:69.7762, 65.6892, 62.5511
3d   AP:69.5488, 65.5737, 62.3185
aos  AP:48.90, 44.27, 41.98
Pedestrian AP_R40@0.50, 0.25, 0.25:
bbox AP:64.2726, 58.7705, 55.7802
bev  AP:70.7739, 65.9955, 62.7884
3d   AP:70.6029, 65.7831, 62.5220
aos  AP:46.84, 41.94, 39.40
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:81.9179, 71.9425, 69.2359
bev  AP:79.0762, 64.1058, 60.6319
3d   AP:74.1696, 60.0801, 55.9461
aos  AP:79.67, 66.70, 64.19
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:83.6821, 73.1250, 69.5147
bev  AP:80.1285, 64.4090, 60.3539
3d   AP:76.1590, 59.7327, 55.7826
aos  AP:81.29, 67.51, 63.98
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:81.9179, 71.9425, 69.2359
bev  AP:81.0571, 69.5623, 66.8585
3d   AP:81.0478, 69.5458, 66.7566
aos  AP:79.67, 66.70, 64.19
Cyclist AP_R40@0.50, 0.25, 0.25:
bbox AP:83.6821, 73.1250, 69.5147
bev  AP:82.5300, 70.6002, 66.9664
3d   AP:82.5258, 70.5420, 66.8761
aos  AP:81.29, 67.51, 63.98

/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (192) < 2 * SM count (196) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
2022-04-29 13:30:16,446   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:90.3784, 88.5270, 86.6963
bev  AP:89.2485, 85.7276, 82.3302
3d   AP:81.3063, 73.8417, 68.0496
aos  AP:90.33, 88.28, 86.28
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:94.7217, 90.4317, 87.9127
bev  AP:91.4451, 86.9826, 84.4608
3d   AP:83.7923, 74.2677, 70.0006
aos  AP:94.66, 90.15, 87.48
Car AP@0.70, 0.50, 0.50:
bbox AP:90.3784, 88.5270, 86.6963
bev  AP:90.4970, 89.6160, 88.8703
3d   AP:90.4872, 89.4479, 88.5171
aos  AP:90.33, 88.28, 86.28
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:94.7217, 90.4317, 87.9127
bev  AP:95.0794, 93.7696, 92.8515
3d   AP:95.0419, 93.2440, 90.9077
aos  AP:94.66, 90.15, 87.48
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:64.0186, 59.0220, 56.2975
bev  AP:55.7574, 50.6846, 46.7843
3d   AP:50.2807, 44.5688, 40.4476
aos  AP:50.28, 45.73, 43.55
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:64.0854, 58.7899, 55.6389
bev  AP:55.1161, 49.1819, 45.0759
3d   AP:48.2914, 42.4445, 38.0089
aos  AP:48.17, 43.22, 40.58
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:64.0186, 59.0220, 56.2975
bev  AP:69.1060, 65.6181, 62.2244
3d   AP:69.0773, 65.4816, 62.1463
aos  AP:50.28, 45.73, 43.55
Pedestrian AP_R40@0.50, 0.25, 0.25:
bbox AP:64.0854, 58.7899, 55.6389
bev  AP:70.3956, 65.9414, 62.4300
3d   AP:70.3693, 65.8184, 62.3050
aos  AP:48.17, 43.22, 40.58
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:81.1125, 69.1905, 66.5259
bev  AP:76.7686, 62.7295, 58.6516
3d   AP:72.9670, 58.1170, 54.5138
aos  AP:78.68, 65.08, 62.46
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:82.6984, 69.9382, 66.2684
bev  AP:77.5158, 62.2732, 58.3883
3d   AP:74.1421, 57.2995, 53.4289
aos  AP:80.00, 65.42, 61.96
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:81.1125, 69.1905, 66.5259
bev  AP:80.5602, 67.1841, 64.2535
3d   AP:80.5602, 66.8748, 64.0881
aos  AP:78.68, 65.08, 62.46
Cyclist AP_R40@0.50, 0.25, 0.25:
bbox AP:82.6984, 69.9382, 66.2684
bev  AP:82.3109, 67.6949, 64.3206
3d   AP:82.3087, 67.5018, 63.8726
aos  AP:80.00, 65.42, 61.96

```
## PV-RCNN
### 第一次标准结构
#### 运行代码
```
(openpcdet5.0_votr) lixusheng@cqu100:~/openpcdet5.0_votr/OpenPCDet/tools$ CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml
```
#### 结果
```
2022-04-30 11:09:26,336   INFO  **********************Start evaluation kitti_models/pv_rcnn(default)**********************
2022-04-30 11:09:26,339   INFO  Loading KITTI dataset
2022-04-30 11:09:26,451   INFO  Total samples for KITTI dataset: 3769
2022-04-30 11:09:26,454   INFO  ==> Loading parameters from checkpoint /home/lixusheng/openpcdet5.0_votr/OpenPCDet/output/kitti_models/pv_rcnn/default/ckpt/checkpoint_epoch_80.pth to GPU
2022-04-30 11:09:26,583   INFO  ==> Checkpoint trained from version: pcdet+0.5.2+846cf3e+pyeda56a0
2022-04-30 11:09:26,600   INFO  ==> Done (loaded 367/367)
2022-04-30 11:09:26,604   INFO  *************** EPOCH 80 EVALUATION *****************
eval: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1885/1885 [04:56<00:00,  6.36it/s, recall_0.3=(16921, 16929) / 17558]
2022-04-30 11:14:23,070   INFO  *************** Performance of EPOCH 80 *****************
2022-04-30 11:14:23,070   INFO  Generate label finished(sec_per_example: 0.0787 second).
2022-04-30 11:14:23,070   INFO  recall_roi_0.3: 0.963720
2022-04-30 11:14:23,070   INFO  recall_rcnn_0.3: 0.964176
2022-04-30 11:14:23,070   INFO  recall_roi_0.5: 0.918157
2022-04-30 11:14:23,070   INFO  recall_rcnn_0.5: 0.924991
2022-04-30 11:14:23,070   INFO  recall_roi_0.7: 0.681513
2022-04-30 11:14:23,070   INFO  recall_rcnn_0.7: 0.727589
2022-04-30 11:14:23,081   INFO  Average predicted number of objects(3769 samples): 10.180
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (24) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (30) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (20) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (35) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (40) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (28) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (24) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/core/typed_passes.py:330: NumbaPerformanceWarning: 
The keyword argument 'parallel=True' was specified but no transformation for parallel execution was possible.

To find out why, try turning on parallel diagnostics, see https://numba.readthedocs.io/en/stable/user/parallel.html#diagnostics for help.

File "../pcdet/datasets/kitti/kitti_object_eval_python/eval.py", line 122:
@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
^

  state.func_ir.loc))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (30) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (20) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (35) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (24) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (40) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (28) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/core/typed_passes.py:330: NumbaPerformanceWarning: 
The keyword argument 'parallel=True' was specified but no transformation for parallel execution was possible.

To find out why, try turning on parallel diagnostics, see https://numba.readthedocs.io/en/stable/user/parallel.html#diagnostics for help.

File "../pcdet/datasets/kitti/kitti_object_eval_python/eval.py", line 122:
@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
^

  state.func_ir.loc))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (24) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (35) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (30) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (28) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (20) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (40) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
2022-04-30 11:14:49,311   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:96.0272, 89.4206, 89.0583
bev  AP:90.2578, 87.9629, 87.4428
3d   AP:89.3523, 79.2207, 78.4632
aos  AP:95.98, 89.28, 88.83
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:98.3743, 94.1796, 92.0053
bev  AP:94.5650, 90.5077, 88.4698
3d   AP:91.9477, 82.9023, 80.3159
aos  AP:98.32, 93.99, 91.74
Car AP@0.70, 0.50, 0.50:
bbox AP:96.0272, 89.4206, 89.0583
bev  AP:96.0888, 89.4854, 89.1753
3d   AP:96.0452, 89.4436, 89.1229
aos  AP:95.98, 89.28, 88.83
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:98.3743, 94.1796, 92.0053
bev  AP:98.4012, 94.4595, 94.2460
3d   AP:98.3735, 94.3668, 94.0799
aos  AP:98.32, 93.99, 91.74
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:71.8060, 67.2843, 63.5134
bev  AP:65.8714, 57.8472, 53.2132
3d   AP:61.4065, 53.3605, 49.4062
aos  AP:67.15, 62.25, 58.50
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:72.8617, 67.2436, 63.5291
bev  AP:65.6801, 57.1602, 52.5866
3d   AP:60.8726, 52.7628, 47.6392
aos  AP:67.34, 61.43, 57.69
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:71.8060, 67.2843, 63.5134
bev  AP:76.5180, 70.9327, 68.1114
3d   AP:76.3712, 70.6512, 67.9073
aos  AP:67.15, 62.25, 58.50
Pedestrian AP_R40@0.50, 0.25, 0.25:
bbox AP:72.8617, 67.2436, 63.5291
bev  AP:78.4000, 71.6802, 68.2390
3d   AP:78.3196, 71.4075, 67.9987
aos  AP:67.34, 61.43, 57.69
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:88.1458, 81.2365, 76.5453
bev  AP:84.2679, 72.7685, 69.7708
3d   AP:82.7539, 70.6082, 67.2584
aos  AP:87.91, 80.32, 75.70
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:93.2339, 81.9493, 77.9510
bev  AP:86.7274, 74.4850, 70.2299
3d   AP:85.0830, 70.9530, 66.8936
aos  AP:92.90, 80.97, 76.98
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:88.1458, 81.2365, 76.5453
bev  AP:86.4944, 77.7608, 73.2341
3d   AP:86.4944, 77.7587, 73.2341
aos  AP:87.91, 80.32, 75.70
Cyclist AP_R40@0.50, 0.25, 0.25:
bbox AP:93.2339, 81.9493, 77.9510
bev  AP:91.2485, 78.3527, 74.2424
3d   AP:91.2485, 78.3510, 74.2407
aos  AP:92.90, 80.97, 76.98

2022-04-30 11:14:49,319   INFO  Result is save to /home/lixusheng/openpcdet5.0_votr/OpenPCDet/output/kitti_models/pv_rcnn/default/eval/eval_with_train/epoch_80/val
2022-04-30 11:14:49,319   INFO  ****************Evaluation done.*****************
2022-04-30 11:14:49,347   INFO  Epoch 80 has been evaluated
2022-04-30 11:15:19,358   INFO  **********************End evaluation kitti_models/pv_rcnn(default)**********************els/pv_rcnn/default/ckpt 

```
#### 第二次测试
```

eval: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1885/1885 [18:03<00:00,  1.74it/s, recall_0.3=(16893, 16911) / 17558]
2022-05-05 12:18:35,844   INFO  *************** Performance of EPOCH 80 *****************
2022-05-05 12:18:35,845   INFO  Generate label finished(sec_per_example: 0.2876 second).
2022-05-05 12:18:35,845   INFO  recall_roi_0.3: 0.962126
2022-05-05 12:18:35,845   INFO  recall_rcnn_0.3: 0.963151
2022-05-05 12:18:35,845   INFO  recall_roi_0.5: 0.915366
2022-05-05 12:18:35,845   INFO  recall_rcnn_0.5: 0.922144
2022-05-05 12:18:35,845   INFO  recall_roi_0.7: 0.682196
2022-05-05 12:18:35,845   INFO  recall_rcnn_0.7: 0.727987
2022-05-05 12:18:35,855   INFO  Average predicted number of objects(3769 samples): 10.591
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (24) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (30) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (35) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (28) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (40) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (20) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (24) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/core/typed_passes.py:330: NumbaPerformanceWarning:
The keyword argument 'parallel=True' was specified but no transformation for parallel execution was possible.

To find out why, try turning on parallel diagnostics, see https://numba.readthedocs.io/en/stable/user/parallel.html#diagnostics for help.

File "../pcdet/datasets/kitti/kitti_object_eval_python/eval.py", line 122:
@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
^

  state.func_ir.loc))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (30) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (35) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (24) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (28) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (40) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/lixusheng/anaconda3/envs/openpcdet5.0_votr/lib/python3.7/site-packages/numba/cuda/compiler.py:726: NumbaPerformanceWarning: Grid size (20) < 2 * SM count (84) will likely result in GPU under utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
2022-05-05 12:19:00,679   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:96.2557, 89.5147, 89.1083
bev  AP:90.3485, 88.1544, 87.5500
3d   AP:89.4485, 79.3167, 78.6378
aos  AP:96.20, 89.36, 88.89
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:98.3046, 94.2214, 92.0428
bev  AP:93.1197, 90.5582, 88.5637
3d   AP:92.1450, 82.8727, 80.3934
aos  AP:98.25, 94.02, 91.78
Car AP@0.70, 0.50, 0.50:
bbox AP:96.2557, 89.5147, 89.1083
bev  AP:95.5509, 89.5453, 89.2098
3d   AP:95.5123, 89.5208, 89.1681
aos  AP:96.20, 89.36, 88.89
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:98.3046, 94.2214, 92.0428
bev  AP:98.1144, 94.4616, 94.2556
3d   AP:97.9435, 94.4062, 94.1494
aos  AP:98.25, 94.02, 91.78
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:70.5063, 64.2334, 62.5548
bev  AP:62.0722, 55.7543, 51.9143
3d   AP:59.1926, 51.7717, 48.0581
aos  AP:66.17, 60.02, 57.97
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:70.9015, 64.8038, 62.2130
bev  AP:62.1076, 54.6071, 50.3788
3d   AP:58.0226, 50.5617, 45.7835
aos  AP:66.01, 59.92, 57.02
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:70.5063, 64.2334, 62.5548
bev  AP:74.8753, 70.6495, 68.5090
3d   AP:74.7016, 70.1838, 68.0020
aos  AP:66.17, 60.02, 57.97
Pedestrian AP_R40@0.50, 0.25, 0.25:
bbox AP:70.9015, 64.8038, 62.2130
bev  AP:76.4885, 71.1074, 68.4274
3d   AP:76.3000, 70.6048, 67.9962
aos  AP:66.01, 59.92, 57.02
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:95.7660, 77.8832, 76.2600
bev  AP:84.9681, 72.4551, 68.5950
3d   AP:83.6841, 69.9794, 63.8109
aos  AP:95.42, 77.08, 75.38
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:96.2903, 81.0491, 77.8667
bev  AP:89.5123, 73.2079, 69.7496
3d   AP:86.3410, 69.9026, 65.4663
aos  AP:95.94, 80.11, 76.88
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:95.7660, 77.8832, 76.2600
bev  AP:94.2662, 78.5496, 73.3031
3d   AP:94.2662, 78.5496, 73.3031
aos  AP:95.42, 77.08, 75.38
Cyclist AP_R40@0.50, 0.25, 0.25:
bbox AP:96.2903, 81.0491, 77.8667
bev  AP:94.8794, 78.9355, 74.6694
3d   AP:94.8794, 78.9355, 74.6694
aos  AP:95.94, 80.11, 76.88

2022-05-05 12:19:00,689   INFO  Result is save to /home/lixusheng/openpcdet5.0_votr/OpenPCDet/output/kitti_models/pv_rcnn/default/eval/eval_with_train/epoch_80/val
2022-05-05 12:19:00,689   INFO  ****************Evaluation done.*****************
2022-05-05 12:19:00,723   INFO  Epoch 80 has been evaluated
2022-05-05 12:19:30,728   INFO  **********************End evaluation kitti_models/pv_rcnn(default)**********************els/pv_rcnn/default/ckpt
(openpcdet5.0_votr) lixusheng@cqu100:~/openpcdet5.0_votr/OpenPCDet/tools$ CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml

```
### waymo 数据处理
#### 指令
```

```