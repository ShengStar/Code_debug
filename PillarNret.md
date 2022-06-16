## PillarNet部署
```
(base) lixusheng@cqu100~$ conda create --name PillarNet python=3.8
conda activate PillarNet
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3
```
## 环境变量
```
export PYTHONPATH="/home/lixusheng/PillarNet:PATH_TO_CENTERPOINT"
```
## 处理数据
```
(PillarNet) lixusheng@cqu100:~/PillarNet$ python tools/create_data.py nuscenes_data_prep --root_path=./data/nuScenes/ --version="v1.0-trainval" --nsweeps=10
```