## ["DCDet: Dynamic Cross-based 3D Object Detector"](https://arxiv.org/abs/2401.07240)

Thanks for the [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), this implementation of the DCDet is mainly based on the pcdet v0.6.

Abstract: Recently, significant progress has been made in the research of 3D object detection. However, most prior studies have focused on the utilization of center-based or anchor-based label assignment schemes. Alternative label assignment strategies remain unexplored in 3D object detection. We find that the center-based label assignment often fails to generate sufficient positive samples for training, while the anchor-based label assignment tends to encounter an imbalanced issue when handling objects of varying scales. To solve these issues, we introduce a dynamic cross label assignment (DCLA) scheme, which dynamically assigns positive samples for each object from a cross-shaped region, thus providing sufficient and balanced positive samples for training. Furthermore, to address the challenge of accurately regressing objects with varying scales, we put forth a rotation-weighted Intersection over Union (RWIoU) metric to replace the widely used L1 metric in regression loss. Extensive experiments demonstrate the generality and effectiveness of our DCLA and RWIoU-based regression loss.

### 1. Recommended Environment

- Linux (tested on Ubuntu 20.04)
- Python 3.6+
- PyTorch 1.1 or higher (tested on PyTorch 1.13)
- CUDA 9.0 or higher (tested on 11.6)

### 2. Set the Environment

```shell
pip install -r requirement.txt
python setup.py develop
```

### 3. Data Preparation

- Prepare [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and [road planes](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing)

```shell
# Download KITTI and organize it into the following form:
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2

# Generatedata infos:
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

- Prepare [Waymo](https://waymo.com/open/download/) dataset

```shell
# Download Waymo and organize it into the following form:
├── data
│   ├── waymo
│   │   │── ImageSets
│   │   │── raw_data
│   │   │   │── segment-xxxxxxxx.tfrecord
|   |   |   |── ...
|   |   |── waymo_processed_data
│   │   │   │── segment-xxxxxxxx/
|   |   |   |── ...
│   │   │── pcdet_gt_database_train_sampled_xx/
│   │   │── pcdet_waymo_dbinfos_train_sampled_xx.pkl

# Install tf 2.5.0
# Install the official waymo-open-dataset by running the following command:
pip3 install --upgrade pip
pip3 install waymo-open-dataset-tf-2-5-0 --user

# Extract point cloud data from tfrecord and generate data infos:
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos --cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml
```

### 4. Train

- Train with a single GPU

```shell
python tools/train.py --cfg_file ${CONFIG_FILE}

# e.g.,
python tools/train.py --cfg_file tools/cfgs/waymo_models/dcdet.yaml
```

- Train with multiple GPUs or multiple machines

```shell
bash tools/scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
# or 
bash tools/scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_GPUS} --cfg_file ${CONFIG_FILE}

# e.g.,
bash tools/scripts/dist_train.sh 8 --cfg_file tools/cfgs/waymo_models/dcdet.yaml
```

### 5. Test

- Test with a pretrained model:

```shell
python tools/test.py --cfg_file ${CONFIG_FILE} --ckpt ${CKPT}

# e.g., 
python tools/test.py --cfg_file tools/cfgs/waymo_models/dcdet.yaml --ckpt {path}
```

## Paper

Please cite our paper if you find our work useful for your research:

```
@article{liu2024dcdet,
  title={DCDet: Dynamic Cross-based 3D Object Detector},
  author={Liu, Shuai and Li, Boyang and Fang, Zhiyu and Huang, Kai},
  journal={arXiv preprint arXiv:2401.07240},
  year={2024}
}
```
