
# <div align = "center"><img src="figure/IMG_1396.PNG" alt="image-20200927095842317" width="5%" height="5%" /> H3-Mapping: </div>

#### <div align = "center"> Quasi-Heterogeneous Feature Grids for Real-time Dense Mapping Using Hierarchical Hybrid Representation</div>

<div align="center">
<a href="https://ieeexplore.ieee.org/document/10602533"><img src="https://img.shields.io/badge/Paper-IEEE RAL-004088.svg"/></a>
<a href="https://arxiv.org/abs/2403.10821"><img src="https://img.shields.io/badge/ArXiv-2403.10821-004088.svg"/></a>
<a href="https://youtu.be/VtQGvuDZSec">
<img alt="Youtube" src="https://img.shields.io/badge/Video-Youtube-red"/>
</a>
<a href="https://www.bilibili.com/video/BV1By421v7S8">
<img alt="Bilibili" src="https://img.shields.io/badge/Video-Bilibili-blue"/>
</a>
<a ><img alt="PRs-Welcome" src="https://img.shields.io/badge/PRs-Welcome-red" />
</a>
<a href="https://github.com/SYSU-STAR/H3-Mapping/stargazers">
<img alt="stars" src="https://img.shields.io/github/stars/SYSU-STAR/H3-Mapping" />
</a>
<a href="https://github.com/SYSU-STAR/H3-Mapping/network/members">
<img alt="FORK" src="https://img.shields.io/github/forks/SYSU-STAR/H3-Mapping?color=FF8000" />
</a>
<a href="https://github.com/SYSU-STAR/H3-Mapping/issues">
<img alt="Issues" src="https://img.shields.io/github/issues/SYSU-STAR/H3-Mapping?color=0088ff"/>
</a>
</div>


> [Chenxing Jiang*](https://jiang-cx.github.io/), Yiming Luo, [Boyu Zhou †](http://sysu-star.com/people/), [Shaojie Shen](https://uav.hkust.edu.hk/group/)
>
> Submitted to IEEE Robotics and Automation Letters 2024

## Abstract

In recent years, implicit online dense mapping methods have achieved high-quality reconstruction results, showcasing great potential in robotics, AR/VR, and digital twins applications. However, existing methods struggle with slow texture modeling which limits their real-time performance. To address these limitations, we propose a NeRF-based dense mapping method that enables faster and higher-quality reconstruction. To improve texture modeling, we introduce quasi-heterogeneous feature grids, which inherit the fast querying ability of uniform feature grids while adapting to varying levels of texture complexity. Besides, we present a gradient-aided coverage-maximizing strategy for keyframe selection that enables the selected keyframes to exhibit a closer focus on rich-textured regions and a broader scope for weak-textured areas. Experimental results demonstrate that our method surpasses existing NeRF-based approaches in texture fidelity, geometry accuracy, and time consumption. 

[![H3-Mapping: Quasi-Heterogeneous Feature Grids for Real-time Dense Mapping Using Hierarchical Hybrid Representation](https://res.cloudinary.com/marcomontalbano/image/upload/v1710830161/video_to_markdown/images/youtube--VtQGvuDZSec-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://www.youtube.com/watch?v=VtQGvuDZSec&t=27s "H3-Mapping: Quasi-Heterogeneous Feature Grids for Real-time Dense Mapping Using Hierarchical Hybrid Representation")

## Installation
1. Begin by cloning this repository using the following command:

   ```bash
   git clone https://github.com/SYSU-STAR/H3-Mapping
   ```

2. Create an anaconda environment called `h3mapping`. Note that installing the 0.17.0 version of open3d may result in errors during reconstruction evaluation. Please install the 0.16.0 version of open3d instead.
   ```bash
   cd H3-Mapping/mapping
   conda create -n h3mapping python=3.9
   source activate h3mapping
   pip install -r requirements.txt
   bash install.sh
   ```

3. Install the [Pytorch](https://pytorch.org/) manually for your hardware platform.

4. Find and download corresponding torch_scatter package in https://pytorch-geometric.com/whl/ . Install the package like:

   ```bash
   eg. pip3 install torch_scatter-2.1.0+pt112cu113-cp37-cp37m-linux_x86_64.whl
   ```

## Run in dataset (Only Mapping)

1. Replace the filename in `src/mapping.py` with the built library

```bash
torch.classes.load_library("third_party/sparse_octree/build/lib.xxx/svo.xxx.so")
```

### Replica dataset

1. Download the data as below and the data is saved into the `./Datasets/Replica` folder.

```bash
bash mapping/scripts/download_replica.sh
```

2. To execute H3-Mapping, please proceed with the following steps. 

```bash
# take replica room0 dataset as example
cd mapping
python -W ignore demo/run_mapping.py --config configs/replica/room_0.yaml
```

The final reconstructed mesh will be saved in `mapping/logs/{DATASET}/{DATA SEQUENCE}/{FILE_NAME}/mesh`.

## Run in ROS (Full SLAM)

1. Install the [Ubuntu](https://cn.ubuntu.com/download), [ROS](http://wiki.ros.org/ROS/Installation), [Ceres](http://ceres-solver.org/), [OpenCV](https://opencv.org/get-started/). If you successfully run  [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion), you will be able to run our tracking code as well.

   The current version of our platform includes:

   * Ubuntu=20.04

   * ROS=noetic

   * Ceres=1.14.0

   * OpenCV=4.2.0

2. Build our Tracking module

```bash
cd H3-Mapping
catkin_make
```

3. Replace the filename in `src/mapping.py` with the built library

```bash
torch.classes.load_library("third_party/sparse_octree/build/lib.xxx/svo.xxx.so")
```

### Self-captured House dataset

1. Please download the ROS bag containing RGB-D sensor data from the Realsense L515. You can access the file through the provided link: [tower_compress.bag](https://hkustconnect-my.sharepoint.com/:u:/g/personal/cjiangan_connect_ust_hk/EVJ94ajgfaZDiPyJRezrHzkBRW0-N4eytXr2eTjHW2735w?e=Al8oVy); 

2. Decompress the ROS bag

   ```bash
   rosbag decompress tower_compress.bag
   ```

3. Configure the tracking parameter in the designated configuration file. For reference, consider the example file located at `src/dvins/config/uav2022/uav_nerf.yaml`. This configuration file is for the provided `tower_compress.orig.bag`.

4. Configure the `ros_args` parameter in the designated configuration file. For reference, consider the example file located at `configs/realsense/realsense.yaml`. This configuration file is for the provided `tower_compress.orig.bag`.

   ```bash
   # set intrinsic parameters, ROS topics of rgb image, depth image and odometry
   ros_args:
     intrinsic: [601.347290039062, 601.343017578125, 329.519226074219, 238.586654663086] # K[0, 0], K[1, 1], K[0, 2], K[1, 2]
     color_topic: '/camera/color/image_raw'
     depth_topic: '/camera/aligned_depth_to_color/image_raw'
     pose_topic: '/vins_estimator/cam_pose'
   ```

5. Run the mapping module. 

   ```bash
   conda activate h3mapping
   cd H3-Mapping
   source devel/setup.bash
   cd mapping
   python -W ignore demo/run_mapping.py --config configs/realsense/tower.yaml -run_ros
   ```

   Once the mapping module is prepared and operational, you will observe a distinctive sign in the console that reads `" ========== MAPPING START ==========="`. This indicator confirms the initiation of the mapping process.

   **Note**: The default configuration was utilized in our paper's experiment, conducted on NVIDIA Jetson AGX Orin (32GB RAM). In case you encounter memory limitations on your platform, you can attempt reducing the `insert_ratio` parameter in the `configs/realsense/tower.yaml` file, but it may result in inferior outcomes.

6. Run the Tracking module 


```bash
cd H3-Mapping
bash ros_cmd/run_vins_rgbd.sh
```

7. Play the ROS Bag

```bash
rosbag play tower_compress.orig.bag
```

8. In a separate console, execute the command `rosnode kill -a` to terminate all the modules. Afterwards, the marching cube algorithm will be executed to reconstruct a mesh for visualization.

### Use your own RGB-D sequence

1. Modify the mapping configuration file in `mapping/configs/realsense`. Please take note of the following:

   **Note:**

   (1) Ensure that the value assigned to the "offset"  (m) parameter is sufficiently large. This value is utilized to ensure that the coordinates of each point are positive.

   (2) Ensure that the upper bound of the "bound" (m) parameter is suitably large. This parameter defines the boundary of the scene. However,  if the range is too much larger than the scene you want to reconstruct, the performance may degrade. 

   (3) Choose the appropriate value for "num_vertexes" based on the size of your scene. It should be large enough to encompass all the vertices of the octree.

   (4) Ensure that there will be no frame with a minimum depth smaller than max_depth.

2. Modify the tracking configuration file in `src/dvins/config/uav2022/uav_nerf.yaml` to suit your specific device.

3. To execute the code similar to the provided demo.

## Evaluation

### Reconstruction Error

1. Download the culled ground truth Replica meshes: [cull_replica_mesh.zip](https://hkustconnect-my.sharepoint.com/:u:/g/personal/cjiangan_connect_ust_hk/ETMuZRHjcw9DrY-IaZEU1FcBtI20dokIqXMfy2sf6nkOPQ?e=qRmVk8)

2. Replace the filename in `eval/eval_recon.py` with the built library

```bash
torch.classes.load_library("third_party/sparse_octree/build/lib.xxx/svo.xxx.so")
```

3. Then run the command below. The 2D metric requires rendering of 1000 depth images. Use `-2d` to enable 2D metric. Use `-3d` to enable 3D metric. The reconstruction results will be saved in the `$OUTPUT_FOLDER`

```bash
# assign any output_folder and gt mesh you like, here is just an example
cd mapping
OUTPUT_FOLDER=logs/replica/room0/FILE_NAME
GT_MESH=../Datasets/Replica/cull_replica_mesh/room0_culled.ply
python eval/eval_recon.py \
--config $OUTPUT_FOLDER/bak/config.yaml \
--rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_culled.ply \
--gt_mesh $GT_MESH \
--ckpt $OUTPUT_FOLDER/ckpt/final_ckpt.pth \
--out_dir $OUTPUT_FOLDER \
-2d \
-3d
```

### Rendering Error

1. Replace the filename in `src/mapping.py` with the built library

```bash
torch.classes.load_library("third_party/sparse_octree/build/lib.xxx/svo.xxx.so")
```

2. Then run the command below. It will calculate the SSIM and PSNR of the color rendering. Additionally, it will calculate the L1 loss of the depth rendering. The **resulting rendering videos and images** will be automatically saved in the designated `$OUTPUT_FOLDER`.

```bash
# assign any output_folder you like, here is just an example
cd mapping
OUTPUT_FOLDER=logs/replica/room0/FILE_NAME
python eval/eval_color.py \
--config $OUTPUT_FOLDER/bak/config.yaml \
--result_file $OUTPUT_FOLDER
```

**Note: If you want to evaluate the rendering error in your custom dataset, you should emulate the structure of `mapping/src/dataset/replica.py` and create a corresponding `MY_DATA.py` file. Additionally, ensure to include lines in the config file specifying the dataset as `dataset: MY_DATA` and the `data_path: YOUR_DATASET_PATH`.**

## FAQ

1. **There is a conflict between OpenCV and cv_bridge during the build process of our tracking module**

This issue is related to the version of cv_bridge. To resolve it, you can follow these steps (using ROS Noetic and OpenCV 4.5.2 as an example):

(1) Remove the previous cv_bridge package for the ROS version:

```bash
sudo apt-get remove ros-noetic-cv-bridge
```

(2) Download the new version of [cv_bridge](https://github.com/ros-perception/vision_opencv)

(3) Locate the CMakeLists.txt file of cv_bridge and modify the following line to match the version of OpenCV you have installed

```cmake
// In CMakeLists.txt of cv_bridge
find_package(OpenCV 4.5.2 REQUIRED) // Change it to your installed version of OpenCV
```

(4) Build and install the cv_bridge

```bash
mkdir build
cd build
cmake ..
make
sudo make install
```

(5) Add the cmake path for cv_bridge in the `src/dvins/vins_estimator/CMakeLists.txt` and  `src/dvins/pose_graph/CMakeLists.txt` 

```cmake
set(cv_bridge_DIR /usr/local/share/cv_bridge/cmake) // At the beginning
```

## Acknowledgement

We adapted some codes from some remarkable repositories including [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion), [NICE-SLAM](https://github.com/cvg/nice-slam), [Vox-Fusion](https://github.com/zju3dv/Vox-Fusion) and [Tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn). We express our gratitude for the authors' generosity in sharing their code publicly.

## Contact

You can contact the author through email: cjiangan@connect.ust.hk and zhouby23@mail.sysu.edu.cn

## Citing

If you find our work useful, please consider citing:

```bash
@article{jiang2024h3,
  title={H3-Mapping: Quasi-Heterogeneous Feature Grids for Real-time Dense Mapping Using Hierarchical Hybrid Representation},
  author={Jiang, Chenxing and Luo, Yiming and Zhou, Boyu and Shen, Shaojie},
  journal={arXiv preprint arXiv:2403.10821},
  year={2024}
}

@ARTICLE{10243098,
  author={Jiang, Chenxing and Zhang, Hanwen and Liu, Peize and Yu, Zehuan and Cheng, Hui and Zhou, Boyu and Shen, Shaojie},
  journal={IEEE Robotics and Automation Letters}, 
  title={H$_{2}$-Mapping: Real-Time Dense Mapping Using Hierarchical Hybrid Representation}, 
  year={2023},
  volume={8},
  number={10},
  pages={6787-6794},
  doi={10.1109/LRA.2023.3313051}}
```

