# Project

## General
* Type: Master's Thesis
* Author: Nicolaj Schmid (nicolaj.schmid@epfl.ch)
* Supervisors: Cornelius von Einem, Florian Tschopp and Lorenz Hruby
* Professors: Roland Siegwart and Colin Jones
* Universities: ETHZ and EPFL
* Labs: Autonomous Systems Lab and Predictive Control Lab

## Abstract
This research strives to leverage cost-effective sensors for local mapping applications in mobile robotics. 
The study introduces _VIRUS-NeRF_ - _Vision, InfraRed, and UltraSonic based Neural Radiance Fields_. 
While traditional mapping techniques often rely on expensive sensors like light detection and ranging (LiDAR) or 
RGB-D cameras, _VIRUS-NeRF_ integrates low-cost sensors while using neural radiance fields (NeRFs) for scene representation. 
By exhaustive sensor analysis and testing, the URM37 could be identified as the optimal ultrasonic sensor (USS). 
Building upon _Instant Neural Graphics Primitives_ (_Instant-NGP_), _VIRUS-NeRF_ incorporates 
depth measurements from USSs and infrared sensors (IRSs) and advances the occupancy grid utilized for ray marching. 
Experimental evaluation conducted at ETHZ demonstrates that _VIRUS-NeRF_ achieves comparable mapping 
performance to LiDAR point clouds in terms of coverage, and surpasses USS and IRS scans. Notably, in environments with 
optimized parameters, its accuracy aligns with that of LiDAR measurements, while in less optimized settings, it 
exhibits performance akin to USSs. Through an in-depth ablation study, three key factors of NeRF-based mapping are identified: 

* Utilizing _Instant-NGP_ for mapping yields poor results. The assistance of the camera with depth sensors is imperative
given the sparse measurements typical in mobile robotics.
* The proposed occupancy grid in _VIRUS-NeRF_ augments mapping performance and training efficiency. 
* Optimization of the large hyper-parameter space via particle swarm optimization and refinement of poses through bundle
adjustment enhances accuracy.

Limitations such as accuracy constraints, hyper-parameter generalization issues, and convergence speed are discussed and 
accompanied by possible solutions. Overall, _VIRUS-NeRF_ presents a promising approach for cost-effective local mapping 
in mobile robotics, with potential applications in safety and navigation tasks. 

__Keywords__: local mapping, NeRF, implicit neural representation, Instant-NGP, occupancy grid, low-cost sensors, infrared sensor, ultrasonic sensor, camera

# Code
## Installation
### VIRUS-NeRF
1. Navigate to the desired directory
2. Clone this repository
3. Make sure that the requirements indicated in _requirements.txt_ are met

### ETHZ Experiments
1. Install ROS1
2. Create a catkin workspace in _USS_experiments/catkin_ws_
3. Install the following packages inside _USS_experiments/catkin_ws/src_ to create a new dataset:
[BALM](https://github.com/hku-mars/BALM "BALM"),
[KISS-ICP](https://github.com/PRBonn/kiss-icp "Kiss_icp"),
[Rosserial](https://wiki.ros.org/rosserial "rosserial"),
[RS-to-Velodyne](https://github.com/HViktorTsoi/rs_to_velodyne "rs_velodyne") and
[Timed Roslaunch](https://wiki.ros.org/timed_roslaunch "timed_roslaunch")
4. Install the following packages inside _USS_experiments/catkin_ws/src_ to calibrate the sensors:
[Camera-LiDAR Calibration](https://github.com/acfr/cam_lidar_calibration "Camera-LiDAR_Calibration") and
[Kilibr](https://github.com/ethz-asl/kalibr "kalibr")

## Running
### VIRUS-NeRF
Choose the desired hyper-parameters as described below. Then execute one of the following scripts:
* Single run: _run.py_
* PSO optimization: _run_optimization.py_
* Ablation study: _run_ablation.py_
* Relaunch optimization continuously to circumvent memory leak of _Taichi Instant-NGP_ implementation: _watch_optimization.py_
* Relaunch ablation continuously to circumvent memory leak of _Taichi Instant-NGP_ implementation: _watch_ablation.py_

### USS Experiments
1. Connect Arduino and USS
2. Flash Arduino with the desired script
3. Execute _USS_experiments/read_data.py_

### ETHZ Experiments
1. Connect Arduino and sensor stacks (USS, IRS and camera)
2. Flash Arduino with _ETHZ_experimens/Arduino/sensor_stack/sensor_stack.ino_
3. Navigate to: _ETHZ_experimens/catkin_ws/src/sensors/_
4. Launch logging file to collect the data in Rosbags: _roslaunch sensors Launch/stack_log.launch_
5. Crop Rosbag to have correct start and ending time: _rosbag filter <bag name> <new name> <condition>_
6. Create LiDAR poses file and Rosbag with filtered LiDAR pointcloud by launching: _roslaunch sensors Launch/lidarFilter_poseEstimation.launch_
7. Read out filtered LiDAR pointcloud as pcd files: _rosrun pcl_ros bag_to_pcd lidar_filtered.bag /rslidar_filtered ./lidars/filtered_
8. Create dataset for BALM: _run src/data_tools/main_balm.py_
9. Optimize poses with BALM: _roslaunch balm2 ethz_dataset.launch_
10. Create ground truth maps: _run src/pcl_tools/pcl_merge.py_
11. Synchronize data: _run src/data_tools/main_syncData.py_
12. Create final dataset: _run src/data_tools/main_rosbag2dataset.py_
13. Create pose lookup tables _run src/data_tools/main_createPoseLookupTables.py_
14. Visualize dataset: _roslaunch sensors Launch/simulation.launch_

## Hyper-Parameters VIRUS-NeRF
The hyper-parameters can be set in the json files of the directory _args_:

Cathegory | Name | Meaning | Type | Options
| ---: | ---: | :--- | :--- | :--- 
dataset  | name  | dataset name | ETHZ or RH2 (Robot@Home2) |
dataset | spli_ratio | train, validation and test split ratios | dict of floats | must sum up to 1
dataset | keep_N_observations | number of samples to load | int |
dataset | keep_sensor | sensor name to use; only available with RH2 dataset | str | "all" means all sensors; "RGBD", "USS" or "ToF"
dataset | sensors | sensors to load | list of str | "RGBD", "USS" or "ToF"
model | ckpt_path | checkpoint path to load | bool or str | false means to start training from skratch
model | scale | scale of normalized scene | float |
model | encoder_type | encoder type | str | "hash" or "triplane"
model | hash_levels | number of hash levels | int |
model | hash_max_res | resolution of finest hash level | int |
model | grid_type | type of occupancy grid | str | "occ" (VIRUS-NeRF) or "ngp" (Instant-NGP)
model | save | save model after training | bool |
training | batch_size | training batch size | int | 
training | sampling_strategy | sampling strategy for images and pixels | dict | images: "all" or "same"; pixels: "entire_img", "random", "valid_uss" or "valid_tof"
training | sensors | sensors used for training | list of str | "RGBD", "USS" or "ToF"
training | max_steps | maximum amount of training steps | int | 
training | max_time | maximum amount of training time | float | 
training | lr | learning rate | float | 
training | rgbd_loss_w | loss weight of RGBD sensor | float | 
training | tof_loss_w | loss weight of IRS (ToF) sensor | float | 
training | uss_loss_w | loss weight of USS sensor | float | 
training | color_loss_w | loss weight of camera | float | 
training | debug_mode | test intermediate results | bool | 
training | real_time_simulation | simulate measurements been done in real-time experiment | bool | 
evaluation | batch_size | evaluation batch size | int | 
evaluation | res_map | side length resolution of evaluation maps | int | 
evaluation | eval_every_n_steps | intermediate evaluation every given steps | int | 
evaluation | num_color_pts | number of colour images to evaluate after training | int | 
evaluation | num_depth_pts | number of depth scans to evaluate after training | int | 
evaluation | num_plot_pts | number of intermediate depth scans to evaluate during training | int | 
evaluation | height_tolerance | distance to consider above and bellow measurements for evaluation | float | 
evaluation | density_map_thr | density threshold for occupancy grid plots | float | 
evaluation | inlier_threshold | inlier/outlier theshold distance in meters for NND plots | float | 
evaluation | zones | definition of zone ranges | dict of lists | 
evaluation | sensors | sensors to evaluate | list of str | "GT", "USS", "ToF", "LiDAR" or "NeRF"
evaluation | plot_results | wheather to generate plots | bool | 
ngp_grid | update_interval | update grid every given steps | int | 
ngp_grid | warmup_steps | sample all cells for the given first steps | int | 
occ_grid | batch_size | batch size of occupancy grid update | int | 
occ_grid | update_interval | update grid every given steps | int | 
occ_grid | decay_warmup_steps | reduce cell values exponentially for given number of steps | int | 
occ_grid | batch_ratio_ray_update | ratio of _Depth-Update_; the rest will be _NeRF-Update_ | float | between 0 and 1
occ_grid | false_detection_prob_every_m | false detection probability of sensor model (_Depth-Update_) every meter | float | 
occ_grid | std_every_m | standard deviation of sensor model (_Depth-Update_) every meter | float | 
occ_grid | nerf_pos_noise_every_m | position noise added during _NeRF-Update_ | float | 
occ_grid | nerf_threshold_max | maximum density threshold for _NeRF-Update_ | float | 
occ_grid | nerf_threshold_slope | density convertion slope for _NeRF-Update_ | float | 
ethz | dataset_dir | path to dataset directory | str | 
ethz | room | name of envrionment | str | "office", "office2", "commonroom", "commonroom2", "corridor"
ethz | cam_ids | camera identity numbers to load | list of str | "CAM1" or "CAM3"
ethz | use_optimized_poses | use optimized poses | bool | 
RH2 | dataset_dir | path to dataset directory | str | 
RH2 | session | session name | str | 
RH2 | home | home name | str | 
RH2 | room | room name | str | 
RH2 | subsession | subsession name | str | 
RH2 | home_session | home session id | str | 
RGBD | angle_of_view | angle of view of depth camera in degrees | list of float | 
USS | angle_of_view | ellipsoid angle of view of USS in degrees | list of float | 
USS | angle_of_view | angle of view of IRS (ToF) in degrees | list of float | 
USS | matrix | number of beams | list of ints | 
USS | sensor_calibration_error | angular calibration error added to IRS (ToF) measurements in degrees | float | 
USS | sensor_random_error | add random error to IRS (ToF) depth measurement in meters | float | 
LiDAR | angle_min_max | field of view of LiDAR for given rooms | dict of lists | 

# Citations
For a complete list of citations please refer to the report.

## Code
The [Taichi](https://github.com/Linyou/taichi-ngp-renderer "taichi_ngp")  _Instant-NGP_ implementation is used for this project. 
All [Taichi](https://github.com/Linyou/taichi-ngp-renderer "taichi_ngp") code is contained inside the _modules_ directory 
(except of _modules/grid.py_ and _modules/occupancy_grid.py_ which are written by myself).

## Algorithm
_VIRUS-NeRF_ is based on _Instant-NGP_: Instant neural graphics primitives with a multiresolution hash encoding; Müller, Thomas and Evans, 
Alex and Schied, Christoph and Keller, Alexander; ACM Transactions on Graphics (ToG); Year: 2022
 
