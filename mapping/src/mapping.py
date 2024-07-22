import os
import random
import time
from statistics import mean

import message_filters
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image
from tqdm import tqdm

from criterion import Criterion
from frame import RGBDFrame
from functions.initialize_sdf import initemb_sdf
from functions.low_freq_directions import low_freq_directions
from functions.render_helpers import bundle_adjust_frames
from functions.render_helpers import fill_in, render_rays
from functions.weak_texture import weak_texture
from loggers import BasicLogger
from utils.filter_utils import edge_conv2d
from utils.import_util import get_decoder, get_property
from utils.keyframe_util import multiple_max_set_coverage
from utils.mesh_util import MeshExtractor
from utils.sample_util import updownsampling_voxel

torch.classes.load_library(
    "third_party/sparse_octree/build/lib.linux-x86_64-cpython-39/svo.cpython-39-x86_64-linux-gnu.so")


class Mapping:
    def __init__(self, args, logger: BasicLogger, data_stream=None, **kwargs):
        super().__init__()
        self.args = args
        self.logger = logger
        mapper_specs = args.mapper_specs
        debug_args = args.debug_args
        data_specs = args.data_specs
        decoder_specs = args.decoder_specs
        low_freq_directions_specs = args.low_freq_directions_specs

        # get data stream
        if data_stream != None:
            self.data_stream = data_stream
            self.start_frame = mapper_specs["start_frame"]
            self.end_frame = mapper_specs["end_frame"]
            if self.end_frame == -1:
                self.end_frame = len(self.data_stream)
            self.start_frame = min(self.start_frame, len(self.data_stream))
            self.end_frame = min(self.end_frame, len(self.data_stream))

        self.decoder = get_decoder(args).cuda()
        self.loss_criteria = Criterion(args)
        # keyframes set
        self.kf_graph = []
        # used for Coverage-maximizing keyframe selection
        self.kf_seen_voxel = []
        self.kf_seen_voxel_num = []
        self.kf_svo_idx = []
        self.kf_voxel_weight = []
        self.kf_unoptimized_voxels = None
        self.kf_optimized_voxels = None
        self.kf_all_voxels = None
        self.kf_voxel_freq = None

        # optional args
        self.ckpt_freq = get_property(args, "ckpt_freq", -1)
        self.final_iter = get_property(mapper_specs, "final_iter", 0)
        self.mesh_res = get_property(mapper_specs, "mesh_res", 8)
        self.save_data_freq = get_property(debug_args, "save_data_freq", 0)

        # required args
        self.adaptive_ending = mapper_specs["adaptive_ending"]
        self.voxel_size = mapper_specs["voxel_size"]
        self.kf_window_size = mapper_specs["kf_window_size"]
        self.num_iterations = mapper_specs["num_iterations"]
        self.n_rays = mapper_specs["N_rays_each"]
        self.max_voxel_hit = mapper_specs["max_voxel_hit"]
        self.step_size = mapper_specs["step_size"] * self.voxel_size
        self.inflate_margin_ratio = mapper_specs["inflate_margin_ratio"]
        self.kf_selection_random_radio = mapper_specs["kf_selection_random_radio"]
        self.offset = mapper_specs["offset"]
        self.kf_selection_method = mapper_specs["kf_selection_method"]
        self.insert_method = mapper_specs["insert_method"]
        self.insert_ratio = mapper_specs["insert_ratio"]
        self.num_vertexes = mapper_specs["num_vertexes"]
        self.min_point_each_voxel = mapper_specs["min_point_each_voxel"]
        self.new_sdf_strategy = mapper_specs["new_sdf_strategy"]
        self.runtime_analysis = mapper_specs["runtime_analysis"]
        self.cull_mesh = mapper_specs["cull_mesh"]
        self.adaptive_kf_pruning = mapper_specs["adaptive_kf_pruning"]
        self.gradient_aided_coverage_max = mapper_specs["gradient_aided_coverage_max"]

        self.heterogeneous_grids = decoder_specs["heterogeneous_grids"]

        self.use_gt = data_specs["use_gt"]
        self.max_distance = data_specs["max_depth"]

        self.render_freq = debug_args["render_freq"]
        self.render_res = debug_args["render_res"]
        self.mesh_freq = debug_args["mesh_freq"]
        self.save_ckpt_freq = debug_args["save_ckpt_freq"]

        self.sdf_truncation = args.criteria["sdf_truncation"]

        if self.heterogeneous_grids:
            self.max_low_freq_dir_num = low_freq_directions_specs["max_low_freq_dir_num"]
            self.line_similar_th = low_freq_directions_specs["line_similar_th"]
            self.pc_dist_th = low_freq_directions_specs["pc_dist_th"]
            self.ksize = low_freq_directions_specs["ksize"]
            self.valid_line_th = low_freq_directions_specs["valid_line_th"]
            self.update_interval = low_freq_directions_specs["update_interval"]

        if self.runtime_analysis:
            self.runtime_analysis_all_list = []
            self.runtime_analysis_sdf_priors_list = []
            self.runtime_analysis_hetero_list = []
            self.runtime_analysis_kf_list = []
            self.runtime_analysis_optim_list = []
            self.runtime_analysis_data_process_list = []

        self.run_ros = args.run_ros
        if self.run_ros:
            ros_args = args.ros_args
            self.intrinsic = np.eye(3)
            self.intrinsic[0, 0] = ros_args["intrinsic"][0]
            self.intrinsic[1, 1] = ros_args["intrinsic"][1]
            self.intrinsic[0, 2] = ros_args["intrinsic"][2]
            self.intrinsic[1, 2] = ros_args["intrinsic"][3]
            print("intrinsic: ", self.intrinsic)
            self.color_topic = ros_args["color_topic"]
            self.depth_topic = ros_args["depth_topic"]
            self.pose_topic = ros_args["pose_topic"]

        self.mesher = MeshExtractor(args)

        self.sdf_priors = torch.zeros(
            (self.num_vertexes, 1),
            requires_grad=True, dtype=torch.float32,
            device=torch.device("cuda"))

        self.svo = torch.classes.svo.Octree()
        self.svo.init(256, int(self.num_vertexes), self.voxel_size)  # Must be a multiple of 2
        self.optimize_params = [{'params': self.decoder.parameters(), 'lr': 1e-2},
                                {'params': self.sdf_priors, 'lr': 1e-2}]

        self.optim = torch.optim.Adam(self.optimize_params)
        self.scaler = torch.cuda.amp.GradScaler()

        self.frame_poses = []
        self.pose_all = []

        # used for adaptive keyframe pruning
        self.round_num = 0
        self.kf_latest_select_round = []
        self.kf_insert_round = []

    def callback(self, color, depth, pose_vins):
        update_pose = False
        bridge = CvBridge()
        color_image_raw = bridge.imgmsg_to_cv2(color, desired_encoding="passthrough")
        depth_image = bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
        q = pose_vins.pose.pose.orientation
        dcm = Rotation.from_quat(np.array([q.x, q.y, q.z, q.w])).as_matrix()
        trans = pose_vins.pose.pose.position
        trans = np.array([trans.x, trans.y, trans.z])
        pose = np.eye(4)
        pose[:3, :3] = dcm
        pose[:3, 3] = trans
        depth_image = depth_image * 0.001
        if self.max_distance > 0:
            depth_image[(depth_image > self.max_distance)] = 0
        color_image = color_image_raw / 255

        tracked_frame = RGBDFrame(self.idx, color_image, depth_image, K=self.intrinsic, offset=self.offset,
                                  ref_pose=pose, img_raw=color_image_raw)
        if (tracked_frame.depth > 0).sum() > self.n_rays:
            self.mapping_step(self.idx, tracked_frame, update_pose)
            self.idx += 1
            print("idx: ", self.idx)

    def mapping_step(self, frame_id, tracked_frame, update_pose):
        ######################
        self.idx = tracked_frame.stamp
        self.create_voxels(tracked_frame)
        if self.runtime_analysis:
            torch.cuda.synchronize()
            self.start_sdf_priors_2 = time.time()

        self.map_states, self.voxel_initialized, self.vertex_initialized = initemb_sdf(tracked_frame,
                                                                                       self.map_states,
                                                                                       self.sdf_truncation,
                                                                                       voxel_size=self.voxel_size,
                                                                                       voxel_initialized=self.voxel_initialized,
                                                                                       octant_idx=self.octant_idx,
                                                                                       vertex_initialized=self.vertex_initialized,
                                                                                       use_gt=self.use_gt)
        self.sdf_priors = self.map_states["sdf_priors"]
        if self.runtime_analysis:
            torch.cuda.synchronize()
            self.end_sdf_priors_2 = time.time()
            self.runtime_analysis_sdf_priors_list.append(
                (self.end_sdf_priors_1 - self.start_sdf_priors_1) + (self.end_sdf_priors_2 - self.start_sdf_priors_2))

        if self.idx == 0:
            self.insert_kf(tracked_frame)
        self.do_mapping(tracked_frame=tracked_frame, update_pose=update_pose)

        if self.runtime_analysis:
            torch.cuda.synchronize()
            self.start_kf_2 = time.time()

        # Fixed 50 frames to insert pictures(naive)
        if (tracked_frame.stamp - self.current_kf.stamp) > 50 and self.insert_method == "naive":
            self.insert_kf(tracked_frame)
        # The keyframe strategy we designed
        if self.insert_method == "intersection":
            insert_bool = self.voxel_field_insert_kf(self.insert_ratio)
            if insert_bool or (tracked_frame.stamp - self.current_kf.stamp) > 10:
                self.insert_kf(tracked_frame)

        if self.runtime_analysis:
            torch.cuda.synchronize()
            self.end_kf_2 = time.time()
            self.runtime_analysis_kf_list.append(
                float(self.end_kf_1 - self.start_kf_1) + float(self.end_kf_2 - self.start_kf_2))

        if not self.runtime_analysis:
            self.tracked_pose = tracked_frame.get_ref_pose().detach() @ tracked_frame.get_d_pose().detach()
            ref_pose = self.current_kf.get_ref_pose().detach() @ self.current_kf.get_d_pose().detach()
            rel_pose = torch.linalg.inv(ref_pose) @ self.tracked_pose
            self.frame_poses += [(len(self.kf_graph) - 1, rel_pose.cpu())]

            if self.mesh_freq > 0 and (tracked_frame.stamp + 1) % self.mesh_freq == 0:
                self.logger.log_mesh(self.extract_mesh(
                    res=self.mesh_res, clean_mesh=False, map_states=self.map_states),
                    name=f"mesh_{tracked_frame.stamp:06d}.ply")

            if self.save_data_freq > 0 and (tracked_frame.stamp + 1) % self.save_data_freq == 0:
                self.save_debug_data(tracked_frame)

            if self.render_freq > 0 and (frame_id + 1) % self.render_freq == 0:
                self.render_debug_images(tracked_frame)

        if self.save_ckpt_freq > 0 and (tracked_frame.stamp + 1) % self.save_ckpt_freq == 0:
            self.logger.log_ckpt(self, name=f"{tracked_frame.stamp:06d}.pth")

        self.pose_all.append(tracked_frame.ref_pose.detach().cpu().numpy())

    def run(self, first_frame, update_pose):
        self.idx = 0
        self.voxel_initialized = torch.zeros(self.num_vertexes).cuda().bool()
        self.vertex_initialized = torch.zeros(self.num_vertexes).cuda().bool()
        if self.heterogeneous_grids:
            self.line_compress = low_freq_directions(line_similar_th=self.line_similar_th,
                                                     pc_dist_th=self.pc_dist_th,
                                                     ksize=self.ksize,
                                                     valid_line_th=self.update_interval,
                                                     max_low_freq_dir_num=self.max_low_freq_dir_num,
                                                     num_vertexes=self.num_vertexes)
            self.weak_texture = weak_texture(self.min_point_each_voxel, self.num_vertexes)
        if self.run_ros:
            rospy.init_node('listener', anonymous=True)
            # realsense
            color_sub = message_filters.Subscriber(self.color_topic, Image)
            depth_sub = message_filters.Subscriber(self.depth_topic, Image)
            pose_sub = message_filters.Subscriber(self.pose_topic, Odometry)

            ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub, pose_sub], 2, 1 / 10,
                                                             allow_headerless=False)
            print(" ========== MAPPING START ===========")
            ts.registerCallback(self.callback)
            rospy.spin()
        else:
            if self.mesher is not None:
                self.mesher.rays_d = first_frame.get_rays()

            self.create_voxels(first_frame)

            self.map_states, self.voxel_initialized, self.vertex_initialized = initemb_sdf(first_frame,
                                                                                           self.map_states,
                                                                                           self.sdf_truncation,
                                                                                           voxel_size=self.voxel_size,
                                                                                           voxel_initialized=self.voxel_initialized,
                                                                                           octant_idx=self.octant_idx,
                                                                                           vertex_initialized=self.vertex_initialized,
                                                                                           use_gt=self.use_gt)
            self.sdf_priors = self.map_states["sdf_priors"]
            self.insert_kf(first_frame)
            self.do_mapping(tracked_frame=first_frame, update_pose=update_pose)

            self.tracked_pose = first_frame.get_ref_pose().detach() @ first_frame.get_d_pose().detach()
            ref_pose = self.current_kf.get_ref_pose().detach() @ self.current_kf.get_d_pose().detach()
            rel_pose = torch.linalg.inv(ref_pose) @ self.tracked_pose
            self.frame_poses += [(len(self.kf_graph) - 1, rel_pose.cpu())]
            self.pose_all.append(first_frame.ref_pose.detach().cpu().numpy())
            print("mapping started!")

            progress_bar = tqdm(range(self.start_frame, self.end_frame), position=0)
            progress_bar.set_description("mapping frame")
            for frame_id in progress_bar:
                if self.runtime_analysis:
                    torch.cuda.synchronize()
                    start = time.time()
                if self.runtime_analysis:
                    torch.cuda.synchronize()
                    self.start_data_process = time.time()
                fid, rgb, depth, K, ref_pose, img_raw = self.data_stream[frame_id]
                if self.use_gt:
                    tracked_frame = RGBDFrame(fid, rgb, depth, K, offset=self.offset, ref_pose=ref_pose,
                                              img_raw=img_raw)
                else:
                    tracked_frame = RGBDFrame(fid, rgb, depth, K, offset=self.offset,
                                              ref_pose=self.tracked_pose.clone(), img_raw=img_raw)
                if update_pose is False:
                    tracked_frame.d_pose.requires_grad_(False)
                if tracked_frame.ref_pose.isinf().any():
                    continue
                if (tracked_frame.depth > 0).sum() < self.n_rays:
                    continue
                if self.runtime_analysis:
                    torch.cuda.synchronize()
                    self.end_data_process = time.time()
                    self.runtime_analysis_data_process_list.append(self.end_data_process - self.start_data_process)
                self.mapping_step(frame_id, tracked_frame, update_pose)
                if self.runtime_analysis:
                    torch.cuda.synchronize()
                    end = time.time()
                    self.runtime_analysis_all_list.append(float(end - start))

        if self.runtime_analysis:
            res_file_path = self.logger.log_dir + "/time.txt"
            print("Frame Processing Time per frame: ", mean(self.runtime_analysis_all_list))
            with open(res_file_path, 'a') as file:
                file.truncate(0)
                file.write("Frame Processing Time per frame:" + str(mean(self.runtime_analysis_all_list)) + '\n')
                file.write("Time for Map Training per frame:" + str(mean(self.runtime_analysis_optim_list[1:])) + '\n')
                file.write(
                    "Time for Octree SDF priors per frame:" + str(mean(self.runtime_analysis_sdf_priors_list)) + '\n')
                file.write("Time for Keyframe Policy per frame:" + str(mean(self.runtime_analysis_kf_list[1:])) + '\n')
                if self.heterogeneous_grids:
                    file.write("Time for Texture Pattern Determination per 10 frame:" + str(
                        mean(self.runtime_analysis_hetero_list[1:])) + '\n')
                file.write("Time for Data Reading & Prepocessing per frame:" + str(
                    mean(self.runtime_analysis_data_process_list)) + '\n')
        print("******* mapping process died *******")
        print("Keyframe_number: ", len(self.kf_graph))
        print("******* extracting final mesh *******")
        if not self.adaptive_kf_pruning:
            pose = self.get_updated_poses()
        else:
            pose = []
            for ref_frame in self.kf_graph:
                ref_pose = ref_frame.get_ref_pose().detach().cpu() @ ref_frame.get_d_pose().detach().cpu()
                frame_pose = ref_pose
                pose += [frame_pose.detach().cpu().numpy()]
        self.kf_graph = None
        mesh = self.extract_mesh(res=self.mesh_res, clean_mesh=False, map_states=self.map_states)
        self.logger.log_mesh(mesh)
        mesh_out_file = os.path.join(self.logger.mesh_dir, "final_mesh.ply")
        if self.cull_mesh:
            print("******* culling final mesh *******")
            self.mesher.cull_mesh(mesh_out_file, "cuda", self.data_stream, self.sdf_truncation)
        self.logger.log_ckpt(self, name="final_ckpt.pth")
        pose = np.asarray(pose)
        pose[:, 0:3, 3] -= self.offset
        self.logger.log_numpy_data(pose, "frame_poses_kf")
        pose_all = np.asarray(self.pose_all)
        pose_all[:, 0:3, 3] -= self.offset
        self.logger.log_numpy_data(pose_all, "frame_poses")
        self.logger.log_mesh(mesh)
        self.logger.log_numpy_data(self.extract_voxels(map_states=self.map_states), "final_voxels")
        print("******* mapping process died *******")

    def initfirst_onlymap(self):
        init_pose = self.data_stream.get_init_pose(self.start_frame)
        fid, rgb, depth, K, ref_pose, img_raw = self.data_stream[self.start_frame]
        first_frame = RGBDFrame(fid, rgb, depth, K, offset=self.offset, ref_pose=init_pose, img_raw=img_raw)
        first_frame.d_pose.requires_grad_(False)

        print("******* initializing first_frame: %d********" % first_frame.stamp)
        self.last_frame = first_frame
        self.start_frame += 1
        return first_frame

    def do_mapping(self, tracked_frame=None, update_pose=True):
        self.decoder.train()
        if self.runtime_analysis:
            torch.cuda.synchronize()
            self.start_kf_1 = time.time()
        optimize_targets = self.select_optimize_targets(tracked_frame)
        if self.runtime_analysis:
            torch.cuda.synchronize()
            self.end_kf_1 = time.time()

        if self.runtime_analysis:
            torch.cuda.synchronize()
            self.start_opt = time.time()
        bundle_adjust_frames(
            optimize_targets,
            self.map_states,
            self.decoder,
            self.loss_criteria,
            self.voxel_size,
            self.step_size,
            self.n_rays,
            self.num_iterations,
            self.sdf_truncation,
            self.max_voxel_hit,
            self.max_distance,
            update_pose=update_pose,
            optim=self.optim,
            scaler=self.scaler,
            frame_id=tracked_frame.stamp,
            adaptive_ending=self.adaptive_ending,
        )
        if self.runtime_analysis:
            torch.cuda.synchronize()
            self.end_opt = time.time()
            self.runtime_analysis_optim_list.append(float(self.end_opt - self.start_opt))

    def select_optimize_targets(self, tracked_frame=None):
        targets = []
        selection_method = self.kf_selection_method
        if len(self.kf_graph) <= self.kf_window_size:
            targets = self.kf_graph[:]
        elif selection_method == 'random':
            targets = random.sample(self.kf_graph, self.kf_window_size)
        elif selection_method == 'multiple_max_set_coverage':
            targets, self.kf_unoptimized_voxels, self.kf_optimized_voxels, self.kf_all_voxels, \
            self.kf_voxel_freq, self.round_num, self.kf_latest_select_round, self.kf_insert_round, \
            self.kf_seen_voxel, self.kf_voxel_weight = multiple_max_set_coverage(
                self.kf_graph,
                self.kf_seen_voxel_num,
                self.kf_unoptimized_voxels,
                self.kf_optimized_voxels,
                self.kf_window_size,
                self.kf_svo_idx,
                self.kf_all_voxels,
                self.num_vertexes,
                self.kf_voxel_freq,
                self.round_num,
                self.kf_latest_select_round,
                self.kf_insert_round,
                self.kf_seen_voxel,
                self.kf_voxel_weight)
        if tracked_frame is not None and (tracked_frame != self.current_kf):
            targets += [tracked_frame]
        return targets

    def insert_kf(self, frame):
        self.last_kf_observed_num = self.current_seen_voxel_num
        self.current_kf = frame
        self.last_kf_seen_voxel = self.seen_voxel
        self.kf_graph += [frame]
        self.kf_seen_voxel += [self.seen_voxel]
        self.kf_seen_voxel_num += [self.last_kf_observed_num]
        self.kf_svo_idx += [self.svo_idx]
        if self.gradient_aided_coverage_max:
            self.kf_voxel_weight += [self.valid_voxel_color_edge_sum]
        if self.adaptive_kf_pruning:
            self.kf_insert_round += [self.round_num]
            self.kf_latest_select_round += [0]
        if self.kf_all_voxels is None:
            self.kf_all_voxels = torch.zeros(self.num_vertexes).cuda().bool()  # All voxels to be optimized
            self.kf_voxel_freq = torch.zeros(self.num_vertexes).cuda()  # voxel frequency
        self.kf_all_voxels[self.svo_idx.long() + 1] += True
        self.kf_all_voxels[0] = False
        self.kf_voxel_freq[self.svo_idx.long() + 1] += 1

    def voxel_field_insert_kf(self, insert_ratio):
        # compute intersection
        voxel_no_repeat, cout = torch.unique(torch.cat([self.last_kf_seen_voxel,
                                                        self.seen_voxel], dim=0), return_counts=True, sorted=False,
                                             dim=0)
        N_i = voxel_no_repeat[cout > 1].shape[0]
        N_a = voxel_no_repeat.shape[0]
        ratio = N_i / N_a
        if ratio < insert_ratio:
            return True
        return False

    def create_voxels(self, frame):
        if self.runtime_analysis:
            torch.cuda.synchronize()
            self.start_sdf_priors_1 = time.time()

        # get the point clouds
        points_raw, points_valid_mask = frame.get_points_and_mask()
        points_raw = points_raw.cuda()
        points_valid_mask = points_valid_mask.cuda()
        if self.use_gt:
            pose = frame.get_ref_pose().cuda()
        else:
            pose = frame.get_ref_pose().cuda() @ frame.get_d_pose().cuda()
        points_raw = points_raw @ pose[:3, :3].transpose(-1, -2) + pose[:3, 3]
        points = points_raw[points_valid_mask]  # change to world frame (Rx)^T = x^T R^T

        # Voxelization
        voxels = torch.div(points, self.voxel_size, rounding_mode='floor')
        voxels_raw, inverse_indices, counts = torch.unique(voxels, dim=0, return_inverse=True, return_counts=True)
        voxels_vaild = voxels_raw[counts > self.min_point_each_voxel]
        voxels_vaild_before_inflate = voxels_vaild.clone()
        offsets = torch.LongTensor([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]]).to(
            voxels.device)

        # Expanded Voxels Allocation
        inflate_margin_ratio = self.inflate_margin_ratio
        updownsampling_points = updownsampling_voxel(points, inverse_indices, counts, "mean")
        for offset in offsets:
            offset_axis = offset.nonzero().item()
            if offset[offset_axis] > 0:
                margin_mask = updownsampling_points[:, offset_axis] % self.voxel_size > (
                        1 - inflate_margin_ratio) * self.voxel_size
            else:
                margin_mask = updownsampling_points[:,
                              offset_axis] % self.voxel_size < inflate_margin_ratio * self.voxel_size
            margin_vox = voxels_raw[margin_mask * (counts > self.min_point_each_voxel)]
            voxels_vaild = torch.cat((voxels_vaild, torch.clip(margin_vox + offset, min=0)), dim=0)
        inflate_num = voxels_vaild.shape[0] - voxels_vaild_before_inflate.shape[0]
        self.seen_voxel = voxels_vaild
        self.current_seen_voxel_num = voxels_vaild.shape[0]

        # Update the sparse voxel octree
        voxels_svo, children_svo, vertexes_svo, svo_mask, svo_idx, structure_idx = self.svo.insert(
            voxels_vaild.cpu().int())
        svo_mask = svo_mask[:, 0].bool()
        voxels_svo = voxels_svo[svo_mask]
        children_svo = children_svo[svo_mask]
        vertexes_svo = vertexes_svo[svo_mask]
        structure_idx = structure_idx.cuda()
        self.octant_idx = svo_mask.nonzero().cuda()
        self.svo_idx = svo_idx.cuda()
        self.update_grid(voxels_svo, children_svo, vertexes_svo, svo_idx)

        if self.runtime_analysis:
            torch.cuda.synchronize()
            self.end_sdf_priors_1 = time.time()
        if self.heterogeneous_grids:
            if self.runtime_analysis and frame.stamp % self.update_interval == 0:
                torch.cuda.synchronize()
                start_heter = time.time()

        # get the color and depth edge
        color_edge_conv = edge_conv2d(3, "cuda")
        depth_edge_conv = edge_conv2d(1, "cuda")
        # remove the depth edge pixel in the color edge
        color_edge = color_edge_conv(frame.rgb.permute(2, 0, 1).unsqueeze(0))[0, 0].detach().abs()
        depth_edge = depth_edge_conv(frame.depth.unsqueeze(0).unsqueeze(0))[0, 0].detach().abs()
        color_edge[depth_edge > 0.01] = 0
        color_edge = color_edge[points_valid_mask]

        #  Texture Pattern Determination
        if self.heterogeneous_grids:
            if frame.stamp % self.update_interval == 0:
                self.line_compress.texture_pattern_determination(frame.img_raw, points_raw, self.map_states,
                                                                 self.voxel_size, pose, frame.K)
            voxel_color_edge_mean = updownsampling_voxel(color_edge.unsqueeze(1), inverse_indices, counts, mode="mean")
            self.weak_texture.texture_pattern_determination(voxel_color_edge_mean, counts, structure_idx, inflate_num)

        # get the score of each voxel for keyframe selection
        if self.gradient_aided_coverage_max:
            # updownsampling_voxel is follow the order in the value of inverse_indices
            # 24 is because the color_edge_conv
            voxel_color_edge_sum = updownsampling_voxel(color_edge.unsqueeze(1) / 24, inverse_indices, counts,
                                                        mode="sum")
            valid_voxel_color_edge_sum = voxel_color_edge_sum[counts > self.min_point_each_voxel] * counts[
                counts > self.min_point_each_voxel].unsqueeze(-1)
            valid_voxel_color_edge_sum[valid_voxel_color_edge_sum < 1] = 1
            self.valid_voxel_color_edge_sum = torch.ones_like(
                self.svo_idx) * 1  # the weight of inflated voxel will be set to 1
            if inflate_num != 0:
                self.valid_voxel_color_edge_sum[:-inflate_num] = valid_voxel_color_edge_sum
            else:
                self.valid_voxel_color_edge_sum = valid_voxel_color_edge_sum.clone()

        if self.heterogeneous_grids:
            self.map_states["compress_simple_area_mask_voxel"] = self.weak_texture.compress_simple_area_mask_voxel
            self.map_states["compress_rot_info_voxel"] = self.line_compress.compress_rot_info_voxel

        if self.heterogeneous_grids:
            if self.runtime_analysis and frame.stamp % self.update_interval == 0:
                torch.cuda.synchronize()
                end_heter = time.time()
                self.runtime_analysis_hetero_list.append(end_heter - start_heter)

    @torch.enable_grad()
    def update_grid(self, voxels, children, vertexes, svo_idx):

        centres = (voxels[:, :3] + voxels[:, -1:] / 2) * self.voxel_size
        children = torch.cat([children, voxels[:, -1:]], -1)

        centres = centres.cuda().float()
        children = children.cuda().int()

        map_states = {}
        map_states["voxels"] = voxels.cuda()
        map_states["voxel_vertex_idx"] = vertexes.cuda()
        map_states["voxel_center_xyz"] = centres.cuda()
        map_states["voxel_structure"] = children.cuda()
        map_states["sdf_priors"] = self.sdf_priors
        map_states["svo_idx"] = svo_idx.cuda()
        map_states["new_sdf_strategy"] = self.new_sdf_strategy
        map_states["heterogeneous_grids"] = self.heterogeneous_grids

        self.map_states = map_states

    @torch.no_grad()
    def get_updated_poses(self):
        frame_poses = []
        for i in range(len(self.frame_poses)):
            ref_frame_ind, rel_pose = self.frame_poses[i]
            ref_frame = self.kf_graph[ref_frame_ind]
            ref_pose = ref_frame.get_ref_pose().detach().cpu() @ ref_frame.get_d_pose().detach().cpu()
            pose = ref_pose @ rel_pose
            frame_poses += [pose.detach().cpu().numpy()]
        return frame_poses

    """
    Get the mesh at the position of the voxel
    Args:
        res: The number of points collected in each dimension in each voxel.
        clean_mesh: Whether to keep only the mesh of the current frame.
        map_states: state parameters of the map.
    Returns:
        mesh
    """

    @torch.no_grad()
    def extract_mesh(self, res=8, clean_mesh=False, map_states=None):
        sdf_network = self.decoder
        sdf_network.eval()
        vertexes = map_states["voxel_vertex_idx"]
        voxels = map_states["voxels"]

        index = vertexes.eq(-1).any(-1)  # remove no smallest voxel
        voxels = voxels[~index.cpu(), :]
        vertexes = vertexes[~index.cpu(), :]
        centres = (voxels[:, :3] + voxels[:, -1:] / 2) * self.voxel_size

        encoder_states = {}
        encoder_states["voxel_vertex_idx"] = vertexes.cuda()
        encoder_states["voxel_center_xyz"] = centres.cuda()
        encoder_states["sdf_priors"] = map_states["sdf_priors"]
        encoder_states["heterogeneous_grids"] = map_states["heterogeneous_grids"]

        encoder_states_color = {}
        encoder_states_color["voxel_center_xyz"] = map_states["voxel_center_xyz"]
        encoder_states_color["heterogeneous_grids"] = map_states["heterogeneous_grids"]
        if self.heterogeneous_grids:
            encoder_states_color["compress_simple_area_mask_voxel"] = map_states["compress_simple_area_mask_voxel"]
            encoder_states_color["compress_rot_info_voxel"] = map_states["compress_rot_info_voxel"]

        mesh = self.mesher.create_mesh(
            self.decoder, encoder_states, self.voxel_size, voxels,
            frame_poses=None, depth_maps=None,
            clean_mesh=clean_mesh, require_color=True, offset=-self.offset, res=res,
            encoder_states_color=encoder_states_color)
        return mesh

    @torch.no_grad()
    def extract_voxels(self, map_states=None):
        vertexes = map_states["voxel_vertex_idx"]
        voxels = map_states["voxels"]

        index = vertexes.eq(-1).any(-1)
        voxels = voxels[~index.cpu(), :]
        voxels = (voxels[:, :3] + voxels[:, -1:] / 2) * \
                 self.voxel_size - self.offset
        return voxels

    @torch.no_grad()
    def save_debug_data(self, tracked_frame):
        """
        save per-frame voxel, mesh and pose
        """
        if self.use_gt:
            pose = tracked_frame.get_ref_pose().detach().cpu().numpy()
        else:
            pose = tracked_frame.get_ref_pose().detach().cpu().numpy() @ tracked_frame.get_d_pose().detach().cpu().numpy()
        pose[:3, 3] -= self.offset
        frame_poses = self.get_updated_poses()
        mesh = self.extract_mesh(res=8, clean_mesh=True)
        voxels = self.extract_voxels(map_states=self.map_states).detach().cpu().numpy()
        if self.use_gt:
            kf_poses = [p.get_ref_pose().detach().cpu().numpy()
                        for p in self.kf_graph]
        else:
            kf_poses = [p.get_ref_pose().detach().cpu().numpy() @ p.get_d_pose().detach().cpu().numpy()
                        for p in self.kf_graph]

        for f in frame_poses:
            f[:3, 3] -= self.offset
        for kf in kf_poses:
            kf[:3, 3] -= self.offset

        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        color = np.asarray(mesh.vertex_colors)

        self.logger.log_debug_data({
            "pose": pose,
            "updated_poses": frame_poses,
            "mesh": {"verts": verts, "faces": faces, "color": color},
            "voxels": voxels,
            "voxel_size": self.voxel_size,
            "keyframes": kf_poses,
            "is_kf": (tracked_frame == self.current_kf)
        }, tracked_frame.stamp)

    """"
    Used to render a complete picture
    """

    @torch.no_grad()
    def render_debug_images(self, current_frame, batch_size=200000):
        rgb = current_frame.rgb
        depth = current_frame.depth
        rotation = (current_frame.get_ref_pose().cuda() @ current_frame.get_d_pose().cuda())[:3, :3]
        ind = current_frame.stamp
        w, h = self.render_res

        decoder = self.decoder.cuda()
        map_states = {}
        for k, v in self.map_states.items():
            if type(v) == torch.Tensor:
                map_states[k] = v.cuda()
            else:
                map_states[k] = v

        rays_d = current_frame.get_rays(w, h).cuda()
        rays_d = rays_d @ rotation.transpose(-1, -2)

        rays_o = (current_frame.get_ref_pose().cuda() @ current_frame.get_d_pose().cuda())[:3, 3]
        rays_o = rays_o.unsqueeze(0).expand_as(rays_d)

        rays_o = rays_o.reshape(1, -1, 3).contiguous()
        rays_d = rays_d.reshape(1, -1, 3)
        torch.cuda.empty_cache()

        batch_size = batch_size
        ray_mask_list = []
        color_list = []
        depth_list = []
        # To prevent memory overflow, batch_size can be given according to the video memory
        with torch.cuda.amp.autocast():
            for batch_iter in range(0, rays_o.shape[1], batch_size):
                final_outputs = render_rays(
                    rays_o[:, batch_iter:batch_iter + batch_size, :].clone(),
                    rays_d[:, batch_iter:batch_iter + batch_size, :].clone(),
                    map_states,
                    decoder,
                    self.step_size,
                    self.voxel_size,
                    self.sdf_truncation,
                    self.max_voxel_hit,
                    self.max_distance,
                    chunk_size=500000,
                    return_raw=True,
                    eval=True
                )
                if final_outputs["color"] == None:
                    ray_mask_list.append(final_outputs["ray_mask"])
                    continue
                ray_mask_list.append(final_outputs["ray_mask"])
                depth_list.append(final_outputs["depth"])
                color_list.append(final_outputs["color"])

        ray_mask_input = torch.cat(ray_mask_list, dim=1)

        if len(depth_list) == 0:
            return None, None, None

        depth_input = torch.cat(depth_list)
        color_input = torch.cat(color_list, dim=0)

        rdepth = fill_in((h, w, 1),
                         ray_mask_input.view(h, w),
                         depth_input, 0)
        rcolor = fill_in((h, w, 3),
                         ray_mask_input.view(h, w),
                         color_input, 0)
        if self.logger.for_eva:
            ssim, psnr, depth_L1 = self.logger.log_images(ind, rgb, depth, rcolor, rdepth)
            return ssim, psnr, depth_L1
        else:
            self.logger.log_images(ind, rgb, depth, rcolor, rdepth)
