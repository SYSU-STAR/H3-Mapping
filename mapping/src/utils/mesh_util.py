import numpy as np
import open3d as o3d
import torch
from functions.render_helpers import get_scores, eval_points
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
import trimesh
from tqdm import tqdm
import torch.nn.functional as F
from frame import RGBDFrame

class MeshExtractor:
    def __init__(self, args):
        self.voxel_size = args.mapper_specs["voxel_size"]
        self.rays_d = None
        self.depth_points = None
        self.model = args.decoder
        self.args=args

    @torch.no_grad()
    def linearize_id(self, xyz, n_xyz):
        return xyz[:, 2] + n_xyz[-1] * xyz[:, 1] + (n_xyz[-1] * n_xyz[-2]) * xyz[:, 0]

    @torch.no_grad()
    def downsample_points(self, points, voxel_size=0.01):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(voxel_size)
        return np.asarray(pcd.points)

    @torch.no_grad()
    def get_rays(self, w=None, h=None, K=None):
        w = self.w if w == None else w
        h = self.h if h == None else h
        if K is None:
            K = np.eye(3)
            K[0, 0] = self.K[0, 0] * w / self.w
            K[1, 1] = self.K[1, 1] * h / self.h
            K[0, 2] = self.K[0, 2] * w / self.w
            K[1, 2] = self.K[1, 2] * h / self.h
        ix, iy = torch.meshgrid(
            torch.arange(w), torch.arange(h), indexing='xy')
        rays_d = torch.stack(
            [(ix - K[0, 2]) / K[0, 0],
             (iy - K[1, 2]) / K[1, 1],
             torch.ones_like(ix)], -1).float()
        return rays_d

    @torch.no_grad()
    def get_valid_points(self, frame_poses, depth_maps):
        if isinstance(frame_poses, list):
            all_points = []
            print("extracting all points")
            for i in range(0, len(frame_poses), 5):
                pose = frame_poses[i]
                depth = depth_maps[i]
                points = self.rays_d * depth.unsqueeze(-1)
                points = points.reshape(-1, 3)
                points = points @ pose[:3, :3].transpose(-1, -2) + pose[:3, 3]
                if len(all_points) == 0:
                    all_points = points.detach().cpu().numpy()
                else:
                    all_points = np.concatenate(
                        [all_points, points.detach().cpu().numpy()], 0)
            print("downsample all points")
            all_points = self.downsample_points(all_points)
            return all_points
        else:
            pose = frame_poses
            depth = depth_maps
            points = self.rays_d * depth.unsqueeze(-1)
            points = points.reshape(-1, 3)
            points = points @ pose[:3, :3].transpose(-1, -2) + pose[:3, 3]
            if self.depth_points is None:
                self.depth_points = points.detach().cpu().numpy()
            else:
                self.depth_points = np.concatenate(
                    [self.depth_points, points], 0)
            self.depth_points = self.downsample_points(self.depth_points)
        return self.depth_points

    @torch.no_grad()
    def create_mesh(self, decoder, map_states, voxel_size, voxels,
                    frame_poses=None, depth_maps=None, clean_mesh=False,
                    require_color=True, offset=-10, res=8, encoder_states_color=None):

        sdf_grid = get_scores(decoder, map_states, voxel_size, bits=res, model=self.model)  # (num_voxels,res*3,4)

        sdf_grid = sdf_grid.reshape(-1, res, res, res, 1)

        voxel_centres = map_states["voxel_center_xyz"]
        verts, faces, verts_svo_idx = self.marching_cubes(voxel_centres, sdf_grid, map_states)


        if clean_mesh:
            print("********** get points from frames **********")
            all_points = self.get_valid_points(frame_poses, depth_maps)
            print("********** construct kdtree **********")
            kdtree = cKDTree(all_points)
            print("********** query kdtree **********")
            point_mask = kdtree.query_ball_point(
                verts, voxel_size * 0.5, workers=12, return_length=True)
            print("********** finished querying kdtree **********")
            point_mask = point_mask > 0
            face_mask = point_mask[faces.reshape(-1)].reshape(-1, 3).any(-1)

            faces = faces[face_mask]

        if require_color:
            print("********** get color from network **********")
            verts_torch = torch.from_numpy(verts).float().cuda()
            batch_points = torch.split(verts_torch, 10000)
            batch_verts_svo_idx = torch.split(verts_svo_idx, 10000)
            colors = []
            for idx, points in enumerate(batch_points):
                # voxel_pos = points // self.voxel_size
                voxel_pos = torch.div(points, self.voxel_size, rounding_mode='trunc')
                batch_voxels = voxels[:, :3].cuda()
                batch_voxels = batch_voxels.unsqueeze(
                    0).repeat(voxel_pos.shape[0], 1, 1)

                # filter outliers
                nonzeros = (batch_voxels == voxel_pos.unsqueeze(1)).all(-1)
                nonzeros = torch.where(nonzeros, torch.ones_like(
                    nonzeros).int(), -torch.ones_like(nonzeros).int())
                sorted, index = torch.sort(nonzeros, dim=-1, descending=True)
                sorted = sorted[:, 0]
                index = index[:, 0]
                valid = (sorted != -1)
                color_empty = torch.zeros_like(points)
                points = points[valid, :]
                index = index[valid]
                chunk_samples = {}
                if map_states['heterogeneous_grids']:
                    chunk_samples["sampled_point_voxel_idx"] = batch_verts_svo_idx[idx][valid]

                # get color
                if len(points) > 0:
                    color = eval_points(decoder, points, encoder_states_color, chunk_samples).cuda()
                    color_empty[valid] = color.float()
                colors += [color_empty]
            colors = torch.cat(colors, 0)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts + offset)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        if require_color:
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                colors.detach().cpu().numpy())
        mesh.compute_vertex_normals()
        return mesh

    @torch.no_grad()
    def marching_cubes(self, voxels, sdf, map_states):
        voxels = voxels[:, :3]  # (num_voxels,3)
        sdf = sdf[..., 0]  # extract sdf
        res = 1.0 / (sdf.shape[1] - 1)  # 1/(8-1)
        spacing = [res, res, res]

        num_verts = 0
        total_verts = []
        total_faces = []
        total_svo_idx = []
        for i in range(len(voxels)):
            sdf_volume = sdf[i].detach().cpu().numpy()  # (res,res,res)
            if np.min(sdf_volume) > 0 or np.max(sdf_volume) < 0:
                continue
            try:
                verts, faces, _, _ = marching_cubes(sdf_volume, 0, spacing=spacing)
            except:
                continue
            verts -= 0.5
            verts *= self.voxel_size
            verts += voxels[i].detach().cpu().numpy()
            faces += num_verts
            num_verts += verts.shape[0]

            total_verts += [verts]
            total_faces += [faces]
            total_svo_idx += [map_states["voxel_vertex_idx"][i, 0]] * verts.shape[0]

        total_verts = np.concatenate(total_verts)
        total_faces = np.concatenate(total_faces)
        total_svo_idx = torch.tensor(total_svo_idx).cuda()
        return total_verts, total_faces, total_svo_idx
    
    @torch.no_grad()
    def cull_mesh(self, mesh_file, device, data_stream, truncation):
        fid, rgb, depth, K, ref_pose, img_raw = data_stream[0]
        # note: here offset is set to 0
        first_frame = RGBDFrame(fid, rgb, depth, K, offset=0, ref_pose=ref_pose, img_raw=img_raw).cuda()
        W, H = first_frame.rgb.shape[1], first_frame.rgb.shape[0]
        eval_rec = True

        n_imgs = len(data_stream)

        mesh = trimesh.load(mesh_file, process=False)
        pc = mesh.vertices

        whole_mask = np.ones(pc.shape[0]).astype('bool')
        for i in tqdm(range(0, n_imgs, 1)):
            fid, rgb, depth, K, ref_pose, img_raw = data_stream[i]
            # note: here offset is set to 0
            frame = RGBDFrame(fid, rgb, depth, K, offset=0, ref_pose=ref_pose, img_raw=img_raw).cuda()
            depth, c2w = frame.depth.to(device), frame.ref_pose.to(device)

            points = pc.copy()
            points = torch.from_numpy(points).to(device)

            w2c = torch.inverse(c2w)
            K = frame.K.to(device)
            ones = torch.ones_like(points[:, 0]).reshape(-1, 1).to(device)
            homo_points = torch.cat(
                [points, ones], dim=1).reshape(-1, 4, 1).to(device).float()
            cam_cord_homo = w2c @ homo_points
            cam_cord = cam_cord_homo[:, :3]

            uv = K.float() @ cam_cord.float()
            z = uv[:, -1:] + 1e-8
            uv = uv[:, :2] / z
            uv = uv.squeeze(-1)

            grid = uv[None, None].clone()
            grid[..., 0] = grid[..., 0] / W
            grid[..., 1] = grid[..., 1] / H
            grid = 2 * grid - 1
            depth_samples = F.grid_sample(depth[None, None], grid, padding_mode='zeros',
                                          align_corners=True).squeeze()

            edge = 0
            if eval_rec:
                mask = (depth_samples + truncation >= z[:, 0, 0]) & (0 <= z[:, 0, 0]) & (uv[:, 0] < W - edge) & (
                        uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)
            else:
                mask = (0 <= z[:, 0, 0]) & (uv[:, 0] < W - edge) & (uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (
                        uv[:, 1] > edge)

            mask = mask.cpu().numpy()

            whole_mask &= ~mask

        face_mask = whole_mask[mesh.faces].all(axis=1)
        mesh.update_faces(~face_mask)
        mesh.remove_unreferenced_vertices()
        mesh.process(validate=False)

        mesh_ext = mesh_file.split('.')[-1]
        output_file = mesh_file[:-len(mesh_ext) - 1] + '_culled.' + mesh_ext

        mesh.export(output_file)
