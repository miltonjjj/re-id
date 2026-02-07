"""
从深度图生成Voronoi图的独立脚本
只使用几何信息，不依赖语义信息
"""

import numpy as np
import cv2
import skimage.morphology
from skimage import measure
from queue import Queue


# ============= 1. 深度图处理工具函数 =============

def get_camera_matrix(width, height, fov):
    """从图像尺寸和FOV计算相机内参矩阵"""
    xc = (width - 1.) / 2.
    zc = (height - 1.) / 2.
    f = (width / 2.) / np.tan(np.deg2rad(fov / 2.))
    return {'xc': xc, 'zc': zc, 'f': f}


def get_point_cloud_from_z(depth, camera_matrix, scale=1):
    """
    将深度图投影为3D点云
    输入:
        depth: HxW 深度图
        camera_matrix: 相机内参
        scale: 下采样因子
    输出:
        XYZ: HxWx3 点云坐标
    """
    x, z = np.meshgrid(np.arange(depth.shape[-1]),
                       np.arange(depth.shape[-2] - 1, -1, -1))
    
    X = (x[::scale, ::scale] - camera_matrix['xc']) * \
        depth[::scale, ::scale] / camera_matrix['f']
    Z = (z[::scale, ::scale] - camera_matrix['zc']) * \
        depth[::scale, ::scale] / camera_matrix['f']
    
    XYZ = np.stack((X, depth[::scale, ::scale], Z), axis=-1)
    return XYZ


def get_rotation_matrix(axis, angle):
    """生成旋转矩阵"""
    ax = axis / np.linalg.norm(axis)
    if np.abs(angle) > 0.001:
        S_hat = np.array(
            [[0.0, -ax[2], ax[1]], 
             [ax[2], 0.0, -ax[0]], 
             [-ax[1], ax[0], 0.0]],
            dtype=np.float32)
        R = np.eye(3) + np.sin(angle) * S_hat + \
            (1 - np.cos(angle)) * (np.linalg.matrix_power(S_hat, 2))
    else:
        R = np.eye(3)
    return R


def transform_camera_view(XYZ, sensor_height, camera_elevation_degree):
    """
    将点云从相机坐标系转换到世界坐标系
    考虑相机高度和俯仰角
    """
    R = get_rotation_matrix([1., 0., 0.], np.deg2rad(camera_elevation_degree))
    XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape(XYZ.shape)
    XYZ[..., 2] = XYZ[..., 2] + sensor_height
    return XYZ


def transform_pose(XYZ, current_pose):
    """
    根据相机位姿转换点云
    current_pose: (x, y, theta) 位姿
    """
    # 处理2D点云 (H, W, 3)
    original_shape = XYZ.shape
    if len(original_shape) == 3:
        # 展平为 (N, 3)
        XYZ_flat = XYZ.reshape(-1, 3)
        R = get_rotation_matrix([0., 0., 1.], current_pose[2] - np.pi / 2.)
        XYZ_transformed = np.matmul(XYZ_flat, R.T)
        XYZ_transformed[:, 0] = XYZ_transformed[:, 0] + current_pose[0]
        XYZ_transformed[:, 1] = XYZ_transformed[:, 1] + current_pose[1]
        # 恢复形状
        return XYZ_transformed.reshape(original_shape)
    else:
        # 原始逻辑
        R = get_rotation_matrix([0., 0., 1.], current_pose[2] - np.pi / 2.)
        XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape(XYZ.shape)
        XYZ[:, :, 0] = XYZ[:, :, 0] + current_pose[0]
        XYZ[:, :, 1] = XYZ[:, :, 1] + current_pose[1]
        return XYZ


def bin_points(XYZ_cms, map_size, z_bins, xy_resolution):
    """
    将点云按xy-z分箱到体素网格
    输入:
        XYZ_cms: H x W x 3 点云 (单位: cm)
        map_size: 地图尺寸
        z_bins: 高度分层
        xy_resolution: xy平面分辨率
    输出:
        counts: map_size x map_size x (len(z_bins)+1) 体素计数
    """
    n_z_bins = len(z_bins) + 1
    
    isnotnan = np.logical_not(np.isnan(XYZ_cms[:, :, 0]))
    X_bin = np.round(XYZ_cms[:, :, 0] / xy_resolution).astype(np.int32)
    Y_bin = np.round(XYZ_cms[:, :, 1] / xy_resolution).astype(np.int32)
    Z_bin = np.digitize(XYZ_cms[:, :, 2], bins=z_bins).astype(np.int32)
    
    isvalid = np.array([X_bin >= 0, X_bin < map_size, Y_bin >= 0,
                        Y_bin < map_size,
                        Z_bin >= 0, Z_bin < n_z_bins, isnotnan])
    isvalid = np.all(isvalid, axis=0)
    
    ind = (Y_bin * map_size + X_bin) * n_z_bins + Z_bin
    ind[np.logical_not(isvalid)] = 0
    count = np.bincount(ind.ravel(), isvalid.ravel().astype(np.int32),
                        minlength=map_size * map_size * n_z_bins)
    counts = np.reshape(count, [map_size, map_size, n_z_bins])
    
    return counts


# ============= 2. 深度图到2D地图转换 =============

def depth_to_obstacle_map(depth, camera_params, agent_pose, map_params):
    """
    将深度图转换为障碍物地图和探索地图
    
    参数:
        depth: (H, W) 深度图，单位**厘米**（重要！）
        camera_params: dict with keys:
            - 'width': int, 图像宽度（像素）
            - 'height': int, 图像高度（像素）
            - 'fov': float, 水平视场角（度）
            - 'cam_height': float, 相机高度（厘米）
            - 'elevation': float, 相机俯仰角（度）
        agent_pose: (x, y, theta) 机器人位姿
        map_params: dict with keys:
            - 'resolution': int, 分辨率（厘米/格子）
            - 'vision_range': int, 视觉范围（格子数）
            - 'z_min': int, 最小高度（厘米，相对地面）
            - 'z_max': int, 最大高度（厘米，相对地面）
    
    返回:
        obstacle_map: (vision_range, vision_range) 障碍物地图 [0, 1]
        explored_map: (vision_range, vision_range) 探索地图 [0, 1]
    """
    # 1. 获取相机矩阵
    camera_matrix = get_camera_matrix(
        camera_params['width'], 
        camera_params['height'], 
        camera_params['fov']
    )
    
    # 2. 深度图转点云（输入是厘米，需要转换为米用于几何计算）
    depth_m = depth / 100.0  # 厘米 -> 米
    point_cloud_m = get_point_cloud_from_z(depth_m, camera_matrix, scale=1)
    
    # 转换为厘米（与 voronav_pre 保持一致）
    point_cloud_cm = point_cloud_m * 100.0
    
    
    # 修改 depth_to_obstacle_map 函数（第172-195行）

    # 3. 转换到世界坐标系
    sensor_height_cm = camera_params['height']  # 88 cm

    # 应用旋转（如果有俯仰角）
    camera_elevation = camera_params.get('elevation', 0)
    if abs(camera_elevation) > 0.001:
        R = get_rotation_matrix([1., 0., 0.], np.deg2rad(camera_elevation))
        point_cloud_cm = np.matmul(point_cloud_cm.reshape(-1, 3), R.T).reshape(point_cloud_cm.shape)

    # ★关键修改：直接调整Z坐标为相对于地面
    # 点云的Z是相对相机的，相机在地面上方88cm
    # 所以 Z_ground = Z_camera + 88
    point_cloud_cm[:, :, 2] = point_cloud_cm[:, :, 2] + sensor_height_cm

    
    # ★关键：调整Z坐标为相对于地面
    # 减去相机高度，使 Z=0 表示地面
    point_cloud_cm[:, :, 2] = point_cloud_cm[:, :, 2] - sensor_height_cm
    
    
    # 4. 根据机器人位姿转换（移动到视觉范围中心）
    vision_range = map_params['vision_range']  # 格子数
    resolution = map_params['resolution']  # cm/格子
    shift_loc = [vision_range * resolution / 2, 0, np.pi / 2.0]  # [cm, cm, 弧度]
    point_cloud_centered = transform_pose(point_cloud_cm, shift_loc)
    
    
    # 5. 按高度分层投影
    z_bins = [map_params['z_min'], map_params['z_max']]  # [cm, cm]
    voxels = bin_points(
        point_cloud_centered,
        vision_range,  # 格子数
        z_bins,  # cm
        resolution  # cm/格子
    )
    
    
    # # 6. 生成障碍物地图和探索地图
    # # 障碍物：agent高度范围内的点（中间层）
    # obstacle_layer = voxels[:, :, 1]
    # obstacle_map = (obstacle_layer > 0).astype(np.float32)
    
    # # 探索区域：所有高度的点
    # explored_layer = voxels.sum(axis=2)
    # explored_map = (explored_layer > 0).astype(np.float32)
    
    # print(f"  障碍物像素: {obstacle_map.sum()}, 探索像素: {explored_map.sum()}")
    # 6. 生成障碍物地图和探索地图
    # 障碍物：agent高度范围内的点（中间层）
    # 注意：这里直接使用体素计数，不是二值化
    obstacle_layer = voxels[:, :, 1]
    explored_layer = voxels.sum(axis=2)

    # 应用阈值（可选，用于过滤噪声）
    map_threshold = 1.0  # 至少1个体素点
    exp_threshold = 1.0

    # 生成概率地图（归一化）
    obstacle_map = np.clip(obstacle_layer / map_threshold, 0.0, 1.0).astype(np.float32)
    explored_map = np.clip(explored_layer / exp_threshold, 0.0, 1.0).astype(np.float32)

    return obstacle_map, explored_map


# ============= 3. Voronoi图生成 =============
#保留骨架的最大连通分支
'''
def generate_voronoi_skeleton(obstacle_map, explored_map, 
                               morphology_kernel_size=10):
    """
    从障碍物地图和探索地图生成Voronoi骨架
    
    参数:
        obstacle_map: (H, W) 障碍物地图
        explored_map: (H, W) 探索地图
        morphology_kernel_size: 形态学操作的核大小
    
    返回:
        skeleton: (H, W) Voronoi骨架 bool数组
        free_space: (H, W) 自由空间 bool数组
    """
    # 1. 膨胀障碍物地图
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_obstacle = cv2.dilate(obstacle_map.astype(np.float32), kernel)
    
    # 2. 对探索地图进行闭运算（填充小孔）
    kernel_close = np.ones((morphology_kernel_size, morphology_kernel_size), 
                           dtype=np.uint8)
    explored_closed = cv2.morphologyEx(
        explored_map.astype(np.uint8), 
        cv2.MORPH_CLOSE, 
        kernel_close
    )
    
    # 3. 计算自由空间：探索区域 - 障碍物
    free_space = np.maximum(0, explored_closed - dilated_obstacle)
    free_space = (free_space > 0.5).astype(bool)
    
    # 4. 骨架化：生成Voronoi图
    skeleton = skimage.morphology.skeletonize(free_space)
    
    # 5. 找到最大连通骨架
    skeleton_labeled, num_components = measure.label(
        skeleton, connectivity=2, return_num=True
    )
    if num_components > 0:
        # 找到最大的连通分量
        largest_component = 1
        max_size = 0
        for i in range(1, num_components + 1):
            size = np.sum(skeleton_labeled == i)
            if size > max_size:
                max_size = size
                largest_component = i
        skeleton = (skeleton_labeled == largest_component)
    
    return skeleton.astype(bool), free_space
'''
#保留机器人位置最近的分支

def generate_voronoi_skeleton(obstacle_map, explored_map, 
                               morphology_kernel_size=10,
                               robot_pos=None,
                               select_nearest=False):
    """
    从障碍物地图和探索地图生成Voronoi骨架
    
    参数:
        obstacle_map: (H, W) 障碍物地图
        explored_map: (H, W) 探索地图
        morphology_kernel_size: 形态学操作的核大小
        robot_pos: (x, y) 机器人位置，如果为None则默认为底部中心
        select_nearest: bool, 如果True则选择与机器人最近的骨架，否则选最大骨架
    
    返回:
        skeleton: (H, W) Voronoi骨架 bool数组
        free_space: (H, W) 自由空间 bool数组
    """
    # 1. 膨胀障碍物地图
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_obstacle = cv2.dilate(obstacle_map.astype(np.float32), kernel)
    
    # 2. 对探索地图进行闭运算（填充小孔）
    kernel_close = np.ones((morphology_kernel_size, morphology_kernel_size), 
                           dtype=np.uint8)
    explored_closed = cv2.morphologyEx(
        explored_map.astype(np.uint8), 
        cv2.MORPH_CLOSE, 
        kernel_close
    )
    
    # 3. 计算自由空间：探索区域 - 障碍物
    free_space = np.maximum(0, explored_closed - dilated_obstacle)
    free_space = (free_space > 0.5).astype(bool)
    
    # 4. 骨架化：生成Voronoi图
    skeleton = skimage.morphology.skeletonize(free_space)
    
    # 5. 选择骨架分支
    skeleton_labeled, num_components = measure.label(
        skeleton, connectivity=2, return_num=True
    )
    
    if num_components > 0:
        if select_nearest and robot_pos is not None:
            # ★新逻辑：选择与机器人最近的骨架分支
            # 设置机器人位置（如果没提供则使用底部中心）
            if robot_pos is None:
                vision_range = skeleton.shape[0]
                robot_y, robot_x = 0, vision_range // 2
            else:
                robot_x, robot_y = robot_pos[0], robot_pos[1]
            
            # 创建机器人位置掩码
            robot_mask = np.zeros_like(skeleton, dtype=bool)
            robot_mask[robot_y, robot_x] = True
            
            # 膨胀机器人位置和每个骨架分支，检查是否相交
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            robot_mask_dilated = cv2.dilate(robot_mask.astype(np.uint8), kernel_dilate).astype(bool)
            
            selected_component = None
            for i in range(1, num_components + 1):
                component_mask = (skeleton_labeled == i)
                component_dilated = cv2.dilate(component_mask.astype(np.uint8), kernel_dilate).astype(bool)
                
                # 检查膨胀后是否相交
                if np.any(robot_mask_dilated & component_dilated):
                    selected_component = i
                    break
            
            # 如果找到相交的分支，使用它；否则使用最大分支
            if selected_component is not None:
                skeleton = (skeleton_labeled == selected_component)
            else:
                # 回退到最大分支
                largest_component = 1
                max_size = 0
                for i in range(1, num_components + 1):
                    size = np.sum(skeleton_labeled == i)
                    if size > max_size:
                        max_size = size
                        largest_component = i
                skeleton = (skeleton_labeled == largest_component)
        else:
            # ★原逻辑：找到最大的连通分量
            largest_component = 1
            max_size = 0
            for i in range(1, num_components + 1):
                size = np.sum(skeleton_labeled == i)
                if size > max_size:
                    max_size = size
                    largest_component = i
            skeleton = (skeleton_labeled == largest_component)
    
    return skeleton.astype(bool), free_space

# ============= 4. 提取Voronoi图的关节点 =============
#修改前的提取交叉点和端点的函数
# def extract_voronoi_nodes(skeleton):
#     """
#     从Voronoi骨架提取关节点（交叉点和端点）
    
#     参数:
#         skeleton: (H, W) bool数组，Voronoi骨架
    
#     返回:
#         joint_nodes: list of (x, y) 关节点坐标
#     """
#     # 1. 找到交叉点：8邻域内有4个以上骨架点
#     old_joint_mask = np.zeros_like(skeleton, dtype=bool)
#     skeleton_coords = np.argwhere(skeleton)
    
#     for coord in skeleton_coords:
#         x, y = coord[0], coord[1]
#         if x > 0 and x < skeleton.shape[0] - 1 and \
#            y > 0 and y < skeleton.shape[1] - 1:
#             neighbor_count = np.sum(skeleton[x-1:x+2, y-1:y+2])
#             if neighbor_count >= 4:  # 交叉点
#                 old_joint_mask[x, y] = True
    
#     # 2. 找到端点：8邻域内只有2个骨架点，且附近没有交叉点
#     for coord in skeleton_coords:
#         x, y = coord[0], coord[1]
#         if x > 4 and x < skeleton.shape[0] - 5 and \
#            y > 4 and y < skeleton.shape[1] - 5:
#             neighbor_count = np.sum(skeleton[x-1:x+2, y-1:y+2])
#             if neighbor_count == 2:  # 可能是端点
#                 # 检查附近是否有交叉点
#                 nearby_region = old_joint_mask[x-4:x+5, y-4:y+5]
#                 if not np.any(nearby_region):
#                     old_joint_mask[x, y] = True
    
#     # 3. 合并相邻的关节点
#     joint_mask = np.zeros_like(skeleton, dtype=int)
#     for coord in np.argwhere(old_joint_mask):
#         x, y = coord[0], coord[1]
#         if x > 0 and x < skeleton.shape[0] - 1 and \
#            y > 0 and y < skeleton.shape[1] - 1:
#             # 计算局部密度
#             joint_mask[x, y] = (old_joint_mask[x-1, y] + 
#                                old_joint_mask[x+1, y] + 
#                                old_joint_mask[x, y-1] + 
#                                old_joint_mask[x, y+1] + 
#                                old_joint_mask[x, y])
    
#     # 4. 提取局部最大值作为最终关节点
#     joint_nodes = []
#     for coord in np.argwhere(joint_mask > 0):
#         x, y = coord[0], coord[1]
#         if x > 0 and x < skeleton.shape[0] - 1 and \
#            y > 0 and y < skeleton.shape[1] - 1:
#             local_max = np.max(joint_mask[x-1:x+2, y-1:y+2])
#             if joint_mask[x, y] == local_max:
#                 joint_nodes.append((x, y))
#                 # 清除周围以避免重复
#                 joint_mask[x-1:x+2, y-1:y+2] = 0
    
#     return joint_nodes
'''
def extract_voronoi_nodes(skeleton, merge_distance=5):
    """
    从Voronoi骨架提取关节点（交叉点和端点）
    
    参数:
        skeleton: (H, W) bool数组，Voronoi骨架
        merge_distance: int, 合并相邻节点的距离阈值（像素）
    
    返回:
        joint_nodes: list of (x, y) 关节点坐标
    """
    # 1. 找到交叉点：8邻域内有4个以上骨架点
    old_joint_mask = np.zeros_like(skeleton, dtype=bool)
    skeleton_coords = np.argwhere(skeleton)
    
    for coord in skeleton_coords:
        x, y = coord[0], coord[1]
        if x > 0 and x < skeleton.shape[0] - 1 and \
           y > 0 and y < skeleton.shape[1] - 1:
            neighbor_count = np.sum(skeleton[x-1:x+2, y-1:y+2])
            if neighbor_count >= 4:  # 交叉点
                old_joint_mask[x, y] = True
    
    # 2. 改进的端点识别：8邻域内只有2个骨架点（自己+1个邻居）
    # ★ 问题1修复：减小边界检查，允许更多端点被识别
    endpoint_check_margin = 2  # 从4改为2，允许更靠近边界的端点
    
    for coord in skeleton_coords:
        x, y = coord[0], coord[1]
        # ★ 放宽边界限制
        if x > endpoint_check_margin and x < skeleton.shape[0] - endpoint_check_margin - 1 and \
           y > endpoint_check_margin and y < skeleton.shape[1] - endpoint_check_margin - 1:
            neighbor_count = np.sum(skeleton[x-1:x+2, y-1:y+2])
            
            # ★ 改进：识别度数为1的端点（neighbor_count == 2：自己+1个邻居）
            if neighbor_count == 2:
                # 检查附近是否有交叉点（避免在交叉点附近标记端点）
                check_radius = 4
                if x > check_radius and x < skeleton.shape[0] - check_radius - 1 and \
                   y > check_radius and y < skeleton.shape[1] - check_radius - 1:
                    nearby_region = old_joint_mask[x-check_radius:x+check_radius+1, 
                                                   y-check_radius:y+check_radius+1]
                    if not np.any(nearby_region):
                        old_joint_mask[x, y] = True
                else:
                    # 边界附近的端点也要识别
                    old_joint_mask[x, y] = True
    '''
def extract_voronoi_nodes(skeleton, merge_distance=5):
    """
    从Voronoi骨架提取关节点（交叉点和端点）
    """
    # 1. 找到交叉点：8邻域内有4个以上骨架点
    old_joint_mask = np.zeros_like(skeleton, dtype=bool)
    skeleton_coords = np.argwhere(skeleton)
    
    for coord in skeleton_coords:
        x, y = coord[0], coord[1]
        if x > 0 and x < skeleton.shape[0] - 1 and \
           y > 0 and y < skeleton.shape[1] - 1:
            neighbor_count = np.sum(skeleton[x-1:x+2, y-1:y+2])
            if neighbor_count >= 4:  # 交叉点
                old_joint_mask[x, y] = True
    
    # 2. ★ 改进的端点识别：放宽边界限制
    for coord in skeleton_coords:
        x, y = coord[0], coord[1]
        
        # 计算有效的邻域范围（处理边界情况）
        x_min = max(0, x - 1)
        x_max = min(skeleton.shape[0], x + 2)
        y_min = max(0, y - 1)
        y_max = min(skeleton.shape[1], y + 2)
        
        neighborhood = skeleton[x_min:x_max, y_min:y_max]
        neighbor_count = np.sum(neighborhood)  # 包含自身
        
        # ★ 端点条件：只有1个邻居（neighbor_count == 2：自己+1个邻居）
        # 或者在边界上且邻居数<=2
        is_boundary = (x <= 1 or x >= skeleton.shape[0] - 2 or 
                       y <= 1 or y >= skeleton.shape[1] - 2)
        
        if neighbor_count == 2:
            # 检查附近是否已有交叉点（避免重复标记）
            check_x_min = max(0, x - 4)
            check_x_max = min(skeleton.shape[0], x + 5)
            check_y_min = max(0, y - 4)
            check_y_max = min(skeleton.shape[1], y + 5)
            
            nearby_region = old_joint_mask[check_x_min:check_x_max, 
                                           check_y_min:check_y_max]
            
            # ★ 关键修改：边界端点直接标记，不检查附近交叉点
            if is_boundary or not np.any(nearby_region):
                old_joint_mask[x, y] = True
    # ★ 问题2修复：使用更大范围的合并策略
    # 3. 计算每个关节点的局部密度（在更大的邻域内）
    joint_mask = np.zeros_like(skeleton, dtype=int)
    merge_kernel_size = 2 * merge_distance + 1  # 默认11×11
    
    for coord in np.argwhere(old_joint_mask):
        x, y = coord[0], coord[1]
        if x > merge_distance and x < skeleton.shape[0] - merge_distance - 1 and \
           y > merge_distance and y < skeleton.shape[1] - merge_distance - 1:
            # 在更大的邻域内统计关节点密度
            local_region = old_joint_mask[x-merge_distance:x+merge_distance+1, 
                                         y-merge_distance:y+merge_distance+1]
            joint_mask[x, y] = np.sum(local_region)
        else:
            # 边界处理
            joint_mask[x, y] = 1
    
    # 4. 使用非极大值抑制（NMS）提取最终关节点
    # ★ 改进：在更大范围内寻找局部最大值，确保相邻节点被合并
    joint_nodes = []
    visited = np.zeros_like(skeleton, dtype=bool)
    
    # 按密度从高到低排序
    candidates = [(joint_mask[x, y], x, y) for x, y in np.argwhere(joint_mask > 0)]
    candidates.sort(reverse=True)
    
    for density, x, y in candidates:
        if visited[x, y]:
            continue
        
        # 将此点标记为关节点
        joint_nodes.append((x, y))
        
        # 在 merge_distance 范围内抑制其他候选点
        x_min = max(0, x - merge_distance)
        x_max = min(skeleton.shape[0], x + merge_distance + 1)
        y_min = max(0, y - merge_distance)
        y_max = min(skeleton.shape[1], y + merge_distance + 1)
        
        visited[x_min:x_max, y_min:y_max] = True
    
    return joint_nodes

# ============= 5. 构建Voronoi拓扑图 =============

def build_voronoi_graph(skeleton, joint_nodes):
    """
    构建Voronoi图的拓扑结构（节点和边）
    
    参数:
        skeleton: (H, W) bool数组，Voronoi骨架
        joint_nodes: list of (x, y) 关节点坐标
    
    返回:
        graph: dict {node: [neighbor_nodes]} 邻接表
        edges: dict {node: [edge_lengths]} 边长度
    """
    m, n = skeleton.shape
    graph = {tuple(node): [] for node in joint_nodes}
    edges = {tuple(node): [] for node in joint_nodes}
    nodes_set = set(tuple(node) for node in joint_nodes)
    
    # 8方向
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0),
                  (-1, -1), (1, 1), (-1, 1), (1, -1)]
    
    # 从每个关节点BFS找到相邻的关节点
    for start_node in joint_nodes:
        start = tuple(start_node)
        visited = np.zeros_like(skeleton, dtype=bool)
        distance = np.full(skeleton.shape, np.inf)
        
        queue = Queue()
        queue.put(start)
        visited[start[0], start[1]] = True
        distance[start[0], start[1]] = 0
        
        while not queue.empty():
            x, y = queue.get()
            
            # 检查是否遇到另一个关节点
            found_neighbor = False
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n:
                    if (nx, ny) in nodes_set and \
                       not visited[nx, ny] and \
                       (nx, ny) != start:
                        # 找到相邻节点
                        graph[start].append((nx, ny))
                        edges[start].append(distance[x, y] + 1)
                        found_neighbor = True
            
            if found_neighbor:
                continue
            
            # 继续沿骨架搜索
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and \
                   skeleton[nx, ny] and not visited[nx, ny]:
                    distance[nx, ny] = distance[x, y] + 1
                    visited[nx, ny] = True
                    queue.put((nx, ny))
    
    return graph, edges


# ============= 6. 主函数 =============

def depth_to_voronoi(depth, camera_params=None, agent_pose=None, map_params=None):
    """
    从深度图生成Voronoi图的完整流程
    
    参数:
        depth: (H, W) 深度图，单位米
        camera_params: 相机参数 (如果为None使用默认值)
        agent_pose: 机器人位姿 (如果为None使用默认值)
        map_params: 地图参数 (如果为None使用默认值)
    
    返回:
        results: dict包含:
            - obstacle_map: 障碍物地图
            - explored_map: 探索地图
            - free_space: 自由空间
            - skeleton: Voronoi骨架
            - joint_nodes: 关节点列表
            - graph: 拓扑图（邻接表）
            - edges: 边长度
    """
    # 使用默认参数
    if camera_params is None:
        camera_params = {
            'width': 256,
            'height': 256,
            'fov': 90.0,
            'height': 88.0, 
            'elevation': 0.0
        }
    
    if agent_pose is None:
        agent_pose = (0, 0, 0)  # x, y, theta
    
    if map_params is None:
        map_params = {
            'map_size': 240,
            'resolution': 5,  # cm per pixel
            'vision_range': 60,
            'z_min': 25,#高度区间
            'z_max': 138
        }
    
    # 1. 深度图 → 障碍物地图和探索地图
    obstacle_map, explored_map = depth_to_obstacle_map(
        depth, camera_params, agent_pose, map_params
    )
    
    # 2. 生成Voronoi骨架
    skeleton, free_space = generate_voronoi_skeleton(
        obstacle_map, explored_map
    )
    
    # 3. 提取关节点
    joint_nodes = extract_voronoi_nodes(skeleton)
    
    # 4. 构建拓扑图
    graph, edges = build_voronoi_graph(skeleton, joint_nodes)
    
    results = {
        'obstacle_map': obstacle_map,
        'explored_map': explored_map,
        'free_space': free_space,
        'skeleton': skeleton,
        'joint_nodes': joint_nodes,
        'graph': graph,
        'edges': edges
    }
    
    
    return results


# ============= 7. 可视化函数 =============

def visualize_voronoi(results, save_path=None):
    """
    可视化Voronoi图结果
    
    参数:
        results: depth_to_voronoi的返回结果
        save_path: 保存路径（可选）
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 障碍物地图
    axes[0, 0].imshow(results['obstacle_map'], cmap='gray',origin='lower')
    axes[0, 0].set_title('Obstacle Map')
    axes[0, 0].axis('off')
    
    # 2. 探索地图
    axes[0, 1].imshow(results['explored_map'], cmap='gray',origin='lower')
    axes[0, 1].set_title('Explored Map')
    axes[0, 1].axis('off')
    
    # 3. 自由空间 ★只在这里添加机器人标记
    axes[0, 2].imshow(results['free_space'], cmap='gray',origin='lower')
    
    # 计算机器人位置
    vision_range = results['free_space'].shape[0]  # 通常是100
    robot_x = vision_range / 2  # 50（水平中央）
    robot_y = 0  # 0（底部）
    
    # 画红色实心圆
    axes[0, 2].plot(robot_x, robot_y, 'ro', markersize=10, label='Robot')
    
    axes[0, 2].set_title('Free Space')
    axes[0, 2].axis('off')
    axes[0, 2].legend(loc='upper right')
    
    # 4. Voronoi骨架
    axes[1, 0].imshow(results['skeleton'], cmap='gray',origin='lower')
    axes[1, 0].set_title('Voronoi Skeleton')
    axes[1, 0].axis('off')
    
    # 5. 关节点
    axes[1, 1].imshow(results['skeleton'], cmap='gray',origin='lower')
    if len(results['joint_nodes']) > 0:
        nodes = np.array(results['joint_nodes'])
        axes[1, 1].scatter(nodes[:, 1], nodes[:, 0], c='red', s=50)
    axes[1, 1].set_title(f'Joint Nodes ({len(results["joint_nodes"])})')
    axes[1, 1].axis('off')
    
    # 6. 拓扑图
    axes[1, 2].imshow(results['skeleton'], cmap='gray',origin='lower')
    if len(results['joint_nodes']) > 0:
        nodes = np.array(results['joint_nodes'])
        axes[1, 2].scatter(nodes[:, 1], nodes[:, 0], c='red', s=50)
        # 绘制边
        for node, neighbors in results['graph'].items():
            for neighbor in neighbors:
                axes[1, 2].plot([node[1], neighbor[1]], 
                               [node[0], neighbor[0]], 
                               'yellow', alpha=0.9, linewidth=3)
    axes[1, 2].set_title('Voronoi Graph')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    import sys
    
    # 读取原始深度数据
    depth_path = "habitat-lab/test/Denmark_start_depth_raw_2.npy"
    
    # 直接加载numpy数组
    depth_raw = np.load(depth_path)
    
    # ★关键1：处理形状（如果是3维，取第一个通道）
    if len(depth_raw.shape) == 3:
        depth_raw = depth_raw[:, :, 0]
    
    # ★关键2：Habitat的深度是归一化的[0,1]，需要反归一化
    # Gibson数据集的深度范围：min=0.5m, max=5.0m（来自配置）
    min_depth_m = 0.5
    max_depth_m = 3.0
    
    # 检查是否已经归一化
    if depth_raw.max() <= 1.0:
        depth_m = min_depth_m + depth_raw * (max_depth_m - min_depth_m)
    else:
        depth_m = depth_raw
    
    
    # ★关键3：处理异常值（参考voronav_pre的做法）
    # 替换0值
    for i in range(depth_m.shape[1]):
        zero_mask = depth_m[:, i] == 0.
        if zero_mask.any():
            valid_values = depth_m[:, i][~zero_mask]
            if len(valid_values) > 0:
                depth_m[:, i][zero_mask] = valid_values.max()
    
    # 处理过大的值（接近max_depth的值）
    mask_too_far = depth_m > 0.99 * max_depth_m
    depth_m[mask_too_far] = 0.
    
    # 再次处理剩余的0值
    mask_zero = depth_m == 0
    depth_m[mask_zero] = max_depth_m  # 设为最大值
    
    
    # 转换为厘米
    depth_cm = depth_m * 100.0
    
    # 调整到标准尺寸
    target_height, target_width = 256, 256
    if depth_cm.shape != (target_height, target_width):
        depth_cm = cv2.resize(depth_cm, (target_width, target_height), 
                              interpolation=cv2.INTER_LINEAR)

    results = depth_to_voronoi(depth_cm)

    
    if len(results['graph']) > 0:
        total_edges = sum(len(v) for v in results['graph'].values())
        avg_degree = total_edges / len(results['graph']) if len(results['graph']) > 0 else 0
    
    visualize_voronoi(results, save_path='habitat-lab/test/voronoi_result_height_test.png')
