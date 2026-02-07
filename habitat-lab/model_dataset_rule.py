"""
基于规则筛选的数据集生成脚本 - 生成 NoMaD 训练格式数据集
功能：
1. Voronoi路径规划和动作生成
2. 规则质量评估（替代VLM，只保留通过规则筛选的episode）
3. 执行动作序列并收集轨迹数据
4. 生成符合 go_stanford_dataset 格式的数据集
   - 文件夹命名: {scene_name}_{episode_num}
   - 图像文件: 0.jpg, 1.jpg, 2.jpg, ...
   - traj_data.pkl: 包含 position 和 yaw 数组
5. 输出统计指标：
   - 总episode数量、保留/删除/无路径数量

评估维度：
1. Efficiency (权重 0.5): 效率评估（基于tortuosity）
2. Exploration (权重 0.5): 探索价值评估（路径长度、终点距离、终点位置、路径数量）

决策规则：
- 综合分数 >= 5.5: keep
- 综合分数 < 5.5: delete
"""

import os
import numpy as np
import cv2
import habitat
import tqdm
import imageio
import matplotlib.pyplot as plt
from PIL import Image
import skimage.morphology
from skimage import measure
from skimage.graph import route_through_array
from collections import deque
import json
import pickle
import shutil
import tqdm

# 导入voronoi_map模块
from voronoi_map import (
    depth_to_obstacle_map, 
    generate_voronoi_skeleton,
    extract_voronoi_nodes,
    build_voronoi_graph
)

# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


# ============= 规则评估函数 =============
def check_action_smoothness(actions):
    """
    检测动作序列的平滑度
    
    Args:
        actions: 动作序列列表
            - 0: Stop
            - 1: Forward
            - 2: Left
            - 3: Right
    
    Returns:
        dict: 包含 oscillation_rate, avg_run_length的字典
    """
    if len(actions) <= 1:
        return {
            'oscillation_rate': 0.0,
            'avg_run_length': 0.0,
            'has_excessive_rotation': False,
        }
    
    # 排除末尾的 Stop 动作
    action_seq = [a for a in actions if a != 0]
    if len(action_seq) == 0:
        return {
            'oscillation_rate': 0.0,
            'avg_run_length': 0.0,
            'has_excessive_rotation': False,
        }
    
    # 1. 计算震荡率：只统计 Left<->Right 的直接切换
    oscillations = 0
    for i in range(1, len(action_seq)):
        prev_action = action_seq[i-1]
        curr_action = action_seq[i]
        
        # 只有 Left(2) -> Right(3) 或 Right(3) -> Left(2) 才算震荡
        if (prev_action == 2 and curr_action == 3) or \
           (prev_action == 3 and curr_action == 2):
            oscillations += 1
    
    oscillation_rate = oscillations / len(action_seq) if len(action_seq) > 0 else 0.0
    
    # 2. 计算平均步长 (每种动作的平均连续长度)
    run_lengths = []
    current_action = action_seq[0]
    current_length = 1
    
    for i in range(1, len(action_seq)):
        if action_seq[i] == current_action:
            current_length += 1
        else:
            run_lengths.append(current_length)
            current_action = action_seq[i]
            current_length = 1
    run_lengths.append(current_length)
    
    avg_run_length = np.mean(run_lengths) if len(run_lengths) > 0 else 0.0
    
    # 3. 检测连续旋转：连续4个或更多的Left或Right动作
    has_excessive_rotation = False
    consecutive_rotation = 0
    
    for action in action_seq:
        if action == 2 or action == 3:  # Left 或 Right
            consecutive_rotation += 1
            if consecutive_rotation >= 4:
                has_excessive_rotation = True
                break
        else:  # Forward
            consecutive_rotation = 0
    return {
        'oscillation_rate': oscillation_rate,
        'avg_run_length': avg_run_length,
        'has_excessive_rotation': has_excessive_rotation,
    }


def filter_single_trajectory(path_info, actions, map_resolution=5.0,
                            min_length_m=2.0, max_steps=50,
                            min_curvature=1.28,max_curvature=2.5):
    """
    对单条轨迹进行分层筛选
    
    Args:
        path_info: 路径信息字典
        actions: 动作序列
        map_resolution: 地图分辨率 (cm/pixel)
        min_length_m: 最小路径长度 (米)
        max_steps: 最大步数
        max_curvature: 最大曲率 (路径长度/直线距离)
    
    Returns:
        tuple: (is_valid: bool, reason: str, metrics: dict)
    """
    complete_path = path_info['complete_path']
    
    # 计算几何距离
    if len(complete_path) < 2:
        return False, "路径点数少于2", {}
    
    path_array = np.array(complete_path)
    
    # 计算路径几何距离 (沿路径的总长度)
    path_length_pixels = 0.0
    for i in range(len(path_array) - 1):
        segment_length = np.linalg.norm(path_array[i+1] - path_array[i])
        path_length_pixels += segment_length
    
    path_length_m = (path_length_pixels * map_resolution) / 100.0  # 转换为米
    
    # 计算起点到终点的欧氏距离
    straight_distance_pixels = np.linalg.norm(path_array[-1] - path_array[0])
    straight_distance_m = (straight_distance_pixels * map_resolution) / 100.0
    
    # 计算曲率
    curvature = path_length_m / straight_distance_m if straight_distance_m > 0 else float('inf')
    num_steps = len([a for a in actions if a != 0])
    metrics = {
        'path_length_m': path_length_m,
        'straight_distance_m': straight_distance_m,
        'curvature': curvature, 
        'num_steps': num_steps,
    }
    
    # ========== 第一部分：有效性检查 ==========
    
    # (1) 最小长度检查
    if path_length_m < min_length_m:
        return False, f"路径长度({path_length_m:.2f}m)小于最小长度({min_length_m}m)", metrics
    
    # (2) 最大步数检查
    if num_steps > max_steps:
        return False, f"步数({num_steps})超过最大步数({max_steps})", metrics
    
    # ========== 第二部分：几何特征检查 ==========
    
     # (1) 路径曲率检查 - 最小值
    if curvature < min_curvature:
        return False, f"曲率({curvature:.2f})小于最小值({min_curvature})", metrics
    
    # (2) 路径曲率检查 - 最大值
    if curvature > max_curvature:
        return False, f"曲率({curvature:.2f})超过最大值({max_curvature})", metrics
    
    # (3) 平滑度检测
    smoothness_metrics = check_action_smoothness(actions)
    metrics.update(smoothness_metrics)
    
    # 震荡率检查
    if smoothness_metrics['oscillation_rate'] > 0.05:
        return False, f"震荡率({smoothness_metrics['oscillation_rate']:.3f})过高(>0.05)", metrics
    
    # 平均步长检查
    if smoothness_metrics['avg_run_length'] < 1.5:
        return False, f"平均步长({smoothness_metrics['avg_run_length']:.2f})过短(<1.5)", metrics
    
    # 连续旋转检查
    if smoothness_metrics['has_excessive_rotation']:
        return False, "检测到连续4次或更多旋转动作", metrics

    # 通过所有检查
    return True, "通过所有筛选规则", metrics


def filter_all_trajectories_independently(all_paths, skeleton, robot_position, 
                                          map_resolution=5.0,
                                          min_length_m=1.5, 
                                          max_steps=50,
                                          min_curvature=1.28, 
                                          max_curvature=2.5):
    """
    对所有轨迹进行独立筛选
    
    Args:
        all_paths: 所有路径信息列表
        skeleton: 骨架地图
        robot_position: 机器人位置
        map_resolution: 地图分辨率 (cm/pixel)
        min_length_m: 最小路径长度 (米)
        max_steps: 最大步数
        max_curvature: 最大曲率
    
    Returns:
        list: 包含有效路径及其动作序列的列表
            [{
                'path_info': path_info,
                'actions': actions,
                'metrics': metrics,
                'path_index': index
            }, ...]
    """
    valid_trajectories = []
    filter_reasons = {} 


    for i, path_info in enumerate(all_paths):
        complete_path = path_info['complete_path']
        
        # 生成动作序列
        actions, action_details = path_to_actions(
            complete_path,
            initial_orientation=90.0,
            turn_angle=30.0,
            map_resolution=map_resolution,
            forward_step_meters=0.25
        )
        
        # 筛选该轨迹
        is_valid, reason, metrics = filter_single_trajectory(
            path_info, actions, map_resolution,
            min_length_m, max_steps, min_curvature,max_curvature
        )
        
        if is_valid:
            valid_trajectories.append({
                'path_info': path_info,
                'actions': actions,
                'action_details': action_details,
                'metrics': metrics,
                'path_index': i + 1
            })
        else:
             # 统计过滤原因
            filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
    
    return valid_trajectories, filter_reasons



# ============= Habitat 轨迹收集工具函数 =============
def depth_to_grayscale_image(depth):
    """
    将 depth 数组转换为灰度图像（用于保存为 jpg）
    
    Args:
        depth: depth 数组，shape 为 (H, W) 或 (H, W, 1)
    
    Returns:
        gray_rgb: 灰度图像（RGB格式，三通道相同），shape 为 (H, W, 3)
    """
    # 确保是 2D 数组
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]
    
    # 归一化到 0-255
    depth_min = depth.min()
    depth_max = depth.max()
    
    if depth_max > depth_min:
        depth_normalized = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    else:
        depth_normalized = np.zeros_like(depth, dtype=np.uint8)
    
    # 转换为 RGB 格式（三通道相同，实际上是灰度图）
    gray_rgb = np.stack([depth_normalized] * 3, axis=-1)
    
    return gray_rgb


def quaternion_to_yaw(rotation_quat):
    w = rotation_quat.w
    y = rotation_quat.y
    
    yaw = 2.0 * np.arctan2(y, w)
    
    return yaw

def get_agent_state_data(env):
    agent_state = env.sim.get_agent_state()
    pos_3d = agent_state.position  # [x, y, z]
    position = np.array([-pos_3d[2], -pos_3d[0]])
    yaw = quaternion_to_yaw(agent_state.rotation)
    
    return position, yaw

def collect_trajectory_from_actions(env, actions, episode_folder_path):
    """
    执行动作序列并收集完整的轨迹数据
    
    Returns:
        trajectory_data: dict {'position': array, 'yaw': array}
        success: bool
    """
    positions = []
    yaws = []
    
    print(f"    开始执行动作序列并收集数据...")
    print(f"    动作总数: {len(actions)}")
    print(f"    动作分布: Forward={actions.count(1)}, Left={actions.count(2)}, "
          f"Right={actions.count(3)}, Stop={actions.count(0)}")
    
    # 记录初始状态
    position, yaw = get_agent_state_data(env)
    positions.append(position)
    yaws.append([yaw])
    
    # 保存初始 RGB 图像
    obs = env.sim.get_observations_at(
        position=env.sim.get_agent_state().position,
        rotation=env.sim.get_agent_state().rotation
    )
    img_path = os.path.join(episode_folder_path, "0.jpg")
    imageio.imwrite(img_path, obs['rgb'])
    
    # 执行动作并记录轨迹
    time_step = 1
    for i, action in enumerate(actions):
        if action == 0:  # Stop
            break
        
        obs = env.step(action)
        
        position, yaw = get_agent_state_data(env)
        positions.append(position)
        yaws.append([yaw])
        
        img_path = os.path.join(episode_folder_path, f"{time_step}.jpg")
        imageio.imwrite(img_path, obs['rgb'])
        
        time_step += 1
        
        if (i + 1) % 10 == 0:
            print(f"      已执行 {i+1}/{len(actions)} 个动作, 已记录 {time_step} 帧")
    
    trajectory_data = {
        'position': np.array(positions, dtype=np.float32),
        'yaw': np.array(yaws, dtype=np.float32)
    }
    
    traj_data_path = os.path.join(episode_folder_path, 'traj_data.pkl')
    with open(traj_data_path, 'wb') as f:
        pickle.dump(trajectory_data, f)
    
    print(f"    ✓ 轨迹数据收集完成:")
    print(f"      - 时间步数: {len(positions)}")
    print(f"      - 图像文件: 0.jpg ~ {time_step-1}.jpg")
    print(f"      - Position shape: {trajectory_data['position'].shape}")
    print(f"      - Yaw shape: {trajectory_data['yaw'].shape}")
    
    return trajectory_data, True

def validate_trajectory_deltas(trajectory_data, max_delta=0.25, len_traj_pred=8, waypoint_spacing=1):
    """
    验证轨迹中所有可能采样窗口的增量是否在合理范围内
    （与 vint_dataset.py / train_utils.py 中的计算方式一致）
    
    Args:
        trajectory_data: dict {'position': array (N, 2), 'yaw': array (N,)}
        max_delta: 最大允许的单步增量 (米)
        len_traj_pred: 预测轨迹长度 (waypoints数量)
        waypoint_spacing: waypoint采样间隔
    
    Returns:
        tuple: (is_valid: bool, reason: str)
    """
    positions = trajectory_data['position']  # (N, 2) 世界坐标
    yaws = trajectory_data['yaw']            # (N,)
    if len(yaws.shape) == 2:
        yaws = yaws.squeeze(1)
    traj_len = len(positions)
    
    # 需要至少 len_traj_pred * waypoint_spacing + 1 个点才能创建一个有效样本
    min_required = len_traj_pred * waypoint_spacing + 1
    if traj_len < min_required:
        return True, f"轨迹太短 ({traj_len} < {min_required})"
    
    def yaw_rotmat(yaw):
        return np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)],
        ])
    
    def to_local_coords(pos, curr_pos, curr_yaw):
        rotmat = yaw_rotmat(curr_yaw)
        return (pos - curr_pos).dot(rotmat)
    
    def get_delta(actions):
        """与 train_utils.py 中的 get_delta 一致"""
        # actions shape: (num_waypoints, 2)
        ex_actions = np.concatenate([np.zeros((1, actions.shape[-1])), actions], axis=0)
        delta = ex_actions[1:] - ex_actions[:-1]
        return delta
    
    # 检查所有可能的采样窗口
    context_size = 3  # 与 nomad.yaml 中的 context_size 一致
    max_curr_time = traj_len - context_size - 1
    
    for curr_time in range(max_curr_time + 1):
        start_index = curr_time
        end_index = curr_time + len_traj_pred * waypoint_spacing + 1
        
        # 提取该窗口的位置和朝向
        pos_slice = positions[start_index:end_index:waypoint_spacing]
        yaw_slice = yaws[start_index:end_index:waypoint_spacing]
        
        if len(pos_slice) < 2:
            continue
        
        # 转换到 ego-centric 坐标（使用该窗口起始点的 yaw）
        waypoints = to_local_coords(pos_slice, pos_slice[0], yaw_slice[0])
        
        # actions = waypoints[1:] (排除原点，与 vint_dataset 一致)
        actions = waypoints[1:]
        
        if len(actions) == 0:
            continue
        
        # 计算 deltas（与 train_utils.py 的 get_delta 一致）
        deltas = get_delta(actions)
        
        # 检查是否超出阈值
        max_dx = np.abs(deltas[:, 0]).max()
        max_dy = np.abs(deltas[:, 1]).max()
        
        if max_dx > max_delta or max_dy > max_delta:
            return False, f"curr_time={curr_time}: max_dx={max_dx:.4f}, max_dy={max_dy:.4f}"
    
    return True, "通过验证"

# ============= Part 1: Preprocess Functions =============

def fov_mask(arr, y, x, fov_deg, fov_bearing, fov_range):
    """生成视野范围内的掩码"""
    min_theta = fov_deg - fov_bearing / 2
    max_theta = fov_deg + fov_bearing / 2
    
    y_idx, x_idx = np.indices(arr.shape)
    delta_x = x_idx - x
    delta_y = y_idx - y
    dist = np.sqrt(delta_x**2 + delta_y**2)
    angle = -np.arctan2(delta_y, delta_x)
    
    if max_theta > np.pi:
        mask = ((angle >= min_theta) | (angle <= max_theta - 2*np.pi)) & (dist <= fov_range)
    elif min_theta < -np.pi:
        mask = ((angle >= min_theta + 2*np.pi) | (angle <= max_theta)) & (dist <= fov_range)
    else:
        mask = (angle >= min_theta) & (angle <= max_theta) & (dist <= fov_range)
    
    return mask.astype(np.uint8)


def fill_visible_space(explored_map, obstacle_map, robot_pos, fov_angle=90, fov_range=50):
    """填充机器人视野内可见的空间"""
    from skimage.draw import line
    
    y, x = robot_pos[0], robot_pos[1]
    filled_map = explored_map.copy()
    
    fov_deg = np.pi / 2
    fov_bearing = np.deg2rad(fov_angle)
    angle_min = fov_deg - fov_bearing / 2
    angle_max = fov_deg + fov_bearing / 2
    
    angle_resolution = np.deg2rad(0.5)
    num_rays = int((angle_max - angle_min) / angle_resolution) + 1
    ray_angles = np.linspace(angle_min, angle_max, num_rays)
    
    for ray_angle in ray_angles:
        end_x = int(x + fov_range * np.cos(ray_angle))
        end_y = int(y + fov_range * np.sin(ray_angle))
        
        end_x = np.clip(end_x, 0, filled_map.shape[1] - 1)
        end_y = np.clip(end_y, 0, filled_map.shape[0] - 1)
        
        rr, cc = line(y, x, end_y, end_x)
        
        for i in range(len(rr)):
            r, c = rr[i], cc[i]
            if 0 <= r < filled_map.shape[0] and 0 <= c < filled_map.shape[1]:
                if obstacle_map[r, c] > 0.5:
                    break
                filled_map[r, c] = 1.0
    
    return filled_map


def process_exploration_map(explored_map, obstacle_map, robot_pos=None, 
                            fov_angle=90, fov_range=50):
    """对探索地图进行预处理"""
    if robot_pos is None:
        vision_range = explored_map.shape[0]
        robot_pos = [0, vision_range // 2]
    else:
        robot_pos = [robot_pos[1], robot_pos[0]]
    
    processed_explored_map = fill_visible_space(
        explored_map, obstacle_map, robot_pos, 
        fov_angle, fov_range
    )
    
    return processed_explored_map


def remove_narrow_passages(free_space, erosion_kernel_size=3):
    """移除自由空间中的狭窄通道"""
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (erosion_kernel_size, erosion_kernel_size))
    eroded = cv2.erode(free_space.astype(np.uint8), kernel_erode)
    
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                              (erosion_kernel_size, erosion_kernel_size))
    processed = cv2.dilate(eroded, kernel_dilate)
    
    return processed.astype(bool)


# ============= Part 2: Path Generation Functions =============

def identify_leaf_nodes(joint_nodes, voronoi_graph):
    """识别拓扑图中的叶节点"""
    leaf_nodes = []
    
    for node in joint_nodes:
        node_tuple = tuple(node)
        if node_tuple in voronoi_graph:
            if len(voronoi_graph[node_tuple]) == 1:
                leaf_nodes.append(node_tuple)
    
    return leaf_nodes


def find_path_in_graph(start_node, end_node, voronoi_graph):
    """在拓扑图中使用BFS找到路径"""
    start_tuple = tuple(start_node)
    end_tuple = tuple(end_node)
    
    if start_tuple not in voronoi_graph or end_tuple not in voronoi_graph:
        return None
    
    queue = deque([(start_tuple, [start_tuple])])
    visited = {start_tuple}
    
    while queue:
        current, path = queue.popleft()
        
        if current == end_tuple:
            return path
        
        if current in voronoi_graph:
            for neighbor in voronoi_graph[current]:
                neighbor_tuple = tuple(neighbor)
                if neighbor_tuple not in visited:
                    visited.add(neighbor_tuple)
                    queue.append((neighbor_tuple, path + [neighbor_tuple]))
    
    return None


def generate_path_on_skeleton(start_point, end_point, skeleton):
    """使用route_through_array在骨架上生成路径"""
    cost_map = np.where(skeleton, 1, 1000).astype(np.float32)
    
    indices, weight = route_through_array(
        cost_map,
        start_point,
        end_point,
        fully_connected=True
    )
    
    return indices, weight


def generate_complete_path(start_node, end_node, node_path, skeleton):
    """生成完整路径"""
    if node_path is None or len(node_path) < 2:
        return None, float('inf')
    
    complete_path = []
    total_cost = 0
    
    for i in range(len(node_path) - 1):
        segment_start = node_path[i]
        segment_end = node_path[i + 1]
        
        segment_path, segment_cost = generate_path_on_skeleton(
            segment_start, segment_end, skeleton
        )
        
        if i == 0:
            complete_path.extend(segment_path)
        else:
            complete_path.extend(segment_path[1:])
        
        total_cost += segment_cost
    
    return complete_path, total_cost


def generate_all_paths_to_leaves(robot_position, leaf_nodes, voronoi_graph, skeleton, joint_nodes):
    """生成从机器人位置到所有叶节点的路径"""
    min_dist = float('inf')
    start_node = None
    
    for node in joint_nodes:
        dist = (node[0] - robot_position[0])**2 + (node[1] - robot_position[1])**2
        if dist < min_dist:
            min_dist = dist
            start_node = tuple(node)
    
    
    all_paths = []
    
    for i, leaf_node in enumerate(leaf_nodes):
        node_path = find_path_in_graph(start_node, leaf_node, voronoi_graph)
        
        if node_path is None:
            continue
        
        complete_path, total_cost = generate_complete_path(
            start_node, leaf_node, node_path, skeleton
        )
        
        if complete_path is None:
            continue
        
        path_info = {
            'leaf_node': leaf_node,
            'node_path': node_path,
            'complete_path': complete_path,
            'total_cost': total_cost,
            'path_length': len(complete_path)
        }
        
        all_paths.append(path_info)
    
    
    return all_paths, start_node


def calculate_path_metrics(path_coords, map_resolution=5.0):
    """计算路径的实际长度和曲折度"""
    if len(path_coords) < 2:
        return {
            'path_length_cm': 0.0,
            'straight_distance_cm': 0.0,
            'tortuosity': 1.0
        }
    
    path_array = np.array(path_coords)
    
    path_length_pixels = 0.0
    for i in range(len(path_array) - 1):
        segment_length = np.linalg.norm(path_array[i+1] - path_array[i])
        path_length_pixels += segment_length
    
    straight_distance_pixels = np.linalg.norm(path_array[-1] - path_array[0])
    
    path_length_cm = path_length_pixels * map_resolution
    straight_distance_cm = straight_distance_pixels * map_resolution
    
    tortuosity = path_length_cm / straight_distance_cm if straight_distance_cm > 0 else 1.0
    
    return {
        'path_length_cm': path_length_cm,
        'straight_distance_cm': straight_distance_cm,
        'tortuosity': tortuosity
    }


# ============= Part 3: Action Generation Functions =============

def path_to_actions(path, initial_orientation=90.0, turn_angle=30.0, 
                    map_resolution=5.0, forward_step_meters=0.25):
    """将路径坐标序列转换为动作序列"""
    if len(path) < 2:
        return [0], [{'action': 0, 'action_name': 'Stop', 'position': path[0]}]
    
    actions = []
    action_details = []
    current_orientation = initial_orientation
    
    forward_step_cm = forward_step_meters * 100
    pixels_per_forward = int(forward_step_cm / map_resolution)
    
    i = 0
    current_pos = path[0]
    
    while i < len(path) - 1:
        target_idx = i + 1
        cumulative_dist = 0
        
        for j in range(i + 1, len(path)):
            dist = np.sqrt((path[j][0] - path[j-1][0])**2 + 
                          (path[j][1] - path[j-1][1])**2)
            cumulative_dist += dist
            
            if cumulative_dist >= pixels_per_forward * 0.8:
                target_idx = j
                break
            elif j == len(path) - 1:
                target_idx = j
                break
        
        target_pos = path[target_idx]
        
        d_row = target_pos[0] - current_pos[0]
        d_col = target_pos[1] - current_pos[1]
        
        target_angle = np.degrees(np.arctan2(d_col, d_row))
        
        current_orientation = ((current_orientation + 180) % 360) - 180
        target_angle = ((target_angle + 180) % 360) - 180
        
        relative_angle = ((current_orientation - target_angle + 180) % 360) - 180
        
        turn_count = 0
        while abs(relative_angle) > turn_angle / 2.0 and turn_count < 12:
            if relative_angle > 0:
                actions.append(3)
                action_details.append({
                    'action': 3,
                    'action_name': 'Right',
                    'position': current_pos,
                    'orientation': current_orientation
                })
                current_orientation -= turn_angle
            else:
                actions.append(2)
                action_details.append({
                    'action': 2,
                    'action_name': 'Left',
                    'position': current_pos,
                    'orientation': current_orientation
                })
                current_orientation += turn_angle
            
            current_orientation = ((current_orientation + 180) % 360) - 180
            relative_angle = ((current_orientation - target_angle + 180) % 360) - 180
            turn_count += 1
        
        actions.append(1)
        action_details.append({
            'action': 1,
            'action_name': 'Forward',
            'position': current_pos,
            'orientation': current_orientation
        })
        
        current_pos = target_pos
        i = target_idx
    
    actions.append(0)
    action_details.append({
        'action': 0,
        'action_name': 'Stop',
        'position': current_pos,
        'orientation': current_orientation
    })
    
    return actions, action_details


# ============= Voronoi Generation Functions =============

def generate_voronoi_map_from_obs(observations):
    """从观测数据生成 Voronoi 地图"""
    try:
        depth_obs = observations["depth"]
        
        if len(depth_obs.shape) == 3:
            depth_raw = depth_obs[:, :, 0]
        else:
            depth_raw = depth_obs
        
        min_depth_m = 0.5
        max_depth_m = 3.0
        
        if depth_raw.max() <= 1.0:
            depth_m = min_depth_m + depth_raw * (max_depth_m - min_depth_m)
        else:
            depth_m = depth_raw
        
        for i in range(depth_m.shape[1]):
            zero_mask = depth_m[:, i] == 0.
            if zero_mask.any():
                valid_values = depth_m[:, i][~zero_mask]
                if len(valid_values) > 0:
                    depth_m[:, i][zero_mask] = valid_values.max()
        
        mask_too_far = depth_m > 0.99 * max_depth_m
        depth_m[mask_too_far] = 0.
        mask_zero = depth_m == 0
        depth_m[mask_zero] = max_depth_m
        
        depth_cm = depth_m * 100.0
        
        from voronoi_map import depth_to_voronoi
        results_original = depth_to_voronoi(depth_cm)
        obstacle_map = results_original['obstacle_map']
        explored_map = results_original['explored_map']
        
        explored_map_processed = process_exploration_map(
            explored_map, obstacle_map, robot_pos=None,
            fov_angle=90, fov_range=50
        )
        
        compensation_mask = np.logical_and(
            explored_map_processed > 0.5,
            explored_map < 0.5
        ).astype(bool)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_obstacle = cv2.dilate(obstacle_map.astype(np.float32), kernel)
        kernel_close = np.ones((10, 10), dtype=np.uint8)
        explored_closed = cv2.morphologyEx(
            explored_map_processed.astype(np.uint8), 
            cv2.MORPH_CLOSE, 
            kernel_close
        )
        free_space_temp = np.maximum(0, explored_closed - dilated_obstacle)
        free_space_temp = (free_space_temp > 0.5).astype(bool)
        
        free_space_processed = remove_narrow_passages(free_space_temp, erosion_kernel_size=3)
        
        skeleton_processed = skimage.morphology.skeletonize(free_space_processed)
        
        vision_range = skeleton_processed.shape[0]
        robot_y, robot_x = 0, vision_range // 2
        robot_position = (robot_y, robot_x)
        protected_robot_position = robot_position
        
        if not skeleton_processed[robot_y, robot_x]:
            skeleton_coords = np.argwhere(skeleton_processed)
            if len(skeleton_coords) > 0:
                distances = np.sqrt(
                    (skeleton_coords[:, 0] - robot_y)**2 + 
                    (skeleton_coords[:, 1] - robot_x)**2
                )
                nearest_idx = np.argmin(distances)
                nearest_point = tuple(skeleton_coords[nearest_idx])
                
                from skimage.draw import line
                rr, cc = line(robot_y, robot_x, nearest_point[0], nearest_point[1])
                
                for i in range(len(rr)):
                    if (0 <= rr[i] < skeleton_processed.shape[0] and 
                        0 <= cc[i] < skeleton_processed.shape[1]):
                        if free_space_processed[rr[i], cc[i]]:
                            skeleton_processed[rr[i], cc[i]] = True
        
        skeleton_processed[robot_y, robot_x] = True
        
        # 执行骨架剪枝
        for prune_iter in range(15):
            iteration_removed = 0
            skeleton_coords = np.argwhere(skeleton_processed)
            
            junction_points = set()
            for coord in skeleton_coords:
                x, y = coord[0], coord[1]
                if x > 0 and x < skeleton_processed.shape[0] - 1 and \
                   y > 0 and y < skeleton_processed.shape[1] - 1:
                    neighbor_count = np.sum(skeleton_processed[x-1:x+2, y-1:y+2]) - 1
                    if neighbor_count >= 3:
                        junction_points.add((x, y))
            
            endpoints_to_check = []
            for coord in skeleton_coords:
                x, y = coord[0], coord[1]
                if x >= 1 and x < skeleton_processed.shape[0] - 1 and \
                   y >= 1 and y < skeleton_processed.shape[1] - 1:
                    neighbor_count = np.sum(skeleton_processed[x-1:x+2, y-1:y+2]) - 1
                    if neighbor_count == 1:
                        endpoints_to_check.append((x, y))
            
            for start_point in endpoints_to_check:
                if start_point == protected_robot_position:
                    continue
                if not skeleton_processed[start_point[0], start_point[1]]:
                    continue
                
                visited = {start_point}
                queue = deque([start_point])
                path = [start_point]
                end_junction = None
                
                while queue:
                    x, y = queue.popleft()
                    
                    if (x, y) != start_point and (x, y) in junction_points:
                        end_junction = (x, y)
                        break
                    
                    directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
                    neighbors = []
                    
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < skeleton_processed.shape[0] and 
                            0 <= ny < skeleton_processed.shape[1] and 
                            skeleton_processed[nx, ny] and 
                            (nx, ny) not in visited):
                            neighbors.append((nx, ny))
                    
                    if len(neighbors) == 1:
                        next_point = neighbors[0]
                        visited.add(next_point)
                        queue.append(next_point)
                        path.append(next_point)
                    elif len(neighbors) > 1:
                        break
                
                branch_length = len(path)
                points_in_compensation = sum(
                    1 for px, py in path 
                    if compensation_mask[px, py]
                )
                compensation_ratio = points_in_compensation / branch_length if branch_length > 0 else 0
                
                should_remove = False
                endpoint_in_compensation = compensation_mask[start_point[0], start_point[1]]
                
                root_length = min(10, branch_length)
                if end_junction is not None:
                    root_points = path[-root_length:]
                else:
                    root_points = path[:root_length]
                    
                root_compensation_ratio = sum(
                    1 for px, py in root_points if compensation_mask[px, py]
                ) / len(root_points)
                
                max_consecutive = 0
                current = 0
                for px, py in path:
                    if compensation_mask[px, py]:
                        current += 1
                        max_consecutive = max(max_consecutive, current)
                    else:
                        current = 0
                
                if endpoint_in_compensation:
                    should_remove = True
                elif root_compensation_ratio > 0.85:
                    should_remove = True
                elif max_consecutive >= 15:
                    should_remove = True
                elif compensation_ratio > 0.5:
                    should_remove = True
                elif branch_length < 6:
                    should_remove = True
                
                if should_remove:
                    for px, py in path:
                        if (px, py) != protected_robot_position and skeleton_processed[px, py]:
                            skeleton_processed[px, py] = False
                            iteration_removed += 1
            
            if iteration_removed == 0:
                break
        
        skeleton_labeled, num_components = measure.label(
            skeleton_processed, connectivity=2, return_num=True
        )
        
        if num_components > 1:
            robot_component = skeleton_labeled[protected_robot_position[0], 
                                               protected_robot_position[1]]
            if robot_component > 0:
                skeleton_processed = (skeleton_labeled == robot_component)
        
        skeleton_processed[protected_robot_position[0], protected_robot_position[1]] = True
        skeleton_processed = skeleton_processed.astype(bool)
        
        joint_nodes_processed = extract_voronoi_nodes(skeleton_processed)
        
        if protected_robot_position not in [tuple(node) for node in joint_nodes_processed]:
            joint_nodes_processed.append(protected_robot_position)
        
        graph_processed, edges_processed = build_voronoi_graph(
            skeleton_processed, joint_nodes_processed
        )
        
        return {
            'obstacle_map': obstacle_map,
            'explored_map': explored_map_processed,
            'skeleton': skeleton_processed,
            'graph': graph_processed,
            'nodes': joint_nodes_processed,
            'robot_pos': robot_position
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def generate_paths_from_voronoi(voronoi_result):
    """从 Voronoi 结果生成路径"""
    try:
        skeleton = voronoi_result['skeleton']
        voronoi_graph = voronoi_result['graph']
        joint_nodes = voronoi_result['nodes']
        robot_position = voronoi_result['robot_pos']
        
        leaf_nodes = identify_leaf_nodes(joint_nodes, voronoi_graph)
        
        if len(leaf_nodes) == 0:
            return None
        
        all_paths, start_node = generate_all_paths_to_leaves(
            robot_position, leaf_nodes, voronoi_graph, skeleton, joint_nodes
        )
        
        if len(all_paths) == 0:
            return None
        
        return {
            'paths': all_paths,
            'leaf_nodes': leaf_nodes,
            'start_node': start_node
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


# ============= Main Pipeline =============
def process_single_episode_with_new_rules(observations, output_dir, scene_name, 
                                          base_episode_num, env,
                                          min_length_m=1.5, max_steps=50, 
                                          min_curvature=1.28, max_curvature=2.5,
                                        min_wall_distance_m=0.75):
    """
    使用新的筛选规则处理单个 episode，为每条有效轨迹创建独立的 episode
    
    Args:
        observations: 观测数据
        output_dir: 输出目录
        scene_name: 场景名称
        base_episode_num: 基础 episode 编号
        env: Habitat 环境
        min_length_m: 最小路径长度 (米)
        max_steps: 最大步数
        max_curvature: 最大曲率
    
    Returns:
        tuple: (num_valid_paths: int, statistics: dict)
    """
    
    
    # ========== Step 1: 生成 Voronoi 地图 ==========
    
    voronoi_result = generate_voronoi_map_from_obs(observations)
    if voronoi_result is None:
        return {
            'total_paths': 0,
            'valid_paths': 0,
            'filtered_paths': 0,
            'filter_reasons': {},
            'successful_episodes': 0,
            'voronoi_failed': True,
            'no_paths': False
        }
    

    # ========== Step 2: 生成路径 ==========

    
    paths_result = generate_paths_from_voronoi(voronoi_result)
    if paths_result is None:
        return {
            'total_paths': 0,
            'valid_paths': 0,
            'filtered_paths': 0,
            'filter_reasons': {},
            'successful_episodes': 0,
            'voronoi_failed': False,
            'no_paths': True
        }
    
    skeleton = voronoi_result['skeleton']
    all_paths = paths_result['paths']
    robot_position = voronoi_result['robot_pos']
    
    total_paths = len(all_paths)
    
    # ========== Step 3: 独立筛选所有轨迹 ==========
    
    valid_trajectories ,filter_reasons= filter_all_trajectories_independently(
        all_paths, skeleton, robot_position,
        map_resolution=5.0,
        min_length_m=min_length_m,
        max_steps=max_steps,
        min_curvature=min_curvature,
        max_curvature=max_curvature
    )
    valid_paths = len(valid_trajectories)
    filtered_paths = total_paths - valid_paths

    if len(valid_trajectories) == 0:
        return {
            'total_paths': total_paths,
            'valid_paths': 0,
            'filtered_paths': filtered_paths,
            'filter_reasons': filter_reasons,
            'successful_episodes': 0,
            'voronoi_failed': False,
            'no_paths': False
        }
    
    # ========== Step 4: 为每条有效轨迹创建独立的 episode ==========
    
    successful_episodes = 0
    delta_validation_failed = 0  # 新增：增量验证失败计数
    
    for traj_idx, traj_data in enumerate(valid_trajectories):
        # 统一命名格式：<场景名>_<episode序号>_<路径序号>
        episode_folder_name = f"{scene_name}_{base_episode_num}_{traj_data['path_index']}"
        
        episode_folder_path = os.path.join(output_dir, episode_folder_name)
        
        
        # 创建文件夹
        os.makedirs(episode_folder_path, exist_ok=True)
        
        try:
            # 重置环境并执行动作收集轨迹
            env.reset()
            
            # 执行动作并收集轨迹
            trajectory_data, success = collect_trajectory_from_actions(
                env, traj_data['actions'], episode_folder_path
            )
            
            if success:
                # 验证轨迹增量是否在合理范围内
                is_valid, validation_reason = validate_trajectory_deltas(
                    trajectory_data, max_delta=0.25
                )
                
                if is_valid:
                    successful_episodes += 1
                else:
                    # 轨迹验证失败，删除该 episode
                    delta_validation_failed += 1  # 新增
                    tqdm.tqdm.write(f"轨迹验证失败 {episode_folder_name}: {validation_reason}")
                    if os.path.exists(episode_folder_path):
                        shutil.rmtree(episode_folder_path)
            else:
                if os.path.exists(episode_folder_path):
                    shutil.rmtree(episode_folder_path)
        
        except Exception as e:
            if os.path.exists(episode_folder_path):
                shutil.rmtree(episode_folder_path)
    
    result = {
        'total_paths': total_paths,
        'valid_paths': valid_paths,
        'filtered_paths': filtered_paths,
        'filter_reasons': filter_reasons,
        'successful_episodes': successful_episodes,
        'delta_validation_failed': delta_validation_failed,  # 新增
        'voronoi_failed': False,
        'no_paths': False
    }
    
    print(f"\n✓ 观测 {scene_name}_ep{base_episode_num} 处理完成！")
    print(f"  - 生成路径: {total_paths} 条")
    print(f"  - 通过规则筛选: {valid_paths} 条")
    print(f"  - 未通过规则筛选: {filtered_paths} 条")
    print(f"  - 增量验证失败: {delta_validation_failed} 条")  # 新增
    print(f"  - 最终保存: {successful_episodes} 个 episodes")  # 修改
    
    return result

def main(config_path="benchmark/nav/pointnav/pointnav_hm3d.yaml", 
         output_dir='data/my_dataset',
         scene_names=['Adrian', 'Albertville', 'Anaheim'],
         max_episodes_per_scene=20):
    """
    主函数：生成 NoMaD 训练数据集（规则筛选）
    
    统计指标:
        (a) 生成的轨迹总数
        (b) 通过筛选的轨迹数量
        (c) 未通过筛选的轨迹数量
        (d) 由于筛选规则中不同规则被过滤的轨迹数量
    """
    import time
    from collections import defaultdict
    
    print("="*70)
    print("生成 NoMaD 训练数据集 (规则筛选)")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算预估总episode数
    estimated_total = len(scene_names) * max_episodes_per_scene
    

    #统计变量
    total_paths_generated = 0      # (a) 生成的轨迹总数
    total_paths_valid = 0          # (b) 通过规则筛选的轨迹数量
    total_paths_filtered = 0       # (c) 未通过规则筛选的轨迹数量
    total_delta_failed = 0         # (d) 新增：增量验证失败数量
    total_successful = 0           # (e) 新增：最终成功保存数量
    all_filter_reasons = defaultdict(int)
    
    # 辅助统计
    total_episodes_processed = 0
    voronoi_failed_count = 0
    no_paths_count = 0
    too_close_to_wall_count = 0

    # 创建全局进度条
    global_pbar = tqdm.tqdm(
        total=estimated_total,
        desc="总进度",
        unit="ep",
        position=0,
        leave=True,
        dynamic_ncols=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    start_time = time.time()
    
    # 逐个场景处理
    for scene_idx, scene_name in enumerate(scene_names, 1):
        global_pbar.set_description(f"场景 {scene_idx}/{len(scene_names)}: {scene_name}")
        
        scene_data_path = f"data/datasets/pointnav/hm3d/v1/train/content/{scene_name}.json.gz"
        
        config = habitat.get_config(
            config_path=config_path,
            overrides=[f"habitat.dataset.data_path={scene_data_path}"]
        )
        
        try:
            dataset = habitat.make_dataset(
                id_dataset=config.habitat.dataset.type, 
                config=config.habitat.dataset
            )
        except Exception as e:
            tqdm.tqdm.write(f"✗ 加载场景 {scene_name} 失败: {e}")
            global_pbar.update(max_episodes_per_scene)
            continue
        
        scene_total_episodes = len(dataset.episodes)
        episodes_to_process = min(max_episodes_per_scene, scene_total_episodes)
        
        # 场景统计
        scene_total_paths = 0
        scene_valid_paths = 0
        scene_filtered_paths = 0
        
        # 创建仿真环境
        try:
            with habitat.Env(config=config, dataset=dataset) as env:
                for ep_idx in range(episodes_to_process):
                    observations = env.reset()
                    episode_num = ep_idx + 1
                    total_episodes_processed += 1
                    
                    # 处理 episode
                    try:
                        result = process_single_episode_with_new_rules(
                            observations, output_dir,
                            scene_name, episode_num, env, 
                            min_length_m=1.5,
                            max_steps=50,   
                            min_curvature=1.28,
                            max_curvature=2.5
                        )
                        
                        # 累计统计
                        if result.get('too_close_to_wall', False):
                            too_close_to_wall_count += 1
                        elif result['voronoi_failed']:
                            voronoi_failed_count += 1
                        elif result['no_paths']:
                            no_paths_count += 1
                        else:
                            # 核心统计
                            total_paths_generated += result['total_paths']
                            total_paths_valid += result['valid_paths']
                            total_paths_filtered += result['filtered_paths']
                            total_delta_failed += result.get('delta_validation_failed', 0)  # 新增
                            total_successful += result['successful_episodes']  # 新增
                            
                            # 场景统计
                            scene_total_paths += result['total_paths']
                            scene_valid_paths += result['valid_paths']
                            scene_filtered_paths += result['filtered_paths']
                            
                            # 过滤原因统计
                            for reason, count in result['filter_reasons'].items():
                                all_filter_reasons[reason] += count

                    except Exception as e:
                        tqdm.tqdm.write(f"错误 - {scene_name}_ep{episode_num}: {e}")
                        voronoi_failed_count += 1
                    
                    # 更新进度条
                    global_pbar.update(1)
                    global_pbar.set_postfix({
                        '生成': total_paths_generated,
                        '规则通过': total_paths_valid,
                        '最终保存': total_successful  # 修改
                    })
                
                tqdm.tqdm.write(f"\n场景 {scene_name}: 生成={scene_total_paths}, "
                              f"通过={scene_valid_paths}, 过滤={scene_filtered_paths}")
                
        except Exception as e:
            tqdm.tqdm.write(f"✗ 创建环境失败 {scene_name}: {e}")
            global_pbar.update(episodes_to_process)
    
    global_pbar.close()
    
    # 计算总耗时
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
        # ========== 最终统计 ==========
    print("\n" + "="*70)
    print("数据集生成完成!")
    print("="*70)
    
    # (1) 总时间
    print(f"\n总时间: {hours}小时 {minutes}分钟 {seconds}秒")

    # (2) 生成的轨迹数量
    print(f"生成的轨迹数量: {total_paths_generated}")
    
    # (3) 通过规则筛选的轨迹数量
    print(f"通过规则筛选的轨迹数量: {total_paths_valid}")
    
    # (4) 未通过规则筛选的轨迹数量
    print(f"未通过规则筛选的轨迹数量: {total_paths_filtered}")
    
    # (5) 新增：增量验证失败数量
    print(f"增量验证失败数量: {total_delta_failed}")
    
    # (6) 新增：最终成功保存数量
    print(f"最终成功保存数量: {total_successful}")
    
    # 分类统计
    filter_categories = {
        '小于最小路径长度': 0,
        '大于最大步数': 0,
        '小于最小路径比率': 0,
        '大于最大路径比率': 0,
        '超出震荡检测': 0,
        '超出平均动作块长度': 0,
        '超出原地旋转步数': 0,
    }
    
    for reason, count in all_filter_reasons.items():
        if "小于最小长度" in reason:
            filter_categories['小于最小路径长度'] += count
        elif "超过最大步数" in reason:
            filter_categories['大于最大步数'] += count
        elif "小于最小值" in reason:  # 曲率小于最小值
            filter_categories['小于最小路径比率'] += count
        elif "超过最大值" in reason:  # 曲率超过最大值
            filter_categories['大于最大路径比率'] += count
        elif "震荡率" in reason:
            filter_categories['超出震荡检测'] += count
        elif "平均步长" in reason:
            filter_categories['超出平均动作块长度'] += count
        elif "连续4次或更多旋转" in reason:
            filter_categories['超出原地旋转步数'] += count
    
    for category, count in filter_categories.items():
        print(f"  - {category}: {count}")
    
    print("="*70)
    
    # 保存统计结果到JSON
    stats = {
        'total_time_seconds': total_time,
        'total_paths_generated': total_paths_generated,
        'total_paths_valid': total_paths_valid,
        'total_paths_filtered': total_paths_filtered,
        'total_delta_failed': total_delta_failed,  # 新增
        'total_successful': total_successful,       # 新增
        'filter_categories': filter_categories,
    }
    
    stats_path = os.path.join(output_dir, 'dataset_statistics.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n统计结果已保存至: {stats_path}")


if __name__ == "__main__":
    import glob
    
    # 配置参数
    CONFIG_PATH = "benchmark/nav/pointnav/pointnav_hm3d.yaml"
    OUTPUT_DIR = "data/my_dataset_800_40"
    
    # 自动获取所有场景名称
    content_dir = "data/datasets/pointnav/hm3d/v1/train/content"
    scene_files = glob.glob(os.path.join(content_dir, "*.json.gz"))
    
    # 提取场景名称（去掉路径和 .json.gz 后缀）
    SCENE_NAMES = []
    for f in scene_files:
        scene_name = os.path.basename(f).replace('.json.gz', '')
        SCENE_NAMES.append(scene_name)
    
    # 排序确保顺序一致
    SCENE_NAMES = sorted(SCENE_NAMES)#[:50]
    
    print(f"找到 {len(SCENE_NAMES)} 个场景")
    
    # 每个场景最多处理的episodes数
    MAX_EPISODES_PER_SCENE = 40  # 可以调整
    
    # 运行
    main(
        config_path=CONFIG_PATH,
        output_dir=OUTPUT_DIR,
        scene_names=SCENE_NAMES,
        max_episodes_per_scene=MAX_EPISODES_PER_SCENE
    )

