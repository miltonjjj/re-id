"""
批量生成原始和净化后的Voronoi地图对比可视化
结合 depth_to_voronoi 和 preprocess_fov 功能
每个episode创建一个文件夹，包含:
  - depth.npy: 原始深度数据
  - depth.png: 深度可视化
  - rgb.png: RGB图像
  - voronoi_original.png: 原始voronoi图(障碍物、探索、自由空间、骨架)
  - voronoi_processed.png: 净化后voronoi图(障碍物、净化探索、净化自由空间、净化骨架)
"""

import os
import numpy as np
import cv2
import habitat
import tqdm
import imageio
from queue import Queue
import matplotlib.pyplot as plt
from PIL import Image
import skimage.morphology
from skimage import measure

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


# ============= FOV预处理函数（从 covnav_realworld/model.py 移植）=============

def fov_mask(arr, y, x, fov_deg, fov_bearing, fov_range):
    """
    生成视野范围内的掩码
    
    参数:
        arr: 地图数组
        y, x: 机器人位置
        fov_deg: 朝向角度（弧度）
        fov_bearing: 视野角度范围（弧度）
        fov_range: 视野距离范围（格子数）
    """
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
    """
    填充机器人视野内可见的空间（机器人到障碍物之间）
    使用射线投射算法
    
    参数:
        explored_map: (H, W) 原始探索地图
        obstacle_map: (H, W) 障碍物地图
        robot_pos: [y, x] 机器人位置
        fov_angle: 视野角度（度）
        fov_range: 视野范围（格子数）
    
    返回:
        filled_map: 填充后的探索地图
    """
    from skimage.draw import line
    
    y, x = robot_pos[0], robot_pos[1]
    filled_map = explored_map.copy()
    
    # 计算FOV的角度范围
    fov_deg = np.pi / 2  # 朝向前方（Y轴正方向）
    fov_bearing = np.deg2rad(fov_angle)
    angle_min = fov_deg - fov_bearing / 2
    angle_max = fov_deg + fov_bearing / 2
    
    # 生成射线方向（每隔0.5度一条射线）
    angle_resolution = np.deg2rad(0.5)
    num_rays = int((angle_max - angle_min) / angle_resolution) + 1
    ray_angles = np.linspace(angle_min, angle_max, num_rays)
    
    # 对每条射线进行投射
    for ray_angle in ray_angles:
        # 计算射线终点
        end_x = int(x + fov_range * np.cos(ray_angle))
        end_y = int(y + fov_range * np.sin(ray_angle))
        
        # 限制在地图范围内
        end_x = np.clip(end_x, 0, filled_map.shape[1] - 1)
        end_y = np.clip(end_y, 0, filled_map.shape[0] - 1)
        
        # 获取射线路径上的所有点
        rr, cc = line(y, x, end_y, end_x)
        
        # 沿着射线填充，直到遇到障碍物
        for i in range(len(rr)):
            r, c = rr[i], cc[i]
            
            # 检查是否在地图范围内
            if 0 <= r < filled_map.shape[0] and 0 <= c < filled_map.shape[1]:
                # 如果遇到障碍物，停止这条射线
                if obstacle_map[r, c] > 0.5:
                    break
                
                # 填充为已探索区域
                filled_map[r, c] = 1.0
    
    return filled_map


def process_exploration_map(explored_map, obstacle_map, robot_pos=None, 
                            fov_angle=90, fov_range=50, filtered_fov_angle=10):
    """
    对探索地图进行预处理：填充机器人视野内的可见空间
    
    参数:
        explored_map: (H, W) 探索地图
        obstacle_map: (H, W) 障碍物地图
        robot_pos: (x, y) 机器人位置，如果为None则使用地图底部中心
        fov_angle: 视野角度（度）
        fov_range: 视野范围（格子数）
        filtered_fov_angle: 未使用（保留接口兼容性）
    
    返回:
        processed_explored_map: 填充后的探索地图
    """
    if robot_pos is None:
        vision_range = explored_map.shape[0]
        robot_pos = [0, vision_range // 2]  # [y, x]
    else:
        robot_pos = [robot_pos[1], robot_pos[0]]  # 转换为 [y, x]
    
    # 直接使用射线填充，不再进行二次处理
    processed_explored_map = fill_visible_space(
        explored_map, obstacle_map, robot_pos, 
        fov_angle, fov_range
    )
    
    return processed_explored_map

def remove_narrow_passages(free_space, erosion_kernel_size=3):
    """
    移除自由空间中的狭窄通道
    
    参数:
        free_space: (H, W) 自由空间地图
        erosion_kernel_size: 腐蚀核大小，越大移除的通道越宽
    
    返回:
        processed_free_space: 处理后的自由空间
    """
    # 腐蚀操作：移除狭窄通道
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (erosion_kernel_size, erosion_kernel_size))
    eroded = cv2.erode(free_space.astype(np.uint8), kernel_erode)
    
    # 膨胀回原大小（但狭窄通道已经被断开）
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                              (erosion_kernel_size, erosion_kernel_size))
    processed = cv2.dilate(eroded, kernel_dilate)
    
    return processed.astype(bool)

# ============= 可视化函数 =============

def visualize_maps_comparison(obstacle_map, explored_map, free_space, skeleton,
                              explored_map_proc, free_space_proc, skeleton_proc,
                              joint_nodes_proc, graph_proc,  # 新增参数
                              save_path_original, save_path_processed):
    """
    生成原始和处理后的对比可视化
    
    参数:
        obstacle_map: 障碍物地图
        explored_map: 原始探索地图
        free_space: 原始自由空间
        skeleton: 原始骨架
        explored_map_proc: 处理后的探索地图
        free_space_proc: 处理后的自由空间
        skeleton_proc: 处理后的骨架
        joint_nodes_proc: 处理后的关节点  # 新增
        graph_proc: 处理后的拓扑图  # 新增
        save_path_original: 原始图保存路径
        save_path_processed: 处理后图保存路径
    """
    vision_range = obstacle_map.shape[0]
    robot_x = vision_range / 2
    robot_y = 0
    
    # 原始图：2x2布局
    fig1, axes1 = plt.subplots(2, 2, figsize=(10, 10))
    
    axes1[0, 0].imshow(obstacle_map, cmap='gray', origin='lower')
    axes1[0, 0].plot(robot_x, robot_y, 'ro', markersize=8)
    axes1[0, 0].set_title('Obstacle Map')
    axes1[0, 0].axis('off')
    
    axes1[0, 1].imshow(explored_map, cmap='gray', origin='lower')
    axes1[0, 1].plot(robot_x, robot_y, 'ro', markersize=8)
    axes1[0, 1].set_title('Explored Map')
    axes1[0, 1].axis('off')
    
    axes1[1, 0].imshow(free_space, cmap='gray', origin='lower')
    axes1[1, 0].plot(robot_x, robot_y, 'ro', markersize=8)
    axes1[1, 0].set_title('Free Space')
    axes1[1, 0].axis('off')
    
    axes1[1, 1].imshow(skeleton, cmap='gray', origin='lower')
    axes1[1, 1].plot(robot_x, robot_y, 'ro', markersize=8)
    axes1[1, 1].set_title(f'Voronoi Skeleton')
    axes1[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path_original, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # 处理后图：2x3布局（新增2个子图）
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    
    axes2[0, 0].imshow(obstacle_map, cmap='gray', origin='lower')
    axes2[0, 0].plot(robot_x, robot_y, 'ro', markersize=8)
    axes2[0, 0].set_title('Obstacle Map')
    axes2[0, 0].axis('off')
    
    axes2[0, 1].imshow(explored_map_proc, cmap='gray', origin='lower')
    axes2[0, 1].plot(robot_x, robot_y, 'ro', markersize=8)
    axes2[0, 1].set_title('Processed Explored Map')
    axes2[0, 1].axis('off')
    
    axes2[0, 2].imshow(free_space_proc, cmap='gray', origin='lower')
    axes2[0, 2].plot(robot_x, robot_y, 'ro', markersize=8)
    axes2[0, 2].set_title('Processed Free Space')
    axes2[0, 2].axis('off')
    
    axes2[1, 0].imshow(skeleton_proc, cmap='gray', origin='lower')
    axes2[1, 0].plot(robot_x, robot_y, 'ro', markersize=8)
    axes2[1, 0].set_title(f'Processed Voronoi Skeleton')
    axes2[1, 0].axis('off')
    
    # ★新增：关节点可视化
    axes2[1, 1].imshow(skeleton_proc, cmap='gray', origin='lower')
    axes2[1, 1].plot(robot_x, robot_y, 'go', markersize=10, label='Robot', zorder=10)
    if len(joint_nodes_proc) > 0:
        nodes = np.array(joint_nodes_proc)
        axes2[1, 1].scatter(nodes[:, 1], nodes[:, 0], c='red', s=50, zorder=5, label='Joint Nodes')
    axes2[1, 1].set_title(f'Joint Nodes ({len(joint_nodes_proc)})')
    axes2[1, 1].axis('off')
    axes2[1, 1].legend(loc='upper right')
    
    # ★新增：拓扑图可视化
    axes2[1, 2].imshow(skeleton_proc, cmap='gray', origin='lower')
    axes2[1, 2].plot(robot_x, robot_y, 'go', markersize=10, label='Robot', zorder=10)
    if len(joint_nodes_proc) > 0:
        nodes = np.array(joint_nodes_proc)
        axes2[1, 2].scatter(nodes[:, 1], nodes[:, 0], c='red', s=50, zorder=5)
        # 绘制边
        for node, neighbors in graph_proc.items():
            for neighbor in neighbors:
                axes2[1, 2].plot([node[1], neighbor[1]], 
                               [node[0], neighbor[0]], 
                               'yellow', alpha=0.9, linewidth=3, zorder=1)
    axes2[1, 2].set_title('Voronoi Graph')
    axes2[1, 2].axis('off')
    axes2[1, 2].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path_processed, dpi=150, bbox_inches='tight')
    plt.close(fig2)


# ============= 主处理函数 =============

# ============= 主处理函数 =============

def batch_generate_processed_voronoi_maps(output_dir='data/voronoi_processed'):
    """
    批量生成原始和净化后的Voronoi地图对比
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建 habitat config
    config = habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_gibson.yaml"
    )
    
    # 创建数据集
    dataset = habitat.make_dataset(
        id_dataset=config.habitat.dataset.type, 
        config=config.habitat.dataset
    )
    
    total_episodes = len(dataset.episodes)
    
    # 统计场景信息
    unique_scenes = set(os.path.basename(ep.scene_id).split('.')[0] 
                        for ep in dataset.episodes)
    
    print(f"总episode数: {total_episodes}")
    print(f"包含场景数: {len(unique_scenes)}")
    print(f"场景列表: {sorted(unique_scenes)}")
    print(f"输出目录: {output_dir}")
    print("="*70)
    
    processed_count = 0
    skipped_count = 0
    scene_episode_counter = {}
    
    # 创建仿真环境
    print("\n正在创建Habitat仿真环境...")
    with habitat.Env(config=config, dataset=dataset) as env:
        print("✓ 环境创建成功！开始处理episodes...\n")
        
        for ep_idx in tqdm.tqdm(range(min(15, total_episodes)), desc="处理episodes"):
            # 重置环境
            observations = env.reset()
            
            # 获取当前episode信息
            current_episode = env.current_episode
            scene_name = os.path.basename(current_episode.scene_id).split('.')[0]
            
            if scene_name not in scene_episode_counter:
                scene_episode_counter[scene_name] = 1
            else:
                scene_episode_counter[scene_name] += 1
            episode_num = scene_episode_counter[scene_name]
            
            # 创建episode文件夹
            episode_folder_name = f"{scene_name}_ep{episode_num}"
            episode_folder_path = os.path.join(output_dir, episode_folder_name)
            
            # 定义文件路径
            depth_path = os.path.join(episode_folder_path, "depth.npy")
            depth_png_path = os.path.join(episode_folder_path, "depth.png")
            rgb_path = os.path.join(episode_folder_path, "rgb.png")
            voronoi_original_path = os.path.join(episode_folder_path, "voronoi_original.png")
            voronoi_processed_path = os.path.join(episode_folder_path, "voronoi_processed.png")
            
            # 检查是否已存在
            if os.path.exists(episode_folder_path):
                if all(os.path.exists(p) for p in [depth_path, depth_png_path, rgb_path, 
                                                     voronoi_original_path, voronoi_processed_path]):
                    skipped_count += 1
                    continue
            
            os.makedirs(episode_folder_path, exist_ok=True)
            
            try:
                # 1. 保存RGB图像
                rgb_obs = observations["rgb"]
                imageio.imwrite(rgb_path, rgb_obs)
                
                # 2. 获取并预处理深度数据
                depth_obs = observations["depth"]
                np.save(depth_path, depth_obs)
                
                from habitat.utils.visualizations.utils import observations_to_image
                depth_only_obs = {"depth": depth_obs}
                depth_image = observations_to_image(depth_only_obs, {})
                imageio.imwrite(depth_png_path, depth_image)
                
                # 深度预处理
                if len(depth_obs.shape) == 3:
                    depth_raw = depth_obs[:, :, 0]
                else:
                    depth_raw = depth_obs
                
                min_depth_m = 0.5
                max_depth_m = 5.0
                
                if depth_raw.max() <= 1.0:
                    depth_m = min_depth_m + depth_raw * (max_depth_m - min_depth_m)
                else:
                    depth_m = depth_raw
                
                # 异常值处理
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
                
                # 3. 生成原始Voronoi图（与act.py保持完全一致）
                from voronoi_map import depth_to_voronoi
                
                results_original = depth_to_voronoi(depth_cm)
                obstacle_map = results_original['obstacle_map']
                explored_map = results_original['explored_map']
                free_space_original = results_original['free_space']
                skeleton_original = results_original['skeleton']
                
                # 4. 净化探索地图
                explored_map_processed = process_exploration_map(
                    explored_map, 
                    obstacle_map,
                    robot_pos=None,
                    fov_angle=90,
                    fov_range=50,
                    filtered_fov_angle=5
                )

                 # ★ 新增：计算补偿区域（填充的但非真实观测的区域）
                compensation_mask = np.logical_and(
                    explored_map_processed > 0.5,  # 处理后认为已探索
                    explored_map < 0.5              # 但原始数据未观测到
                ).astype(bool)
                
                print(f"    补偿区域占比: {np.sum(compensation_mask) / compensation_mask.size:.2%}")

                # 5. 生成处理后的自由空间（先不做骨架）
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

                # 6. 移除狭窄通道
                free_space_processed = remove_narrow_passages(free_space_temp, erosion_kernel_size=3)

                # 7. 从处理后的自由空间生成骨架（选择与机器人最近的分支）
                skeleton_processed = skimage.morphology.skeletonize(free_space_processed)

                # 选择与机器人最近的骨架分支
                skeleton_labeled, num_components = measure.label(
                    skeleton_processed, connectivity=2, return_num=True
                )

                if num_components > 0:
                    # 机器人位置
                    vision_range = skeleton_processed.shape[0]
                    robot_y, robot_x = 0, vision_range // 2
                    
                    # ... 选择骨架分支的代码保持不变 ...
                    # [保留原有的骨架选择逻辑]

                skeleton_processed = skeleton_processed.astype(bool)
                
                # ============= 新增：连接机器人到骨架 =============
                vision_range = skeleton_processed.shape[0]
                robot_y, robot_x = 0, vision_range // 2
                robot_position = (robot_y, robot_x)
                
                print(f"    机器人位置: ({robot_y}, {robot_x})")
                
                # 1. 检查机器人位置是否已在骨架上
                if not skeleton_processed[robot_y, robot_x]:
                    # 2. 找到最近的骨架点
                    skeleton_coords = np.argwhere(skeleton_processed)
                    if len(skeleton_coords) > 0:
                        # 计算到所有骨架点的距离
                        distances = np.sqrt(
                            (skeleton_coords[:, 0] - robot_y)**2 + 
                            (skeleton_coords[:, 1] - robot_x)**2
                        )
                        nearest_idx = np.argmin(distances)
                        nearest_point = tuple(skeleton_coords[nearest_idx])
                        min_distance = distances[nearest_idx]
                        
                        print(f"    最近骨架点: {nearest_point}, 距离: {min_distance:.2f} 像素")
                        
                        # 3. 使用Bresenham算法连接机器人到最近骨架点
                        from skimage.draw import line
                        rr, cc = line(robot_y, robot_x, nearest_point[0], nearest_point[1])
                        
                        # 只在自由空间内绘制连接线
                        for i in range(len(rr)):
                            if (0 <= rr[i] < skeleton_processed.shape[0] and 
                                0 <= cc[i] < skeleton_processed.shape[1]):
                                # 检查是否在自由空间内
                                if free_space_processed[rr[i], cc[i]]:
                                    skeleton_processed[rr[i], cc[i]] = True
                        
                        print(f"    ✓ 已连接机器人到骨架，连接长度: {len(rr)} 像素")
                else:
                    print(f"    机器人位置已在骨架上")
                
                # 确保机器人位置在骨架上
                skeleton_processed[robot_y, robot_x] = True
                # ============= 连接机器人到骨架结束 =============

                # ★ 在剪枝前保存机器人位置
                protected_robot_position = (robot_y, robot_x)

                # ============= 改进的短枝杈剪枝（保护机器人节点） =============
                from collections import deque
                
                # 参数设置
                max_prune_iterations = 25
                min_branch_length_threshold = 15
                
                print(f"    开始剪枝，最小分支长度阈值: {min_branch_length_threshold} 像素 ({min_branch_length_threshold * 5} cm)")
                print(f"    同时删除端点落在补偿区域的分支")
                print(f"    保护机器人位置节点: {protected_robot_position}")
                
                total_removed_points = 0
                compensation_branches_removed = 0
                
                for prune_iter in range(max_prune_iterations):
                    iteration_removed = 0
                    comp_removed_this_iter = 0
                    
                    # 1. 识别交叉点（度数 >= 3）
                    junction_points = set()
                    skeleton_coords = np.argwhere(skeleton_processed)
                    
                    for coord in skeleton_coords:
                        x, y = coord[0], coord[1]
                        if x > 0 and x < skeleton_processed.shape[0] - 1 and \
                           y > 0 and y < skeleton_processed.shape[1] - 1:
                            neighbor_count = np.sum(skeleton_processed[x-1:x+2, y-1:y+2]) - 1
                            if neighbor_count >= 3:
                                junction_points.add((x, y))
                    
                    # 2. 识别端点（度数 == 1）
                    endpoints_to_check = []
                    
                    for coord in skeleton_coords:
                        x, y = coord[0], coord[1]
                        if x >= 1 and x < skeleton_processed.shape[0] - 1 and \
                           y >= 1 and y < skeleton_processed.shape[1] - 1:
                            neighbor_count = np.sum(skeleton_processed[x-1:x+2, y-1:y+2]) - 1
                            if neighbor_count == 1:
                                endpoints_to_check.append((x, y))
                    
                    print(f"      迭代 {prune_iter + 1}: 发现 {len(endpoints_to_check)} 个端点, {len(junction_points)} 个交叉点")
                    
                    # 3. 对每个端点进行检查和删除
                    for start_point in endpoints_to_check:
                        # 保护机器人位置节点
                        if start_point == protected_robot_position:
                            continue
                        
                        if not skeleton_processed[start_point[0], start_point[1]]:
                            continue  # 已被删除
                        
                        # ★ 修改：BFS沿分支前进，只到最近的交叉点（邻居节点）
                        queue = deque([start_point])
                        visited = {start_point}
                        path = [start_point]
                        path_touches_robot = False
                        reached_junction = False  # 标记是否到达交叉点
                        
                        while queue:
                            x, y = queue.popleft()
                            
                            # 检查路径是否经过机器人位置
                            if (x, y) == protected_robot_position:
                                path_touches_robot = True
                            
                            # 查找未访问的骨架邻居
                            directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
                            neighbors = []
                            
                            for dx, dy in directions:
                                nx, ny = x + dx, y + dy
                                if (0 <= nx < skeleton_processed.shape[0] and 
                                    0 <= ny < skeleton_processed.shape[1] and 
                                    skeleton_processed[nx, ny] and 
                                    (nx, ny) not in visited):
                                    neighbors.append((nx, ny))
                            
                            # ★ 关键修改：如果当前点（非起点）是交叉点，停止
                            if (x, y) != start_point and (x, y) in junction_points:
                                reached_junction = True
                                # 交叉点也加入路径（因为它是这段路径的终点）
                                break
                            
                            # 如果只有一个邻居，继续
                            if len(neighbors) == 1:
                                next_point = neighbors[0]
                                visited.add(next_point)
                                queue.append(next_point)
                                path.append(next_point)
                            elif len(neighbors) > 1:
                                # 遇到分叉（不应该在端点到交叉点之间出现）
                                break
                            # len(neighbors) == 0: 已到尽头（孤立端点）
                        
                        # 如果路径经过机器人，不删除
                        if path_touches_robot:
                            continue
                        
                        # ★★★ 统计从端点到邻居交叉点的路径在补偿区域的比例 ★★★
                        branch_length = len(path)
                        points_in_compensation = sum(
                            1 for px, py in path 
                            if compensation_mask[px, py]
                        )
                        compensation_ratio_in_branch = points_in_compensation / branch_length if branch_length > 0 else 0
                        
                        # 判断是否需要删除分支
                        should_remove = False
                        remove_reason = ""
                        
                        # ★ 条件1：从端点到邻居节点的路径主要在补偿区域
                        compensation_threshold = 0.2 
                        if compensation_ratio_in_branch > compensation_threshold:
                            should_remove = True
                            remove_reason = f"到邻居节点路径补偿占比{compensation_ratio_in_branch:.1%}"
                            comp_removed_this_iter += 1
                        
                        # 条件2：端点本身在补偿区域
                        elif compensation_mask[start_point[0], start_point[1]]:
                            should_remove = True
                            remove_reason = "端点在补偿区域"
                            comp_removed_this_iter += 1
                        
                        # 条件3：到邻居节点的路径太短
                        elif branch_length < min_branch_length_threshold:
                            should_remove = True
                            remove_reason = f"到邻居节点距离{branch_length}过短"
                        
                        # 删除分支段
                        if should_remove:
                            for px, py in path:
                                if (px, py) != protected_robot_position and skeleton_processed[px, py]:
                                    skeleton_processed[px, py] = False
                                    iteration_removed += 1
                            
                            # 可选：详细日志
                            if False:  # 调试时设为True
                                junction_status = "到达交叉点" if reached_junction else "孤立端点"
                                print(f"        删除分支段: 长度{branch_length}, "
                                      f"{junction_status}, "
                                      f"补偿占比{compensation_ratio_in_branch:.1%}, "
                                      f"原因: {remove_reason}")
                    
                    total_removed_points += iteration_removed
                    compensation_branches_removed += comp_removed_this_iter
                    
                    if iteration_removed > 0:
                        print(f"      迭代 {prune_iter + 1}: 移除 {iteration_removed} 个点 "
                              f"(补偿区域分支: {comp_removed_this_iter} 个)")
                    
                    if iteration_removed == 0:
                        print(f"    剪枝完成，共 {prune_iter + 1} 次迭代，移除 {total_removed_points} 个点")
                        print(f"    其中补偿区域分支: {compensation_branches_removed} 个")
                        break
                
                # 4. 清理孤立点（但保护机器人）
                print(f"    开始清理孤立点...")
                for cleanup_round in range(5):
                    cleaned = 0
                    for coord in np.argwhere(skeleton_processed):
                        x, y = coord[0], coord[1]
                        
                        # ★ 保护机器人位置
                        if (x, y) == protected_robot_position:
                            continue
                        
                        if x > 0 and x < skeleton_processed.shape[0] - 1 and \
                           y > 0 and y < skeleton_processed.shape[1] - 1:
                            neighbor_count = np.sum(skeleton_processed[x-1:x+2, y-1:y+2]) - 1
                            if neighbor_count <= 1:
                                skeleton_processed[x, y] = False
                                cleaned += 1
                    
                    if cleaned > 0:
                        print(f"      清理轮次 {cleanup_round + 1}: 移除 {cleaned} 个孤立点")
                    else:
                        break
                
                # ★ 最终确保机器人位置仍在骨架上
                skeleton_processed[protected_robot_position[0], protected_robot_position[1]] = True
                
                print(f"    ✓ 剪枝和清理全部完成，机器人节点已保护")
                # ============= 消除短枝杈结束 =============
                # ============= 新增：保留与机器人连接的骨架分量 =============
                print(f"    检查骨架连通性...")
                
                # 1. 对剪枝后的骨架进行连通性分析
                skeleton_labeled, num_components = measure.label(
                    skeleton_processed, connectivity=2, return_num=True
                )
                
                print(f"    发现 {num_components} 个独立骨架分量")
                
                if num_components > 1:
                    # 2. 找到包含机器人位置的连通分量
                    robot_component = skeleton_labeled[protected_robot_position[0], 
                                                       protected_robot_position[1]]
                    
                    if robot_component > 0:
                        # 只保留机器人所在的分量
                        skeleton_processed = (skeleton_labeled == robot_component)
                        print(f"    ✓ 保留机器人所在的分量 {robot_component}，移除其他 {num_components - 1} 个分量")
                    else:
                        # 如果机器人不在任何分量上（理论上不应该发生）
                        print(f"    警告：机器人位置不在任何骨架分量上，寻找最近分量...")
                        
                        # 找到距离机器人最近的分量
                        min_distance = float('inf')
                        closest_component = 1
                        
                        for comp_id in range(1, num_components + 1):
                            component_mask = (skeleton_labeled == comp_id)
                            component_coords = np.argwhere(component_mask)
                            
                            if len(component_coords) > 0:
                                distances = np.sqrt(
                                    (component_coords[:, 0] - protected_robot_position[0])**2 + 
                                    (component_coords[:, 1] - protected_robot_position[1])**2
                                )
                                comp_min_dist = np.min(distances)
                                
                                if comp_min_dist < min_distance:
                                    min_distance = comp_min_dist
                                    closest_component = comp_id
                        
                        skeleton_processed = (skeleton_labeled == closest_component)
                        print(f"    ✓ 选择最近分量 {closest_component}（距离: {min_distance:.2f} 像素）")
                        
                        # 重新连接机器人到选择的骨架
                        skeleton_coords = np.argwhere(skeleton_processed)
                        if len(skeleton_coords) > 0:
                            distances = np.sqrt(
                                (skeleton_coords[:, 0] - protected_robot_position[0])**2 + 
                                (skeleton_coords[:, 1] - protected_robot_position[1])**2
                            )
                            nearest_idx = np.argmin(distances)
                            nearest_point = tuple(skeleton_coords[nearest_idx])
                            
                            # 连接
                            from skimage.draw import line
                            rr, cc = line(protected_robot_position[0], protected_robot_position[1], 
                                        nearest_point[0], nearest_point[1])
                            
                            for i in range(len(rr)):
                                if (0 <= rr[i] < skeleton_processed.shape[0] and 
                                    0 <= cc[i] < skeleton_processed.shape[1]):
                                    if free_space_processed[rr[i], cc[i]]:
                                        skeleton_processed[rr[i], cc[i]] = True
                            
                            print(f"    ✓ 重新连接机器人到骨架")
                    
                    # 确保机器人位置在骨架上
                    skeleton_processed[protected_robot_position[0], protected_robot_position[1]] = True
                    skeleton_processed = skeleton_processed.astype(bool)
                    
                    # 统计最终骨架
                    final_skeleton_points = np.sum(skeleton_processed)
                    print(f"    最终骨架点数: {final_skeleton_points}")
                else:
                    print(f"    ✓ 只有一个连通骨架分量，无需过滤")
                
                # ============= 保留与机器人连接的骨架分量结束 =============

                # 8. 提取处理后骨架的关节点
                joint_nodes_processed = extract_voronoi_nodes(skeleton_processed)
                
                # ★ 确保机器人位置在关节点列表中
                if protected_robot_position not in [tuple(node) for node in joint_nodes_processed]:
                    joint_nodes_processed.append(protected_robot_position)
                    print(f"    ✓ 机器人位置已添加到关节点列表")
                # 9. 构建拓扑图
                graph_processed, edges_processed = build_voronoi_graph(
                    skeleton_processed, joint_nodes_processed
                )

                # 10. 生成可视化
                visualize_maps_comparison(
                    obstacle_map, explored_map, free_space_original, skeleton_original,
                    explored_map_processed, free_space_processed, skeleton_processed,
                    joint_nodes_processed, graph_processed,
                    voronoi_original_path, voronoi_processed_path
                )
                # ★ 新增：保存Voronoi数据供paths.py使用
                voronoi_data_path = os.path.join(episode_folder_path, 'voronoi_data.npz')
                np.savez(
                    voronoi_data_path,
                    skeleton=skeleton_processed,
                    joint_nodes=joint_nodes_processed,
                    voronoi_graph=graph_processed,
                    allow_pickle=True
                )
                print(f"    ✓ Voronoi数据已保存: voronoi_data.npz")
                
                processed_count += 1
                
                if processed_count % 5 == 0:
                    print(f"\n已处理: {processed_count}/{min(15, total_episodes)}, "
                          f"跳过: {skipped_count}, "
                          f"当前: {episode_folder_name}")
                
            except Exception as e:
                print(f"\n错误 - {episode_folder_name}: {e}")
                if os.path.exists(episode_folder_path):
                    import shutil
                    shutil.rmtree(episode_folder_path)
                continue
    
    # 最终统计
    print("\n" + "="*70)
    print("处理完成!")
    print(f"  总episodes: {min(15, total_episodes)}")
    print(f"  成功处理: {processed_count}")
    print(f"  跳过(已存在): {skipped_count}")
    print(f"  失败: {min(15, total_episodes) - processed_count - skipped_count}")
    print(f"  输出目录: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='批量生成原始和净化后的Voronoi地图对比')
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='data/voronoi_processed',
        help='输出根目录'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("批量Voronoi地图生成系统（原始 vs 净化后对比）")
    print("="*70)
    
    batch_generate_processed_voronoi_maps(output_dir=args.output_dir)
    
    print("\n✓ 全部完成！")