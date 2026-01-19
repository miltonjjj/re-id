"""
完整的Voronoi路径规划和动作生成Pipeline
整合 preprocess.py, paths.py, actions.py 的功能
从yaml配置文件加载数据集，自动执行完整流程
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


# ============= Part 1: Preprocess Functions (from preprocess.py) =============

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
    """填充机器人视野内可见的空间（机器人到障碍物之间）"""
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
    """对探索地图进行预处理：填充机器人视野内的可见空间"""
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


def visualize_maps_comparison(obstacle_map, explored_map, free_space, skeleton,
                              explored_map_proc, free_space_proc, skeleton_proc,
                              joint_nodes_proc, graph_proc,
                              save_path_original, save_path_processed):
    """生成原始和处理后的对比可视化"""
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
    
    # 处理后图：2x3布局
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
    
    axes2[1, 1].imshow(skeleton_proc, cmap='gray', origin='lower')
    axes2[1, 1].plot(robot_x, robot_y, 'go', markersize=10, label='Robot', zorder=10)
    if len(joint_nodes_proc) > 0:
        nodes = np.array(joint_nodes_proc)
        axes2[1, 1].scatter(nodes[:, 1], nodes[:, 0], c='red', s=50, zorder=5, label='Joint Nodes')
    axes2[1, 1].set_title(f'Joint Nodes ({len(joint_nodes_proc)})')
    axes2[1, 1].axis('off')
    axes2[1, 1].legend(loc='upper right')
    
    axes2[1, 2].imshow(skeleton_proc, cmap='gray', origin='lower')
    axes2[1, 2].plot(robot_x, robot_y, 'go', markersize=10, label='Robot', zorder=10)
    if len(joint_nodes_proc) > 0:
        nodes = np.array(joint_nodes_proc)
        axes2[1, 2].scatter(nodes[:, 1], nodes[:, 0], c='red', s=50, zorder=5)
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


# ============= Part 2: Path Generation Functions (from paths.py) =============

def identify_leaf_nodes(joint_nodes, voronoi_graph):
    """识别拓扑图中的叶节点（端点，度数为1）"""
    leaf_nodes = []
    
    for node in joint_nodes:
        node_tuple = tuple(node)
        if node_tuple in voronoi_graph:
            if len(voronoi_graph[node_tuple]) == 1:
                leaf_nodes.append(node_tuple)
    
    return leaf_nodes


def find_path_in_graph(start_node, end_node, voronoi_graph):
    """在拓扑图中使用BFS找到从起点到终点经过的交叉节点序列"""
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
    """生成完整路径：沿着节点序列，在骨架上连接每对相邻节点"""
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
    
    print(f"    起点节点: {start_node} (距离机器人 {np.sqrt(min_dist):.2f} 像素)")
    print(f"    叶节点总数: {len(leaf_nodes)}")
    
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
    
    print(f"    成功生成 {len(all_paths)} 条路径")
    
    return all_paths, start_node


def visualize_all_paths(skeleton, all_paths, start_node, robot_position, 
                        joint_nodes, leaf_nodes, save_path=None):
    """可视化所有路径：左图显示骨架，右图显示彩色路径"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    n_paths = len(all_paths)
    colors = plt.cm.hsv(np.linspace(0, 0.9, n_paths))
    
    axes[0].imshow(skeleton, cmap='gray', origin='lower')
    
    if len(joint_nodes) > 0:
        nodes_array = np.array(joint_nodes)
        axes[0].scatter(nodes_array[:, 1], nodes_array[:, 0],
                       c='red', s=50, alpha=0.8, zorder=5,
                       label='Joint Nodes')
    
    if len(leaf_nodes) > 0:
        leaves_array = np.array(leaf_nodes)
        axes[0].scatter(leaves_array[:, 1], leaves_array[:, 0],
                       c='orange', s=100, marker='*',
                       edgecolors='white', linewidths=1,
                       label='Leaf Nodes', zorder=8)
    
    axes[0].scatter(start_node[1], start_node[0],
                   c='cyan', s=200, marker='o',
                   edgecolors='white', linewidths=2,
                   label='Start Node', zorder=9)
    
    axes[0].scatter(robot_position[1], robot_position[0],
                   c='lime', s=300, marker='s',
                   edgecolors='white', linewidths=2,
                   label='Robot', zorder=10)
    
    axes[0].set_title(f'Input Skeleton ({np.sum(skeleton)} points, {len(joint_nodes)} nodes)')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].axis('off')
    
    black_bg = np.zeros_like(skeleton, dtype=np.uint8)
    axes[1].imshow(black_bg, cmap='gray', vmin=0, vmax=255, origin='lower')
    
    for i, path_info in enumerate(all_paths):
        path_coords = path_info['complete_path']
        if len(path_coords) > 0:
            path_array = np.array(path_coords)
            axes[1].plot(path_array[:, 1], path_array[:, 0],
                        color=colors[i], linewidth=3, alpha=0.9,
                        label=f'Path {i+1}')
    
    if len(leaf_nodes) > 0:
        leaves_array = np.array(leaf_nodes)
        axes[1].scatter(leaves_array[:, 1], leaves_array[:, 0],
                       c='red', s=150, marker='*',
                       edgecolors='white', linewidths=2,
                       label='Leaf Nodes', zorder=8)
    
    axes[1].scatter(start_node[1], start_node[0],
                   c='cyan', s=200, marker='o',
                   edgecolors='white', linewidths=2,
                   label='Start', zorder=9)
    
    axes[1].scatter(robot_position[1], robot_position[0],
                   c='lime', s=300, marker='s',
                   edgecolors='white', linewidths=3,
                   label='Robot', zorder=10)
    
    axes[1].set_title(f'Generated Paths ({n_paths} paths)')
    axes[1].legend(loc='upper right', fontsize=9, ncol=2)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close(fig)

def visualize_paths_simple(skeleton, all_paths, start_node, robot_position, save_path=None):
    """
    简洁的路径可视化：所有路径使用相同颜色，标注起点和终点
    
    Args:
        skeleton: 骨架图
        all_paths: 所有路径信息列表
        start_node: 起始节点（机器人位置对应的拓扑节点）
        robot_position: 机器人实际位置
        save_path: 保存路径
    """
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # 黑色背景
    black_bg = np.zeros_like(skeleton, dtype=np.uint8)
    ax.imshow(black_bg, cmap='gray', vmin=0, vmax=255, origin='lower')
    
    # 绘制所有路径（统一白色，不区分）
    for path_info in all_paths:
        path_coords = path_info['complete_path']
        if len(path_coords) > 0:
            path_array = np.array(path_coords)
            ax.plot(path_array[:, 1], path_array[:, 0],
                   color='white', linewidth=2, alpha=0.8)
    
    # 标注所有路径的终点（红色）
    leaf_nodes = []
    for path_info in all_paths:
        leaf_node = path_info['leaf_node']
        leaf_nodes.append(leaf_node)
    
    if len(leaf_nodes) > 0:
        leaves_array = np.array(leaf_nodes)
        ax.scatter(leaves_array[:, 1], leaves_array[:, 0],
                  c='red', s=150, marker='o',
                  edgecolors='white', linewidths=2,
                  label='End Points', zorder=10)
    
    # 标注起点（绿色 - 机器人位置对应的拓扑节点）
    ax.scatter(start_node[1], start_node[0],
              c='lime', s=200, marker='o',
              edgecolors='white', linewidths=2,
              label='Start Node', zorder=11)
    
    # 可选：也标注机器人实际位置（绿色方块）
    # ax.scatter(robot_position[1], robot_position[0],
    #           c='lime', s=250, marker='s',
    #           edgecolors='white', linewidths=2,
    #           label='Robot', zorder=12)
    
    ax.set_title(f'All Paths ({len(all_paths)} paths)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
    
    plt.close(fig)

# ============= Part 3: Action Generation Functions (from actions.py) =============

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
                actions.append(3)  # Right
                action_details.append({
                    'action': 3,
                    'action_name': 'Right',
                    'position': current_pos,
                    'orientation': current_orientation
                })
                current_orientation -= turn_angle
            else:
                actions.append(2)  # Left
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
        
        actions.append(1)  # Forward
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


def visualize_action_sequence_all_paths(skeleton, all_paths_data, start_node, 
                                        robot_position, leaf_nodes, save_path=None):
    """可视化所有路径的动作序列"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    n_paths = len(all_paths_data)
    colors = plt.cm.hsv(np.linspace(0, 0.9, n_paths))
    
    axes[0].imshow(skeleton, cmap='gray', origin='lower')
    
    if len(leaf_nodes) > 0:
        leaves_array = np.array(leaf_nodes)
        axes[0].scatter(leaves_array[:, 1], leaves_array[:, 0],
                       c='red', s=100, marker='*',
                       edgecolors='white', linewidths=1,
                       label='Leaf Nodes', zorder=8)
    
    axes[0].scatter(start_node[1], start_node[0],
                   c='cyan', s=200, marker='o',
                   edgecolors='white', linewidths=2,
                   label='Start Node', zorder=9)
    
    axes[0].scatter(robot_position[1], robot_position[0],
                   c='lime', s=300, marker='s',
                   edgecolors='white', linewidths=2,
                   label='Robot', zorder=10)
    
    axes[0].set_title(f'Skeleton & Nodes')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].axis('off')
    
    black_bg = np.zeros_like(skeleton, dtype=np.uint8)
    axes[1].imshow(black_bg, cmap='gray', vmin=0, vmax=255, origin='lower')
    
    for i, path_data in enumerate(all_paths_data):
        path_coords = path_data['path_coords']
        if len(path_coords) > 0:
            path_array = np.array(path_coords)
            axes[1].plot(path_array[:, 1], path_array[:, 0],
                        color=colors[i], linewidth=3, alpha=0.9,
                        label=f'Path {i+1}')
    
    if len(leaf_nodes) > 0:
        leaves_array = np.array(leaf_nodes)
        axes[1].scatter(leaves_array[:, 1], leaves_array[:, 0],
                       c='red', s=150, marker='*',
                       edgecolors='white', linewidths=2, zorder=8)
    
    axes[1].scatter(start_node[1], start_node[0],
                   c='cyan', s=200, marker='o',
                   edgecolors='white', linewidths=2, zorder=9)
    
    axes[1].scatter(robot_position[1], robot_position[0],
                   c='lime', s=300, marker='s',
                   edgecolors='white', linewidths=3, zorder=10)
    
    axes[1].set_title(f'Paths ({n_paths} paths)')
    axes[1].legend(loc='upper right', fontsize=8, ncol=2)
    axes[1].axis('off')
    
    axes[2].imshow(black_bg, cmap='gray', vmin=0, vmax=255, origin='lower')
    
    for i, path_data in enumerate(all_paths_data):
        path_coords = path_data['path_coords']
        actions = path_data['actions']
        action_details = path_data['action_details']
        
        if len(path_coords) > 0:
            path_array = np.array(path_coords)
            axes[2].plot(path_array[:, 1], path_array[:, 0],
                        color=colors[i], linewidth=2, alpha=0.3)
        
        forward_pos = []
        left_pos = []
        right_pos = []
        
        for detail in action_details:
            if 'position' in detail:
                pos = detail['position']
                if detail['action'] == 1:
                    forward_pos.append(pos)
                elif detail['action'] == 2:
                    left_pos.append(pos)
                elif detail['action'] == 3:
                    right_pos.append(pos)
        
        if len(forward_pos) > 0:
            forward_array = np.array(forward_pos)
            axes[2].scatter(forward_array[:, 1], forward_array[:, 0],
                           c='lime', s=60, marker='^',
                           edgecolors='darkgreen', linewidths=1,
                           alpha=0.9, zorder=5)
        
        if len(left_pos) > 0:
            left_array = np.array(left_pos)
            axes[2].scatter(left_array[:, 1], left_array[:, 0],
                           c='dodgerblue', s=60, marker='<',
                           edgecolors='darkblue', linewidths=1,
                           alpha=0.9, zorder=5)
        
        if len(right_pos) > 0:
            right_array = np.array(right_pos)
            axes[2].scatter(right_array[:, 1], right_array[:, 0],
                           c='orange', s=60, marker='>',
                           edgecolors='darkorange', linewidths=1,
                           alpha=0.9, zorder=5)
    
    if len(leaf_nodes) > 0:
        leaves_array = np.array(leaf_nodes)
        axes[2].scatter(leaves_array[:, 1], leaves_array[:, 0],
                       c='red', s=150, marker='*',
                       edgecolors='white', linewidths=2, zorder=8)
    
    axes[2].scatter(start_node[1], start_node[0],
                   c='cyan', s=200, marker='o',
                   edgecolors='white', linewidths=2, zorder=9)
    
    axes[2].scatter(robot_position[1], robot_position[0],
                   c='lime', s=300, marker='s',
                   edgecolors='white', linewidths=3, zorder=10)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='lime', 
               markersize=10, label='Forward', markeredgecolor='darkgreen'),
        Line2D([0], [0], marker='<', color='w', markerfacecolor='dodgerblue', 
               markersize=10, label='Left', markeredgecolor='darkblue'),
        Line2D([0], [0], marker='>', color='w', markerfacecolor='orange', 
               markersize=10, label='Right', markeredgecolor='darkorange')
    ]
    axes[2].legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    axes[2].set_title('Action Sequence')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close(fig)

def calculate_path_metrics(path_coords, map_resolution=5.0):
    """
    计算路径的实际长度和曲折度
    
    Args:
        path_coords: 路径坐标列表 [(y1, x1), (y2, x2), ...]
        map_resolution: 地图分辨率 (cm/像素)
    
    Returns:
        dict: {
            'path_length_cm': 路径总长度(厘米),
            'straight_distance_cm': 起点到终点的直线距离(厘米),
            'tortuosity': 曲折度(路径长度/直线距离)
        }
    """
    if len(path_coords) < 2:
        return {
            'path_length_cm': 0.0,
            'straight_distance_cm': 0.0,
            'tortuosity': 1.0
        }
    
    path_array = np.array(path_coords)
    
    # 计算路径总长度（像素）
    path_length_pixels = 0.0
    for i in range(len(path_array) - 1):
        segment_length = np.linalg.norm(path_array[i+1] - path_array[i])
        path_length_pixels += segment_length
    
    # 计算起点到终点的直线距离（像素）
    straight_distance_pixels = np.linalg.norm(path_array[-1] - path_array[0])
    
    # 转换为厘米
    path_length_cm = path_length_pixels * map_resolution
    straight_distance_cm = straight_distance_pixels * map_resolution
    
    # 计算曲折度
    tortuosity = path_length_cm / straight_distance_cm if straight_distance_cm > 0 else 1.0
    
    return {
        'path_length_cm': path_length_cm,
        'straight_distance_cm': straight_distance_cm,
        'tortuosity': tortuosity
    }


def overlay_paths_on_rgb(rgb_image, all_paths, skeleton_shape, line_width=2):
    """
    将路径叠加在RGB图像上
    
    Args:
        rgb_image: RGB图像 (H, W, 3)
        all_paths: 路径信息列表，每个包含 'complete_path'
        skeleton_shape: Voronoi骨架的形状 (H_skeleton, W_skeleton)
        line_width: 路径线宽（默认2）
    
    Returns:
        overlayed_image: 叠加路径后的图像
    """
    import cv2
    
    # 复制图像避免修改原图
    result_image = rgb_image.copy()
    
    # 计算缩放比例
    rgb_h, rgb_w = rgb_image.shape[:2]
    skeleton_h, skeleton_w = skeleton_shape
    
    scale_y = rgb_h / skeleton_h
    scale_x = rgb_w / skeleton_w
    
    # 所有路径使用白色
    path_color_bgr = (255, 255, 255)  # 白色（BGR格式）
    
    # 绘制每条路径
    for i, path_info in enumerate(all_paths):
        path_coords = path_info['complete_path']
        
        if len(path_coords) < 2:
            continue
        
        # 转换路径坐标到RGB图像坐标系
        # skeleton坐标是(y, x)格式，其中y=0在上方，需要翻转y轴使起点在底部
        scaled_path = []
        for y, x in path_coords:
            # 缩放到RGB图像尺寸
            rgb_x = int(x * scale_x)
            # 翻转y轴：skeleton的y=0对应RGB的底部
            rgb_y = int((skeleton_h - 1 - y) * scale_y)
            # 限制在图像范围内
            rgb_x = np.clip(rgb_x, 0, rgb_w - 1)
            rgb_y = np.clip(rgb_y, 0, rgb_h - 1)
            scaled_path.append((rgb_x, rgb_y))
        
        # 绘制路径（白色）
        for j in range(len(scaled_path) - 1):
            cv2.line(result_image, scaled_path[j], scaled_path[j+1], 
                    path_color_bgr, thickness=line_width, lineType=cv2.LINE_AA)
        
        # 在终点旁边绘制标记：白色实心圆 + 红色边框 + 黑色数字
        endpoint = scaled_path[-1]
        path_number = i + 1
        circle_radius = 8  # 圆半径
        offset_distance = 15  # 圆心到终点的距离
        
        # 尝试多个位置（按优先级）：右上、右下、左上、左下
        possible_positions = [
            (offset_distance, -offset_distance),   # 右上
            (offset_distance, offset_distance),    # 右下
            (-offset_distance, -offset_distance),  # 左上
            (-offset_distance, offset_distance),   # 左下
        ]
        
        # 找到第一个不会超出边界的位置
        circle_center = None
        for offset_x, offset_y in possible_positions:
            candidate_center = (endpoint[0] + offset_x, endpoint[1] + offset_y)
            
            # 检查圆是否完全在图像范围内（考虑圆的半径）
            if (circle_radius <= candidate_center[0] < rgb_w - circle_radius and
                circle_radius <= candidate_center[1] < rgb_h - circle_radius):
                circle_center = candidate_center
                break
        
        # 如果所有位置都超出边界，使用终点本身（兜底方案）
        if circle_center is None:
            # 将圆心限制在安全范围内
            safe_x = np.clip(endpoint[0], circle_radius, rgb_w - circle_radius - 1)
            safe_y = np.clip(endpoint[1], circle_radius, rgb_h - circle_radius - 1)
            circle_center = (safe_x, safe_y)
        
        # 1. 绘制白色实心圆
        cv2.circle(result_image, circle_center, radius=circle_radius, 
                  color=(255, 255, 255), thickness=-1)  # 白色实心
        
        # 2. 绘制红色边框（BGR格式：红色是(0, 0, 255)）
        cv2.circle(result_image, circle_center, radius=circle_radius, 
                  color=(255, 0, 0), thickness=1)  # 红色边框
        
        # 3. 在圆心绘制黑色数字
        # 计算文字尺寸以居中
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        text = str(path_number)
        
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, font_thickness
        )
        
        # 文字位置：圆心 - 文字尺寸的一半（让文字居中）
        text_x = circle_center[0] - text_width // 2
        text_y = circle_center[1] + text_height // 2
        
        cv2.putText(result_image, text, 
                   (text_x, text_y),
                   font, 
                   font_scale,
                   (0, 0, 0),  # 黑色文字
                   font_thickness, 
                   cv2.LINE_AA)
    
    return result_image
# ============= Main Pipeline =============

def process_single_episode(observations, episode_folder_path, scene_name, episode_num):
    """处理单个episode的完整流程"""
    
    print(f"\n{'='*70}")
    print(f"处理 {scene_name}_ep{episode_num}")
    print(f"{'='*70}")
    
    os.makedirs(episode_folder_path, exist_ok=True)
    
    # ========== Step 1: Preprocess - 生成Voronoi图 ==========
    print("\n[步骤1] 生成Voronoi地图...")
    
    try:
        # 保存RGB和深度数据
        rgb_obs = observations["rgb"]
        depth_obs = observations["depth"]
        
        rgb_path = os.path.join(episode_folder_path, "rgb.png")
        depth_path = os.path.join(episode_folder_path, "depth.npy")
        depth_png_path = os.path.join(episode_folder_path, "depth.png")
        
        imageio.imwrite(rgb_path, rgb_obs)
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
        
        # 生成原始Voronoi图
        from voronoi_map import depth_to_voronoi
        results_original = depth_to_voronoi(depth_cm)
        obstacle_map = results_original['obstacle_map']
        explored_map = results_original['explored_map']
        free_space_original = results_original['free_space']
        skeleton_original = results_original['skeleton']
        
        # 净化探索地图
        explored_map_processed = process_exploration_map(
            explored_map, obstacle_map, robot_pos=None,
            fov_angle=90, fov_range=50
        )
        
        # 计算补偿区域
        compensation_mask = np.logical_and(
            explored_map_processed > 0.5,
            explored_map < 0.5
        ).astype(bool)
        
        # 生成处理后的自由空间
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
        
        # 连接机器人到骨架（简化版）
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
        
        # 剪枝（使用简化逻辑以节省时间）
        print("    执行骨架剪枝...")
        for prune_iter in range(15):  # 减少迭代次数
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
                
                queue = deque([start_point])
                visited = {start_point}
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
                remove_reason = ""
                # if compensation_ratio > 0.5:
                #     should_remove = True
                # elif branch_length < 10:
                #     should_remove = True
                # 计算指标
                endpoint_in_compensation = compensation_mask[start_point[0], start_point[1]]
                junction_in_compensation = False
                if end_junction is not None:
                    junction_in_compensation = compensation_mask[end_junction[0], end_junction[1]]

                # 根部检查（靠近交叉点的部分）
                root_length = min(10, branch_length)
                if end_junction is not None:
                    root_points = path[-root_length:]  # 最后N个点（靠近交叉点）
                else:
                    root_points = path[:root_length]   # 前N个点
                    
                root_compensation_ratio = sum(
                    1 for px, py in root_points if compensation_mask[px, py]
                ) / len(root_points)

                # 连续补偿区域检查
                max_consecutive = 0
                current = 0
                for px, py in path:
                    if compensation_mask[px, py]:
                        current += 1
                        max_consecutive = max(max_consecutive, current)
                    else:
                        current = 0

                # 第1层：端点直接在补偿区域
                if endpoint_in_compensation:
                    should_remove = True
                    remove_reason = "端点在补偿区域"

                # 第2层：根部大部分在补偿区域（说明从补偿区域长出来）
                elif root_compensation_ratio > 0.85:  # 70%阈值
                    should_remove = True
                    remove_reason = f"根部在补偿区域 ({root_compensation_ratio:.1%})"

                # 第3层：连续经过大片补偿区域
                elif max_consecutive >= 15:
                    should_remove = True
                    remove_reason = f"连续穿过补偿区域 ({max_consecutive}点)"

                # 第4层：整体补偿占比高
                elif compensation_ratio > 0.5:  # 保持原阈值
                    should_remove = True
                    remove_reason = f"补偿占比高 ({compensation_ratio:.1%})"

                # 第5层：短分支
                elif branch_length < 6:
                    should_remove = True
                    remove_reason = "分支过短"
                
                # 执行删除
                if should_remove:
                    for px, py in path:
                        if (px, py) != protected_robot_position and skeleton_processed[px, py]:
                            skeleton_processed[px, py] = False
                            iteration_removed += 1
            
            if iteration_removed == 0:
                break
        
        # 保留与机器人连接的骨架分量
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
        
        # 提取关节点和拓扑图
        joint_nodes_processed = extract_voronoi_nodes(skeleton_processed)
        
        if protected_robot_position not in [tuple(node) for node in joint_nodes_processed]:
            joint_nodes_processed.append(protected_robot_position)
        
        graph_processed, edges_processed = build_voronoi_graph(
            skeleton_processed, joint_nodes_processed
        )
        
        # 生成可视化
        voronoi_original_path = os.path.join(episode_folder_path, "voronoi_original.png")
        voronoi_processed_path = os.path.join(episode_folder_path, "voronoi_processed.png")
        
        visualize_maps_comparison(
            obstacle_map, explored_map, free_space_original, skeleton_original,
            explored_map_processed, free_space_processed, skeleton_processed,
            joint_nodes_processed, graph_processed,
            voronoi_original_path, voronoi_processed_path
        )
        
        # 保存Voronoi数据
        voronoi_data_path = os.path.join(episode_folder_path, 'voronoi_data.npz')
        np.savez(
            voronoi_data_path,
            skeleton=skeleton_processed,
            joint_nodes=joint_nodes_processed,
            voronoi_graph=graph_processed,
            allow_pickle=True
        )
        
        print(f"    ✓ Voronoi地图生成完成")
        
    except Exception as e:
        print(f"    ✗ Voronoi生成失败: {e}")
        return False
    
    # ========== Step 2: 生成路径 ==========
    print("\n[步骤2] 生成路径...")
    
    try:
        skeleton = skeleton_processed
        joint_nodes = joint_nodes_processed
        voronoi_graph = graph_processed
        
        # 识别叶节点
        leaf_nodes = identify_leaf_nodes(joint_nodes, voronoi_graph)
        print(f"    识别到 {len(leaf_nodes)} 个叶节点")
        
        if len(leaf_nodes) == 0:
            print(f"    ✗ 没有叶节点，跳过路径生成")
            return False
        
        # 生成所有路径
        all_paths, start_node = generate_all_paths_to_leaves(
            robot_position, leaf_nodes, voronoi_graph, skeleton, joint_nodes
        )
        
        if len(all_paths) == 0:
            print(f"    ✗ 未能生成任何路径")
            return False
        
        # 保存路径结果
        output_path = os.path.join(episode_folder_path, 'paths_results.npz')
        results = {
            'robot_position': robot_position,
            'start_node': start_node,
            'leaf_nodes': leaf_nodes,
            'all_paths': all_paths,
            'skeleton': skeleton,
            'joint_nodes': joint_nodes,
            'voronoi_graph': voronoi_graph
        }
        np.savez(output_path, **results, allow_pickle=True)
        
        # 可视化路径
        viz_path = os.path.join(episode_folder_path, 'paths_visualization.png')
        visualize_all_paths(
            skeleton, all_paths, start_node, robot_position,
            joint_nodes, leaf_nodes, save_path=viz_path
        )

        # 新增：简洁的路径可视化
        paths_simple_path = os.path.join(episode_folder_path, 'paths.png')
        visualize_paths_simple(
            skeleton, all_paths, start_node, robot_position, 
            save_path=paths_simple_path
        )
        print(f"    ✓ 路径生成完成，共 {len(all_paths)} 条路径")
        
        # ========== 新增：路径分析和可视化 ==========
        print("\n[路径分析] 计算路径指标...")
        
        map_resolution = 5.0  # cm/像素
        
        # 准备JSON数据
        paths_metrics = {}
        
        for i, path_info in enumerate(all_paths, 1):
            complete_path = path_info['complete_path']
            
            # 计算路径指标
            metrics = calculate_path_metrics(complete_path, map_resolution)
            
            # 保存到path_info中
            path_info['metrics'] = metrics
            
            # 添加到JSON数据（以米为单位）
            paths_metrics[f"path_{i}"] = {
                "path_number": i,
                "path_length_m": round(metrics['path_length_cm'] / 100, 2),
                "tortuosity": round(metrics['tortuosity'], 3)
            }
        
        # 保存路径指标为JSON文件
        metrics_json_path = os.path.join(episode_folder_path, 'path_metrics.json')
        with open(metrics_json_path, 'w', encoding='utf-8') as f:
            json.dump(paths_metrics, f, indent=2, ensure_ascii=False)
        print(f"    ✓ 已保存路径指标: path_metrics.json")
        
        # 将路径叠加到RGB图像上
        print("\n[可视化] 生成路径叠加RGB图像...")
        paths_on_rgb = overlay_paths_on_rgb(
            rgb_obs, 
            all_paths, 
            skeleton.shape,
            line_width=2
        )
        
        # 保存叠加图像
        paths_rgb_path = os.path.join(episode_folder_path, 'paths_on_rgb.png')
        imageio.imwrite(paths_rgb_path, paths_on_rgb)
        print(f"    ✓ 已保存路径叠加图像: paths_on_rgb.png")
        
        # 更新 paths_results.npz 以包含指标
        output_path = os.path.join(episode_folder_path, 'paths_results.npz')
        results = {
            'robot_position': robot_position,
            'start_node': start_node,
            'leaf_nodes': leaf_nodes,
            'all_paths': all_paths,  # 现在包含metrics
            'skeleton': skeleton,
            'joint_nodes': joint_nodes,
            'voronoi_graph': voronoi_graph
        }
        np.savez(output_path, **results, allow_pickle=True)
    except Exception as e:
        print(f"    ✗ 路径生成失败: {e}")
        return False
    
    # ========== Step 3: 生成动作序列 ==========
    print("\n[步骤3] 生成动作序列...")
    
    try:
        # 参数设置
        map_resolution = 5  # cm/像素
        forward_step = 0.25  # 米
        turn_angle = 30  # 度
        initial_orientation = 90.0  # 度
        
        all_paths_with_actions = []
        
        for i, path_info in enumerate(all_paths):
            leaf_node = path_info['leaf_node']
            complete_path = path_info['complete_path']
            
            actions, action_details = path_to_actions(
                complete_path,
                initial_orientation=initial_orientation,
                turn_angle=turn_angle,
                map_resolution=map_resolution,
                forward_step_meters=forward_step
            )
            
            path_with_actions = {
                'leaf_node': leaf_node,
                'path_coords': complete_path,
                'path_length': len(complete_path),
                'actions': actions,
                'action_details': action_details
            }
            
            all_paths_with_actions.append(path_with_actions)
        
        # 保存动作序列
        action_results = {}
        for i, path_data in enumerate(all_paths_with_actions):
            path_key = f'path_{i+1}'
            forward_count = path_data['actions'].count(1)
            total_distance = forward_count * 0.25
            
            # 转换numpy类型为Python原生类型
            leaf_node = path_data['leaf_node']
            if isinstance(leaf_node, (tuple, list)):
                leaf_node = [int(x) if hasattr(x, 'item') else x for x in leaf_node]
            
            # 转换actions列表中的numpy类型
            actions_list = [int(a) if hasattr(a, 'item') else a for a in path_data['actions']]
            
            action_results[path_key] = {
                'leaf_node': leaf_node,
                'actions': actions_list,
                'action_count': {
                    'forward': int(path_data['actions'].count(1)),
                    'left': int(path_data['actions'].count(2)),
                    'right': int(path_data['actions'].count(3)),
                    'stop': int(path_data['actions'].count(0))
                },
                'total_distance_meters': float(total_distance),
                'path_length': int(path_data['path_length']) if hasattr(path_data['path_length'], 'item') else path_data['path_length']
            }
        
        json_path = os.path.join(episode_folder_path, 'actions_sequence.json')
        with open(json_path, 'w') as f:
            json.dump(action_results, f, indent=2)
        
        npz_path = os.path.join(episode_folder_path, 'actions_sequence.npz')
        np.savez(npz_path, **action_results, allow_pickle=True)
        
        # 可视化动作序列
        viz_path = os.path.join(episode_folder_path, 'actions_visualization.png')
        visualize_action_sequence_all_paths(
            skeleton, all_paths_with_actions, start_node,
            robot_position, leaf_nodes, save_path=viz_path
        )
        
        print(f"    ✓ 动作序列生成完成，共 {len(all_paths_with_actions)} 条路径")
        
    except Exception as e:
        print(f"    ✗ 动作序列生成失败: {e}")
        return False
    
    print(f"\n✓ Episode {scene_name}_ep{episode_num} 处理完成！")
    return True

def main(config_path="benchmark/nav/pointnav/pointnav_hm3d.yaml", 
         output_dir='data/voronoi_processed_hm3d',
         scene_names=['Adrian', 'Albertville', 'Anaheim'],  # 新增：指定要加载的场景
         max_episodes_per_scene=20):
    """
    主函数：完整的pipeline
    
    参数:
        config_path: habitat配置文件路径
        output_dir: 输出根目录
        scene_names: 要处理的场景名称列表
        max_episodes_per_scene: 每个场景最多处理的episode数量
    """
    
    print("="*70)
    print("完整Voronoi路径规划和动作生成Pipeline")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n目标场景: {scene_names}")
    print(f"每个场景最多处理: {max_episodes_per_scene} episodes")
    print(f"输出目录: {output_dir}")
    print("="*70)
    
    processed_count = 0
    failed_count = 0
    
    # 逐个场景处理
    for scene_idx, scene_name in enumerate(scene_names, 1):
        print(f"\n{'='*70}")
        print(f"处理场景 {scene_idx}/{len(scene_names)}: {scene_name}")
        print(f"{'='*70}")
        
        # 动态修改配置以加载特定场景
        config = habitat.get_config(config_path=config_path)
        
        # 修改数据路径为特定场景
        scene_data_path = f"data/datasets/pointnav/hm3d/v1/train/content/{scene_name}.json.gz"
        config = habitat.get_config(
            config_path=config_path,
            overrides=[f"habitat.dataset.data_path={scene_data_path}"]
        )
        
        print(f"加载数据集: {scene_data_path}")
        
        try:
            dataset = habitat.make_dataset(
                id_dataset=config.habitat.dataset.type, 
                config=config.habitat.dataset
            )
        except Exception as e:
            print(f"✗ 加载场景 {scene_name} 失败: {e}")
            continue
        
        total_episodes = len(dataset.episodes)
        print(f"  场景 {scene_name} 包含 {total_episodes} 个episodes")
        print(f"  将处理: {min(max_episodes_per_scene, total_episodes)} 个episodes")
        
        # 创建仿真环境
        print(f"\n正在创建 {scene_name} 的仿真环境...")
        with habitat.Env(config=config, dataset=dataset) as env:
            print("✓ 环境创建成功！开始处理episodes...\n")
            
            scene_processed = 0
            scene_failed = 0
            
            for ep_idx in tqdm.tqdm(range(min(max_episodes_per_scene, total_episodes)), 
                                    desc=f"{scene_name} episodes"):
                # 重置环境
                observations = env.reset()
                
                episode_num = ep_idx + 1
                
                # 创建episode文件夹
                episode_folder_name = f"{scene_name}_ep{episode_num}"
                episode_folder_path = os.path.join(output_dir, episode_folder_name)
                
                # 检查是否已存在完整结果
                required_files = [
                    'depth.npy', 'depth.png', 'rgb.png',
                    'paths.png',
                    'voronoi_original.png', 'voronoi_processed.png',
                    'paths_results.npz', 'paths_visualization.png',
                    'actions_sequence.json', 'actions_sequence.npz',
                    'actions_visualization.png'
                ]
                
                if os.path.exists(episode_folder_path):
                    if all(os.path.exists(os.path.join(episode_folder_path, f)) 
                           for f in required_files):
                        print(f"\n跳过 {episode_folder_name} (已存在完整结果)")
                        scene_processed += 1
                        continue
                
                # 处理episode
                try:
                    success = process_single_episode(
                        observations, episode_folder_path, 
                        scene_name, episode_num
                    )
                    
                    if success:
                        scene_processed += 1
                    else:
                        scene_failed += 1
                        if os.path.exists(episode_folder_path):
                            import shutil
                            shutil.rmtree(episode_folder_path)
                            print(f"    ✗ 已删除失败的episode文件夹")
                        
                except Exception as e:
                    print(f"\n错误 - {episode_folder_name}: {e}")
                    scene_failed += 1
                    if os.path.exists(episode_folder_path):
                        import shutil
                        shutil.rmtree(episode_folder_path)
                        print(f"    ✗ 已删除异常的episode文件夹")
                    continue
            
            print(f"\n场景 {scene_name} 处理完成:")
            print(f"  成功: {scene_processed}")
            print(f"  失败: {scene_failed}")
            
            processed_count += scene_processed
            failed_count += scene_failed
    
    # 最终统计
    print("\n" + "="*70)
    print("所有场景处理完成!")
    print(f"  处理的场景数: {len(scene_names)}")
    for scene in scene_names:
        print(f"    - {scene}")
    print(f"  总成功处理: {processed_count}")
    print(f"  总失败: {failed_count}")
    print(f"  输出目录: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    # 配置参数
    CONFIG_PATH = "benchmark/nav/pointnav/pointnav_hm3d.yaml"
    OUTPUT_DIR = "data/voronoi_processed_hm3d"
    
    # 指定要处理的场景（从content目录中选择）
    #SCENE_NAMES = ['Adrian']
    SCENE_NAMES = ['1EiJpeRNEs1']
    
    # 每个场景最多处理的episodes数
    MAX_EPISODES_PER_SCENE = 10
    
    # 运行完整pipeline
    main(
        config_path=CONFIG_PATH,
        output_dir=OUTPUT_DIR,
        scene_names=SCENE_NAMES,
        max_episodes_per_scene=MAX_EPISODES_PER_SCENE
    )