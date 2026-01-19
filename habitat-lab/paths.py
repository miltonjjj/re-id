"""
基于预处理后的Voronoi图生成从机器人位置到所有端点的路径
输入：preprocess.py生成的骨架、节点和拓扑图
"""

import numpy as np
import cv2
from skimage.graph import route_through_array
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import deque


def load_preprocessed_voronoi(episode_path):
    """
    从preprocess.py生成的episode文件夹中加载预处理后的Voronoi数据
    
    参数:
        episode_path: str, episode文件夹路径（例如 'data/voronoi_processed/Adrian_ep4'）
    
    返回:
        skeleton: (H, W) bool数组，处理后的骨架
        joint_nodes: list of (x, y)，关节节点
        voronoi_graph: dict，拓扑图邻接表
        robot_position: (x, y)，机器人位置
    """
    import os
    
    # 读取保存的numpy文件
    # 注意：需要在preprocess.py中保存这些数据
    # 这里假设数据已保存为 voronoi_data.npz
    data_path = os.path.join(episode_path, 'voronoi_data.npz')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到预处理数据: {data_path}\n"
                               f"请确保在preprocess.py中保存了voronoi_data.npz")
    
    data = np.load(data_path, allow_pickle=True)
    
    skeleton = data['skeleton']
    joint_nodes = data['joint_nodes'].tolist()
    voronoi_graph = data['voronoi_graph'].item()  # dict类型
    
    # 机器人位置：底部中心
    vision_range = skeleton.shape[0]
    robot_position = (0, vision_range // 2)
    
    print(f"加载预处理数据成功:")
    print(f"  骨架点数: {np.sum(skeleton)}")
    print(f"  关节节点数: {len(joint_nodes)}")
    print(f"  拓扑图节点数: {len(voronoi_graph)}")
    print(f"  机器人位置: {robot_position}")
    
    return skeleton, joint_nodes, voronoi_graph, robot_position


def identify_leaf_nodes(joint_nodes, voronoi_graph):
    """
    识别拓扑图中的叶节点（端点，度数为1）
    
    参数:
        joint_nodes: list of (x, y) 所有关节节点
        voronoi_graph: dict 拓扑图邻接表
    
    返回:
        leaf_nodes: list of (x, y) 叶节点列表
    """
    leaf_nodes = []
    
    for node in joint_nodes:
        node_tuple = tuple(node)
        if node_tuple in voronoi_graph:
            # 度数为1的是叶节点
            if len(voronoi_graph[node_tuple]) == 1:
                leaf_nodes.append(node_tuple)
    
    return leaf_nodes


def find_path_in_graph(start_node, end_node, voronoi_graph):
    """
    在拓扑图中使用BFS找到从起点到终点经过的交叉节点序列
    
    参数:
        start_node: (x, y) 起点
        end_node: (x, y) 终点
        voronoi_graph: dict 拓扑图邻接表
    
    返回:
        node_path: list of (x, y) 节点序列，如果不可达返回None
    """
    start_tuple = tuple(start_node)
    end_tuple = tuple(end_node)
    
    if start_tuple not in voronoi_graph or end_tuple not in voronoi_graph:
        return None
    
    # BFS搜索
    queue = deque([(start_tuple, [start_tuple])])
    visited = {start_tuple}
    
    while queue:
        current, path = queue.popleft()
        
        # 到达终点
        if current == end_tuple:
            return path
        
        # 探索邻居
        if current in voronoi_graph:
            for neighbor in voronoi_graph[current]:
                neighbor_tuple = tuple(neighbor)
                if neighbor_tuple not in visited:
                    visited.add(neighbor_tuple)
                    queue.append((neighbor_tuple, path + [neighbor_tuple]))
    
    return None  # 不可达


def generate_path_on_skeleton(start_point, end_point, skeleton):
    """
    使用route_through_array在骨架上生成路径
    骨架点代价=1，非骨架点代价=1000
    
    参数:
        start_point: (x, y) 起点
        end_point: (x, y) 终点
        skeleton: (H, W) bool数组，Voronoi骨架
    
    返回:
        path_coords: list of (x, y) 路径坐标序列
        total_cost: float 总代价
    """
    # 创建代价地图：骨架=1，非骨架=1000
    cost_map = np.where(skeleton, 1, 1000).astype(np.float32)
    
    # 使用 route_through_array 找到最优路径
    indices, weight = route_through_array(
        cost_map,
        start_point,
        end_point,
        fully_connected=True
    )
    
    return indices, weight


def generate_complete_path(start_node, end_node, node_path, skeleton):
    """
    生成完整路径：沿着节点序列，在骨架上连接每对相邻节点
    
    参数:
        start_node: (x, y) 起点
        end_node: (x, y) 终点
        node_path: list of (x, y) 拓扑节点序列
        skeleton: (H, W) bool数组
    
    返回:
        complete_path: list of (x, y) 完整路径坐标
        total_cost: float 总代价
    """
    if node_path is None or len(node_path) < 2:
        return None, float('inf')
    
    complete_path = []
    total_cost = 0
    
    # 连接每对相邻节点
    for i in range(len(node_path) - 1):
        segment_start = node_path[i]
        segment_end = node_path[i + 1]
        
        # 在骨架上生成这一段路径
        segment_path, segment_cost = generate_path_on_skeleton(
            segment_start, segment_end, skeleton
        )
        
        # 添加到完整路径（避免重复节点）
        if i == 0:
            complete_path.extend(segment_path)
        else:
            complete_path.extend(segment_path[1:])  # 跳过起点，避免重复
        
        total_cost += segment_cost
    
    return complete_path, total_cost


def generate_all_paths_to_leaves(robot_position, leaf_nodes, voronoi_graph, skeleton, joint_nodes):
    """
    生成从机器人位置到所有叶节点的路径
    
    参数:
        robot_position: (x, y) 机器人位置
        leaf_nodes: list of (x, y) 所有叶节点
        voronoi_graph: dict 拓扑图
        skeleton: (H, W) Voronoi骨架
        joint_nodes: list of (x, y) 所有关节节点
    
    返回:
        all_paths: list of dict，每个dict包含路径信息
        start_node: (x, y) 起点节点
    """
    # 找到距离机器人最近的节点作为起点
    min_dist = float('inf')
    start_node = None
    
    for node in joint_nodes:
        dist = (node[0] - robot_position[0])**2 + (node[1] - robot_position[1])**2
        if dist < min_dist:
            min_dist = dist
            start_node = tuple(node)
    
    print(f"\n起点节点: {start_node} (距离机器人 {np.sqrt(min_dist):.2f} 像素)")
    print(f"叶节点总数: {len(leaf_nodes)}")
    
    all_paths = []
    
    for i, leaf_node in enumerate(leaf_nodes):
        print(f"\n处理叶节点 {i+1}/{len(leaf_nodes)}: {leaf_node}")
        
        # 1. 在拓扑图中找到节点序列
        node_path = find_path_in_graph(start_node, leaf_node, voronoi_graph)
        
        if node_path is None:
            print(f"  警告: 无法到达叶节点 {leaf_node}")
            continue
        
        print(f"  拓扑路径: {len(node_path)} 个节点")
        
        # 2. 在骨架上生成完整路径
        complete_path, total_cost = generate_complete_path(
            start_node, leaf_node, node_path, skeleton
        )
        
        if complete_path is None:
            print(f"  警告: 路径生成失败")
            continue
        
        print(f"  完整路径: {len(complete_path)} 个点, 总代价: {total_cost:.2f}")
        
        # 存储路径信息
        path_info = {
            'leaf_node': leaf_node,
            'node_path': node_path,
            'complete_path': complete_path,
            'total_cost': total_cost,
            'path_length': len(complete_path)
        }
        
        all_paths.append(path_info)
    
    print(f"\n成功生成 {len(all_paths)} 条路径")
    
    return all_paths, start_node


def visualize_all_paths(skeleton, all_paths, start_node, robot_position, 
                        joint_nodes, leaf_nodes, save_path=None):
    """
    可视化所有路径：左图显示骨架，右图显示彩色路径
    
    参数:
        skeleton: (H, W) Voronoi骨架
        all_paths: list of path_info dicts
        start_node: (x, y) 起点节点
        robot_position: (x, y) 机器人位置
        joint_nodes: list of all joint nodes
        leaf_nodes: list of leaf nodes
        save_path: 保存路径（可选）
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 生成颜色映射
    n_paths = len(all_paths)
    colors = plt.cm.hsv(np.linspace(0, 0.9, n_paths))
    
    # 左图：输入的骨架 + 节点
    axes[0].imshow(skeleton, cmap='gray', origin='lower')
    
    # 绘制所有关节节点
    if len(joint_nodes) > 0:
        nodes_array = np.array(joint_nodes)
        axes[0].scatter(nodes_array[:, 1], nodes_array[:, 0],
                       c='red', s=50, alpha=0.8, zorder=5,
                       label='Joint Nodes')
    
    # 绘制叶节点
    if len(leaf_nodes) > 0:
        leaves_array = np.array(leaf_nodes)
        axes[0].scatter(leaves_array[:, 1], leaves_array[:, 0],
                       c='orange', s=100, marker='*',
                       edgecolors='white', linewidths=1,
                       label='Leaf Nodes', zorder=8)
    
    # 绘制起点节点
    axes[0].scatter(start_node[1], start_node[0],
                   c='cyan', s=200, marker='o',
                   edgecolors='white', linewidths=2,
                   label='Start Node', zorder=9)
    
    # 绘制机器人位置
    axes[0].scatter(robot_position[1], robot_position[0],
                   c='lime', s=300, marker='s',
                   edgecolors='white', linewidths=2,
                   label='Robot', zorder=10)
    
    axes[0].set_title(f'Input Skeleton ({np.sum(skeleton)} points, {len(joint_nodes)} nodes)')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].axis('off')
    
    # 右图：黑背景 + 彩色路径
    black_bg = np.zeros_like(skeleton, dtype=np.uint8)
    axes[1].imshow(black_bg, cmap='gray', vmin=0, vmax=255, origin='lower')
    
    # 绘制所有路径（彩色线条）
    for i, path_info in enumerate(all_paths):
        path_coords = path_info['complete_path']
        if len(path_coords) > 0:
            path_array = np.array(path_coords)
            axes[1].plot(path_array[:, 1], path_array[:, 0],
                        color=colors[i], linewidth=3, alpha=0.9,
                        label=f'Path {i+1}')
    
    # 绘制叶节点
    if len(leaf_nodes) > 0:
        leaves_array = np.array(leaf_nodes)
        axes[1].scatter(leaves_array[:, 1], leaves_array[:, 0],
                       c='red', s=150, marker='*',
                       edgecolors='white', linewidths=2,
                       label='Leaf Nodes', zorder=8)
    
    # 绘制起点节点
    axes[1].scatter(start_node[1], start_node[0],
                   c='cyan', s=200, marker='o',
                   edgecolors='white', linewidths=2,
                   label='Start', zorder=9)
    
    # 绘制机器人
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
        print(f"\n路径可视化已保存到: {save_path}")
    
    plt.show()


def main(episode_path=None):
    """
    主函数：从预处理结果生成路径
    
    参数:
        episode_path: str, episode文件夹路径（可选）
    """
    
    print("="*70)
    print("Voronoi 多路径生成系统（基于预处理结果）")
    print("="*70)
    
    # 设置默认路径
    if episode_path is None:
        episode_path = "data/voronoi_processed/Adrian_ep10"
    
    print(f"\nEpisode路径: {episode_path}")
    
    # 1. 加载预处理的Voronoi数据
    print("\n步骤1: 加载预处理的Voronoi数据...")
    skeleton, joint_nodes, voronoi_graph, robot_position = load_preprocessed_voronoi(episode_path)
    
    # 2. 识别叶节点
    print("\n步骤2: 识别叶节点...")
    leaf_nodes = identify_leaf_nodes(joint_nodes, voronoi_graph)
    print(f"  叶节点数: {len(leaf_nodes)}")
    
    if len(leaf_nodes) == 0:
        print("  警告: 没有找到叶节点！")
        return None
    
    # 3. 生成所有路径
    print("\n步骤3: 生成从机器人到所有叶节点的路径...")
    all_paths, start_node = generate_all_paths_to_leaves(
        robot_position, leaf_nodes, voronoi_graph, skeleton, joint_nodes
    )
    
    # 4. 统计信息
    print("\n步骤4: 路径统计...")
    print(f"  成功生成路径数: {len(all_paths)}")
    
    for i, path_info in enumerate(all_paths):
        print(f"  路径 {i+1}:")
        print(f"    目标叶节点: {path_info['leaf_node']}")
        print(f"    拓扑节点数: {len(path_info['node_path'])}")
        print(f"    路径总点数: {path_info['path_length']}")
        print(f"    总代价: {path_info['total_cost']:.2f}")
    
    # 5. 保存结果
    print("\n步骤5: 保存结果...")
    
    import os
    output_path = os.path.join(episode_path, 'paths_results.npz')
    
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
    print(f"  结果已保存到: {output_path}")
    
    # 6. 可视化
    print("\n步骤6: 生成可视化...")
    viz_path = os.path.join(episode_path, 'paths_visualization.png')
    
    visualize_all_paths(
        skeleton,
        all_paths,
        start_node,
        robot_position,
        joint_nodes,
        leaf_nodes,
        save_path=viz_path
    )
    
    print("\n" + "="*70)
    print("✓ 完成！")
    print("="*70)
    
    return results


if __name__ == "__main__":
    import sys
    
    # 可以从命令行参数指定episode路径，例如python paths.py data/voronoi_processed/Adrian_ep8
    if len(sys.argv) > 1:
        episode_path = sys.argv[1]
    else:
        episode_path = "data/voronoi_processed/Adrian_ep10"
    
    results = main(episode_path)