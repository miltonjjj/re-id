"""
完整的数据集生成脚本 - 生成 NoMaD 训练格式数据集
功能：
1. Voronoi路径规划和动作生成
2. VLM质量评估（只保留VLM认为可以保留的episode）
3. 执行动作序列并收集轨迹数据
4. 生成符合 go_stanford_dataset 格式的数据集
   - 文件夹命名: {scene_name}_{episode_num}
   - 图像文件: 0.jpg, 1.jpg, 2.jpg, ...
   - traj_data.pkl: 包含 position 和 yaw 数组
5. 输出统计指标：
   - 总episode数量、保留/删除/无路径数量
   - Token消耗统计
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
import base64
import pickle
import quaternion  # pip install numpy-quaternion

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


# ============= Habitat 轨迹收集工具函数 =============

def quaternion_to_yaw(rotation_quat):
    """
    将 Habitat 四元数转换为 yaw 角度（弧度）
    """
    w = rotation_quat.w
    x = rotation_quat.x
    y = rotation_quat.y
    z = rotation_quat.z
    
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return yaw


def get_agent_state_data(env):
    """
    从 Habitat 环境中获取 agent 的位置和朝向
    """
    agent_state = env.sim.get_agent_state()
    position = np.array(agent_state.position)  # [x, y, z]
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


def overlay_paths_on_rgb(rgb_image, all_paths, skeleton_shape, line_width=2):
    """将路径叠加在RGB图像上"""
    result_image = rgb_image.copy()
    
    rgb_h, rgb_w = rgb_image.shape[:2]
    skeleton_h, skeleton_w = skeleton_shape
    
    scale_y = rgb_h / skeleton_h
    scale_x = rgb_w / skeleton_w
    
    path_color_bgr = (255, 255, 255)
    
    for i, path_info in enumerate(all_paths):
        path_coords = path_info['complete_path']
        
        if len(path_coords) < 2:
            continue
        
        scaled_path = []
        for y, x in path_coords:
            rgb_x = int(x * scale_x)
            rgb_y = int((skeleton_h - 1 - y) * scale_y)
            rgb_x = np.clip(rgb_x, 0, rgb_w - 1)
            rgb_y = np.clip(rgb_y, 0, rgb_h - 1)
            scaled_path.append((rgb_x, rgb_y))
        
        for j in range(len(scaled_path) - 1):
            cv2.line(result_image, scaled_path[j], scaled_path[j+1], 
                    path_color_bgr, thickness=line_width, lineType=cv2.LINE_AA)
        
        endpoint = scaled_path[-1]
        path_number = i + 1
        circle_radius = 8
        offset_distance = 15
        
        possible_positions = [
            (offset_distance, -offset_distance),
            (offset_distance, offset_distance),
            (-offset_distance, -offset_distance),
            (-offset_distance, offset_distance),
        ]
        
        circle_center = None
        for offset_x, offset_y in possible_positions:
            candidate_center = (endpoint[0] + offset_x, endpoint[1] + offset_y)
            
            if (circle_radius <= candidate_center[0] < rgb_w - circle_radius and
                circle_radius <= candidate_center[1] < rgb_h - circle_radius):
                circle_center = candidate_center
                break
        
        if circle_center is None:
            safe_x = np.clip(endpoint[0], circle_radius, rgb_w - circle_radius - 1)
            safe_y = np.clip(endpoint[1], circle_radius, rgb_h - circle_radius - 1)
            circle_center = (safe_x, safe_y)
        
        cv2.circle(result_image, circle_center, radius=circle_radius, 
                  color=(255, 255, 255), thickness=-1)
        cv2.circle(result_image, circle_center, radius=circle_radius, 
                  color=(0, 0, 255), thickness=1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        text = str(path_number)
        
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, font_thickness
        )
        
        text_x = circle_center[0] - text_width // 2
        text_y = circle_center[1] + text_height // 2
        
        cv2.putText(result_image, text, 
                   (text_x, text_y),
                   font, 
                   font_scale,
                   (0, 0, 0),
                   font_thickness, 
                   cv2.LINE_AA)
    
    return result_image


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


# ============= Part 4: VLM Evaluation Functions =============

def evaluate_paths_with_vlm_efficient(paths_rgb_path, path_metrics):
    """
    使用VLM高效评估paths_on_rgb图像质量
    
    Returns:
        dict: {
            'decision': 'keep' or 'delete',
            'score': float (0-10),
            'explanation': str,
            'token_usage': dict,
            'sub_scores': dict
        }
    """
    try:
        from openai import OpenAI
        import re
        
        # API配置
        API_KEY = "sk-zk24390385d11aba6430c32a49e645dc3ee695b79880a6ab"
        BASE_URL = "https://api.zhizengzeng.com/v1/"
        MODEL = "gpt-4o-mini"
        
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        
        # System Prompt
        system_prompt = (
            "You are an embodied wheeled robot assistant, with an RGB image sensor. "
            "You observe the image with overlaid navigation paths and evaluate path quality for exploration. "
            "You prefer smooth, flat surfaces and must avoid stairs, steps, and uneven terrain. "
        )

        # 加载图像
        with open(paths_rgb_path, 'rb') as f:
            img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')

        # Score Prompt
        path_data_lines = []
        for path_key, metrics in sorted(path_metrics.items()):
            num = metrics['path_number']
            length = metrics['path_length_m']
            tort = metrics['tortuosity']
            path_data_lines.append(f"Path {num}: {length}m, tortuosity {tort}")
        path_data_str = "\n".join(path_data_lines)

        score_prompt = f"""You are a professional auditor for robot navigation data. Evaluate the paths based on THREE specific dimensions. 

Path Data:
{path_data_str}

Scoring Criteria (0-10 each):

1. Feasibility (Weight: 0.4):
   - Check for collisions. Do paths hit walls, furniture, or obstacles?
   - For a wheeled robot, paths MUST be on flat ground. Avoid stairs or steep gaps.
   - If ANY path clearly collides with an object, this score must be < 4.0.

2. Efficiency (Weight: 0.3):
   - Analyze tortuosity. If it's high (> 1.2), is there a visual justification? 
   - Justified: Moving around a table or through a narrow door.
   - Unjustified: Shaking or unnecessary zigzagging on open ground.
   - High tortuosity without a clear terrain reason should result in a low score.

3. Exploration (Weight: 0.3):
   - Does the path lead to 'frontier' areas (unexplored corridors, open rooms)?
   - Deduct points if paths lead into dead ends, corners, or already well-observed areas.

Output Requirement:
Perform a step-by-step analysis for each path first, then provide the three sub-scores. Keep the analysis concise.

Output Format (Strict JSON):
{{
  "analysis": {{
    "feasibility": "<reasoning about collisions/terrain>",
    "efficiency": "<reasoning about tortuosity vs environment>",
    "exploration": "<reasoning about exploration value>"
  }},
  "sub_scores": {{
    "feasibility_score": <float 0-10>,
    "efficiency_score": <float 0-10>,
    "exploration_score": <float 0-10>
  }}
}}"""

        message_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": score_prompt},
        ]

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message_content},
            ],
            max_tokens=500,
            temperature=0.4,
        )

        answer = response.choices[0].message.content
        
        # 获取token使用量
        token_usage = {}
        if hasattr(response, 'usage') and response.usage:
            token_usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
        
        try:
            # 提取 JSON 部分
            json_str_match = re.search(r'\{.*\}', answer, re.DOTALL)
            if not json_str_match:
                raise ValueError("未找到JSON结构")
            
            json_str = json_str_match.group()
            json_str = re.sub(r',\s*([\]\}])', r'\1', json_str)
            
            parsed_data = json.loads(json_str)
            
            sub = parsed_data.get('sub_scores', {})
            f_score = float(sub.get('feasibility_score', 0.0))
            ef_score = float(sub.get('efficiency_score', 0.0))
            ex_score = float(sub.get('exploration_score', 0.0))
            
            total_score = round((0.4 * f_score) + (0.3 * ef_score) + (0.3 * ex_score), 2)
            
            analysis = parsed_data.get('analysis', {})
            explanation = (f"F:{f_score} {analysis.get('feasibility','')} | "
                           f"Ef:{ef_score} {analysis.get('efficiency','')} | "
                           f"Ex:{ex_score} {analysis.get('exploration','')}")
            
            result = {
                'sub_scores': sub,
                'score': total_score,
                'explanation': explanation,
                'token_usage': token_usage,
                'decision': 'keep' if total_score >= 6 else 'delete'
            }
            return result

        except Exception as e:
            print(f"    ✗ 解析VLM回复失败: {e}")
            return {
                'sub_scores': {},
                'score': 0.0,
                'explanation': f"Parse Error: {answer}",
                'token_usage': token_usage,
                'decision': 'delete'
            }
        
    except Exception as e:
        print(f"    ✗ VLM评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_vlm_api():
    """测试 VLM API 是否可用"""
    print("\n" + "="*70)
    print("测试 VLM API 连接...")
    print("="*70)
    
    try:
        from openai import OpenAI
        
        API_KEY = "sk-zk24390385d11aba6430c32a49e645dc3ee695b79880a6ab"
        BASE_URL = "https://api.zhizengzeng.com/v1/"
        MODEL = "gpt-4o-mini"
        
        print(f"API: {BASE_URL}")
        print(f"模型: {MODEL}")
        print(f"正在测试...\n")
        
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": "Hello, are you working?"}
            ],
            max_tokens=10,
            temperature=0.2
        )
        
        if hasattr(response, 'error') and response.error:
            error_info = response.error
            error_msg = error_info.get('message', 'Unknown') if isinstance(error_info, dict) else str(error_info)
            print(f"✗ VLM API 调用失败: {error_msg}")
            print("="*70 + "\n")
            return False
        
        if response and hasattr(response, 'choices') and response.choices:
            if response.choices[0] and hasattr(response.choices[0], 'message'):
                answer = response.choices[0].message.content
                
                if answer and len(answer.strip()) > 0:
                    print(f"✅ VLM API 可用")
                    print(f"测试响应: {answer[:50]}")
                    print("="*70 + "\n")
                    return True
        
        print(f"✗ VLM API 响应异常")
        print("="*70 + "\n")
        return False
        
    except Exception as e:
        print(f"✗ VLM API 调用失败")
        print(f"错误: {str(e)[:200]}")
        print("="*70 + "\n")
        return False


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
        print(f"    ✗ Voronoi地图生成失败: {e}")
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
        print(f"    ✗ 路径生成异常: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============= Main Pipeline =============

def process_single_episode(observations, episode_folder_path, scene_name, episode_num, env, rgb_obs):
    """
    处理单个episode的完整流程
    
    Returns:
        tuple: (success: bool, vlm_result: dict or None)
        - success=True, vlm_result: 成功保留
        - success=False, vlm_result=None: 路径生成失败（no_path）
        - success=False, vlm_result有值: VLM拒绝（deleted）
    """
    
    print(f"\n{'='*70}")
    print(f"处理 {scene_name}_ep{episode_num}")
    print(f"{'='*70}")
    
    # ========== Step 1: 生成 Voronoi 地图 ==========
    print("\n[步骤1] 生成Voronoi地图...")
    
    voronoi_result = generate_voronoi_map_from_obs(observations)
    if voronoi_result is None:
        print(f"    ✗ Voronoi生成失败")
        return False, None
    
    print(f"    ✓ Voronoi地图生成完成")
    
    # ========== Step 2: 生成路径 ==========
    print("\n[步骤2] 生成路径...")
    
    paths_result = generate_paths_from_voronoi(voronoi_result)
    if paths_result is None:
        print(f"    ✗ 路径生成失败（无叶节点或无法生成路径）")
        return False, None
    
    skeleton = voronoi_result['skeleton']
    all_paths = paths_result['paths']
    start_node = paths_result['start_node']
    robot_position = voronoi_result['robot_pos']
    
    print(f"    ✓ 路径生成完成，共 {len(all_paths)} 条路径")
    
    # ========== Step 3: 计算路径指标并叠加到RGB ==========
    print("\n[步骤3] 计算路径指标...")
    
    map_resolution = 5.0
    paths_metrics = {}
    
    for i, path_info in enumerate(all_paths, 1):
        complete_path = path_info['complete_path']
        metrics = calculate_path_metrics(complete_path, map_resolution)
        path_info['metrics'] = metrics
        
        paths_metrics[f"path_{i}"] = {
            "path_number": i,
            "path_length_m": round(metrics['path_length_cm'] / 100, 2),
            "tortuosity": round(metrics['tortuosity'], 3)
        }
    
    # 生成路径叠加图
    paths_on_rgb = overlay_paths_on_rgb(
        rgb_obs, 
        all_paths, 
        skeleton.shape,
        line_width=2
    )
    
    # 保存临时图像用于VLM评估
    import tempfile
    temp_dir = tempfile.mkdtemp()
    paths_rgb_path = os.path.join(temp_dir, 'paths_on_rgb.png')
    imageio.imwrite(paths_rgb_path, paths_on_rgb)
    
    print(f"    ✓ 路径指标计算完成")
    
    # ========== Step 4: VLM评估 ==========
    print("\n[步骤4] VLM评估路径质量...")
    
    vlm_result = evaluate_paths_with_vlm_efficient(paths_rgb_path, paths_metrics)
    
    # 清理临时文件
    import shutil
    shutil.rmtree(temp_dir)
    
    if vlm_result is None:
        print(f"    ⚠️  VLM评估失败，跳过此episode")
        return False, None
    
    print(f"    VLM评分: {vlm_result['score']:.1f}/10")
    print(f"    决策: {vlm_result['decision'].upper()}")
    print(f"    原因: {vlm_result['explanation'][:100]}...")
    
    if vlm_result['decision'] == 'delete':
        print(f"    ⚠️  VLM建议删除此episode")
        return False, vlm_result
    
    # ========== Step 5: 生成动作序列 ==========
    print("\n[步骤5] 生成动作序列...")
    
    selected_path_info = all_paths[0]
    complete_path = selected_path_info['complete_path']
    
    actions, action_details = path_to_actions(
        complete_path,
        initial_orientation=90.0,
        turn_angle=30.0,
        map_resolution=5.0,
        forward_step_meters=0.25
    )
    
    print(f"    ✓ 动作序列生成完成")
    print(f"      - 总动作数: {len(actions)}")
    print(f"      - Forward: {actions.count(1)}, Left: {actions.count(2)}, Right: {actions.count(3)}")
    
    # ========== Step 6: 执行动作并收集轨迹数据 ==========
    print("\n[步骤6] 执行动作序列并收集轨迹数据...")
    
    os.makedirs(episode_folder_path, exist_ok=True)
    
    try:
        trajectory_data, success = collect_trajectory_from_actions(
            env, actions, episode_folder_path
        )
        
        if not success:
            print(f"    ✗ 轨迹收集失败")
            if os.path.exists(episode_folder_path):
                shutil.rmtree(episode_folder_path)
            return False, vlm_result
        
        print(f"\n✓ Episode {scene_name}_ep{episode_num} 处理完成！")
        print(f"  - 轨迹长度: {len(trajectory_data['position'])} 帧")
        print(f"  - 文件夹: {episode_folder_path}")
        
        return True, vlm_result
        
    except Exception as e:
        print(f"    ✗ 轨迹收集失败: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(episode_folder_path):
            shutil.rmtree(episode_folder_path)
        return False, vlm_result


def main(config_path="benchmark/nav/pointnav/pointnav_hm3d.yaml", 
         output_dir='data/nomad_training_dataset',
         scene_names=['Adrian', 'Albertville', 'Anaheim'],
         max_episodes_per_scene=20,
         test_vlm_first=True):
    """
    主函数：生成 NoMaD 训练数据集
    只保留VLM认为可以保留的episode
    
    输出格式:
        output_dir/
        ├── Adrian_1/
        │   ├── 0.jpg, 1.jpg, ...
        │   └── traj_data.pkl
        └── ...
    
    统计指标:
        - 总episode数量
        - 保留的episode数量
        - 删除的episode数量（VLM拒绝）
        - 没有生成路径的episode数量
        - Token消耗统计
    """
    
    print("="*70)
    print("生成 NoMaD 训练数据集 (go_stanford 格式)")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n配置:")
    print(f"  目标场景: {scene_names}")
    print(f"  每场景episodes: {max_episodes_per_scene}")
    print(f"  输出目录: {output_dir}")
    print("="*70)

    # 测试 VLM API
    if test_vlm_first:
        vlm_ok = test_vlm_api()
        if not vlm_ok:
            print("\n" + "="*70)
            print("❌ VLM API 不可用，程序终止")
            print("="*70)
            return
        else:
            print("✅ VLM API 测试通过，开始数据收集\n")

    # 统计变量
    total_episodes = 0          # 总episode数量
    kept_episodes = 0           # 保留的episode数量
    deleted_episodes = 0        # VLM删除的episode数量
    no_path_episodes = 0        # 没有生成路径的episode数量
    total_tokens = 0            # 总token消耗
    vlm_evaluated_count = 0     # VLM评估过的episode数量（用于计算平均token）
    
    # 逐个场景处理
    for scene_idx, scene_name in enumerate(scene_names, 1):
        print(f"\n{'='*70}")
        print(f"处理场景 {scene_idx}/{len(scene_names)}: {scene_name}")
        print(f"{'='*70}")
        
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
            print(f"✗ 加载场景 {scene_name} 失败: {e}")
            continue
        
        scene_total_episodes = len(dataset.episodes)
        print(f"  场景包含 {scene_total_episodes} 个episodes")
        print(f"  将处理: {min(max_episodes_per_scene, scene_total_episodes)} 个episodes")
        
        # 创建仿真环境
        print(f"\n正在创建 {scene_name} 的仿真环境...")
        with habitat.Env(config=config, dataset=dataset) as env:
            print("✓ 环境创建成功！\n")
            
            scene_kept = 0
            scene_deleted = 0
            scene_no_path = 0
            scene_tokens = 0
            scene_vlm_count = 0
            
            episodes_to_process = min(max_episodes_per_scene, scene_total_episodes)
            
            for ep_idx in tqdm.tqdm(range(episodes_to_process), desc=f"{scene_name}"):
                observations = env.reset()
                episode_num = ep_idx + 1
                total_episodes += 1
                
                traj_folder_name = f"{scene_name}_{episode_num}"
                traj_folder_path = os.path.join(output_dir, traj_folder_name)
                
                # 检查是否已存在
                if os.path.exists(traj_folder_path):
                    required_files = ['traj_data.pkl', '0.jpg']
                    if all(os.path.exists(os.path.join(traj_folder_path, f)) 
                           for f in required_files):
                        print(f"\n跳过 {traj_folder_name} (已存在)")
                        kept_episodes += 1
                        scene_kept += 1
                        continue
                
                # 获取RGB观测
                rgb_obs = observations["rgb"]
                
                # 处理 episode
                try:
                    success, vlm_result = process_single_episode(
                        observations, traj_folder_path, 
                        scene_name, episode_num, env, rgb_obs
                    )
                    
                    # 统计token消耗
                    if vlm_result is not None and 'token_usage' in vlm_result:
                        token_usage = vlm_result['token_usage']
                        if 'total_tokens' in token_usage:
                            total_tokens += token_usage['total_tokens']
                            scene_tokens += token_usage['total_tokens']
                            vlm_evaluated_count += 1
                            scene_vlm_count += 1
                    
                    if success:
                        kept_episodes += 1
                        scene_kept += 1
                    else:
                        if vlm_result is None:
                            # 路径生成失败
                            no_path_episodes += 1
                            scene_no_path += 1
                        else:
                            # VLM拒绝
                            deleted_episodes += 1
                            scene_deleted += 1
                        
                        # 删除失败的文件夹
                        if os.path.exists(traj_folder_path):
                            import shutil
                            shutil.rmtree(traj_folder_path)
                
                except Exception as e:
                    print(f"\n错误 - {traj_folder_name}: {e}")
                    no_path_episodes += 1
                    scene_no_path += 1
                    if os.path.exists(traj_folder_path):
                        import shutil
                        shutil.rmtree(traj_folder_path)
            
            print(f"\n场景 {scene_name} 完成:")
            print(f"  处理总数: {episodes_to_process}")
            print(f"  保留: {scene_kept}")
            print(f"  VLM删除: {scene_deleted}")
            print(f"  无路径: {scene_no_path}")
            print(f"  Token消耗: {scene_tokens}")
            if scene_vlm_count > 0:
                print(f"  平均Token/episode: {scene_tokens / scene_vlm_count:.1f}")
    
    # ========== 最终统计 ==========
    print("\n" + "="*70)
    print("数据集生成完成!")
    print("="*70)
    
    print(f"\n【Episode统计】")
    print(f"  总Episode数量: {total_episodes}")
    print(f"  ├── 保留的Episode数量: {kept_episodes}")
    print(f"  ├── 删除的Episode数量（VLM拒绝）: {deleted_episodes}")
    print(f"  └── 无路径的Episode数量: {no_path_episodes}")
    
    # 验证总数
    assert kept_episodes + deleted_episodes + no_path_episodes == total_episodes, \
        f"统计不一致: {kept_episodes} + {deleted_episodes} + {no_path_episodes} != {total_episodes}"
    print(f"  [验证] 三者之和 = {kept_episodes + deleted_episodes + no_path_episodes} ✓")
    
    print(f"\n【Token消耗统计】")
    print(f"  总Token消耗: {total_tokens}")
    print(f"  VLM评估的Episode数: {vlm_evaluated_count}")
    if vlm_evaluated_count > 0:
        avg_tokens = total_tokens / vlm_evaluated_count
        print(f"  平均每Episode Token消耗: {avg_tokens:.2f}")
    else:
        print(f"  平均每Episode Token消耗: N/A (无VLM评估)")
    
    print(f"\n【输出信息】")
    print(f"  输出目录: {output_dir}")
    print(f"  处理的场景: {scene_names}")
    print("="*70)
    
    # 保存统计结果到JSON
    stats = {
        'total_episodes': total_episodes,
        'kept_episodes': kept_episodes,
        'deleted_episodes': deleted_episodes,
        'no_path_episodes': no_path_episodes,
        'total_tokens': total_tokens,
        'vlm_evaluated_count': vlm_evaluated_count,
        'avg_tokens_per_episode': total_tokens / vlm_evaluated_count if vlm_evaluated_count > 0 else 0,
        'scenes': scene_names,
        'max_episodes_per_scene': max_episodes_per_scene
    }
    
    stats_path = os.path.join(output_dir, 'dataset_statistics.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n统计结果已保存至: {stats_path}")


if __name__ == "__main__":
    # 配置参数
    CONFIG_PATH = "benchmark/nav/pointnav/pointnav_hm3d.yaml"
    OUTPUT_DIR = "data/nomad_training_dataset"
    
    # 指定要处理的场景
    #SCENE_NAMES = ['Adrian', 'Albertville', 'Anaheim', 'Andover', 'Angiola']
    # SCENE_NAMES = ['Adrian', 'Albertville', 'Anaheim', 'Andover', 'Angiola',
    #     'Annawan','Applewold','Arkansaw','Avonia','Azusa',
    #     'Ballou','Beach']
    SCENE_NAMES = ['1EiJpeRNEs1']

    # 每个场景最多处理的episodes数
    MAX_EPISODES_PER_SCENE = 10
    
    # 是否先测试VLM API
    TEST_VLM_FIRST = True
    
    # 运行
    main(
        config_path=CONFIG_PATH,
        output_dir=OUTPUT_DIR,
        scene_names=SCENE_NAMES,
        max_episodes_per_scene=MAX_EPISODES_PER_SCENE,
        test_vlm_first=TEST_VLM_FIRST
    )