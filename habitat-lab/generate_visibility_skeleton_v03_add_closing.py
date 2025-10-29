import os
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.transform import Rotation as R
from skimage.morphology import skeletonize, closing, disk

import habitat
from habitat.config.default_structured_configs import (
    TopDownMapMeasurementConfig,
    FogOfWarConfig,
)
from habitat.datasets.pointnav.multi_path_generator import (
    get_first_floor_height,
    sample_start_position,
)
from habitat.utils.visualizations import maps

# 静默日志
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

# ============ 配置参数 ============
CONFIG_PATH = "benchmark/nav/pointnav/pointnav_habitat_test.yaml"
OUTPUT_DIR = "output_6"
NUM_SAMPLES = 30  # 采样次数

# 相机参数
FOV = 90  # 视场角（度）
CLIP_FAR = 3.0  # 最远可视距离（米）
CAMERA_HEIGHT = 0.88  # 相机高度（米）

# 网格投影参数
GRID_RESOLUTION = 0.03  # 网格分辨率（米/像素）
GRID_SIZE_METERS = 6.0  # 网格覆盖范围（米）

# 岛屿半径限制
ISLAND_RADIUS_LIMIT = 1.0


def project_depth_to_grid(
    depth_image: np.ndarray,
    agent_state,
    grid_res: float,
    grid_size_m: float,
    fov_deg: float,
    clip_far: float
) -> np.ndarray:
    """将深度图投影到2D网格"""
    h, w = depth_image.shape
    grid_dim = int(grid_size_m / grid_res)
    grid = np.zeros((grid_dim, grid_dim), dtype=np.uint8)
    
    # 获取agent的旋转
    agent_rot = agent_state.rotation
    quat = np.array([agent_rot.x, agent_rot.y, agent_rot.z, agent_rot.w])
    rot_matrix = R.from_quat(quat).as_matrix()
    
    # Agent前向为-Z
    forward_vector = rot_matrix @ np.array([0, 0, -1])
    agent_yaw = np.arctan2(forward_vector[0], -forward_vector[2])
    
    # 遍历深度图像素
    fov_rad = np.radians(fov_deg)
    for i in range(h):
        for j in range(w):
            depth = depth_image[i, j]
            if depth <= 0 or depth >= clip_far:
                continue
            
            # 计算像素对应的角度
            u = (j - w / 2.0) / (w / 2.0)
            v = (i - h / 2.0) / (h / 2.0)
            
            theta_h = u * fov_rad / 2.0
            theta_v = v * fov_rad / 2.0
            
            # 投影到3D空间（相机坐标系）
            x_cam = depth * np.tan(theta_h)
            y_cam = depth * np.tan(theta_v)
            z_cam = depth
            
            # 转换到世界坐标系
            angle_world = agent_yaw + theta_h
            x_world = z_cam * np.sin(angle_world)
            z_world = z_cam * np.cos(angle_world)
            
            # 转换到网格坐标
            grid_x = int(grid_dim / 2 + x_world / grid_res)
            grid_z = int(grid_dim / 2 + z_world / grid_res)
            
            if 0 <= grid_x < grid_dim and 0 <= grid_z < grid_dim:
                grid[grid_z, grid_x] = 255
    
    return grid
'''
def fill_near_region(
    grid_projection: np.ndarray,
    agent_yaw: float,
    fill_radius_meters: float = 0.03,
    grid_res: float = GRID_RESOLUTION
) -> np.ndarray:
    """填充机器人附近的黑色区域为白色"""
    filled_grid = grid_projection.copy()
    grid_dim = filled_grid.shape[0]
    center_x, center_y = grid_dim // 2, grid_dim // 2  # 网格中心（机器人位置）
    fill_radius = int(fill_radius_meters / grid_res)  # 填充半径（像素）
    
    # 在机器人前方填充一个扇形区域
    for r in range(fill_radius):
        for theta in range(-45, 46):  # 扇形角度范围，可根据需要调整
            theta_rad = np.radians(theta)
            # 计算机器人朝向方向的填充点
            x = int(center_x + r * np.sin(agent_yaw + theta_rad))
            y = int(center_y + r * np.cos(agent_yaw + theta_rad))
            if 0 <= x < grid_dim and 0 <= y < grid_dim:
                filled_grid[y, x] = 255
    
    return filled_grid
'''
def fill_near_region(
    grid_projection: np.ndarray,
    agent_yaw: float,
    fill_radius_meters: float = 0.5,
    grid_res: float = GRID_RESOLUTION,
    fov_deg: float = FOV,
) -> np.ndarray:
    """
    使用凸包操作连接机器人位置和扇形区域内的白色区域
    
    步骤：
    1. 创建扇形掩码
    2. 提取扇形内的白色像素并添加机器人位置
    3. 计算凸包
    4. 约束到扇形内
    5. 合并到原始图像
    
    参数：
      grid_projection: uint8 二值化深度投影（0/255）
      agent_yaw: 机器人朝向（弧度）
      fill_radius_meters: 扇形半径（米）
      grid_res: 网格分辨率（米/像素）
      fov_deg: 深度相机的水平视场角（度）
    返回：
      filled_grid: 填充后的投影图
    """
    from skimage.morphology import convex_hull_image
    
    filled_grid = grid_projection.copy()
    grid_dim = filled_grid.shape[0]
    center_x = grid_dim // 2
    center_y = grid_dim // 2
    max_r_pix = max(1, int(fill_radius_meters / grid_res))

    # 步骤1: 创建扇形掩码
    sector_mask = np.zeros((grid_dim, grid_dim), dtype=bool)
    
    half_fov = np.radians(fov_deg / 2.0)
    
    # 遍历扇形区域内的所有像素
    for dy in range(-max_r_pix, max_r_pix + 1):
        for dx in range(-max_r_pix, max_r_pix + 1):
            # 计算像素相对于机器人的极坐标
            r = np.sqrt(dx**2 + dy**2)
            if r < 1 or r > max_r_pix:
                continue
            
            # 计算该像素的角度（世界坐标系）
            pixel_angle_world = np.arctan2(dx, dy)
            
            # 转换到相对于机器人朝向的角度
            relative_angle = pixel_angle_world - agent_yaw
            
            # 归一化到[-pi, pi]
            while relative_angle > np.pi:
                relative_angle -= 2 * np.pi
            while relative_angle < -np.pi:
                relative_angle += 2 * np.pi
            
            # 检查是否在FOV扇形范围内
            if abs(relative_angle) <= half_fov:
                y = center_y + dy
                x = center_x + dx
                if 0 <= x < grid_dim and 0 <= y < grid_dim:
                    sector_mask[y, x] = True

    # 步骤2: 准备待填充图像 - 提取扇形内的白色像素并添加机器人位置
    sector_white_pixels = np.zeros((grid_dim, grid_dim), dtype=bool)
    
    # 提取扇形内的原始白色像素
    sector_white_pixels = sector_mask & (grid_projection > 0)
    
    # 手动将机器人位置设为白色（这是关键步骤）
    sector_white_pixels[center_y, center_x] = True
    
    # 步骤3: 计算凸包
    # 只有当扇形内有白色像素时才计算凸包
    if np.any(sector_white_pixels):
        try:
            convex_hull_result = convex_hull_image(sector_white_pixels)
            
            # 步骤4: 约束与合并 - 将凸包结果限制在扇形掩码内
            constrained_hull = convex_hull_result & sector_mask
            
            # 步骤5: 最终合并 - 将新填充的区域与原始图像合并
            filled_grid[constrained_hull] = 255
            
        except Exception as e:
            # 如果凸包计算失败，打印警告但不中断程序
            print(f"Warning: Convex hull calculation failed: {e}")
            # 至少确保机器人位置是白色的
            filled_grid[center_y, center_x] = 255
    else:
        # 如果扇形内没有白色像素，至少将机器人位置设为白色
        filled_grid[center_y, center_x] = 255

    return filled_grid

def visualize_sample(
    env: habitat.Env,
    start_pos: List[float],
    start_rot: List[float],
    sample_idx: int,
    output_dir: str
):
    """可视化单次采样结果"""
    # 设置agent状态
    env.sim.set_agent_state(start_pos, start_rot)
    
    # 重置环境以获取观测和metrics
    observations = env.reset()
    metrics = env.get_metrics()
    
    # 获取观测
    rgb_img = observations["rgb"]
    depth_img = observations["depth"].squeeze()
    
    # 获取topdown map
    topdown_map = metrics["top_down_map"]["map"]
    agent_map_coord = metrics["top_down_map"]["agent_map_coord"]
    agent_angle = metrics["top_down_map"]["agent_angle"]
    
    # 类型转换：确保 agent_map_coord 是简单的 list [x, y]
    if isinstance(agent_map_coord, np.ndarray):
        agent_map_coord = agent_map_coord.flatten().tolist()
    elif isinstance(agent_map_coord, list):
        # 处理嵌套列表的情况，如 [[x, y]] -> [x, y]
        while isinstance(agent_map_coord, list) and len(agent_map_coord) == 1 and isinstance(agent_map_coord[0], (list, tuple)):
            agent_map_coord = agent_map_coord[0]
        if isinstance(agent_map_coord, tuple):
            agent_map_coord = list(agent_map_coord)
    
    # 类型转换：确保 agent_angle 是标量 - 改进的处理逻辑
    def extract_scalar(x):
        """递归提取标量值"""
        if isinstance(x, (int, float, np.integer, np.floating)):
            # 已经是标量
            return float(x)
        elif isinstance(x, np.ndarray):
            # numpy数组
            if x.ndim == 0:
                # 0维数组（标量数组）
                return float(x.item())
            else:
                # 多维数组，递归处理第一个元素
                return extract_scalar(x.flatten()[0])
        elif isinstance(x, (list, tuple)):
            # 列表或元组
            if len(x) == 0:
                raise ValueError("Empty list/tuple")
            return extract_scalar(x[0])
        else:
            # 尝试直接转换
            return float(x)
    
    agent_angle = extract_scalar(agent_angle)

    # 获取agent状态
    agent_state = env.sim.get_agent_state()
    
    # 深度图投影到2D
    grid_projection = project_depth_to_grid(
        depth_img,
        agent_state,
        GRID_RESOLUTION,
        GRID_SIZE_METERS,
        FOV,
        CLIP_FAR
    )
    
    # 获取agent旋转信息用于填充
    agent_rot = agent_state.rotation
    quat = np.array([agent_rot.x, agent_rot.y, agent_rot.z, agent_rot.w])
    rot_matrix = R.from_quat(quat).as_matrix()
    forward_vector = rot_matrix @ np.array([0, 0, -1])
    agent_yaw = np.arctan2(forward_vector[0], -forward_vector[2])
    
    # 填充机器人附近的黑色区域
    filled_grid = fill_near_region(grid_projection, agent_yaw, fill_radius_meters=0.5)
    
    # 骨架化原始投影和填充后的投影
    binary_grid = (grid_projection > 0).astype(bool)
    skeleton = skeletonize(binary_grid).astype(np.uint8) * 255
    
    binary_filled_grid = (filled_grid > 0).astype(bool)
    filled_skeleton = skeletonize(binary_filled_grid).astype(np.uint8) * 255
    
    # 创建结构元素（圆盘形，半径可调整）
    selem = disk(3)  # 半径为3像素的圆盘结构元素

    # 对二值化的填充网格执行闭操作
    # 闭操作 = 先膨胀后腐蚀，可以填补小的空洞和平滑边界
    closed_grid = closing(binary_filled_grid, selem)
    
    # 转换回uint8格式用于显示
    closed_grid_display = closed_grid.astype(np.uint8) * 255
    
    # 对闭操作后的结果提取骨架
    closed_skeleton = skeletonize(closed_grid).astype(np.uint8) * 255
    
    # 获取topdown_map中的最大值
    max_value = topdown_map.max()
    
    # 创建足够大的颜色映射
    recolor_map = np.array(
        [[255, 255, 255],  # 0: 被占用区域 -> 白色
         [128, 128, 128],  # 1: 可通行区域 -> 灰色
         [0, 0, 0],        # 2: 边界 -> 黑色
         [255, 0, 0],      # 3: 起点 -> 红色
         [0, 255, 0],      # 4: 目标点 -> 绿色
         [0, 0, 255],      # 5: 路径点 -> 蓝色
         [255, 255, 0],    # 6: 其他 -> 黄色
         [255, 0, 255],    # 7: 其他 -> 品红
         [0, 255, 255],    # 8: 其他 -> 青色
         [128, 0, 0],      # 9: 其他 -> 深红
        ],
        dtype=np.uint8
    )
    
    # 如果topdown_map中有更大的值，扩展颜色映射
    if max_value >= len(recolor_map):
        # 添加更多随机颜色
        extra_colors = np.random.randint(0, 256, size=(max_value - len(recolor_map) + 1, 3), dtype=np.uint8)
        recolor_map = np.vstack([recolor_map, extra_colors])
    
    colored_topdown = recolor_map[topdown_map]
    
    # 绘制agent
    maps.draw_agent(
        colored_topdown,
        agent_map_coord,
        agent_angle,
        agent_radius_px=8
    )
    
    # habitat的坐标的索引是(y,x), 但matplotlib的wedge格式是(x,y)
    grid_y, grid_x = agent_map_coord
    
    # 计算FOV参数
    bounds = env.sim.pathfinder.get_bounds()
    lower_bound, upper_bound = bounds
    
    map_height, map_width = topdown_map.shape
    x_range = upper_bound[0] - lower_bound[0]
    z_range = upper_bound[2] - lower_bound[2]
    
    meters_per_pixel = max(x_range / map_width, z_range / map_height)
    fov_radius = CLIP_FAR / meters_per_pixel
    
    # agent_angle是从+X轴逆时针的角度（弧度）
    # 转换为度数
    agent_angle_deg = np.degrees(agent_angle)
    
    # 为origin='upper'计算角度
    theta_center = agent_angle_deg
    fov_half = FOV / 2.0
    theta1 = theta_center - fov_half
    theta2 = theta_center + fov_half
    # ============ 创建可视化 ============
    fig, axes = plt.subplots(1, 9, figsize=(45, 5))  # 从5个子图增加到7个
    
    # 子图1: Top-down map with agent and FOV
    ax1 = axes[0]
    ax1.imshow(colored_topdown, origin='upper')
    
    # 绘制FOV扇形
    wedge = mpatches.Wedge(
        (grid_x, grid_y),
        fov_radius,
        theta1,
        theta2,
        alpha=0.3,
        color='yellow',
        label='FOV',
        zorder=4
    )
    ax1.add_patch(wedge)
    ax1.set_title('Top-Down Map with Agent', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.axis('off')
    
    # 子图2: RGB
    ax2 = axes[1]
    ax2.imshow(rgb_img)
    ax2.set_title('RGB Image', fontsize=12)
    ax2.axis('off')
    
    # 子图3: Depth
    ax3 = axes[2]
    depth_display = np.clip(depth_img, 0, CLIP_FAR)
    ax3.imshow(depth_display, cmap='plasma')
    ax3.set_title('Depth Image', fontsize=12)
    ax3.axis('off')
    
    # 子图4: Depth Projection
    ax4 = axes[3]
    ax4.imshow(grid_projection, cmap='gray', origin='lower')
    ax4.set_title('Depth Projection (2D Grid)', fontsize=12)

    # ============ 添加起点和FOV扇形边界 ============
    grid_dim = grid_projection.shape[0]
    agent_center_x = grid_dim / 2  # agent在网格中心的X坐标
    agent_center_y = grid_dim / 2  # agent在网格中心的Y坐标

    # 1. 绘制起点（红色圆点）
    from matplotlib.patches import Circle
    agent_circle = Circle(
        (agent_center_x, agent_center_y),
        radius=2,  # 圆点半径（像素）
        color='red',
        fill=True,
        zorder=5
    )
    ax4.add_patch(agent_circle)

    # 2. 计算FOV扇形参数
    fov_radius_pixels = CLIP_FAR / GRID_RESOLUTION  # FOV半径（像素）

    # 获取agent状态来计算朝向
    agent_state = env.sim.get_agent_state()
    agent_rot = agent_state.rotation
    quat = np.array([agent_rot.x, agent_rot.y, agent_rot.z, agent_rot.w])
    rot_matrix = R.from_quat(quat).as_matrix()

    # Agent前向为-Z
    forward_vector = rot_matrix @ np.array([0, 0, -1])
    agent_yaw = np.arctan2(forward_vector[0], -forward_vector[2])
    agent_yaw_deg = np.degrees(agent_yaw)

    # 计算扇形角度范围
    # 注意：matplotlib Wedge的角度是从+X轴逆时针测量
    # 但由于origin='lower'，需要调整角度系统
    theta_center = 90 - agent_yaw_deg  # 转换到matplotlib坐标系
    fov_half = FOV / 2.0
    theta1 = theta_center - fov_half
    theta2 = theta_center + fov_half

    # 3. 绘制FOV扇形边界（红色边框，透明填充）
    from matplotlib.patches import Wedge
    fov_wedge = Wedge(
        (agent_center_x, agent_center_y),
        fov_radius_pixels,
        theta1,
        theta2,
        facecolor='none',  # 无填充
        edgecolor='red',   # 红色边框
        linewidth=2,
        linestyle='-',
        zorder=4
    )
    ax4.add_patch(fov_wedge)

    # 4. 绘制FOV的两条射线边界（可选，让边界更清晰）
    # 计算两条边界射线的端点
    theta1_rad = np.radians(theta1)
    theta2_rad = np.radians(theta2)

    # 第一条射线
    x1_end = agent_center_x + fov_radius_pixels * np.cos(theta1_rad)
    y1_end = agent_center_y + fov_radius_pixels * np.sin(theta1_rad)
    ax4.plot(
        [agent_center_x, x1_end],
        [agent_center_y, y1_end],
        'r-',
        linewidth=2,
        zorder=4
    )

    # 第二条射线
    x2_end = agent_center_x + fov_radius_pixels * np.cos(theta2_rad)
    y2_end = agent_center_y + fov_radius_pixels * np.sin(theta2_rad)
    ax4.plot(
        [agent_center_x, x2_end],
        [agent_center_y, y2_end],
        'r-',
        linewidth=2,
        zorder=4
    )

    ax4.axis('off')


    # 子图5: Skeleton
    ax5 = axes[4]
    ax5.imshow(skeleton, cmap='gray', origin='lower')
    ax5.set_title('Skeleton', fontsize=12)

    # ============ 添加起点和FOV扇形边界 ============
    grid_dim = grid_projection.shape[0]
    agent_center_x = grid_dim / 2  # agent在网格中心的X坐标
    agent_center_y = grid_dim / 2  # agent在网格中心的Y坐标

    # 1. 绘制起点（红色圆点）
    from matplotlib.patches import Circle
    agent_circle = Circle(
        (agent_center_x, agent_center_y),
        radius=2,  # 圆点半径（像素）
        color='red',
        fill=True,
        zorder=5
    )
    ax5.add_patch(agent_circle)

    # 2. 计算FOV扇形参数
    fov_radius_pixels = CLIP_FAR / GRID_RESOLUTION  # FOV半径（像素）

    # 获取agent状态来计算朝向
    agent_state = env.sim.get_agent_state()
    agent_rot = agent_state.rotation
    quat = np.array([agent_rot.x, agent_rot.y, agent_rot.z, agent_rot.w])
    rot_matrix = R.from_quat(quat).as_matrix()

    # Agent前向为-Z
    forward_vector = rot_matrix @ np.array([0, 0, -1])
    agent_yaw = np.arctan2(forward_vector[0], -forward_vector[2])
    agent_yaw_deg = np.degrees(agent_yaw)

    # 计算扇形角度范围
    # 注意：matplotlib Wedge的角度是从+X轴逆时针测量
    # 但由于origin='lower'，需要调整角度系统
    theta_center = 90 - agent_yaw_deg  # 转换到matplotlib坐标系
    fov_half = FOV / 2.0
    theta1 = theta_center - fov_half
    theta2 = theta_center + fov_half

    # 3. 绘制FOV扇形边界（红色边框，透明填充）
    from matplotlib.patches import Wedge
    fov_wedge = Wedge(
        (agent_center_x, agent_center_y),
        fov_radius_pixels,
        theta1,
        theta2,
        facecolor='none',  # 无填充
        edgecolor='red',   # 红色边框
        linewidth=2,
        linestyle='-',
        zorder=4
    )
    ax5.add_patch(fov_wedge)

    # 4. 绘制FOV的两条射线边界（可选，让边界更清晰）
    # 计算两条边界射线的端点
    theta1_rad = np.radians(theta1)
    theta2_rad = np.radians(theta2)

    # 第一条射线
    x1_end = agent_center_x + fov_radius_pixels * np.cos(theta1_rad)
    y1_end = agent_center_y + fov_radius_pixels * np.sin(theta1_rad)
    ax5.plot(
        [agent_center_x, x1_end],
        [agent_center_y, y1_end],
        'r-',
        linewidth=2,
        zorder=4
    )

    # 第二条射线
    x2_end = agent_center_x + fov_radius_pixels * np.cos(theta2_rad)
    y2_end = agent_center_y + fov_radius_pixels * np.sin(theta2_rad)
    ax5.plot(
        [agent_center_x, x2_end],
        [agent_center_y, y2_end],
        'r-',
        linewidth=2,
        zorder=4
    )
    ax5.axis('off')
    # 子图6: 填充后的深度投影
    ax6 = axes[5]
    ax6.imshow(filled_grid, cmap='gray', origin='lower')
    ax6.set_title('Filled Depth Projection', fontsize=12)

    # 添加相同的起点和FOV边界
    from matplotlib.patches import Circle
    agent_circle = Circle(
        (agent_center_x, agent_center_y),
        radius=2,
        color='red',
        fill=True,
        zorder=5
    )
    ax6.add_patch(agent_circle)
    
    fov_wedge = Wedge(
        (agent_center_x, agent_center_y),
        fov_radius_pixels,
        theta1,
        theta2,
        facecolor='none',
        edgecolor='red',
        linewidth=2,
        linestyle='-',
        zorder=4
    )
    ax6.add_patch(fov_wedge)
    
    ax6.plot(
        [agent_center_x, x1_end],
        [agent_center_y, y1_end],
        'r-',
        linewidth=2,
        zorder=4
    )
    
    ax6.plot(
        [agent_center_x, x2_end],
        [agent_center_y, y2_end],
        'r-',
        linewidth=2,
        zorder=4
    )
    ax6.axis('off')
    
    # 子图7: 填充后的骨架
    ax7 = axes[6]
    ax7.imshow(filled_skeleton, cmap='gray', origin='lower')
    ax7.set_title('Filled Skeleton', fontsize=12)
    
    # 添加相同的起点和FOV边界
    agent_circle = Circle(
        (agent_center_x, agent_center_y),
        radius=2,
        color='red',
        fill=True,
        zorder=5
    )
    ax7.add_patch(agent_circle)
    
    fov_wedge = Wedge(
        (agent_center_x, agent_center_y),
        fov_radius_pixels,
        theta1,
        theta2,
        facecolor='none',
        edgecolor='red',
        linewidth=2,
        linestyle='-',
        zorder=4
    )
    ax7.add_patch(fov_wedge)
    
    ax7.plot(
        [agent_center_x, x1_end],
        [agent_center_y, y1_end],
        'r-',
        linewidth=2,
        zorder=4
    )
    
    ax7.plot(
        [agent_center_x, x2_end],
        [agent_center_y, y2_end],
        'r-',
        linewidth=2,
        zorder=4
    )
    ax7.axis('off')
    
    #子图8：对子图6做闭操作
    ax8 = axes[7]
    ax8.imshow(closed_grid_display, cmap='gray', origin='lower')
    ax8.set_title('Closed Depth Projection', fontsize=12)
    
    # 添加相同的起点和FOV边界
    agent_circle = Circle(
        (agent_center_x, agent_center_y),
        radius=2,
        color='red',
        fill=True,
        zorder=5
    )
    ax8.add_patch(agent_circle)
    
    fov_wedge = Wedge(
        (agent_center_x, agent_center_y),
        fov_radius_pixels,
        theta1,
        theta2,
        facecolor='none',
        edgecolor='red',
        linewidth=2,
        linestyle='-',
        zorder=4
    )
    ax8.add_patch(fov_wedge)
    
    ax8.plot(
        [agent_center_x, x1_end],
        [agent_center_y, y1_end],
        'r-',
        linewidth=2,
        zorder=4
    )
    
    ax8.plot(
        [agent_center_x, x2_end],
        [agent_center_y, y2_end],
        'r-',
        linewidth=2,
        zorder=4
    )
    ax8.axis('off')
    
    #子图9：从子图8提取骨架
    ax9 = axes[8]
    ax9.imshow(closed_skeleton, cmap='gray', origin='lower')
    ax9.set_title('Closed Skeleton', fontsize=12)
    
    # 添加相同的起点和FOV边界
    agent_circle = Circle(
        (agent_center_x, agent_center_y),
        radius=2,
        color='red',
        fill=True,
        zorder=5
    )
    ax9.add_patch(agent_circle)
    
    fov_wedge = Wedge(
        (agent_center_x, agent_center_y),
        fov_radius_pixels,
        theta1,
        theta2,
        facecolor='none',
        edgecolor='red',
        linewidth=2,
        linestyle='-',
        zorder=4
    )
    ax9.add_patch(fov_wedge)
    
    ax9.plot(
        [agent_center_x, x1_end],
        [agent_center_y, y1_end],
        'r-',
        linewidth=2,
        zorder=4
    )
    
    ax9.plot(
        [agent_center_x, x2_end],
        [agent_center_y, y2_end],
        'r-',
        linewidth=2,
        zorder=4
    )
    ax9.axis('off')

    plt.tight_layout()
    
    # 保存
    output_path = os.path.join(output_dir, f"sample_{sample_idx:03d}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"保存可视化结果: {output_path}")
    plt.close()

def main():
    """主函数"""
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print(f"加载配置: {CONFIG_PATH}")
    
    # 创建habitat config
    config = habitat.get_config(config_path=CONFIG_PATH)
    
    # 使用 read_write 上下文动态添加 TopDownMap 测量配置
    with habitat.config.read_write(config):

        # 更新相机参数
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = 512
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = 512
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = FOV
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position = [0.0, CAMERA_HEIGHT, 0.0]
        
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width = 512
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height = 512
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov = FOV
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth = CLIP_FAR
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.min_depth = 0.0
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.position = [0.0, CAMERA_HEIGHT, 0.0]
        
        # 添加 TopDownMap 测量
        config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=False,
                    draw_view_points=False,
                    draw_goal_positions=False,
                    draw_goal_aabbs=False,
                    fog_of_war=FogOfWarConfig(
                        draw=False,
                        visibility_dist=5.0,
                        fov=90,
                    ),
                ),
            }
        )
    
    # 创建数据集
    dataset = habitat.make_dataset(
        id_dataset=config.habitat.dataset.type,
        config=config.habitat.dataset
    )
    
    print(f"\n开始生成 {NUM_SAMPLES} 个采样可视化...\n")
    
    # 创建环境
    with habitat.Env(config=config, dataset=dataset) as env:
        # 获取场景的一楼高度
        start_height = get_first_floor_height(env.sim)
        
        # 多次采样
        for i in range(NUM_SAMPLES):
            print(f"处理采样 {i+1}/{NUM_SAMPLES}...")
            
            # 采样起点
            start_pos = sample_start_position(
                start_height,
                env.sim,
                min_island_radius=ISLAND_RADIUS_LIMIT
            )
            
            if start_pos is None:
                print(f"  采样 {i+1} 失败: 无法找到合适的起始位置")
                continue
            
            # 随机采样朝向
            angle = np.random.uniform(0, 2 * np.pi)
            start_rot = [0.0, np.sin(angle / 2), 0.0, np.cos(angle / 2)]
            
            # 可视化
            try:
                visualize_sample( 
                    env,
                    start_pos,
                    start_rot,
                    i,
                    OUTPUT_DIR
                )
            except Exception as e:
                print(f"  采样 {i+1} 处理失败: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n所有可视化已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()