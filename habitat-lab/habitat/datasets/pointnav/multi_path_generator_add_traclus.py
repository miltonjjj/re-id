from typing import Dict, Generator, List, Optional, Sequence, Tuple, Union, Any

import numpy as np

from habitat.core.simulator import ShortestPathPoint
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal

try:
    from habitat_sim.errors import GreedyFollowerError
except ImportError:
    GreedyFollower = BaseException

from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.geometry_utils import quaternion_to_list
import habitat_sim

import math
import attr
from habitat.core.dataset import Episode
from habitat.core.utils import not_none_validator
import magnum as mn
import os

from habitat.utils.traclus import traclus
import copy

import logging
logger = logging.getLogger(__name__)

def random_gaussian(k, std_dev):
  noise = np.random.normal(0, std_dev)
  return k + noise

ISLAND_RADIUS_LIMIT = 1.#最小岛屿半径限制
num_angle_samples=30#采样目标点个数
#sampling_distance=5.

def get_action_shortest_path(
    sim: "HabitatSim",
    source_position: List[float],
    source_rotation: List[float],
    goal_position: List[float],
    success_distance: float = 0.05,
    max_episode_steps: int = 1000,
) -> Tuple[List[ShortestPathPoint],int]:
    flag=1
    sim.reset()
    sim.set_agent_state(source_position, source_rotation)
    follower = ShortestPathFollower(sim, success_distance, False)

    shortest_path = []
    step_count = 0
    action = follower.get_next_action(goal_position)
    while (
        action is not HabitatSimActions.stop and step_count < max_episode_steps
    ):
        state = sim.get_agent_state()
        shortest_path.append(
            ShortestPathPoint(
                state.position.tolist(),
                quaternion_to_list(state.rotation),
                action,
            )
        )
        
        sim.step(action)
        step_count += 1
        action = follower.get_next_action(goal_position)

    if step_count == max_episode_steps:#达到最大步数而终止
        logger.warning("beyond max timesteps!")
        flag=0
    return shortest_path,flag


@attr.s(auto_attribs=True, kw_only=True)
class NavigationPath:
    
    goal: List[NavigationGoal]
    shortest_path: List[List[ShortestPathPoint]]
    geodesic_distance: float

"""定义子类MultiPathEpisode,继承自NavigationEpisode"""
@attr.s(auto_attribs=True, kw_only=True)
class MultiPathEpisode(NavigationEpisode):
    
    # 覆盖父类的 goals 属性
    goals: List[NavigationGoal]=attr.ib(
        default=None,
        validator=not_none_validator,
        on_setattr=Episode._reset_shortest_path_cache_hook,
    )
    
    # 覆盖父类的 shortest_paths 属性
    shortest_paths: Optional[List[List[ShortestPathPoint]]] = None
    
    # 记录目标点数量
    num_targets: float = attr.ib(
        default=None,
        validator=not_none_validator
    )
    
    # 新的路径结构
    paths: List[NavigationPath] = attr.ib(
        default=None,
        validator=not_none_validator
    )



"""在一楼高度采样起始点"""
def get_first_floor_height(sim: "HabitatSim") -> float:
    bounds = sim.pathfinder.get_bounds()
    first_floor_height = float(bounds[0][1])  # 一楼高度
    return first_floor_height

def sample_start_position(
    start_position_height: float,
    sim: "HabitatSim",
    min_island_radius: float = ISLAND_RADIUS_LIMIT,
    max_attempts: int = 2000
) -> Optional[List[float]]:
    """从指定高度的导航网格mask中采样起始位置"""
    attempt=0
    while attempt<max_attempts:
        source_position=sim.sample_navigable_point()
        if np.any(np.isnan(source_position)):
            attempt+=1
            continue
        source_position[1]=start_position_height
        #print(f"navigable island radius is {source_position}")
        if sim.is_navigable(source_position) and sim.island_radius(source_position)>=min_island_radius:
            return source_position
        else:
            snapped_source_position=list(sim.pathfinder.snap_point(mn.Vector3(*source_position)))
            snapped_source_position[1]=start_position_height
            if sim.is_navigable(snapped_source_position) and sim.island_radius(snapped_source_position)>=min_island_radius:
                #print(f"snapped_source_position is {snapped_source_position}")
                return snapped_source_position
            else:
                attempt+=1
                #print(f"attempt = {attempt}")
    return None


"""360度采样目标点"""
def sample_target_points( 
    start_position: List[float], 
    sampling_distance: float,
    sim: "HabitatSim",
) -> List[List[float]]:
    original_navigable_target_count=0
    snapped_navigable_target_count=0
    not_navigable_target_count=0
    target_points = []
    agent_pos = np.array(start_position)
    for i in range(num_angle_samples):
        sampling_distance=random_gaussian(sampling_distance,0.03*sampling_distance)
        angle = 2 * math.pi * i / num_angle_samples
        dx = sampling_distance * math.cos(angle)  
        dz = sampling_distance * math.sin(angle)
        
        target_position = [
            agent_pos[0] + dx,
            agent_pos[1],
            agent_pos[2] + dz
        ]
        #print(f"scene {sim.habitat_config.scene} target_position is {target_position}")
        if sim.is_navigable(target_position):
            target_points.append(target_position)
            original_navigable_target_count+=1
            #print(f"target_position {target_position} is navigable")
        else:
            snapped_target_position=list(sim.pathfinder.snap_point(mn.Vector3(*target_position)))
            #print(f"snapped_target_position is {snapped_target_position}")
            if abs(snapped_target_position[1]-agent_pos[1])<0.5 and sim.is_navigable(snapped_target_position):
                target_points.append(snapped_target_position)
                snapped_navigable_target_count+=1
                #print(f"target_position {target_position} is changed into snapped_target_position {snapped_target_position} and is navigable")
            else:
                not_navigable_target_count+=1

    #print(f"target_points are {target_points}")
    return target_points, original_navigable_target_count, snapped_navigable_target_count, not_navigable_target_count


def _ratio_sample_rate(ratio: float, ratio_threshold: float) -> float:
    r"""Sampling function for aggressive filtering of straight-line
    episodes with shortest path geodesic distance to Euclid distance ratio
    threshold.

    :param ratio: geodesic distance ratio to Euclid distance
    :param ratio_threshold: geodesic shortest path to Euclid
    distance ratio upper limit till aggressive sampling is applied.
    :return: value between 0.008 and 0.144 for ratio [1, 1.1]
    """
    assert ratio < ratio_threshold
    return 20 * (ratio - 0.98) ** 2 
    #ratio越高，采样的频率越高，并且在ratio接近1是几乎完全拒绝简单的路径


def is_compatible_episode(
    s: Sequence[float],
    t: Sequence[float],
    near_dist: float,
    far_dist: float,
    geodesic_to_euclid_ratio: float,
    sim: "HabitatSim",
) -> Union[Tuple[bool, float], Tuple[bool, int]]:
    euclid_dist = np.power(np.power(np.array(s) - np.array(t), 2).sum(0), 0.5)
    if np.abs(s[1] - t[1]) > 0.5:  # check height difference to assure s and 
        #  t are from same floor
        return False, 0
        
    d_separation = sim.geodesic_distance(s, [t])
    #print(f"geodesic distance is {d_separation}")
    if d_separation == np.inf:
        return False, 0
    if not near_dist <= d_separation <= far_dist:#确保episode既不太简单也不太困难
        return False, 0
    distances_ratio = d_separation / euclid_dist
    #print(f"distances_ratio is {distances_ratio}")
    if distances_ratio < geodesic_to_euclid_ratio and (
        np.random.rand()
        > _ratio_sample_rate(distances_ratio, geodesic_to_euclid_ratio)
    ):
        return False, 0#拒绝采样
        #参考分布是均匀分布
    if sim.island_radius(s) < ISLAND_RADIUS_LIMIT:
        return False, 0
    #岛屿是navmesh中一个连通的区域（两个通过走廊连接的房间是一个岛屿），每个岛屿对应一个岛屿半径
    #岛屿半径等于岛屿中所有多边形顶点的几何中心到距离最远的多边形顶点的欧式距离
    return True, d_separation

def extract_3d_to_2d(paths: List[Dict]) -> List[np.ndarray]:

    trajectories_2d = []
    
    for path in paths:
        shortest_path = path.get('shortest_path', [])
        if shortest_path:
            waypoints_2d = []
            for point in shortest_path:
                pos = point.position
                waypoints_2d.append([pos[0], pos[2]])
            
            if len(waypoints_2d) >= 2:  # 至少需要2个点
                trajectories_2d.append(np.array(waypoints_2d))
            else:
                print(f"traj is too short")
    return trajectories_2d

def revert_2d_to_3d(traj_2d: np.ndarray, original_height: float) -> List[List[float]]:
    
    if traj_2d.shape[0] == 0:
        print(f"traj_2d is null")
        return []
    
    waypoints_3d = []
    for point_2d in traj_2d:
        waypoints_3d.append([float(point_2d[0]), original_height, float(point_2d[1])])
    
    return waypoints_3d

def generate_new_shortest_path_with_actions(
    waypoints_3d: List[List[float]], 
    start_rotation: List[float],
    sim: "HabitatSim",
    success_distance: float = 0.02
    ) -> List[ShortestPathPoint]:
    
    if not waypoints_3d or len(waypoints_3d) == 0:
        print(f"traj_3d is null")
        return []
    
    new_shortest_path = []
    current_rotation = start_rotation

    sim.reset()
    sim.set_agent_state(waypoints_3d[0], start_rotation)

    for i, waypoint in enumerate(waypoints_3d):
        if i == len(waypoints_3d) - 1:
            action = HabitatSimActions.stop
            shortest_path_point = ShortestPathPoint(
                position=waypoint,
                rotation=current_rotation,
                action=action
            )
            new_shortest_path.append(shortest_path_point)
        else:
            try:
                sim.set_agent_state(waypoint, start_rotation)
                
                # 创建follower并获取下一个action
                follower = ShortestPathFollower(sim, success_distance, False)
                next_waypoint = waypoints_3d[i + 1]
                action = follower.get_next_action(next_waypoint)

                shortest_path_point = ShortestPathPoint(
                    position=waypoint,
                    rotation=current_rotation,
                    action=action
                )
                new_shortest_path.append(shortest_path_point)

                if action != HabitatSimActions.stop:
                    sim.step(action)
                    state_after_action = sim.get_agent_state()
                    current_rotation = quaternion_to_list(state_after_action.rotation)
            except Exception as e:
                logger.warning(f"Error calculating action for waypoint {i}: {str(e)}")
                # 使用默认action
                action = HabitatSimActions.move_forward
                shortest_path_point = ShortestPathPoint(
                    position=waypoint,
                    rotation=current_rotation,
                    action=action
                )
                new_shortest_path.append(shortest_path_point)
    
    return new_shortest_path
'''
def generate_trajectory_with_rotation_and_action(
    sim:"HabitatSim",
    waypoints_3d: List[List[float]],
    source_position: List[float],
    source_rotation: List[float],
    goal_position: List[float],
    success_distance: float = 0.02,
    max_episode_steps: int = 1000,
    )->List[ShortestPathPoint]:
    
    path_with_rotation_and_action=[]
    for waypoint in waypoints_3d[1:]:
        shortest_path,success=get_action_shortest_path(
        sim,
        source_position=source_position,
        source_rotation=source_rotation,
        goal_position=waypoint,
        success_distance=success_distance,
        max_episode_steps=max_episode_steps,
        )
        path_with_rotation_and_action
'''   
def calculate_trajectory_geodesic_distance(waypoints_3d: List[List[float]], 
                                         sim: "HabitatSim") -> float:
    """
    计算轨迹的geodesic距离
    """
    if len(waypoints_3d) < 2:
        return 0.0
    
    total_distance = 0.0
    for i in range(len(waypoints_3d) - 1):
        start_pos = waypoints_3d[i]
        end_pos = waypoints_3d[i + 1]
        
        # 使用simulator计算geodesic距离
        distance = sim.geodesic_distance(start_pos, [end_pos])
        if distance != np.inf:
            total_distance += distance
        else:
            # 如果geodesic距离无效，使用欧氏距离作为备选
            euclidean_dist = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
            total_distance += euclidean_dist
    
    return total_distance
'''
def apply_trajectory_clustering(paths: List[Dict], 
                              start_rotation: List[float],
                              sim: "HabitatSim",
                              max_eps: None,
                              min_samples: int = 2) -> List[Dict]:
    """
    对paths中的轨迹进行TRACLUS聚类并更新
    """
    if not paths:
        return paths
    
    # 提取2D轨迹数据
    trajectories_2d = extract_3d_to_2d(paths)
    
    if len(trajectories_2d) < min_samples:
        logger.warning(f"Not enough trajectories for clustering: {len(trajectories_2d)} < {min_samples}")
        return paths
    
    # 检查轨迹长度分布
    traj_lengths = [len(traj) for traj in trajectories_2d]
    logger.info(f"Trajectory lengths - min: {min(traj_lengths)}, max: {max(traj_lengths)}, avg: {np.mean(traj_lengths):.1f}")
    
    try:
        # 运行TRACLUS算法
        logger.info(f"Running TRACLUS with max_eps={max_eps}, min_samples={min_samples}")
        
        partitions, segments, dist_matrix, clusters, cluster_assignments, representative_trajectories = traclus(
            trajectories=trajectories_2d,
            max_eps=max_eps,
            min_samples=min_samples,
            directional=True,
            use_segments=True,
            progress_bar=False
        )
        
        # 分析聚类结果
        unique_clusters = set(cluster_assignments)
        logger.info(f"TRACLUS clustering completed:")
        logger.info(f"  - Total trajectories: {len(trajectories_2d)}")
        logger.info(f"  - Number of clusters: {len(unique_clusters)}")
        logger.info(f"  - Cluster IDs: {unique_clusters}")
        logger.info(f"  - Representative trajectories: {len(representative_trajectories)}")
        
        # 统计每个聚类的大小
        from collections import Counter
        cluster_sizes = Counter(cluster_assignments)
        logger.info(f"  - Cluster sizes: {dict(cluster_sizes)}")
        
        # 如果聚类数等于轨迹数，说明没有合并
        if len(unique_clusters) == len(trajectories_2d):
            logger.warning("WARNING: No trajectories were merged! Each trajectory is in its own cluster.")
            logger.warning("Consider:")
            logger.warning("  1. Increasing max_eps parameter")
            logger.warning("  2. Decreasing min_samples parameter")
            logger.warning("  3. Checking if trajectories are too diverse")

        # 更新paths
        new_paths = []
        valid_traj_idx = 0
        
        for i, path in enumerate(paths):
            # 检查是否有有效的shortest_path
            if path.get('shortest_path') and len(trajectories_2d) > valid_traj_idx:
                cluster_id = cluster_assignments[valid_traj_idx] if valid_traj_idx < len(cluster_assignments) else -1
                
                # 如果属于有效聚类且有代表性轨迹
                if cluster_id >= 0 and cluster_id < len(representative_trajectories):
                    traj_2d = representative_trajectories[cluster_id]
                    
                    if traj_2d.shape[0] > 0:
                        # 获取原始高度（使用第一个waypoint的y坐标）
                        original_shortest_path = path['shortest_path']
                        if original_shortest_path:
                            if hasattr(original_shortest_path[0], 'position'):
                                original_height = original_shortest_path[0].position[1]
                            else:
                                original_height = original_shortest_path[0]['position'][1]
                        else:
                            original_height = 0.0
                        
                        # 转换为3D坐标
                        rep_waypoints_3d = revert_2d_to_3d(
                            traj_2d, original_height
                        )
                        
                        # 生成新的shortest_path with actions
                        
                        new_shortest_path = generate_new_shortest_path_with_actions(
                            rep_waypoints_3d, start_rotation, sim
                        )

                        # 计算新的geodesic距离
                        new_geodesic_distance = calculate_trajectory_geodesic_distance(
                            rep_waypoints_3d, sim
                        )
                        
                        # 创建新的path
                        new_path = copy.deepcopy(path)
                        new_path['shortest_path'] = new_shortest_path
                        new_path['geodesic_distance'] = new_geodesic_distance
                        
                        new_paths.append(new_path)
                        logger.debug(f"Updated path {i} with clustered trajectory")
                    else:
                        # 代表性轨迹为空，保持原始path
                        new_paths.append(path)
                else:
                    # 噪声点或无效聚类，保持原始path
                    new_paths.append(path)
                
                valid_traj_idx += 1
            else:
                # 无效轨迹，保持原始path
                new_paths.append(path)
        
        return new_paths
        
    except Exception as e:
        logger.error(f"TRACLUS clustering failed: {str(e)}")
        return paths
'''
def apply_trajectory_clustering(paths: List[Dict], 
                              start_rotation: List[float],
                              sim: "HabitatSim",
                              max_eps: float = 15.0,
                              min_samples: int = 2) -> List[Dict]:
    """
    对paths中的轨迹进行TRACLUS聚类并更新
    """
    if not paths:
        logger.warning("No paths to cluster")
        return paths
    
    # 提取2D轨迹数据
    trajectories_2d = extract_3d_to_2d(paths)
    
    logger.info(f"Extracted {len(trajectories_2d)} 2D trajectories from {len(paths)} paths")
    
    if len(trajectories_2d) < min_samples:
        logger.warning(f"Not enough trajectories for clustering: {len(trajectories_2d)} < {min_samples}")
        return paths
    
    # === 添加详细的距离诊断 ===
    print("\n" + "="*60)
    print("TRAJECTORY CLUSTERING DIAGNOSTICS")
    print("="*60)
    
    # 统计轨迹长度
    traj_lengths = [len(traj) for traj in trajectories_2d]
    print(f"Trajectory counts: {len(trajectories_2d)}")
    print(f"Trajectory lengths - min: {min(traj_lengths)}, max: {max(traj_lengths)}, avg: {np.mean(traj_lengths):.1f}")
    
    # 计算轨迹间的距离
    from scipy.spatial.distance import euclidean
    
    # 方法1: 起点和终点的平均距离
    start_distances = []
    end_distances = []
    avg_distances = []
    
    for i in range(len(trajectories_2d)):
        for j in range(i+1, len(trajectories_2d)):
            start_dist = euclidean(trajectories_2d[i][0], trajectories_2d[j][0])
            end_dist = euclidean(trajectories_2d[i][-1], trajectories_2d[j][-1])
            avg_dist = (start_dist + end_dist) / 2
            
            start_distances.append(start_dist)
            end_distances.append(end_dist)
            avg_distances.append(avg_dist)
    
    if avg_distances:
        print(f"\nDistance Statistics (meters):")
        print(f"  Start point distances:")
        print(f"    - Min: {np.min(start_distances):.2f}")
        print(f"    - Max: {np.max(start_distances):.2f}")
        print(f"    - Mean: {np.mean(start_distances):.2f}")
        print(f"    - Median: {np.median(start_distances):.2f}")
        
        print(f"  End point distances:")
        print(f"    - Min: {np.min(end_distances):.2f}")
        print(f"    - Max: {np.max(end_distances):.2f}")
        print(f"    - Mean: {np.mean(end_distances):.2f}")
        print(f"    - Median: {np.median(end_distances):.2f}")
        
        print(f"  Average distances:")
        print(f"    - Min: {np.min(avg_distances):.2f}")
        print(f"    - Max: {np.max(avg_distances):.2f}")
        print(f"    - Mean: {np.mean(avg_distances):.2f}")
        print(f"    - Median: {np.median(avg_distances):.2f}")
        
        # 建议的max_eps值
        percentiles = [25, 50, 75, 90]
        print(f"  Distance percentiles:")
        for p in percentiles:
            val = np.percentile(avg_distances, p)
            print(f"    - {p}th: {val:.2f}")
        
        suggested_eps = np.percentile(avg_distances, 75)
        print(f"\n  Current max_eps: {max_eps:.2f}")
        print(f"  Suggested max_eps (75th percentile): {suggested_eps:.2f}")
        
        # 统计有多少对轨迹在max_eps范围内
        pairs_within_eps = sum(1 for d in avg_distances if d <= max_eps)
        total_pairs = len(avg_distances)
        print(f"  Pairs within max_eps: {pairs_within_eps}/{total_pairs} ({100*pairs_within_eps/total_pairs:.1f}%)")
        
        if pairs_within_eps == 0:
            print(f"\n  ⚠️  WARNING: NO trajectory pairs are within max_eps={max_eps}!")
            print(f"  ⚠️  Clustering will fail to merge any trajectories!")
            print(f"  ⚠️  Consider increasing max_eps to at least {np.min(avg_distances):.2f}")
    
    print("="*60 + "\n")
    
    try:
        # 运行TRACLUS算法
        logger.info(f"Running TRACLUS with max_eps={max_eps}, min_samples={min_samples}")
        
        partitions, segments, dist_matrix, clusters, cluster_assignments, representative_trajectories = traclus(
            trajectories=trajectories_2d,
            max_eps=max_eps,
            min_samples=min_samples,
            directional=True,
            use_segments=True,
            progress_bar=False
        )
        
        # 分析聚类结果
        unique_clusters = set(cluster_assignments)
        logger.info(f"TRACLUS clustering completed:")
        logger.info(f"  - Total trajectories: {len(trajectories_2d)}")
        logger.info(f"  - Number of clusters: {len(unique_clusters)}")
        logger.info(f"  - Cluster IDs: {unique_clusters}")
        logger.info(f"  - Representative trajectories: {len(representative_trajectories)}")
        
        # 统计每个聚类的大小
        from collections import Counter
        cluster_sizes = Counter(cluster_assignments)
        logger.info(f"  - Cluster sizes: {dict(cluster_sizes)}")
        
        # 如果聚类数等于轨迹数，说明没有合并
        if len(unique_clusters) == len(trajectories_2d):
            logger.warning("WARNING: No trajectories were merged! Each trajectory is in its own cluster.")
            logger.warning("Possible reasons:")
            logger.warning(f"  1. max_eps ({max_eps}) is too small")
            logger.warning(f"  2. Trajectories are too diverse")
            logger.warning(f"  3. Try increasing max_eps to {suggested_eps:.2f} or higher")
        
        # 更新paths (保持原有逻辑)
        new_paths = []
        valid_traj_idx = 0
        
        for i, path in enumerate(paths):
            if path.get('shortest_path') and len(trajectories_2d) > valid_traj_idx:
                cluster_id = cluster_assignments[valid_traj_idx] if valid_traj_idx < len(cluster_assignments) else -1
                
                if cluster_id >= 0 and cluster_id < len(representative_trajectories):
                    traj_2d = representative_trajectories[cluster_id]
                    
                    if traj_2d.shape[0] > 0:
                        # 获取原始高度
                        original_shortest_path = path['shortest_path']
                        if original_shortest_path:
                            first_point = original_shortest_path[0]
                            if isinstance(first_point, ShortestPathPoint):
                                original_height = first_point.position[1]
                            elif isinstance(first_point, dict):
                                original_height = first_point['position'][1]
                            else:
                                original_height = 0.0
                        else:
                            original_height = 0.0
                        
                        # 转换为3D坐标
                        rep_waypoints_3d = revert_2d_to_3d(traj_2d, original_height)
                        
                        # 生成新的shortest_path with actions
                        new_shortest_path = generate_new_shortest_path_with_actions(
                            rep_waypoints_3d, start_rotation, sim
                        )

                        # 计算新的geodesic距离
                        new_geodesic_distance = calculate_trajectory_geodesic_distance(
                            rep_waypoints_3d, sim
                        )
                        
                        # 创建新的path
                        new_path = copy.deepcopy(path)
                        new_path['shortest_path'] = new_shortest_path
                        new_path['geodesic_distance'] = new_geodesic_distance
                        
                        new_paths.append(new_path)
                    else:
                        new_paths.append(path)
                else:
                    new_paths.append(path)
                
                valid_traj_idx += 1
            else:
                new_paths.append(path)
        
        return new_paths
        
    except Exception as e:
        logger.error(f"TRACLUS clustering failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return paths

def apply_simple_endpoint_clustering(paths: List[Dict], 
                                     start_rotation: List[float],
                                     sim: "HabitatSim",
                                     distance_threshold: float = 0.5,
                                     min_cluster_size: int = 2) -> List[Dict]:
    """
    基于终点位置的简单聚类
    适合同起点、不同终点的轨迹聚类
    """
    if not paths or len(paths) < 2:
        return paths
    
    print("\n" + "="*60)
    print("ENDPOINT-BASED CLUSTERING")
    print("="*60)
    
    # 提取所有终点位置
    endpoints = []
    valid_paths = []
    
    for path in paths:
        shortest_path = path.get('shortest_path', [])
        if shortest_path:
            last_point = shortest_path[-1]
            if isinstance(last_point, ShortestPathPoint):
                endpoint = np.array(last_point.position)
            elif isinstance(last_point, dict):
                endpoint = np.array(last_point['position'])
            else:
                continue
            endpoints.append(endpoint)
            valid_paths.append(path)
    
    if len(endpoints) < min_cluster_size:
        print(f"Not enough valid paths: {len(endpoints)}")
        return paths
    
    endpoints = np.array(endpoints)
    
    # 使用DBSCAN对终点进行聚类
    from sklearn.cluster import DBSCAN
    
    clustering = DBSCAN(
        eps=distance_threshold,
        min_samples=min_cluster_size,
        metric='euclidean'
    ).fit(endpoints)
    
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"Paths: {len(valid_paths)}")
    print(f"Clusters found: {n_clusters}")
    print(f"Noise points: {n_noise}")
    print(f"Distance threshold: {distance_threshold:.2f}m")
    
    # 统计每个聚类的大小
    from collections import Counter
    cluster_sizes = Counter(labels)
    print(f"Cluster sizes: {dict(cluster_sizes)}")
    
    # 对每个聚类，选择代表性路径（例如选择最短的）
    clustered_paths = []
    processed_clusters = set()
    
    for i, label in enumerate(labels):
        if label == -1:
            # 噪声点，保留原始路径
            clustered_paths.append(valid_paths[i])
        elif label not in processed_clusters:
            # 找到该聚类中的所有路径
            cluster_indices = [j for j, l in enumerate(labels) if l == label]
            cluster_paths = [valid_paths[j] for j in cluster_indices]
            
            # 选择该聚类的代表性路径（geodesic距离最短的）
            representative_path = min(
                cluster_paths,
                key=lambda p: p.get('geodesic_distance', float('inf'))
            )
            
            clustered_paths.append(representative_path)
            processed_clusters.add(label)
            
            print(f"Cluster {label}: merged {len(cluster_indices)} paths into 1")
    
    print(f"Result: {len(valid_paths)} paths -> {len(clustered_paths)} paths")
    print("="*60 + "\n")
    
    return clustered_paths
'''

def apply_adaptive_endpoint_clustering_with_target(paths: List[Dict], 
                                                   start_rotation: List[float],
                                                   sim: "HabitatSim",
                                                   target_min_clusters: int = 3,
                                                   target_max_clusters: int = 5,
                                                   min_cluster_size: int = 2) -> List[Dict]:
    """
    自适应调整distance_threshold以达到3-5个簇
    """
    if not paths or len(paths) < target_min_clusters:
        return paths
    
    print("\n" + "="*60)
    print("ADAPTIVE ENDPOINT CLUSTERING (Target: 3-5 clusters)")
    print("="*60)
    
    # 提取所有终点位置
    endpoints = []
    valid_paths = []
    
    for path in paths:
        shortest_path = path.get('shortest_path', [])
        if shortest_path:
            last_point = shortest_path[-1]
            if isinstance(last_point, ShortestPathPoint):
                endpoint = np.array(last_point.position)
            elif isinstance(last_point, dict):
                endpoint = np.array(last_point['position'])
            else:
                continue
            endpoints.append(endpoint)
            valid_paths.append(path)
    
    if len(endpoints) < target_min_clusters:
        print(f"Not enough paths: {len(endpoints)}")
        return paths
    
    endpoints = np.array(endpoints)
    
    # 计算终点之间的距离分布
    from scipy.spatial.distance import pdist
    distances = pdist(endpoints)
    
    print(f"Paths: {len(valid_paths)}")
    print(f"Endpoint distance statistics:")
    print(f"  Min: {np.min(distances):.3f}m")
    print(f"  Max: {np.max(distances):.3f}m")
    print(f"  Mean: {np.mean(distances):.3f}m")
    print(f"  Median: {np.median(distances):.3f}m")
    
    # 尝试不同的eps值，找到最接近目标簇数的
    from sklearn.cluster import DBSCAN
    
    # 更密集的候选eps值（从5th到95th百分位）
    candidate_eps = np.percentile(distances, [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95])
    
    best_eps = None
    best_n_clusters = 0
    best_labels = None
    best_score = float('inf')  # 用于评估与目标的接近程度
    
    print(f"\nTrying different eps values to find optimal clustering...")
    
    for eps in candidate_eps:
        clustering = DBSCAN(
            eps=eps,
            min_samples=min_cluster_size,
            metric='euclidean'
        ).fit(endpoints)
        
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # 计算得分：优先考虑簇数在目标范围内，其次考虑噪声点少
        if target_min_clusters <= n_clusters <= target_max_clusters:
            # 在目标范围内，得分=噪声点占比
            score = n_noise / len(endpoints)
        else:
            # 不在目标范围内，得分=与目标范围的距离 + 噪声点占比
            if n_clusters < target_min_clusters:
                score = (target_min_clusters - n_clusters) * 10 + n_noise / len(endpoints)
            else:
                score = (n_clusters - target_max_clusters) * 10 + n_noise / len(endpoints)
        
        print(f"  eps={eps:.3f}m -> {n_clusters} clusters, {n_noise} noise, score={score:.2f}")
        
        # 选择得分最低的配置
        if score < best_score:
            best_eps = eps
            best_n_clusters = n_clusters
            best_labels = labels
            best_score = score
    
    if best_labels is None:
        print("Failed to find suitable clustering parameters")
        return paths
    
    labels = best_labels
    n_noise = list(labels).count(-1)
    
    print(f"\n✅ Selected eps={best_eps:.3f}m")
    print(f"   Clusters: {best_n_clusters} (target: {target_min_clusters}-{target_max_clusters})")
    print(f"   Noise points: {n_noise}/{len(endpoints)} ({100*n_noise/len(endpoints):.1f}%)")
    
    # 如果噪声点太多（>30%），尝试将它们分配到最近的簇
    if n_noise / len(endpoints) > 0.3:
        print(f"\n⚠️  Too many noise points ({n_noise}), reassigning to nearest clusters...")
        
        # 找到所有非噪声点的簇中心
        cluster_centers = {}
        for cluster_id in set(labels):
            if cluster_id != -1:
                cluster_points = endpoints[labels == cluster_id]
                cluster_centers[cluster_id] = np.mean(cluster_points, axis=0)
        
        # 将噪声点分配到最近的簇
        noise_indices = np.where(labels == -1)[0]
        for idx in noise_indices:
            noise_point = endpoints[idx]
            min_dist = float('inf')
            nearest_cluster = -1
            
            for cluster_id, center in cluster_centers.items():
                dist = np.linalg.norm(noise_point - center)
                if dist < min_dist:
                    min_dist = dist
                    nearest_cluster = cluster_id
            
            if min_dist < best_eps * 2:  # 只重新分配距离不太远的噪声点
                labels[idx] = nearest_cluster
        
        n_noise_after = list(labels).count(-1)
        print(f"   Reassigned {n_noise - n_noise_after} noise points")
        print(f"   Remaining noise: {n_noise_after}")
    
    # 统计簇大小
    from collections import Counter
    cluster_sizes = Counter(labels)
    print(f"Final cluster sizes: {dict(cluster_sizes)}")
    print("="*60 + "\n")
    
    # 选择代表性路径
    clustered_paths = []
    processed_clusters = set()
    
    for i, label in enumerate(labels):
        if label == -1:
            clustered_paths.append(valid_paths[i])
        elif label not in processed_clusters:
            cluster_indices = [j for j, l in enumerate(labels) if l == label]
            cluster_paths = [valid_paths[j] for j in cluster_indices]
            
            representative_path = min(
                cluster_paths,
                key=lambda p: p.get('geodesic_distance', float('inf'))
            )
            
            clustered_paths.append(representative_path)
            processed_clusters.add(label)
            
            print(f"Cluster {label}: merged {len(cluster_indices)} paths into 1")
    
    print(f"Result: {len(valid_paths)} paths -> {len(clustered_paths)} paths\n")
    
    return clustered_paths
'''
def _create_multi_path_episode(#需要修改返回的格式，因为一个episode中可能有多条path
    episode_id: Union[int, str],
    scene_id: str,
    start_position: List[float],
    start_rotation: List[float],
    goals: List[NavigationGoal],
    paths: List[Dict[str, Any]],
    num_targets: float
) -> MultiPathEpisode:
    return MultiPathEpisode(
        episode_id=str(episode_id),
        goals=goals,
        scene_id=scene_id,
        start_position=start_position,
        start_rotation=start_rotation,
        num_targets=num_targets,
        paths=paths,
    )


def generate_multi_path_episode(
    sim: "HabitatSim",
    num_episodes: int = -1,
    is_gen_shortest_path: bool = True,
    shortest_path_success_distance: float = 0.2,
    shortest_path_max_steps: int = 2000,#最大时间步长
    closest_dist_limit: float = 0.1,
    furthest_dist_limit: float = 50,
    geodesic_to_euclid_min_ratio: float = 1.0,#1.1
    trajectory_cluster: bool =False,
) -> Generator[Tuple[MultiPathEpisode, Optional[MultiPathEpisode]], None, None]:
    
    #scene_name = sim.habitat_config.scene.split("/")[-1].split(".")[0]
    
    episode_count = 0
    source_position_count=0
    target_points_count=0
    paths_count=0
    true_episode_count=num_episodes
    source_failure_count=0
    target_failure_count=0
    action_failure_count=0
    while episode_count < num_episodes or num_episodes < 0:
        start_position_height=get_first_floor_height(sim)
        source_position = sample_start_position(start_position_height,sim=sim)
        if source_position is None:
            #print(f"source_position is None")
            #重新进行有限次数的起始点采样
            source_position_count+=1
            if source_position_count>19:
                source_failure_count+=1
                print(f"episode {episode_count}:source failure")
                episode_count+=1
                
            continue
        #print(f"scene {sim.habitat_config.scene} source_position is {source_position}")

        #随机采样起始朝向
        angle = np.random.uniform(0, 2 * np.pi)
        source_rotation = [0.0, np.sin(angle / 2), 0.0, np.cos(angle / 2)]
        #print(f"sampling distance island_radius is {source_position}")
        radius=0.08*sim.island_radius(source_position)
        std_dev=0.05*radius
        sampling_distance=random_gaussian(radius,std_dev)
        target_points,original_navigable_target_count,snapped_navigable_target_count,not_navigable_target_count=sample_target_points(
            source_position,
            sampling_distance,
            sim=sim)#采样路径的尺寸大小
        
        if target_points is None:
            #重新有限次数的采样起始点
            target_points_count+=1
            if target_points_count>19:
                target_failure_count+=1
                print(f"episode {episode_count}:target failure")
                episode_count+=1
            continue
        
        goals=[]
        paths=[]
        num_targets=0
        not_compatible_count=0
        not_gen_path_count=0
    
        for target_position in target_points:
            path={}
            # 检查兼容性
            is_compatible, dist = is_compatible_episode(
                source_position,
                target_position,  
                near_dist=closest_dist_limit,
                far_dist=furthest_dist_limit,
                geodesic_to_euclid_ratio=geodesic_to_euclid_min_ratio,
                sim=sim
            )

            if is_compatible:
                #print(f"scene {sim.habitat_config.scene} episode {episode_count} target {target_position} compatible")
                if is_gen_shortest_path:
                    shortest_path,success = get_action_shortest_path(
                        sim,
                        source_position=source_position,
                        source_rotation=source_rotation,
                        goal_position=target_position,
                        success_distance=shortest_path_success_distance,
                        max_episode_steps=shortest_path_max_steps,
                    )
                    
                    if success:
                        path['goal']= NavigationGoal(position=target_position, radius=shortest_path_success_distance)
                        path['shortest_path']=shortest_path
                        path['geodesic_distance']=dist
                        paths.append(path)
                        goals.append(NavigationGoal(position=target_position, radius=shortest_path_success_distance))
                        #(f"num_targets {num_targets} is success")
                        num_targets+=1
    
                    else:
                        not_gen_path_count+=1
            else:
                not_compatible_count+=1
        if paths:
            original_episode = _create_multi_path_episode(
                episode_id=episode_count,
                scene_id=sim.habitat_config.scene,
                start_position=source_position,
                start_rotation=source_rotation,
                num_targets=num_targets,
                paths=copy.deepcopy(paths),  # 深拷贝保留原始数据
                goals=copy.deepcopy(goals),
            )
            
            clustered_episode = None
            
            if trajectory_cluster:
                logger.info(f"Applying trajectory clustering to {len(paths)} paths")
                # 打印聚类前的轨迹统计
                traj_lengths = []
                for path in paths:
                    if path.get('shortest_path'):
                        traj_lengths.append(len(path['shortest_path']))
                
                if traj_lengths:
                    logger.info(f"Before clustering - trajectory points: min={min(traj_lengths)}, max={max(traj_lengths)}, avg={np.mean(traj_lengths):.1f}")
        
                clustered_paths = apply_simple_endpoint_clustering(
                paths=paths,
                start_rotation=source_rotation,
                sim=sim,
                distance_threshold=0.2,  # 0.5米内的终点视为相同
                min_cluster_size=2        # 至少2条路径才能聚类
                )
            
                logger.info(f"Clustering completed, {len(clustered_paths)} paths remain")
                
                if len(clustered_paths) < len(paths):
                    logger.info(f"Successfully merged {len(paths) - len(clustered_paths)} paths")
                    print(f"✅ Clustering SUCCESS! Original: {len(paths)}, Clustered: {len(clustered_paths)}")
                else:
                    logger.warning("No paths were merged during clustering!")
                    print(f"⚠️  No merging occurred. Original: {len(paths)}, Clustered: {len(clustered_paths)}")
                '''
            if trajectory_cluster:
                logger.info(f"Applying ADAPTIVE clustering to {len(paths)} paths (target: 3-5 clusters)")
                
                # 使用自适应聚类，确保簇数在3-5之间
                clustered_paths = apply_adaptive_endpoint_clustering_with_target(
                    paths=paths,
                    start_rotation=source_rotation,
                    sim=sim,
                    target_min_clusters=3,
                    target_max_clusters=5,
                    min_cluster_size=2
                )
                
                logger.info(f"Clustering completed, {len(clustered_paths)} paths remain")
                
                if len(clustered_paths) < len(paths):
                    logger.info(f"Successfully merged {len(paths) - len(clustered_paths)} paths")
                    print(f"✅ Clustering SUCCESS! Original: {len(paths)}, Clustered: {len(clustered_paths)}")
                else:
                    logger.warning("No paths were merged during clustering!")
                    print(f"⚠️  No merging occurred. Original: {len(paths)}, Clustered: {len(clustered_paths)}")
                '''
                # 创建聚类后的episode
                clustered_episode = _create_multi_path_episode(
                    episode_id=episode_count,
                    scene_id=sim.habitat_config.scene,
                    start_position=source_position,
                    start_rotation=source_rotation,
                    num_targets=len(clustered_paths),
                    paths=clustered_paths,
                    goals=goals,  # goals保持不变
                )
                
            episode_count += 1
            yield (original_episode, clustered_episode)
            
        else:
            #print(f"scene {sim.habitat_config.scene} episode {episode_count} fails")
            #重新有限次数采样起始点
            paths_count+=1
            if paths_count>19:
                action_failure_count+=1
                print(f"episode {episode_count}:action failure")
                episode_count+=1
            continue
        print(f"scene {sim.habitat_config.scene} episode {episode_count-1}\naimed_targets_count:{num_angle_samples}\ntrue_targets_count:{num_targets}\n")
        print(f"sampled_target_count:{original_navigable_target_count+snapped_navigable_target_count}\noriginal_navigable_target_count:{original_navigable_target_count}\nsnapped_navigable_target_count:{snapped_navigable_target_count}\n")
        print(f"not_navigable_target_count:{not_navigable_target_count}\nnot_compatible_count:{not_compatible_count}\nnot_gen_path_count:{not_gen_path_count}\n")
    true_episode_count=num_episodes-(source_failure_count+target_failure_count+action_failure_count)
    print(f"scene {sim.habitat_config.scene}:{true_episode_count}/{num_episodes} true episodes")
    print(f"total failure:{source_failure_count+target_failure_count+action_failure_count}\nsource failure:{source_failure_count}\ntarget failure:{target_failure_count}\naction failure:{action_failure_count}")
