from typing import Dict, Generator, List, Optional, Sequence, Tuple, Union, Any

import numpy as np

from habitat.core.simulator import ShortestPathPoint
#from habitat.datasets.utils import get_action_shortest_path
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

import logging
logger = logging.getLogger(__name__)

ISLAND_RADIUS_LIMIT = 1.#最小岛屿半径限制
num_angle_samples=20#采样目标点个数
#sampling_distance=5.

def get_action_shortest_path(
    sim: "HabitatSim",
    source_position: List[float],
    source_rotation: List[float],
    goal_position: List[float],
    success_distance: float = 0.05,
    max_episode_steps: int = 500,
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
) -> Generator[MultiPathEpisode, None, None]:
    
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
        target_points,original_navigable_target_count,snapped_navigable_target_count,not_navigable_target_count=sample_target_points(source_position, sampling_distance=0.08*sim.island_radius(source_position),sim=sim)#采样路径的尺寸大小
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
            episode = _create_multi_path_episode(
                episode_id=episode_count,
                scene_id=sim.habitat_config.scene,
                start_position=source_position,
                start_rotation=source_rotation,
                num_targets=num_targets,
                paths=paths,
                goals=goals,
            )
            
            #print(f"scene {sim.habitat_config.scene} episode {episode_count} succeeds")
            episode_count+=1
            yield episode
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
