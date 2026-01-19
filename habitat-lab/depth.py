"""
生成agent起始pose的rgb和depth image
"""
import os
from typing import TYPE_CHECKING, Union, cast

import matplotlib.pyplot as plt
import numpy as np

import habitat
# from habitat.config.default_structured_configs import (
#     CollisionsMeasurementConfig,
#     FogOfWarConfig,
#     TopDownMapMeasurementConfig,
# )
from habitat.core.agent import Agent
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
#from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import (
    #images_to_video,
    observations_to_image,
    #overlay_frame,
)
#from habitat_sim.utils import viz_utils as vut
import tqdm,imageio
# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

if TYPE_CHECKING:
    from habitat.core.simulator import Observations
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim


#agent起始pose的rgb image
output_path_start_rgb_image = "data/pics/start_rgb_image/"
#agent起始pose的depth image
output_path_start_depth_image = "data/pics/start_depth_image/"

for path in [
    output_path_start_rgb_image,
    output_path_start_depth_image,
]:
    os.makedirs(path, exist_ok=True)

class ShortestPathFollowerAgent(Agent):
    r"""Implementation of the :ref:`habitat.core.agent.Agent` interface that
    uses :ref`habitat.tasks.nav.shortest_path_follower.ShortestPathFollower` utility class
    for extracting the action on the shortest path to the goal.
    """

    def __init__(self, env: habitat.Env, goal_radius: float):
        self.env = env
        self.shortest_path_follower = ShortestPathFollower(
            sim=cast("HabitatSim", env.sim),
            goal_radius=goal_radius,
            return_one_hot=False,
        )

    def act(self, observations: "Observations") -> Union[int, np.ndarray]:
        return self.shortest_path_follower.get_next_action(
            cast(NavigationEpisode, self.env.current_episode).goals[0].position
        )

    def reset(self) -> None:
        pass

def rgb_and_depth_image():
    # Create habitat config
    config = habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_gibson.yaml"
    )
    # Add habitat.tasks.nav.nav.TopDownMap and habitat.tasks.nav.nav.Collisions measures
    # with habitat.config.read_write(config):
        # config.habitat.task.measurements.update(
        #     {
        #         "top_down_map": TopDownMapMeasurementConfig(
        #             map_padding=3,
        #             map_resolution=1024,
        #             draw_source=True,
        #             draw_border=True,
        #             draw_shortest_path=True,
        #             draw_view_points=True,
        #             draw_goal_positions=True,
        #             draw_goal_aabbs=True,
        #             fog_of_war=FogOfWarConfig(
        #                 draw=True,
        #                 visibility_dist=5.0,
        #                 fov=90,
        #             ),
        #         ),
        #         "collisions": CollisionsMeasurementConfig(),
        #     }
        # )
    # Create dataset
    dataset = habitat.make_dataset(
        id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
    )
    # Create simulation environment
    with habitat.Env(config=config, dataset=dataset) as env:
        # Create ShortestPathFollowerAgent agent
        agent = ShortestPathFollowerAgent(
            env=env,
            goal_radius=config.habitat.task.measurements.success.success_distance,
        )
        saved_depth_keys = set()
        
        total_episodes = len(dataset.episodes)

        # Create video of agent navigating in the first episode
        #num_episodes = 1
        #for _ in range(num_episodes):
        for _ in tqdm.tqdm(range(total_episodes), desc="Processing episodes"):

            # Load the first episode and reset agent
            observations = env.reset()
            agent.reset()

            # Get metrics
            info = env.get_metrics()

            current_episode = env.current_episode
            #current_ep_index = extract_current_episode_index(current_episode.episode_id)
            scene_name = os.path.basename(current_episode.scene_id).split('.')[0]
            #创建唯一键
            #depth_key = (scene_name, current_ep_index)
            depth_key = scene_name
            if depth_key not in saved_depth_keys:
                rgb_only_obs = {"rgb": observations["rgb"]}
                start_rgb_image_frame = observations_to_image(rgb_only_obs, {})
                depth_only_obs = {"depth": observations["depth"]}
                start_depth_image_frame = observations_to_image(depth_only_obs, {})

                '''存储start_rgb_image'''
                start_rgb_image_name = f"{scene_name}_start_rgb_image.png"
                start_rgb_image_path = os.path.join(output_path_start_rgb_image, start_rgb_image_name)
                imageio.imwrite(start_rgb_image_path, start_rgb_image_frame)

                '''存储start_depth_image'''
                start_depth_image_name = f"{scene_name}_start_depth_image.png"
                start_depth_image_path = os.path.join(output_path_start_depth_image, start_depth_image_name)
                imageio.imwrite(start_depth_image_path, start_depth_image_frame)
                
                '''存储原始depth数据 (numpy数组)'''
                start_depth_raw_name = f"{scene_name}_start_depth_raw.npy"
                start_depth_raw_path = os.path.join(output_path_start_depth_image, start_depth_raw_name)
                np.save(start_depth_raw_path, observations["depth"])
                
                #存储唯一键
                saved_depth_keys.add(depth_key)
            # # Concatenate RGB-D observation and topdowm map into one image
            # frame = observations_to_image(observations, info)

            # # Remove top_down_map from metrics
            # info.pop("top_down_map")
            # # Overlay numeric metrics onto frame
            # frame = overlay_frame(frame, info)
            # # Add fame to vis_frames
            # vis_frames = [frame]

            # # Repeat the steps above while agent doesn't reach the goal
            # while not env.episode_over:
            #     # Get the next best action
            #     action = agent.act(observations)
            #     if action is None:
            #         break

            #     # Step in the environment
            #     observations = env.step(action)
            #     info = env.get_metrics()
            #     frame = observations_to_image(observations, info)

            #     info.pop("top_down_map")
            #     frame = overlay_frame(frame, info)
            #     vis_frames.append(frame)

if __name__ == "__main__":
    rgb_and_depth_image()