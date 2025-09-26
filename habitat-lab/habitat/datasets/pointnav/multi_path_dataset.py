#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os
import pickle
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from habitat.config import read_write
from habitat.core.dataset import ALL_SCENES_MASK, Dataset
from habitat.core.registry import registry
from habitat.datasets.pointnav.multi_path_generator import MultiPathEpisode,NavigationPath
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    ShortestPathPoint,
)
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1

if TYPE_CHECKING:
    from omegaconf import DictConfig


CONTENT_SCENES_PATH_FIELD = "content_scenes_path"
DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"


@registry.register_dataset(name="MultiPath-v1")
class MultiPathNavDatasetV1(PointNavDatasetV1):
    episodes: List[MultiPathEpisode]
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"
   
    @staticmethod
    def check_config_paths_exist(config: "DictConfig") -> bool:
        return os.path.exists(
            config.data_path.format(split=config.split)
        ) and os.path.exists(config.scenes_dir)

    @classmethod
    def get_scenes_to_load(cls, config: "DictConfig") -> List[str]:
        r"""Return list of scene ids for which dataset has separate files with
        episodes.
        """
        dataset_dir = os.path.dirname(
            config.data_path.format(split=config.split)
        )
        if not cls.check_config_paths_exist(config):
            raise FileNotFoundError(
                f"Could not find dataset file `{dataset_dir}`"
            )

        cfg = config.copy()
        with read_write(cfg):
            cfg.content_scenes = []
            dataset = cls(cfg)
            has_individual_scene_files = os.path.exists(
                dataset.content_scenes_path.split("{scene}")[0].format(
                    data_path=dataset_dir
                )
            )
            if has_individual_scene_files:
                return cls._get_scenes_from_folder(
                    content_scenes_path=dataset.content_scenes_path,
                    dataset_dir=dataset_dir,
                )
            else:
                # Load the full dataset, things are not split into separate files
                cfg.content_scenes = [ALL_SCENES_MASK]
                dataset = cls(cfg)
                return list(map(cls.scene_from_scene_path, dataset.scene_ids))

    @staticmethod
    def _get_scenes_from_folder(
        content_scenes_path: str, dataset_dir: str
    ) -> List[str]:
        scenes: List[str] = []
        content_dir = content_scenes_path.split("{scene}")[0]
        scene_dataset_ext = content_scenes_path.split("{scene}")[1]
        content_dir = content_dir.format(data_path=dataset_dir)
        if not os.path.exists(content_dir):
            return scenes

        for filename in os.listdir(content_dir):
            if filename.endswith(scene_dataset_ext):
                scene = filename[: -len(scene_dataset_ext)]
                scenes.append(scene)
        scenes.sort()
        return scenes

    def _load_from_file(self, fname: str, scenes_dir: str) -> None:
        """
        Load the data from a file into `self.episodes`. This can load `.pickle`
        or `.json.gz` file formats.
        """

        with gzip.open(fname, "rt") as f:
            self.from_json(f.read(), scenes_dir=scenes_dir)

    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        self.episodes = []

        if config is None:
            return

        datasetfile_path = config.data_path.format(split=config.split)

        self._load_from_file(datasetfile_path, config.scenes_dir)

        # Read separate file for each scene
        dataset_dir = os.path.dirname(datasetfile_path)
        has_individual_scene_files = os.path.exists(
            self.content_scenes_path.split("{scene}")[0].format(
                data_path=dataset_dir
            )
        )
        if has_individual_scene_files:
            scenes = config.content_scenes
            if ALL_SCENES_MASK in scenes:
                scenes = self._get_scenes_from_folder(
                    content_scenes_path=self.content_scenes_path,
                    dataset_dir=dataset_dir,
                )

            for scene in scenes:
                scene_filename = self.content_scenes_path.format(
                    data_path=dataset_dir, scene=scene
                )

                self._load_from_file(scene_filename, config.scenes_dir)

        else:
            self.episodes = list(
                filter(self.build_content_scenes_filter(config), self.episodes)
            )
   
    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        for episode in deserialized["episodes"]:
            multi_episode = MultiPathEpisode(**episode)

            if scenes_dir is not None:
                if multi_episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    multi_episode.scene_id = multi_episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                multi_episode.scene_id = os.path.join(scenes_dir, multi_episode.scene_id)
                
            for g_index, goal in enumerate(multi_episode.goals):
                multi_episode.goals[g_index] = NavigationGoal(**goal)
                #print(episode.goals)
            for p_index,path in enumerate(multi_episode.paths):
                multi_episode.paths[p_index] = NavigationPath(**path) 
                #print(multi_episode.paths)
            #print(multi_episode)
            
            #将MultiPathEpisode转换为NavigationEpisode
            episodes=self.convert_episodes(multi_episode)
            self.episodes.extend(episodes)
            #print(self.episodes)

    def convert_episodes(self, multi_episode: MultiPathEpisode) -> List[NavigationEpisode]:
        
        navigation_episodes = []
        num_targets = multi_episode.num_targets
        goals = multi_episode.goals
        paths = multi_episode.paths
        
        # 如果goals或paths数量不足，将num_targets改成len(goals)和len(paths)中的最小值
        actual_targets = min(len(goals), len(paths), num_targets)
        if actual_targets<num_targets:
            print(f"num_targets:{num_targets}, actual targets:{actual_targets}")
    
        for i in range(actual_targets):
            new_episode_id = f"{multi_episode.episode_id}_{i}"
            
            current_goal = goals[i] 
            current_goals = [current_goal] 
            
            current_shortest_paths = None
            if hasattr(paths[i], 'shortest_path'):
                if paths[i].shortest_path:
                    #检验path[i].goal和current_goal是否一致
                    path_goal = paths[i].goal
                
                    path_x = path_goal['position'][0]
                    goal_x = current_goal.position[0]
                    
                    # 比较x坐标是否相等
                    if path_x != goal_x:
                        print(f"path & goal do not match")
                        continue
                    
                    shortest_path_points = []
                    for point_data in paths[i].shortest_path:
                        if isinstance(point_data, dict):
                            point = ShortestPathPoint(
                                position=point_data["position"],
                                rotation=point_data["rotation"],
                                action=point_data["action"]
                            )
                        else:
                            point = point_data
                        shortest_path_points.append(point)
                    #current_shortest_paths = [shortest_path_points]
            
            navigation_episode = NavigationEpisode(
                episode_id=new_episode_id,
                scene_id=multi_episode.scene_id,
                scene_dataset_config=multi_episode.scene_dataset_config,
                additional_obj_config_paths=multi_episode.additional_obj_config_paths,
                start_position=multi_episode.start_position,
                start_rotation=multi_episode.start_rotation,
                goals=current_goals,
                shortest_paths=shortest_path_points,
            )
            #print(navigation_episode)
            navigation_episodes.append(navigation_episode)

        return navigation_episodes      

        