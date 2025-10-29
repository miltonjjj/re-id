#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import gzip
import json
import argparse
import os
from os import path as osp

import tqdm

import habitat
from habitat.config.default import get_agent_config
from habitat.datasets.pointnav.multi_path_generator import (
    generate_multi_path_episode,
)

from habitat.core.registry import registry
from habitat.datasets.pointnav.multi_path_dataset import MultiPathNavDatasetV1

registry.register_dataset(name="MultiPath-v1")(MultiPathNavDatasetV1)

NUM_EPISODES_PER_SCENE = int(10)#int(1e4)
QUAL_THRESH = 2                   


def safe_mkdir(path):
    
    try:
        os.mkdir(path)
    except OSError:
        pass

'''
def _generate_fn(scene,trajectory_cluster):#生成单一场景的episodes
    
    cfg = habitat.get_config(
        "benchmark/nav/pointnav/pointnav_habitat_test.yaml"
    )

    with habitat.config.read_write(cfg):
        cfg.habitat.simulator.scene = scene

        agent_config = get_agent_config(cfg.habitat.simulator)
        agent_config.sim_sensors.clear() 

    sim = habitat.sims.make_sim("Sim-v0", config=cfg.habitat.simulator)

    dset = habitat.datasets.make_dataset("MultiPath-v1")

    dset.episodes = list(
        generate_multi_path_episode(
            sim, 
            NUM_EPISODES_PER_SCENE,     
            is_gen_shortest_path=True,
            trajectory_cluster=trajectory_cluster,   
        )
    )
    print(f"each scene should collect {NUM_EPISODES_PER_SCENE} episodes")

    
    for ep in dset.episodes:
        ep.scene_id = ep.scene_id[len("./data/scene_datasets/") :]

    scene_key = scene.split("/")[-1].split(".")[0]
    
    #out_file = (
    #    f"./data/datasets/pointnav/gibson/v2/multi_path/content/"
    #    f"{scene_key}.json.gz"
    #)
    # 根据是否聚类选择不同的输出路径
    if trajectory_cluster:
        out_file = (
            f"./data/datasets/pointnav/gibson/v2/multi_path/content/"
            f"{scene_key}_cluster.json.gz"
        )
    else:
        out_file = (
            f"./data/datasets/pointnav/gibson/v2/multi_path/content/"
            f"{scene_key}.json.gz"
        )

    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with gzip.open(out_file, "wt") as f:
        f.write(dset.to_json()) 
'''
def _generate_fn(scene, trajectory_cluster):
    cfg = habitat.get_config(
        "benchmark/nav/pointnav/pointnav_habitat_test.yaml"
    )

    with habitat.config.read_write(cfg):
        cfg.habitat.simulator.scene = scene
        agent_config = get_agent_config(cfg.habitat.simulator)
        agent_config.sim_sensors.clear() 

    sim = habitat.sims.make_sim("Sim-v0", config=cfg.habitat.simulator)

    # 创建两个数据集：原始和聚类后
    dset_original = habitat.datasets.make_dataset("MultiPath-v1")
    dset_clustered = habitat.datasets.make_dataset("MultiPath-v1") if trajectory_cluster else None

    original_episodes = []
    clustered_episodes = []

    for original_ep, clustered_ep in generate_multi_path_episode(
        sim, 
        NUM_EPISODES_PER_SCENE,     
        is_gen_shortest_path=True,
        trajectory_cluster=trajectory_cluster,   
    ):
        original_episodes.append(original_ep)
        if clustered_ep is not None:
            clustered_episodes.append(clustered_ep)
    
    dset_original.episodes = original_episodes
    print(f"Scene {scene}: Generated {len(original_episodes)} original episodes")

    # 处理scene_id
    for ep in dset_original.episodes:
        ep.scene_id = ep.scene_id[len("./data/scene_datasets/"):]

    scene_key = scene.split("/")[-1].split(".")[0]
    
    # 保存原始episode
    out_file_original = (
        f"./data/datasets/pointnav/gibson/v2/multi_path/content/"
        f"{scene_key}_original.json.gz"
    )
    os.makedirs(osp.dirname(out_file_original), exist_ok=True)
    with gzip.open(out_file_original, "wt") as f:
        f.write(dset_original.to_json())
    print(f"Saved original episodes to {out_file_original}")

    # 如果有聚类，保存聚类后的episode
    if trajectory_cluster and clustered_episodes:
        dset_clustered.episodes = clustered_episodes
        
        for ep in dset_clustered.episodes:
            ep.scene_id = ep.scene_id[len("./data/scene_datasets/"):]
        
        out_file_clustered = (
            f"./data/datasets/pointnav/gibson/v2/multi_path/content/"
            f"{scene_key}_clustered.json.gz"
        )
        with gzip.open(out_file_clustered, "wt") as f:
            f.write(dset_clustered.to_json())
        print(f"Saved clustered episodes to {out_file_clustered}")
        print(f"Scene {scene}: Generated {len(clustered_episodes)} clustered episodes")
'''
def generate_multi_path_dataset(trajectory_cluster=False):#多进程并行收集数据
    
    with open(
        osp.join(osp.dirname(__file__), "gibson_dset_with_qual.json"), "r"
    ) as f:
        dataset_statistics = json.load(f)

    gibson_large_scene_keys = []
    for k, v in dataset_statistics.items():
        qual = v["qual"] 
        
        if (
            v["split_full+"] == "train"
            and qual is not None
            and qual >= QUAL_THRESH
        ):
            gibson_large_scene_keys.append(k)
    #print(f"keys are: {gibson_large_scene_keys}")

    scenes = glob.glob("./data/scene_datasets/gibson/gibson/*.glb")
    #print(f"find the scenes: {scenes}")

    _fltr = lambda x: x.split("/")[-1].split(".")[0] in gibson_large_scene_keys
    scenes = list(filter(_fltr, scenes))
    scenes = sorted(scenes)
    scenes = scenes[:8]#必须超过pool数量
    
    print(f"Total number of training scenes: {len(scenes)}")

    safe_mkdir("./data/datasets/pointnav/gibson/v2/multi_path")
    
    #多进程
    with multiprocessing.Pool(8) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
        #for _ in pool.imap_unordered(_generate_fn, scenes):
        #    pbar.update()
        # 使用functools.partial传递额外参数
        from functools import partial
        generate_fn_with_cluster = partial(_generate_fn, trajectory_cluster=trajectory_cluster)
        for _ in pool.imap_unordered(generate_fn_with_cluster, scenes):
            pbar.update() 
    
    #单进程
    import tqdm
    for scene in tqdm.tqdm(scenes):
        _generate_fn(scene, trajectory_cluster)

    path = "./data/datasets/pointnav/gibson/v2/multi_path/multi_path.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump(dict(episodes=[]), f)
'''

def generate_multi_path_dataset(trajectory_cluster=False,multiprocess=False):
    """
    多进程或单进程并行收集数据
    """
    
    with open(
        osp.join(osp.dirname(__file__), "gibson_dset_with_qual.json"), "r"
    ) as f:
        dataset_statistics = json.load(f)

    gibson_large_scene_keys = []
    for k, v in dataset_statistics.items():
        qual = v["qual"] 
        
        if (
            v["split_full+"] == "train"
            and qual is not None
            and qual >= QUAL_THRESH
        ):
            gibson_large_scene_keys.append(k)

    scenes = glob.glob("./data/scene_datasets/gibson/gibson/*.glb")

    _fltr = lambda x: x.split("/")[-1].split(".")[0] in gibson_large_scene_keys
    scenes = list(filter(_fltr, scenes))
    scenes = sorted(scenes)
    scenes = scenes[:8]  # 必须超过pool数量
    
    print(f"Total number of training scenes: {len(scenes)}")
    print(f"Trajectory clustering: {'Enabled' if trajectory_cluster else 'Disabled'}")

    safe_mkdir("./data/datasets/pointnav/gibson/v2/multi_path")
    safe_mkdir("./data/datasets/pointnav/gibson/v2/multi_path/content")
    
    # 选择多进程或单进程模式
    #USE_MULTIPROCESS = False  # 设置为True使用多进程，False使用单进程
    
    if multiprocess:
        print("Using multiprocessing mode...")
        import multiprocessing
        from functools import partial
        
        # 使用functools.partial传递额外参数
        generate_fn_with_cluster = partial(_generate_fn, trajectory_cluster=trajectory_cluster)
        
        with multiprocessing.Pool(8) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
            for _ in pool.imap_unordered(generate_fn_with_cluster, scenes):
                pbar.update()
    else:
        print("Using single process mode...")
        for scene in tqdm.tqdm(scenes):
            _generate_fn(scene, trajectory_cluster)

    # 创建空的主数据集文件（用于兼容性）
    if trajectory_cluster:
        # 创建两个主文件
        path_original = "./data/datasets/pointnav/gibson/v2/multi_path/multi_path_original.json.gz"
        path_clustered = "./data/datasets/pointnav/gibson/v2/multi_path/multi_path_clustered.json.gz"
        
        with gzip.open(path_original, "wt") as f:
            json.dump(dict(episodes=[]), f)
        
        with gzip.open(path_clustered, "wt") as f:
            json.dump(dict(episodes=[]), f)
        
        print(f"\nDataset generation completed!")
        print(f"Generated files:")
        print(f"  - Original paths: ./data/datasets/pointnav/gibson/v2/multi_path/content/*_original.json.gz")
        print(f"  - Clustered paths: ./data/datasets/pointnav/gibson/v2/multi_path/content/*_clustered.json.gz")
    else:
        path = "./data/datasets/pointnav/gibson/v2/multi_path/multi_path.json.gz"
        with gzip.open(path, "wt") as f:
            json.dump(dict(episodes=[]), f)
        
        print(f"\nDataset generation completed!")
        print(f"Generated files:")
        print(f"  - Original paths: ./data/datasets/pointnav/gibson/v2/multi_path/content/*_original.json.gz")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multi-path navigation dataset")
    parser.add_argument(
        "--trajectory_cluster",
        action="store_true",
        default=False,
        help="Enable trajectory clustering using TRACLUS algorithm"
    )
    parser.add_argument(
        "--multiprocess",
        action="store_true",
        default=False,
        help="Use multiprocessing to speed up dataset generation"
    )
    args = parser.parse_args()
    
    generate_multi_path_dataset(
        trajectory_cluster=args.trajectory_cluster,
        multiprocess=args.multiprocess)
    #generate_multi_path_dataset()