"""
批量生成Voronoi地图可视化
结合 depth.py 和 voronoi_map.py 的功能
每个episode创建一个文件夹，包含depth、rgb和voronoi图
"""

import os
import numpy as np
import habitat
from typing import cast
import tqdm
import imageio

# 导入voronoi_map模块
from voronoi_map import depth_to_voronoi, visualize_voronoi

# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


def batch_generate_voronoi_maps(output_dir='data/voronoi_map'):
    """
    批量生成Voronoi地图可视化
    每个episode创建独立文件夹，包含depth.npy, rgb.png, voronoi.png
    
    参数:
        output_dir: 输出根目录路径
    """
    # 创建输出目录
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
        
        #for ep_idx in tqdm.tqdm(range(total_episodes), desc="处理episodes"):
        for ep_idx in tqdm.tqdm(range(min(15, total_episodes)), desc="处理episodes"):    
            # 重置环境到当前episode
            observations = env.reset()
            
            # 获取当前episode信息
            current_episode = env.current_episode
            scene_name = os.path.basename(current_episode.scene_id).split('.')[0]
            
            # 场景内episode编号从1递增
            if scene_name not in scene_episode_counter:
                scene_episode_counter[scene_name] = 1
            else:
                scene_episode_counter[scene_name] += 1
            episode_num = scene_episode_counter[scene_name]
            
            # 创建episode文件夹
            episode_folder_name = f"{scene_name}_ep{episode_num}"
            episode_folder_path = os.path.join(output_dir, episode_folder_name)
            
            # 定义文件的路径
            depth_path = os.path.join(episode_folder_path, "depth.npy")
            depth_png_path = os.path.join(episode_folder_path, "depth.png")
            rgb_path = os.path.join(episode_folder_path, "rgb.png")
            voronoi_path = os.path.join(episode_folder_path, "voronoi.png")
            
            # 检查文件夹是否已存在且包含所有文件（避免重复处理）
            if os.path.exists(episode_folder_path):
                if (os.path.exists(depth_path) and 
                    os.path.exists(depth_png_path) and
                    os.path.exists(rgb_path) and 
                    os.path.exists(voronoi_path)):
                    skipped_count += 1
                    continue
            
            # 创建episode文件夹
            os.makedirs(episode_folder_path, exist_ok=True)
            
            try:
                # 1. 保存RGB图像
                rgb_obs = observations["rgb"]
                imageio.imwrite(rgb_path, rgb_obs)
                
                # 2. 获取并预处理深度数据
                depth_obs = observations["depth"]
                
                # 保存原始深度数据
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
                
                # 3. 生成Voronoi图
                results = depth_to_voronoi(depth_cm)
                visualize_voronoi(results, save_path=voronoi_path)
                
                processed_count += 1
                
                # 每处理10个输出一次进度
                if processed_count % 5 == 0:
                    print(f"\n已处理: {processed_count}/{total_episodes}, "
                          f"跳过: {skipped_count}, "
                          f"当前: {episode_folder_name}")
                
            except Exception as e:
                print(f"\n错误 - {episode_folder_name}: {e}")
                # 删除可能部分写入的文件夹
                if os.path.exists(episode_folder_path):
                    import shutil
                    shutil.rmtree(episode_folder_path)
                continue
    
    # 最终统计
    print("\n" + "="*70)
    print("处理完成!")
    print(f"  总episodes: {total_episodes}")
    print(f"  成功处理: {processed_count}")
    print(f"  跳过(已存在): {skipped_count}")
    print(f"  失败: {total_episodes - processed_count - skipped_count}")
    print(f"  输出目录: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='批量生成Voronoi地图')
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='data/voronoi_map',
        help='输出根目录'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("批量Voronoi地图生成系统（按Episode组织）")
    print("="*70)
    
    batch_generate_voronoi_maps(output_dir=args.output_dir)
    
    print("\n✓ 全部完成！")