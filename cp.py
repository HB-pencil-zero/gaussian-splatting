import os
import shutil

source_dir = '/root/autodl-tmp/PhysGaussian/gaussian-splatting/data/ICCV_2025/watermelon_game_ready__2k_pbr'
target_dir = '/root/autodl-tmp/PhysGaussian/gaussian-splatting/data/process_data/watermelon_game_ready__2k_pbr/images'

# 确保目标目录存在
os.makedirs(target_dir, exist_ok=True)

# 遍历源目录中的文件
for filename in os.listdir(source_dir):
    if filename.endswith('.png'):
        # 从原文件名中提取数字
        num = int(filename.split('.')[0])
        # 创建新的6位数文件名
        new_filename = f'{num:06d}.png'
        
        # 构建完整的源路径和目标路径
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(target_dir, new_filename)
        
        # 复制文件
        shutil.copy2(src_path, dst_path)
