from PIL import Image
import os

def change_black_background_to_white(folder_path):
    if not os.path.exists(folder_path):
        print(f"文件夹路径不存在: {folder_path}")
        return

    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            img = Image.open(file_path)
            
            # 确保图像是 RGBA 模式
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            
            # 获取图像数据
            data = img.getdata()
            
            # 创建一个新的图像数据列表
            new_data = []
            for item in data:
                # 如果像素是黑色（或接近黑色），则改为白色
                if item[0] >190  and item[1] > 190 and item[2] > 190:  # 接近黑色的阈值
                    new_data.append((0, 0, 0, item[3]))  # 改为白色，保留透明度
                else:
                    new_data.append(item)  # 其他颜色保持不变
            
            # 更新图像数据
            img.putdata(new_data)
            
            # 保存修改后的图像
            img.save(file_path, "PNG")
            print(f"已处理并保存: {filename}")

# 使用示例
folder_path = "/root/autodl-tmp/PhysGaussian/gaussian-splatting/dataset/train"
change_black_background_to_white(folder_path)