import os
import shutil

# 指定要遍历的目标文件夹
target_directory = "./"
cache = {"__pycache__"}

targets = []

# 定义遍历文件夹的函数
def list_files_in_directory(directory):
    
    for root, dirs, files in os.walk(directory):
        # for file in files:
        #     file_path = os.path.join(root, file)
        #     print("文件:", file_path)
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            if subdir in cache:
                targets.append(subdir_path)
                # print("子文件夹:", subdir_path)
            
            list_files_in_directory(subdir_path)
            
    return targets



# 调用函数进行遍历
cache_dirs = list_files_in_directory(target_directory)
print("子文件夹:")
for directory_to_delete in cache_dirs:
    # 使用 shutil 模块的 rmtree() 函数来递归删除目录及其内容
    try:
        shutil.rmtree(directory_to_delete)
        print(f"成功删除目录: {directory_to_delete}")
    except OSError as e:
        print(f"删除目录失败: {e}")