# import numpy as np
# import os
#
# data_filenames_list = ['D:\\Files\\OneDrive - stu.hit.edu.cn\\codes\\python\\MSIM-MPSS-tiff\\data\\raw_images\\0.tiff']
# mem_path = os.path.splitext(data_filenames_list[0])[0]
# mem_path = mem_path+".dat"
#
# print(mem_path)

import os
import shutil


def delete_items(items):
    for item in items:
        if os.path.isfile(item):
            os.remove(item)
        elif os.path.isdir(item):
            shutil.rmtree(item)


# 示例用法
items_to_delete = ['file1.txt', 'folder1']
delete_items(items_to_delete)
