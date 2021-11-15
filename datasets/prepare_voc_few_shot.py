"""
基于VOC，生成多组（29组）不同seed的few-shot dataset

代码逻辑如下：
遍历训练集（07、12的train和val合起来）的标注。对于每个class，知道哪些图片包含（至少有1个）属于该class的object
对于某个seed、某个class的多个k-shot：3-shot dataset包括1-shot dataset、5-shot dataset包括3-shot dataset，以此类推……
对于某个seed、某个class、某个k-shot（以5-shot为例）：
    基于上个shot（3-shot）选取的图片（m张图片，最多3张，可以少于3张，最少1张；n个object，最少3个，最多不限量）。Note：这里有个bug，详见代码（可搜索TODO）
    先再随机（random seed为当前seed）选取diff_shot张（5-3=2张）图片（因为每张图片至少有1个属于该class的object，所以最多使用diff_shot=2张图片就能得到至少3+2=5个object）。
    遍历这diff_shot=2张图片（不一定全部遍历完），先将当前所遍历到的图片纳入该class的few-shot dataset，然后判断目前所纳入的图片中属于该class的object的数量是否大于等于diff_shot，是的话则不再遍历后续图片（最多遍历diff_shot张图片）
"""

import argparse
import copy
import os
import random
import xml.etree.ElementTree as ET

import numpy as np
from fsdet.utils.file_io import PathManager

# VOC的20个class
VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor']  # fmt: skip


def parse_args():
    parser = argparse.ArgumentParser()
    # 生成多少组few-shot dataset，range(1,30)即生成1到29
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1, 30], help="Range of seeds"
    )
    args = parser.parse_args()
    return args


def generate_seeds(args):
    #### VOC2007、VOC2012中train set和val set的图片的id
    data = []
    for year in [2007, 2012]:
        data_file = "datasets/VOC{}/ImageSets/Main/trainval.txt".format(year) # 该文件中保存了train set和val set中图片的id
        with PathManager.open(data_file) as f: # 打开文件
            fileids = np.loadtxt(f, dtype=np.str).tolist() # 读取图片的id保存在list中
        data.extend(fileids)


    #### 每个class对应的标注文件路径（如果某图片中有属于class c的object，则该图片的标注文件的路径就会存入该class c的list）
    data_per_cat = {c: [] for c in VOC_CLASSES}
    for fileid in data: # 遍历所有图片的id
        # 读取对应标注（xml）文件
        year = "2012" if "_" in fileid else "2007" # 年份（VOC2007的图片id例如“009946”，VOC2012的图片id例如“2011_000971”）
        dirname = os.path.join("datasets", "VOC{}".format(year)) # 文件夹路径
        anno_file = os.path.join(dirname, "Annotations", fileid + ".xml") # 标注文件路径
        tree = ET.parse(anno_file) # 读取xml文件
        # 该图片中有哪些class的object
        clses = []
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            clses.append(cls)
        # 该图片中有属于该class的object，则将该图片的标注文件路径存入该class对应的list
        for cls in set(clses):
            data_per_cat[cls].append(anno_file)


    #### 生成所有seed情况下所有class的所有k-shot dataset
    result = {cls: {} for cls in data_per_cat.keys()} # 双层dict，外层dict的key为cls、内层dict的key为k(-shot)、内层dict的value为对应所有图片的文件路径（list）
    shots = [1, 2, 3, 5, 10] # k(-shots)：1, 2, 3, 5, 10
    for i in range(args.seeds[0], args.seeds[1]): # 遍历每个seed
        random.seed(i) # 设置seed
        # 生成当前seed情况下所有class的所有k-shot dataset
        for c in data_per_cat.keys(): # 遍历每个class
            c_data = [] # 该class对应的所有（不超过max_k个）图片（文件路径）
            ## IMPORTANT: 生成当前seed情况下当前class的所有k-shot dataset
            for j, shot in enumerate(shots): # 遍历每个k(-shot)
                ## 先采样一组（diff_shot个）图片（但不一定都纳入few-shot dataset）。因为每张图片中至少有1个属于该class的object，所以这组图片至少包含diff_shot个属于该class的object
                diff_shot = shots[j] - shots[j - 1] if j != 0 else 1 # diff_shot: 当前k-shot比上个k-shot多多少
                shots_c = random.sample(data_per_cat[c], diff_shot) # 采样diff_shot张图片（标注文件），
                
                ## IMPORTANT：遍历这diff_shot张图片并先将当前所遍历到的图片纳入该class的few-shot dataset，然后判断目前所纳入的图片中属于该class的object的数量大于等于diff_shot（如果是，则停止遍历）
                num_objs = 0 # 当目前新纳入的图片中属于该class的object的数量大于等于diff_shot，就肯定够k个object了，不再遍历后续图片
                for s in shots_c: # 遍历每张图片
                    # TODO：这行代码应该是错了，s是标注文件路径，c_data是图片文件路径的list，怎么可能相等呢？这行代码本意应该是避免重复采样相同图片或避免重复采样相同标注
                    # TODO：如果本意是避免重复采样相同图片/标注，那如果新采样的所有 图片/标注 在之前都已经被采样过，那按这个逻辑走下去，当前的k-shot dataset中object数量会不足k个，所以这是个bug。
                    # TODO：更好的方式是像`prepare_coco_few_shot.py`那样，先预采样k个object，无论是否基于上个shot生成当前shot的dataset。
                    if s not in c_data:
                        tree = ET.parse(s) # 读取该图片的标注文件
                        file = tree.find("filename").text # 如"009392.jpg"
                        year = tree.find("folder").text # 如"VOC2007"
                        name = "datasets/{}/JPEGImages/{}".format(year, file) # 该图片的文件路径
                        c_data.append(name) # 将该图片纳入该class的few-shot dataset
                        for obj in tree.findall("object"): # 遍历该图片中的所有object，统计属于该class的object的数量
                            if obj.find("name").text == c:
                                num_objs += 1
                        if num_objs >= diff_shot: # IMPORTANT: 当目前新纳入的图片中属于该class的object的数量大于等于diff_shot，就肯定够k个object了，不再遍历后续图片
                            break
                result[c][shot] = copy.deepcopy(c_data) # 某class某k-shot对应的所有图片文件的路径。双层dict，外层dict的key为cls、内层dict的key为k(-shot)、内层dict的value为对应所有图片的文件路径（list）
        
        # 保存当前seed情况下所有class的所有k-shot dataset
        save_path = "datasets/vocsplit/seed{}".format(i)
        os.makedirs(save_path, exist_ok=True) # 创建该seed对应的文件夹
        for c in result.keys(): # 遍历所有class
            for shot in result[c].keys(): # 遍历所有k(-shot)
                filename = "box_{}shot_{}_train.txt".format(shot, c) # 文件路径
                # 当前seed情况下该class的该k-shot dataset（图片数量不固定）
                with open(os.path.join(save_path, filename), "w") as fp:
                    fp.write("\n".join(result[c][shot]) + "\n") # 每行为1个图片的文件路径
        print('Generated : ', save_path)


if __name__ == "__main__":
    args = parse_args()
    generate_seeds(args)
