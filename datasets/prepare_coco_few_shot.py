"""
基于COCO，生成多组（9组）不同seed的few-shot dataset

代码逻辑如下：
对于某个seed、某个class的多个k-shot：3-shot dataset包括1-shot dataset、5-shot dataset包括3-shot dataset，以此类推……
对于某个seed、某个class、某个k-shot（以5-shot为例）：
    基于上个shot（3-shot）选取的图片（m张图片，最多3张，可以少于3张，最少1张；3个object）
    先随机（random seed为当前seed）选取k=5张图片（因为每张图片至少有1个属于该class的object，所以最多使用5张图片就能得到至少5个object）
    遍历这k=5张包含属于该class的object的图片（不一定遍历完所有图片，持续遍历直到刚刚好采样到k个属于该class的object）
        如果如果当前图片中的某个object已被采样过，则跳过这张图片；
        如果 该图片中属于该class的object数量 + 已采样属于该class的object数量 > 需采样属于该class的标注数量(k)，则跳过该图片
        如果不是以上两种情况中的任意一种，则采样该图片和其中所有属于该class的object
"""

import argparse
import json
import os
import random

def parse_args():
    parser = argparse.ArgumentParser()
    # 生成多少组few-shot dataset，range(1,10)即生成1到9
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1, 10], help="Range of seeds"
    )
    args = parser.parse_args()
    return args


def generate_seeds(args):
    #### 读取所使用的COCO训练集
    # 训练集
    data_path = "datasets/cocosplit/datasplit/trainvalno5k.json"
    data = json.load(open(data_path))
    # 各个class（类型为dict）
    new_all_cats = []
    for cat in data["categories"]:
        new_all_cats.append(cat) # 每个dict如{'supercategory': 'person', 'id': 1, 'name': 'person'}
    # 图片id对应的图片（类型为dict）
    id2img = {} # dict，key为img_id，value为img(dict)
    for i in data["images"]:
        id2img[i["id"]] = i # 每个dict如{'license': 5, 'file_name': 'COCO_train2014_00000...057870.jpg', 'coco_url': 'http://images.cocoda...057870.jpg', 'height': 480, 'width': 640, 'date_captured': '2013-11-14 16:28:13', 'flickr_url': 'http://farm4.staticf...0b83_z.jpg', 'id': 57870}
    # 各个class对应的所有标注（类型为list[dict]）
    anno = {i: [] for i in ID2CLASS.keys()} # dict，key为cls_id，value为标注(list[dict])
    for a in data["annotations"]:
        if a["iscrowd"] == 1: # 忽略crowd标注
            continue
        anno[a["category_id"]].append(a) # 每个dict如{'segmentation': [[...]], 'area': 54652.9556, 'iscrowd': 0, 'image_id': 480023, 'bbox': [116.95, 305.86, 285.3, 266.03], 'category_id': 58, 'id': 86}


    #### 生成并保存所有seed、所有class、所有k-shot的dataset
    for i in range(args.seeds[0], args.seeds[1]): # 遍历每个seed
        random.seed(i) # 设置新的seed
        for c in ID2CLASS.keys(): # c: cls_id，遍历每个class
            ## 获取每张图片对应的所有属于该class的标注
            img_ids = {} # dict，key为img_id，value为所有属于该class的标注(list[dict])
            for a in anno[c]: # a: 标注（类型为dict），遍历该class的每个标注，分配给其对应图片
                if a["image_id"] in img_ids:
                    img_ids[a["image_id"]].append(a)
                else:
                    img_ids[a["image_id"]] = [a]

            ## 采样该seed、该class的所有k-shot dataset（5-shot dataset就是基于3-shot dataset再选择2个instance）并保存
            sample_shots = [] # list[dict]，所有k-shot中采样的所有标注（类型为dict）
            sample_imgs = [] # list[dict]，所有k-shot中采样的所有图片（类型为dict）
            for shots in [1, 2, 3, 5, 10, 30]: # 遍历每个k-shot
                ## 生成该seed、该class、该k-shot的dataset
                while True:
                    ## 先随机选择k张包含属于该class的标注的图片（img_id）
                    imgs = random.sample(list(img_ids.keys()), shots)

                    ## IMPORTANT：遍历这k张包含属于该class的标注的图片（不一定遍历完所有图片，持续遍历直到刚刚好采样到k个属于该class的标注）
                    for img in imgs: # 遍历这k张包含属于该class的标注的图片（img_id）
                        ## IMPORTANT：如果当前图片img中的某个标注s已被采样过，则跳过这张图片
                        skip = False # 是否跳过这张图片
                        for s in sample_shots: # 遍历所有k-shot中采样的所有标注（类型为dict）
                            if img == s["image_id"]: # 如果当前图片img中的某个标注s已被采样过，则跳过这张图片
                                skip = True
                                break
                        if skip: # 跳过这张图片
                            continue
                        
                        ## IMPORTANT：如果 该图片中属于该class的标注数量 + 已采样属于该class的标注数量 > 需采样属于该class的标注数量(k)，则跳过该图片
                        if len(img_ids[img]) + len(sample_shots) > shots:
                            continue

                        ## IMPORTANT：如果不是以上两种情况中的任意一种，则采样该图片和其中所有属于该class的object
                        sample_shots.extend(img_ids[img])
                        sample_imgs.append(id2img[img])

                        ## 如果采样该图片后正好采样够k个属于该class的标注，则停止遍历这k张包含属于该class的标注的图片
                        if len(sample_shots) == shots:
                            break
                    ## 如果采样该图片后正好采样够k个属于该class的标注，则停止遍历这k张包含属于该class的标注的图片
                    if len(sample_shots) == shots:
                        break
                
                ## 保存该seed、该class、该k-shot的dataset
                new_data = { # 需要保存的数据
                    "info": data["info"],
                    "licenses": data["licenses"],
                    "images": sample_imgs,
                    "annotations": sample_shots,
                }
                save_path = get_save_path_seeds( # 生成save path
                    data_path, ID2CLASS[c], shots, i
                )
                new_data["categories"] = new_all_cats # 所有类别
                with open(save_path, "w") as f:
                    json.dump(new_data, f)


def get_save_path_seeds(path, cls, shots, seed):
    prefix = "full_box_{}shot_{}_trainval".format(shots, cls)
    save_dir = os.path.join("datasets", "cocosplit", "seed" + str(seed))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, prefix + ".json")
    return save_path


if __name__ == "__main__":
    # 1个dict，key为cls_id，value为对应的cls_name
    ID2CLASS = {
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        5: "airplane",
        6: "bus",
        7: "train",
        8: "truck",
        9: "boat",
        10: "traffic light",
        11: "fire hydrant",
        13: "stop sign",
        14: "parking meter",
        15: "bench",
        16: "bird",
        17: "cat",
        18: "dog",
        19: "horse",
        20: "sheep",
        21: "cow",
        22: "elephant",
        23: "bear",
        24: "zebra",
        25: "giraffe",
        27: "backpack",
        28: "umbrella",
        31: "handbag",
        32: "tie",
        33: "suitcase",
        34: "frisbee",
        35: "skis",
        36: "snowboard",
        37: "sports ball",
        38: "kite",
        39: "baseball bat",
        40: "baseball glove",
        41: "skateboard",
        42: "surfboard",
        43: "tennis racket",
        44: "bottle",
        46: "wine glass",
        47: "cup",
        48: "fork",
        49: "knife",
        50: "spoon",
        51: "bowl",
        52: "banana",
        53: "apple",
        54: "sandwich",
        55: "orange",
        56: "broccoli",
        57: "carrot",
        58: "hot dog",
        59: "pizza",
        60: "donut",
        61: "cake",
        62: "chair",
        63: "couch",
        64: "potted plant",
        65: "bed",
        67: "dining table",
        70: "toilet",
        72: "tv",
        73: "laptop",
        74: "mouse",
        75: "remote",
        76: "keyboard",
        77: "cell phone",
        78: "microwave",
        79: "oven",
        80: "toaster",
        81: "sink",
        82: "refrigerator",
        84: "book",
        85: "clock",
        86: "vase",
        87: "scissors",
        88: "teddy bear",
        89: "hair drier",
        90: "toothbrush",
    }
    # 1个dict，key为cls_name，value为对应的cls_id
    CLASS2ID = {v: k for k, v in ID2CLASS.items()}

    args = parse_args()
    generate_seeds(args)
