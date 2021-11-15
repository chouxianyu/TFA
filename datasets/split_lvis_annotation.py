# LVIS数据集中json文件格式的说明：https://www.lvisdataset.org/dataset

"""
把`lvis_v0.5_train.json`分成`lvis_v0.5_train_{freq,common,rare}.json`
"""
import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    # 源文件和输出文件的路径
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/lvis/lvis_v0.5_train.json",
        help="path to the annotation file",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="datasets/lvis",
        help="path to the save directory",
    )
    args = parser.parse_args()
    return args


def split_annotation(args):
    # 打开源json文件
    with open(args.data) as fp:
        ann_train = json.load(fp)

    # 遍历3个frequency，生成`lvis_v0.5_train_{freq,common,rare}.json`
    for s, name in [("f", "freq"), ("c", "common"), ("r", "rare")]:
        # annotation structure
        ann_s = {
            "info": ann_train["info"],
            # 'images': ann_train['images'],
            "categories": ann_train["categories"],
            "licenses": ann_train["licenses"],
        }

        # 当前所遍历到的frequency的categories的id
        ids = [
            cat["id"] # 该category的id
            for cat in ann_train["categories"] # 遍历所有category
            if cat["frequency"] == s # 判断该category的frequency是否等于当前所遍历到的frequency
        ]

        # 读取当前所遍历到的frequency的categories的annotation
        ann_s["annotations"] = [
            ann
            for ann in ann_train["annotations"]
            if ann["category_id"] in ids
        ]

        # 读取当前所遍历到的frequency的categories的img的id
        img_ids = set([ann["image_id"] for ann in ann_s["annotations"]])
        # 读取当前所遍历到的frequency的categories的img
        new_images = [
            img for img in ann_train["images"] if img["id"] in img_ids
        ]
        ann_s["images"] = new_images

        # 保存该frequency对应的annotation
        save_path = os.path.join(
            args.save_dir, "lvis_v0.5_train_{}.json".format(name)
        )
        print("Saving {} annotations to {}.".format(name, save_path))
        with open(save_path, "w") as fp:
            json.dump(ann_s, fp)


if __name__ == "__main__":
    args = parse_args()
    split_annotation(args)
