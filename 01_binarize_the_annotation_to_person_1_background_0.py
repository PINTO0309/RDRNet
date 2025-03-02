import os
import shutil
import random
from PIL import Image
import numpy as np

# -- 1) 元フォルダ・出力先フォルダなどの設定 --
voc_root = "data/VOCdevkit/VOC2012"

original_anno_dir = os.path.join(voc_root, "SegmentationClass")
original_img_dir  = os.path.join(voc_root, "JPEGImages")

# Person 専用のアノテーション＆画像を保存する先
person_anno_dir = os.path.join(voc_root, "SegmentationClassPerson")
person_img_dir  = os.path.join(voc_root, "JPEGImagesPerson")

os.makedirs(person_anno_dir, exist_ok=True)
os.makedirs(person_img_dir, exist_ok=True)

# Person のクラス ID (VOC では通常 15)
PERSON_ID = 15

# -- 2) Person が含まれる画像だけをコピー＆2値マスク生成 --
ann_list = [f for f in os.listdir(original_anno_dir) if f.endswith(".png")]

person_image_ids = []

for ann_name in ann_list:
    ann_path = os.path.join(original_anno_dir, ann_name)
    mask = Image.open(ann_path)
    mask_np = np.array(mask, dtype=np.uint8)

    # Person (ID=15) が含まれているか判定
    if (mask_np == PERSON_ID).any():
        # 二値化: Person=1, それ以外=0
        bin_mask = np.where(mask_np == PERSON_ID, 1, 0).astype(np.uint8)
        bin_mask_img = Image.fromarray(bin_mask)

        # SegmentationClassPerson に二値マスクを保存
        out_ann_path = os.path.join(person_anno_dir, ann_name)
        bin_mask_img.save(out_ann_path)

        # JPEGImagesPerson に対応する画像をコピー
        base_id = os.path.splitext(ann_name)[0]   # 例: "2007_000032"
        jpg_name = base_id + ".jpg"
        src_jpg = os.path.join(original_img_dir, jpg_name)
        dst_jpg = os.path.join(person_img_dir,  jpg_name)

        if os.path.exists(src_jpg):
            shutil.copy(src_jpg, dst_jpg)
            person_image_ids.append(base_id)

print("Done. Person images count:", len(person_image_ids))

# -- 3) Person を含む全画像を 8:2 に分割する (train/val) --
# ランダム性を固定したい場合は seed を設定
random.seed(0)

# シャッフル
random.shuffle(person_image_ids)

# 8:2 に分割
train_ratio = 0.8
train_count = int(len(person_image_ids) * train_ratio)
train_ids = person_image_ids[:train_count]
val_ids   = person_image_ids[train_count:]

print("Train images:", len(train_ids))
print("Val images:  ", len(val_ids))

# -- 4) train_person.txt, val_person.txt を作成 --
train_person_path = os.path.join(voc_root, "ImageSets", "Segmentation", "train_person.txt")
val_person_path   = os.path.join(voc_root, "ImageSets", "Segmentation", "val_person.txt")

with open(train_person_path, "w") as f:
    for img_id in train_ids:
        f.write(img_id + "\n")

with open(val_person_path, "w") as f:
    for img_id in val_ids:
        f.write(img_id + "\n")

print(f"Saved: {train_person_path}, {val_person_path}")
