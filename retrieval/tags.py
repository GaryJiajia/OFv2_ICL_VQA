import os
import pickle
import json
from tqdm import tqdm
import numpy as np
from vqa_splits import COCOVQADataset
import heapq
import random


def get_img_id(img_name):
    return int(img_name.split(".")[0].split('_')[-1])


# load relation and object
input_dir = "/IETrans-SGG.pytorch/coco_vis/"
total_dic = {}
for i in tqdm(range(31)):
    demo_dir = os.path.join(input_dir, "demo_{}".format(i))
    pred_file = os.path.join(demo_dir, "custom_prediction.pk")
    dic = pickle.load(open(pred_file, "rb"))
    dic = {os.path.basename(k): v for k, v in dic.items()}
    total_dic.update(dic)
print(len(total_dic.keys()))

# load train split
splits = COCOVQADataset()
train_id2name = splits.get_train_data()
train_filenames = list(splits.get_train_data().values())
train_ids = list(splits.get_train_data().keys())
test_id2name = splits.get_test_data()
test_filenames = list(splits.get_test_data().values())
test_ids = list(splits.get_test_data().keys())

# load attr
attr_dir = "/feature_extraction/scene_graph_benchmark/coco_attr"

# load vocab
vocab = json.load(open("/IETrans-SGG.pytorch/datasets/vg/1000/VG-dicts.json"))
relation_dim = len(vocab['idx_to_predicate'].keys())
obejct_dim = len(vocab['idx_to_label'].keys())
print("got {} relations {} objects".format(relation_dim, obejct_dim))

vocab_2 = json.load(open("/feature_extraction/scene_graph_benchmark/model/VG-SGG-dicts-vgoi6-clipped.json"))
attribute_dim = len(vocab_2['idx_to_attribute'].keys())
class_dim = len(vocab_2['idx_to_label'].keys())
print("got {} relations {} objects".format(attribute_dim, class_dim))

# split train and test
train_dic = {}
test_dic = {}
for img, data in tqdm(list(total_dic.items())):
    id = get_img_id(img)
    if id in train_ids:
        data['attr_labels'] = json.load(
            open(os.path.join(attr_dir, os.path.basename(train_id2name[id]).split(".")[0] + "_attr.json")))
        data['class_labels'] = json.load(
            open(os.path.join(attr_dir, os.path.basename(train_id2name[id]).split(".")[0] + "_class.json")))

        attr_list = []
        for d in data['attr_labels']:
            attr_list.extend(d)

        data['rel_labels'] = [int(d) for d in data['rel_labels']]
        data['bbox_labels'] = [int(d) + relation_dim for d in data['bbox_labels']]
        data['attr_labels'] = [int(d) + relation_dim + obejct_dim for d in attr_list]
        data['class_labels'] = [int(d) + relation_dim + obejct_dim + attribute_dim for d in data['class_labels']]
        train_dic[id] = data
    if id in test_ids:
        data['attr_labels'] = json.load(
            open(os.path.join(attr_dir, os.path.basename(test_id2name[id]).split(".")[0] + "_attr.json")))
        data['class_labels'] = json.load(
            open(os.path.join(attr_dir, os.path.basename(test_id2name[id]).split(".")[0] + "_class.json")))

        attr_list = []
        for d in data['attr_labels']:
            attr_list.extend(d)
        data['rel_labels'] = [int(d) for d in data['rel_labels']]
        data['bbox_labels'] = [int(d) + relation_dim for d in data['bbox_labels']]
        data['attr_labels'] = [int(d) + relation_dim + obejct_dim for d in attr_list]
        data['class_labels'] = [int(d) + relation_dim + obejct_dim + attribute_dim for d in data['class_labels']]
        test_dic[id] = data

print(len(train_dic.keys()), len(test_dic.keys()))

output_dir = "./SGG_tags/test_tags_topn_DIIR_TR"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# cut the input into same length
search_imgs = list(train_dic.items())
for img_t, data_t in tqdm(search_imgs):
    rel_list = data_t['rel_labels']
    data_t['rel_labels'] = list(set(rel_list))
    data_t['rel_labels'].sort(key=rel_list.index)

    bbox_list = data_t['bbox_labels']
    data_t['bbox_labels'] = list(set(bbox_list))
    data_t['bbox_labels'].sort(key=bbox_list.index)

    attr_list = data_t['attr_labels']
    data_t['attr_labels'] = list(set(attr_list))
    data_t['attr_labels'].sort(key=attr_list.index)

    class_list = data_t['class_labels']
    data_t['class_labels'] = list(set(class_list))
    data_t['class_labels'].sort(key=class_list.index)


for img_t, data_t in tqdm(search_imgs):
    data_t['rel_labels'] = set(data_t['rel_labels'])
    data_t['bbox_labels'] = set(data_t['bbox_labels'])
    data_t['attr_labels'] = set(data_t['attr_labels'])
    data_t['class_labels'] = set(data_t['class_labels'])

# generate random tags index
total_len = relation_dim + relation_dim + obejct_dim + attribute_dim
total_list = [i for i in range(total_len)]
random.shuffle(total_list)
step = int(len(total_list) / 4)
print(len(total_list))
tag_splits = [set(total_list[i:i + step]) for i in range(0, len(total_list), step)]

for img, data in tqdm(list(test_dic.items())):
    relations = set([int(d) for d in data['rel_labels']])
    objects = set([int(d) for d in data["bbox_labels"]])
    attrs = set([int(a) for a in data["attr_labels"]])
    classes = set([int(d) for d in data["class_labels"]])

    total_dis = [[len(relations & data_t['rel_labels']), len(objects & data_t['bbox_labels']),
                  len(attrs & data_t['attr_labels']),
                  len(classes & data_t['class_labels']
                      )] for img_t, data_t in search_imgs]

    total_dis = np.array(total_dis)
    print(total_dis.shape)

    total_dis_mean = np.mean(total_dis, axis=0)
    print(total_dis_mean.shape)

    total_dis = total_dis / total_dis_mean
    print(total_dis.shape)

    total_dis = np.sum(total_dis, axis=1)
    print(total_dis.shape)

    train_idx = heapq.nlargest(32, range(len(total_dis)), total_dis.__getitem__)
    train_num = heapq.nlargest(32, total_dis)

    items = [search_imgs[idx] for idx in train_idx]
    output = [[key, dis] for (key, _), dis in zip(items, train_num)]

    json.dump(output, open(os.path.join(output_dir, "{}_sim.json".format(img)), "w"))
