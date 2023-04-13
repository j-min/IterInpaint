import os,math,re,json
import numpy as np
import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import random
from transformers import CLIPTokenizer, CLIPTextModel
import ldm.data.transforms as T

import re
from pathlib import Path
import pandas as pd

from ldm.data.layout import LayoutDataset

class CLEVRDataset(LayoutDataset):
    def __init__(self,
                old_clevr=False,
                data_ratio=1.0,
                *args,
                **kwargs
                ):

        super().__init__(*args, **kwargs)

        clevr_dir = self.data_root
        split = self.set
        self.split = split

        self.old_clevr = old_clevr
        
        clevr_dir = Path(clevr_dir)
        print('clevr_dir', clevr_dir)

        print('Split:' , split)

        self.data_ratio = data_ratio

        if self.old_clevr:
            clevr_df_path = clevr_dir / f'{split}_ann.json'
            clevr_df = pd.read_json(clevr_df_path)
            print('Loaded ', clevr_df_path, ' | shape: ' , clevr_df.shape)

            if self.max_num_objects > 0:
                clevr_df = clevr_df[clevr_df.objects.apply(len) <= self.max_num_objects]
                print('Filtered to ', self.max_num_objects, ' objects | shape: ', clevr_df.shape)
                clevr_df = clevr_df.reset_index(drop=True)
            self.clevr_df = clevr_df

            if data_ratio < 1.0:
                clevr_df = clevr_df.sample(frac=data_ratio, random_state=42)
                print('Sampled ', data_ratio, ' | shape: ', clevr_df.shape)
                clevr_df = clevr_df.reset_index(drop=True)
                self.clevr_df = clevr_df

            if split == 'train':
                self.image_dir = clevr_dir / 'CLEVR_v1.0/images' / 'train'
            else:
                self.image_dir = clevr_dir / 'CLEVR_v1.0/images' / 'val'
            self.num_images = len(self.clevr_df)
        else:
            scene_path = clevr_dir / split / 'scenes.json'
            with open(scene_path) as f:
                scenes = json.load(f)
            print('Loaded ', scene_path, ' | shape: ' , len(scenes['scenes']))
                # output = {
                # 'info': {
                #     'date': args.date,
                #     'version': args.version,
                #     'split': args.split,
                #     'license': args.license,
                # },
                # 'scenes': all_scenes
                # }

            self.image_dir = clevr_dir / split / 'images'

            self.scenes = scenes['scenes']
            self.num_images = len(self.scenes)
        
        self._length = self.num_images

        # print('========================')
        print('number of samples:',self._length)
        # print('========================')


        self.all_objects = clevr_all_objects

        self.class_name_to_token = {}
        self.token_to_class_name = {}

        for i, obj in enumerate(self.all_objects):
            self.class_name_to_token[obj] = f'<class{str(i).zfill(3)}>'
            self.token_to_class_name[f'<class{str(i).zfill(3)}>'] = obj

        assert len(self.class_name_to_token) == self.num_classes, f'num_classes should be {len(self.class_name_to_token)}'


        # self.background_instruction = "Add gray background"


    def __len__(self):
        return self._length

    def load_image_ann(self, index):
        if self.old_clevr:
            _datum = self.clevr_df.iloc[index]
            # train_df.iloc[0].id
            #     'CLEVR_train_060674.png'

            # xyxy
            # train_df.iloc[0].objects
            #     [{'text': 'red rubber cube', 'box': [270, 80, 344, 170]},
            #     {'text': 'yellow rubber sphere', 'box': [387, 131, 423, 169]},
            #     {'text': 'brown rubber cylinder', 'box': [346, 161, 388, 221]},
            #     {'text': 'purple rubber cube', 'box': [141, 55, 205, 133]}]

            id = _datum['id']

            img_fname = _datum['id']
            img_path = self.image_dir / img_fname
            img_path = str(img_path)
            
            objects = _datum['objects']

        else:
            _datum = self.scenes[index]

            id = _datum['image_filename']

            img_path = self.image_dir / id
            img_path = str(img_path)

            # xywh -> xyxy
            boxes = [obj['bbox'] for obj in _datum['objects']]
            boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in boxes]

            obj_names = [f"{obj['color']} {obj['material']} {obj['shape']}" for obj in _datum['objects']]

            objects = []
            for obj_name, box in zip(obj_names, boxes):
                objects.append({
                    'text': obj_name,
                    'box': box
                })

        # Assert that the bbox in xyxy, not xywh
        for obj in objects:
            assert len(obj['box']) == 4

            assert obj['box'][2] > obj['box'][0], f'x2 should be greater than x1, {obj["box"]}'
            assert obj['box'][3] > obj['box'][1], f'y2 should be greater than y1, {obj["box"]}'
        

        return {
            'id': id,
            'img_path': img_path,
            'objects': objects,
        }



clevr_all_objects = [
    'blue metal cube',
    'blue metal cylinder',
    'blue metal sphere',
    'blue rubber cube',
    'blue rubber cylinder',
    'blue rubber sphere',
    'brown metal cube',
    'brown metal cylinder',
    'brown metal sphere',
    'brown rubber cube',
    'brown rubber cylinder',
    'brown rubber sphere',
    'cyan metal cube',
    'cyan metal cylinder',
    'cyan metal sphere',
    'cyan rubber cube',
    'cyan rubber cylinder',
    'cyan rubber sphere',
    'gray metal cube',
    'gray metal cylinder',
    'gray metal sphere',
    'gray rubber cube',
    'gray rubber cylinder',
    'gray rubber sphere',
    'green metal cube',
    'green metal cylinder',
    'green metal sphere',
    'green rubber cube',
    'green rubber cylinder',
    'green rubber sphere',
    'purple metal cube',
    'purple metal cylinder',
    'purple metal sphere',
    'purple rubber cube',
    'purple rubber cylinder',
    'purple rubber sphere',
    'red metal cube',
    'red metal cylinder',
    'red metal sphere',
    'red rubber cube',
    'red rubber cylinder',
    'red rubber sphere',
    'yellow metal cube',
    'yellow metal cylinder',
    'yellow metal sphere',
    'yellow rubber cube',
    'yellow rubber cylinder',
    'yellow rubber sphere'
]