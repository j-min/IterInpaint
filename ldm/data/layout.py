import numpy as np
import PIL
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import random
import ldm.data.transforms as T

import re

from ldm.gen_utils import prepare_text, prepare_clip_tokenizer
from ldm.task_utils import ALL_TASKS, prepare_task


class LayoutDataset(Dataset):
    def __init__(self,
                 data_root=None,
                 size=None,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.5,
                 set="train",
                 placeholder_token="dog",
                 per_image_tokens=False,
                 center_crop=False,
                 mixing_prob=0.25,
                 coarse_class_text=None,
                 reg = False,
                #  coco_path = 'dataset/coco/all2014',
                 with_bbox = False,
                 num_bins = 1000,
                 max_src_length = 77,
                 box_descp = 'tag',
                 spatial_word = None,

                #  split='train',
                #  tokenizer=None,
                 max_num_objects=-1,
                # #  image_resolution=512,
                # #  num_bins=1000,
                 inpaint=None,
                 text_reco=True,
                 box_crop=False,
                #  verbose=False,
                 tasks=None,
                 tasks_weights=None,
                 prefix_plan=False,
                 prefix_global_caption=True,

                 with_class_embedding=False,
                 num_classes=48,

                 has_global_caption=False,
                 generation_order='random',  

                 background_instruction="Add gray background",
                 ):

        self.data_root = data_root
        # self.coco_path = coco_path
        self.max_src_length = max_src_length

        self.set = set

        self.with_bbox = with_bbox
        self.num_bins = num_bins
        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob
        self.size = size
        self.coarse_class_text = coarse_class_text

        self.max_num_objects = max_num_objects
        self.generation_order = generation_order
        self.has_global_caption = has_global_caption

        self.with_class_embedding = with_class_embedding
        self.num_classes = num_classes

        self.inpaint = inpaint
        self.text_reco = text_reco
        self.box_crop = box_crop

        self.spatial_word = spatial_word

        if tasks is None:
            tasks = [
                'add_object_given_background',
                'add_object_given_empty_background',
                'add_background_given_object',
            ]
        if tasks_weights is None:
            # tasks_weights = [
            #     0.1,
            #     0.8,
            #     0.1
            # ]

            tasks_weights = [
                0.0,
                0.9,
                0.1
            ]

        print('tasks', tasks)
        print('tasks_weights', tasks_weights)

        assert len(tasks) == len(tasks_weights), 'tasks and tasks_weights must be of same length'
        for task in tasks:
            assert task in ALL_TASKS, f'{task} not in {ALL_TASKS}'

        self.tasks = tasks
        self.tasks_weights = tasks_weights
        self.prefix_plan = prefix_plan
        self.prefix_global_caption = prefix_global_caption

        self.background_instruction = background_instruction

        self.image_resolution = self.size

        # if per_image_tokens:
        #     assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        if self.with_bbox:
            self.tokenizer = prepare_clip_tokenizer(
                with_bbox=with_bbox,
                num_bins=num_bins,
                max_length=max_src_length,
                with_class_embedding=with_class_embedding,
                num_classes=num_classes,
                )
        
        # self.coco_hw = json.load(open('dataset/coco_wh.json', 'r'))
        # if self.with_bbox:
        #     self.box_descp = box_descp
        #     self.spatial_word = spatial_word
        #     if self.spatial_word is not None:
        #         # self.box_descp = 'tag'
        #         self.box_descp = 'caption'
        #     self.cliptext_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        #     self.box_offset = len(self.cliptext_tokenizer)
        #     self.coco_od = json.load(open('dataset/coco_allbox_gitcaption.json', 'r')) ## 'tag' 'pad_caption' 'crop_caption'  ## GIT-COCO ckpt
        #     # self.coco_od = json.load(open('dataset/coco_allbox_gitcaption_coco.json', 'r')) ## GIT-COCO ckpt
        #     num_withbox = 0
        #     for key in self.coco_od:
        #         if 'box' in self.coco_od[key]:
        #             if len(self.coco_od[key]['box'])!=0: num_withbox+=1
        #     print('images with box',num_withbox,'images',len(self.coco_od),'dataset len',self._length)

        ##
        print('always using Image.BICUBIC in resizing.')
        if set == "train":
            self._transforms = T.Compose(
                [ 
                    T.RandomHorizontalFlip(p=flip_p),
                    T.RandomResize([self.size]),
                    T.RandomCrop((self.size,self.size)), 
                ]
            )
        else:
            self._transforms = T.Compose(
                [
                    T.RandomResize([self.size]),
                    T.CenterCrop((self.size,self.size)),
                ]
            )

        self.tensor_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize([0.5], [0.5], box_cxcywh=False),
            ]
        )

    def reverse_image_transform(self, image_tensor):
        """Reverse the tensor_transform to get the original image"""
        image_tensor = image_tensor.clone().detach()
        image_tensor = image_tensor * 0.5 + 0.5
        image_tensor = image_tensor * 255
        image_tensor = image_tensor.permute(1, 2, 0)
        image_tensor = image_tensor.numpy().astype(np.uint8)
        image = Image.fromarray(image_tensor)
        return image


    def __len__(self):
        return self._length

    def load_image_ann(self, index):
        """
        Load image and bounding boxes info from a image annotation dict

        Returns:
        {
            'id': 
            'img_path': image_path,
            'objects': list of objects {'text': 'purple rubber cube', 'box': [141, 55, 205, 133]}
        }
        """
        raise NotImplementedError

    def __getitem__(self, i):

        datum = self.load_image_ann(i)

        img_path = datum['img_path']
        image = Image.open(img_path).convert('RGB')
        datum['img_path'] = img_path

        orig_W, orig_H = image.size

        # # Store the original PIL image
        datum['orig_pil_image'] = image

        region_captions = [d['text'] for d in datum['objects']]
        bboxes = [d['box'] for d in datum['objects']]

        # Randomly shuffle the objects
        if 'train' in self.split:
            idxs = np.random.permutation(len(bboxes))
            bboxes = [bboxes[i] for i in idxs]
            region_captions = [region_captions[i] for i in idxs]

            # Drop boxes and box captions whose box areas are too small
            idxs = [i for i, box in enumerate(bboxes) if (box[2] - box[0]) * (box[3] - box[1]) > 10] 
            bboxes = [bboxes[i] for i in idxs]
            region_captions = [region_captions[i] for i in idxs]

        target = {
            "image_id": datum['id'],
            "boxes": torch.tensor(bboxes, dtype=torch.float32).reshape(-1, 4),
            "box_caption": region_captions,
        }

        # # print('target:', target)
        # normalized_bboxes = []
        # for box in bboxes:
        #     x1, y1, x2, y2 = box
        #     # w, h = x2 - x1, y2 - y1
        #     x1, y1, x2, y2 = x1 / image.width, y1 / image.height, x2 / image.width, y2 / image.height
        #     normalized_bboxes.append([x1, y1, x2, y2])
            
        # viz_img = viz_utils.fig2img(viz_utils.plot_results(
        #     image,
        #     bboxes,
        #     region_captions,
        # ))
        # datum['viz_img'] = viz_img

        global_caption = None
        if self.has_global_caption:
            assert 'caption' in datum, 'has_global_caption is True but no caption is found'
            target['caption'] = datum['caption']
            global_caption = target["caption"]
        datum['global_caption'] = global_caption

        # Crop a single box and return that image
        if self.box_crop == 'single':
            box, box_caption = target["boxes"][0], target["box_caption"][0]

            image = image.crop(box.long().tolist())

            # Resize to full resolution
            image = image.resize((self.image_resolution, self.image_resolution), Image.Resampling.BICUBIC)
            # datum['crop_pil_image'] = image

            # image_tensor, _ = self.tensor_transform(image, {})
            # datum['target_image'] = image_tensor

            image = np.array(image).astype(np.uint8)
            datum["image"] = (image / 127.5 - 1.0).astype(np.float32)

            text = box_caption
            # datum['text'] = text

            datum["caption"] = text
            # datum["str_caption"] = text
            return datum

        image, target = self._transforms(image, target)

        # datum['processed_pil_image'] = image
    
        generation_order = self.generation_order

        box_indices = list(range(target["boxes"].shape[0]))

        if generation_order == "bottom_to_top":
            # Generate from bottom to top by y1 coordinate (reverse=True: descending)
            box_indices = sorted(box_indices, key=lambda x: target["boxes"][x][3], reverse=True)

        elif generation_order == "top_to_bottom":
            # Generate from top to bottom by y0 coordinate
            box_indices = sorted(box_indices, key=lambda x: target["boxes"][x][1])

        elif generation_order == "random":
            # Randomly shuffle the boxes
            random.shuffle(box_indices)
        
        else:
            raise ValueError("Invalid generation_order: {}".format(generation_order))

        # print('target:', target)
        # print('generation_order:', generation_order)
        # print("bboxes:", bboxes)
        # print('target boxes:', target["boxes"])
        # print('box_indices:', box_indices)

        box_captions = [target["box_caption"][box_sample_ii] for box_sample_ii in box_indices]
        boxes_unnormalized = [target["boxes"][box_sample_ii].tolist() for box_sample_ii in box_indices]

        boxes_normalized = []
        for box_sample_ii in box_indices:
            box = target["boxes"][box_sample_ii].tolist()
            box = [float(x) / self.image_resolution for x in box]
            boxes_normalized.append(box)

        boxes_normalized = torch.tensor(boxes_normalized, dtype=torch.float32).reshape(-1, 4)
        boxes_unnormalized = torch.tensor(boxes_unnormalized, dtype=torch.float32).reshape(-1, 4)

        datum['normalized_boxes'] = boxes_normalized
        datum['unnormalized_boxes'] = boxes_unnormalized
        # datum['quantized_boxes'] = boxes_quantized
        datum['box_captions'] = box_captions

        # datum['box_captions_with_coords'] = box_captions_with_coords
        # datum['box_bins'] = box_bins

        if self.inpaint is None:
            target_image = image

            # Store the processed original PIL image
            datum['processed_pil_image'] = image

            # Target images
            image_tensor, target = self.tensor_transform(target_image, target)
            datum['target_image'] = image_tensor


            # (global caption +) [ 'xyxy' + 'box caption' ] * n_boxes
            text = prepare_text(
                box_captions=box_captions,
                box_normalized=boxes_normalized,
                global_caption=global_caption if self.prefix_global_caption else None,
                text_reco=self.text_reco,
                num_bins=self.num_bins,
            )

            datum['text'] = text
        else:
            # Iterative inpainting
            if self.inpaint == 'iterative':

                d = datum

                # mask_img = image.copy().convert('L')
                # mask_draw = ImageDraw.Draw(mask_img)
                # context_img = image.copy().convert('RGB')
                # context_draw = ImageDraw.Draw(context_img)

                # text_tokens = []

                # mask_img
                # White -> repainted (to be masked)
                # Black -> preserved
                
                # Sample task
                tasks = self.tasks
                weights = self.tasks_weights

                assert sum(weights) == 1, f'weights must sum to 1, but sum to {sum(weights)}'
                task = np.random.choice(tasks, p=weights)

                task_datum = prepare_task(
                    image,
                    datum,
                    task=task,
                    # verbose=True
                    verbose=False,
                    prefix_plan=self.prefix_plan,
                    background_instruction = self.background_instruction,
                    )

                # 'text': text,
                # 'target_image': target_image,
                # 'context_image': context_img,
                # 'mask_image': mask_img,

                text = task_datum['text']
                target_image = task_datum['target_image']
                context_img = task_datum['context_image']
                mask_img = task_datum['mask_image']

                # current step task aption
                # step_box_caption = task_datum['box_caption']

                datum['task'] = task

                if self.prefix_global_caption:
                    EOS_token = '<|endoftext|>'
                    text = f'{global_caption} {EOS_token} {text}'

                datum['text'] = text

            # Store the processed original PIL image
            datum['processed_pil_image'] = image

            # Target images
            image_tensor, target = self.tensor_transform(target_image, target)
            datum['target_image'] = image_tensor
            datum['target_pil_image'] = self.reverse_image_transform(image_tensor)

            datum['context_pil_image'] = context_img
            datum['mask_pil_image'] = mask_img

            mask = np.array(mask_img.convert("L"))
            mask = mask.astype(np.float32) / 255.0
            # print(mask.shape)
            # mask = mask[None, None]
            mask = mask.reshape(1, mask.shape[0], mask.shape[1])
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            mask = torch.from_numpy(mask)

            masked_image = image_tensor * (mask < 0.5)

            # Input context image tensor
            datum['masked_image'] = masked_image
            datum['masked_pil_image'] = self.reverse_image_transform(masked_image)
            datum['mask'] = mask

        # Needs to return 
        # caption
        # image

        # # Target image (H, W, C)
        # image = np.array(image).astype(np.uint8)
        # datum["image"] = (image / 127.5 - 1.0).astype(np.float32)

        # LDM codebase expects HWC instead of CHW
        datum["target_image"] = datum["target_image"].permute(1, 2, 0)

        if self.inpaint is not None:
            datum["mask"] = datum['mask'].permute(1, 2, 0)
            datum['masked_image'] = datum['masked_image'].permute(1, 2, 0)

        if self.with_class_embedding:
            assert self.class_name_to_token is not None, 'class_name_to_token must be provided'

            for orig_class_name, new_token_name in self.class_name_to_token.items():

                if orig_class_name in text:
                    text = text.replace(orig_class_name, new_token_name)

                    assert new_token_name in self.tokenizer.get_vocab(), f'{new_token_name} not in tokenizer vocab'

        input_ids = self.tokenizer(text,
            truncation=True,
            padding='max_length',
            max_length=self.max_src_length,
            return_length=True,
            return_tensors='pt').input_ids
        input_ids = input_ids.squeeze(0)
        assert input_ids.shape == (self.max_src_length, ), f"input_ids.shape: {input_ids.shape}"

        datum["caption"] = input_ids

        # datum["str_caption"] = text

        # print(datum)
        # for k, v in datum.items():
        #     # print size or length of each key
        #     print('key: ', k, 'size: ', v.size() if isinstance(v, torch.Tensor) else len(v))

        return datum

    def collate_fn(self, batch):
        """Collate a batch of data."""

        out = {}
        # out['image'] = torch.stack([torch.from_numpy(datum['image']) for datum in batch], dim=0)
        out['image'] = torch.stack([datum['target_image'] for datum in batch], dim=0)
        out['caption'] = torch.stack([datum['caption'] for datum in batch], dim=0)

        # For inpainting model only
        if self.inpaint == 'iterative':
            # concat_keys=("mask", "masked_image"),
            out['mask'] = torch.stack([datum['mask'] for datum in batch], dim=0)
            out['masked_image'] = torch.stack([datum['masked_image'] for datum in batch], dim=0)
            # out['mask'] = torch.stack([torch.from_numpy(datum['mask']) for datum in batch], dim=0)
            # out['masked_image'] = torch.stack([torch.from_numpy(datum['masked_image']) for datum in batch], dim=0)

        # for k in batch[0].keys():
        #     if k not in ['image', 'caption', 'target_image', 'mask', 'masked_image']:
        #         out[k] = [datum[k] for datum in batch]

        return out





        # if not self.with_bbox:
        #     example["caption"] = target["caption"]
        #     example["str_caption"] = target["caption"]
        # else:
        #     tokenized_text = []
        #     str_caption = target["caption"]
        #     text = [self.pre_caption(target["caption"].lower(), self.max_src_length)]
        #     text_enc = self.cliptext_tokenizer(text, truncation=True, max_length=self.max_src_length, return_length=True,
        #                                 return_overflowing_tokens=False, padding=False, return_tensors="pt")["input_ids"]
        #     tokenized_text.append(text_enc[0,:])
        #     # tokenized_text, str_caption = [self.cliptext_tokenizer([''], truncation=True, max_length=self.max_src_length, return_length=True, return_overflowing_tokens=False, padding=False, return_tensors="pt")["input_ids"][0,:]], ''

        #     if len(box_list)!=0:
        #         imagehw = self.coco_hw[sample_idx]
        #         # for box_sample_ii in range(len(box_list)):
        #         for box_sample_ii in range(target["boxes"].shape[0]):
        #             if self.spatial_word is None:
        #                 box, caption = target["boxes"][box_sample_ii], target["box_caption"][box_sample_ii]
        #                 # box = self.process_centercrop(box, imagehw, self.center_crop)
        #                 box = [float(x)/self.size for x in box]
        #                 quant_x0 = int(round((box[0] * (self.num_bins - 1)))) + self.box_offset
        #                 quant_y0 = int(round((box[1] * (self.num_bins - 1)))) + self.box_offset
        #                 quant_x1 = int(round((box[2] * (self.num_bins - 1)))) + self.box_offset
        #                 quant_y1 = int(round((box[3] * (self.num_bins - 1)))) + self.box_offset
        #                 region_coord = torch.tensor([quant_x0,quant_y0,quant_x1,quant_y1]).to(text_enc.device)
        #                 caption = self.pre_caption(caption.lower(), self.max_src_length)
        #                 region_text = self.cliptext_tokenizer(caption, truncation=True, max_length=self.max_src_length, return_length=True,
        #                                             return_overflowing_tokens=False, padding=False, return_tensors="pt")["input_ids"]
        #                 tokenized_text.append(region_coord)
        #                 tokenized_text.append(region_text[0,:])
        #                 str_caption += ' <%d> <%d> <%d> <%d> '%(quant_x0-self.box_offset,quant_y0-self.box_offset,quant_x1-self.box_offset,quant_y1-self.box_offset) + caption
        #             else:
        #                 box, caption = target["boxes"][box_sample_ii], target["box_caption"][box_sample_ii]
        #                 box = [float(x)/self.size for x in box]
        #                 region_word = self.process_spatial_word(box, caption, self.spatial_word)
        #                 region_word = self.pre_caption(region_word.lower(), self.max_src_length)
        #                 region_text = self.cliptext_tokenizer(region_word, truncation=True, max_length=self.max_src_length, return_length=True,
        #                                             return_overflowing_tokens=False, padding=False, return_tensors="pt")["input_ids"]
        #                 tokenized_text.append(region_text[0,:])
        #                 str_caption += region_word

        #     # if self.box_descp=='caption' and self.max_src_length==77:
        #     #     tokenized_text = torch.cat(tokenized_text, dim=0)[:self.max_src_length*8]
        #     #     pad_tokenized_text = torch.tensor([self.box_offset-1]*self.max_src_length*8).to(text_enc.device)    ## manual set extended pos emd length
        #     # else:
        #     tokenized_text = torch.cat(tokenized_text, dim=0)[:self.max_src_length]
        #     pad_tokenized_text = torch.tensor([self.box_offset-1]*self.max_src_length).to(text_enc.device)
        #     pad_tokenized_text[:len(tokenized_text)] = tokenized_text
        #     # print(pad_tokenized_text)
        #     # print(str_caption)
        #     # print('====')
        #     example["caption"] = pad_tokenized_text
        #     example["str_caption"] = str_caption
        # return example

    # #####################################
    # ## 32*32 / (640*480) = 0.003333333
    # ## 96*96 / (640*480) = 0.03
    # def process_spatial_word(self, box, caption, spatial_word_mode='all'):
    #     if spatial_word_mode == 'tag' or spatial_word_mode == 'caption':
    #         return '%s '%caption
    #     box_w, box_h = box[2]-box[0], box[3]-box[1]
    #     aspect, size = box_w / box_h, box_w*box_h
    #     box_cx, box_cy = (box[2]+box[0])/2., (box[3]+box[1])/2.
    #     size_word, aspect_word, location_word, tag_word = '', '', '', ''
    #     tag_word = caption

    #     if size<0.003333333: size_word = 'small'
    #     elif size>0.03: size_word = 'large'
    #     else: size_word = 'medium'

    #     if aspect>2.: aspect_word = 'long'
    #     elif aspect<0.5: aspect_word = 'tall'
    #     else: aspect_word = 'square'

    #     if box_cx<(1./3):
    #         if box_cy<(1./3): location_word = 'top left'
    #         elif box_cy>(2./3): location_word = 'bottom left'
    #         else: location_word = 'left'
    #     elif box_cx>(2./3):
    #         if box_cy<(1./3): location_word = 'top right'
    #         elif box_cy>(2./3): location_word = 'bottom right'
    #         else: location_word = 'right'
    #     else:
    #         if box_cy<(1./3): location_word = 'top'
    #         elif box_cy>(2./3): location_word = 'bottom'
    #         else: location_word = 'center'
    #     # prompt = '%s %s %s in the %s.'%(size_word, aspect_word, tag_word, location_word)
    #     prompt = '%s %s in the %s, %s '%(size_word, aspect_word, location_word, tag_word)
    #     if spatial_word_mode == 'all':
    #         return prompt

    # #####################################
    # def process_centercrop(self, boxes, imagehw, centercrop):
    #     h, w = float(imagehw[0]), float(imagehw[1])
    #     if centercrop:
    #         if w>=h:
    #             boxes[0] = (boxes[0]+h/2-w/2)/h
    #             boxes[2] = (boxes[2]+h/2-w/2)/h
    #             boxes[1], boxes[3] = boxes[1]/h, boxes[3]/h
    #         else:
    #             boxes[1] = (boxes[1]+w/2-h/2)/w
    #             boxes[3] = (boxes[3]+w/2-h/2)/w
    #             boxes[0], boxes[2] = boxes[0]/w, boxes[2]/w
    #     else:
    #         boxes[0], boxes[2] = boxes[0]/w, boxes[2]/w
    #         boxes[1], boxes[3] = boxes[1]/h, boxes[3]/h
    #     boxes[0], boxes[1] = max(0., boxes[0]), max(0., boxes[1])
    #     boxes[2], boxes[3] = max(0., boxes[2]), max(0., boxes[3])
    #     boxes[0], boxes[1] = min(1., boxes[0]), min(1., boxes[1])
    #     boxes[2], boxes[3] = min(1., boxes[2]), min(1., boxes[3])
    #     return boxes

    # def pre_caption(self, caption, max_words):
    #     caption = caption.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    #     caption = re.sub(
    #         r"\s{2,}",
    #         ' ',
    #         caption,
    #     )
    #     caption = caption.rstrip('\n')
    #     caption = caption.strip(' ')

    #     # truncate caption
    #     caption_words = caption.split(' ')
    #     if len(caption_words) > max_words:
    #         caption = ' '.join(caption_words[:max_words])

    #     return caption


