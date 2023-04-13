import argparse, os, sys, glob, json, re, random, math
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
import time
from pytorch_lightning import seed_everything
from datetime import datetime
from pathlib import Path
import pandas as pd

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from transformers import CLIPTokenizer, CLIPTextModel

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.eval()
    return model

def image_save_retry(img, path, n_retries=10, timeout=1):
    """
    Retry saving an image if it fails.
    img: PIL image
    """
    for i in range(n_retries):
        try:
            img.save(path)
            break
        except OSError:
            print(f"{path}, OSError, retrying in {timeout} seconds - {i+1}/{n_retries}")
            time.sleep(timeout)

        except SystemError as e:
            print(f'{path}, {e} - image size: {img.size}')
            exit()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--centercrop",
        action='store_true',
        help="centercrop",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--box_descp",
        type=str,
        default='tag',
        help="box_descp",
    )
    parser.add_argument(
        "--spatial_word",
        type=str,
        default=None,
        help="spatial_word",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference-reco.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    # parser.add_argument(
    #     "--cocogt",
    #     type=str,
    #     default="dataset/coco/coco_test30k",
    #     help="also crop gt coco image",
    # )    

    # parser.add_argument(
    #     "--embedding_path", 
    #     type=str, 
        # help="Path to a pre-trained embedding manager checkpoint")


    parser.add_argument(
        '--eval_data',
        type=str,
        default='clevr',
        choices=['clevr', 'clevr320', 'layoutbench_v0', 'layoutbench_v1', 'coco30k', 'coco_val2017'],
        help='Dataset to evaluate on')

    # parser.add_argument(
    #     '--old_clevr',
    #     action='store_true',
    #     help='Use old CLEVR dataset (v1.0)')

    parser.add_argument(
        "--clevr_dir", 
        type=str, 
        default="datasets/clevr_data",
        # default="datasets/layoutbench/clevr_blender293_w320h320/",
        help="Path to CLEVR dataset")

    parser.add_argument(
        "--clevr_dump_dir", 
        type=str, 
        default="eval_images_dump/clevr",
        help="Path to save GT / generated CLEVR images for Evaluation")

    # parser.add_argument(
    #     '--old_layoutbench',
    #     action='store_true',
    #     help='Use old LayoutBench dataset (v0.0)')

    parser.add_argument(
        "--layoutbench_dir", 
        type=str, 
        default="datasets/layoutbench",
        help="Path to LayoutBench dataset")

    parser.add_argument(
        "--skill_split",
        type=str,
        default='number_0-2',
        choices=[
        
        # layoutbench_v0
        'number_few', 'number_many',
        'position_boundary', 'position_center',
        'size_tiny', 'size_verylarge',
        'shape_horizontal', 'shape_vertical',
        'allskills_allsplits',

        # layoutbench_v1
        # 'number_0-2' 'number_3-5' 'number_6-8' 'number_9-10' 'number_11-13' 'number_14-16' 'position_boundary' 'position_center' 'position_random' 'size_020' 'size_035' 'size_050' 'size_070' 'size_090' 'size_110' 'size_130' 'size_150' 'shape_H3W1' 'shape_H2W1' 'shape_H1W1' 'shape_H1W2' 'shape_H1W3'
        'number_0-2_200', 'number_3-5_200', 'number_6-8_200', 'number_9-10_200', 'number_11-13_200', 'number_14-16_200', 'position_boundary_200', 'position_center_200', 'position_random_200', 'size_020_200', 'size_035_200', 'size_050_200', 'size_070_200', 'size_090_200', 'size_110_200', 'size_130_200', 'size_150_200', 'shape_H3W1_200', 'shape_H2W1_200', 'shape_H1W1_200', 'shape_H1W2_200', 'shape_H1W3_200',
        'number_few', 'number_many', 'position_boundary', 'position_center', 'size_tiny', 'size_large', 'shape_horizontal', 'shape_vertical'
    
        ],
        help='Skill split to evaluate on'
    )

    parser.add_argument(
        "--layoutbench_dump_dir",
        type=str,
        default="eval_images_dump/layoutbench/",
        help="Path to save GT / generated LayoutBench images for Evaluation")
    
    parser.add_argument(
        "--coco30k_dump_dir",
        type=str,
        default="eval_images_dump/coco30k/",
        help="Path to save GT / generated COCO30k images for Evaluation")
    
    # coco_val2017_dump_dir
    parser.add_argument(
        "--coco_val2017_dump_dir",
        type=str,
        default="eval_images_dump/coco_val2017/",
        help="Path to save GT / generated COCO_val2017 images for Evaluation")

    parser.add_argument(
        '--save_gt',
        action='store_true',
        help='Save GT images for CLEVR')

    parser.add_argument(
        '--save_bbox_viz',
        action='store_true',
        help='Save bbox visualization for CLEVR')
    
    parser.add_argument(
        '--save_intermediate',
        action='store_true',
        help='Save intermediate images for IterInpainting')

    parser.add_argument(
        '--name',
        type=str,
        default='reco',
    )

    parser.add_argument(
        '--iterinpaint_nopaste',
        action='store_true',
        help='Do not paste the intermediate image back to the original image')

    parser.add_argument(
        '--gt_only',
        action='store_true',
        help='Only save GT images')
    
    parser.add_argument(
        '--box_generation_order',
        type=str,
        default=None,
        choices=['random', 'top-down', 'bottom-up',
                 'large-to-small', 'small-to-large',
                 ],
    )

    parser.add_argument(
        '--gt_shuffle',
        action='store_true',
    )

    parser.add_argument(
        '--use_git_captions',
        action='store_true',
    )

    parser.add_argument(
        '--min_obj_size',
        type=int,
        default=5,
        help="Skip objects smaller than this size"
    )

    # parser.add_argument(
    #     '--batch_sizse',
    #     type=int,
    #     default=1,
    # )

    opt = parser.parse_args()

    print(opt)

    if opt.iterinpaint_nopaste:
        print("IterInpaint ---- No paste")

    # if opt.laion400m:
    #     print("Falling back to LAION 400M model...")
    #     opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
    #     opt.ckpt = "models/ldm/text2img-large/model.ckpt"
    #     opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)
    
    if not opt.gt_only:

        config = OmegaConf.load(f"{opt.config}")
        # max_src_length = 77
        # if opt.spatial_word is not None:
        #     # opt.box_descp = 'tag'
        #     opt.box_descp = 'caption'
        # # if opt.box_descp!='tag':
        # #     config.model.params.cond_stage_config.params['extend_outputlen']=385
        # #     config.model.params.cond_stage_config.params['max_length']=385
        # #     max_src_length = 385
        # if opt.config != 'configs/stable-diffusion/v1-inference.yaml':
        #     max_src_length = 1232 if '1232' in opt.ckpt else 616
        #     config.model.params.cond_stage_config.params['extend_outputlen']=max_src_length
        #     config.model.params.cond_stage_config.params['max_length']=max_src_length
        model = load_model_from_config(config, f"{opt.ckpt}")
        #model.embedding_manager.load(opt.embedding_path)

    rank = os.environ.get('RANK', '0')
    local_rank = os.environ.get('LOCAL_RANK', '0')
    world_size = os.environ.get('WORLD_SIZE', '1')

    rank = int(rank)
    local_rank = int(local_rank)
    world_size = int(world_size)

    print('==============================================')
    print(f"Model prepared  - global rank: {rank} / local rank: {local_rank} / world size: {world_size}")
    print('==============================================')
    # time.sleep(1)

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)

    device = torch.device(f"cuda:{local_rank}")

    # device = torch.device(local_rank)

    # device = torch.device("cuda")

    # print(device)

    torch.cuda.set_device(device)

    # device = 'cuda'

    if not opt.gt_only:

        model = model.to(device)

        if opt.plms:
            sampler = PLMSSampler(model)
        else:
            sampler = DDIMSampler(model)

    # os.makedirs(opt.outdir, exist_ok=True)
    # outpath = opt.outdir
    # batch_size = opt.n_samples

    print('==============================================')
    print(f"Data preprocessing  - global rank: {rank} / local rank: {local_rank} / world size: {world_size}")
    print('==============================================')
    # time.sleep(1)

    # Distributed Inference -> split data into world_size


    data = []
    if opt.eval_data in ['clevr', 'clevr320']:

        from pathlib import Path
        import pandas as pd

        clevr_dir = Path(opt.clevr_dir)
        print('clevr_dir', clevr_dir)

        split = 'val'
        print('Split:', split)

        # if opt.old_clevr:
        if opt.eval_data == 'clevr':
            
            clevr_df_path = clevr_dir / f'{split}_ann.json'
            clevr_df = pd.read_json(clevr_df_path)
            print('Loaded ', clevr_df_path, ' | shape: ' , clevr_df.shape)

            for i in range(len(clevr_df)):
                _datum = clevr_df.iloc[i]

                img_fname = _datum['id']
                img_path = clevr_dir / 'CLEVR_v1.0/images' / 'val' / img_fname

                objects = _datum['objects']

                # Assert that the bbox in xyxy, not xywh
                for obj in objects:
                    assert len(obj['box']) == 4

                    assert obj['box'][2] > obj['box'][0], f'x2 should be greater than x1, {obj["box"]}'
                    assert obj['box'][3] > obj['box'][1], f'y2 should be greater than y1, {obj["box"]}'

                data.append({
                    'img_path': img_path,
                    'objects': objects,
                })
        elif opt.eval_data == 'clevr320':
            import json

            scene_path = clevr_dir / split / 'scenes.json'
            with open(scene_path) as f:
                scenes = json.load(f)
            print('Loaded ', scene_path, ' | shape: ', len(scenes['scenes']))

            image_dir = clevr_dir / split / 'images'

            for datum in scenes['scenes']:
                img_fname = datum['image_filename']
                img_path = image_dir / img_fname

                # xywh -> xyxy
                boxes = [obj['bbox'] for obj in datum['objects']]
                boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in boxes]

                obj_names = [f"{obj['color']} {obj['material']} {obj['shape']}" for obj in datum['objects']]

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


                data.append({
                    'img_path': img_path,
                    'objects': objects,
                })


    # elif opt.eval_data == 'layoutbench':
    elif opt.eval_data in ['layoutbench_v0', 'layoutbench_v1']:

        from pathlib import Path
        import json

        layoutbench_dir = Path(opt.layoutbench_dir)

        # if opt.old_layoutbench:
        if opt.eval_data == 'layoutbench_v0':

            skill, split = opt.skill_split.split('_')
            print(f"Skil: {skill} / split: {split}")            
            
            scene_dir = layoutbench_dir / 'v0_blender293_w320h320_1000images' / skill / split 
            scenes_path = scene_dir / 'scenes.json'
            scenes_data = json.load(open(scenes_path))
            print('Loaded ', scenes_path, ' | shape: ' , len(scenes_data['scenes']))

            image_dir = scene_dir / 'images'
        # else:
        elif opt.eval_data == 'layoutbench_v1':

            assert opt.skill_split in ['number_0-2_200', 'number_3-5_200', 'number_6-8_200', 'number_9-10_200', 'number_11-13_200', 'number_14-16_200', 'position_boundary_200', 'position_center_200', 'position_random_200', 'size_020_200', 'size_035_200', 'size_050_200', 'size_070_200', 'size_090_200', 'size_110_200', 'size_130_200', 'size_150_200', 'shape_H3W1_200', 'shape_H2W1_200', 'shape_H1W1_200', 'shape_H1W2_200', 'shape_H1W3_200', 'number_few', 'number_many', 'position_boundary', 'position_center', 'size_tiny', 'size_large', 'shape_horizontal', 'shape_vertical'], f"Invalid skill_split: {opt.skill_split}"
            
            skill, subsplit = opt.skill_split.split('_', 1)

            # datasets/layoutbench/v1_blender293_w320h320_200images/val/number/scenes_14-16.json

            split = 'val'
            scene_dir = layoutbench_dir / 'v1_blender293_w320h320_2000images' / split / skill
            scenes_path = scene_dir / f'scenes_{opt.skill_split}.json'

            scenes_data = json.load(open(scenes_path))
            print('Loaded ', scenes_path, ' | shape: ', len(scenes_data['scenes']))

            image_dir = scene_dir / 'images'

        for scene_datum in scenes_data['scenes']:

            # {'split': 'vertical',
            # 'image_index': 2,
            # 'image_filename': 'CLEVR_vertical_000002.png',
            # 'objects': [{'shape': 'sphere',
            # 'size': 'large',
            # 'material': 'rubber',
            # '3d_coords': [0.7150912880897522, -2.186058282852173, 3.49804949760437],
            # 'rotation': 34.010319868420616,
            # 'pixel_coords': [104, 58, 7.829171657562256],
            # 'color': 'yellow',
            # 'scale': 0.7,
            # 'bbox': [67, 0, 69, 193]},
            # {'shape': 'cube',

            img_path = image_dir / scene_datum['image_filename']
            objects = []
            for obj in scene_datum['objects']:
                text = f"{obj['color']} {obj['material']} {obj['shape']}"
                x1, y1, w, h = obj['bbox']

                # xywh -> xyxy
                box = [x1, y1, x1+w, y1+h]

                objects.append({'text': text, 'box': box})

            for obj in objects:
                assert len(obj['box']) == 4

                assert obj['box'][2] > obj['box'][0], f'x2 should be greater than x1, {obj["box"]}'
                assert obj['box'][3] > obj['box'][1], f'y2 should be greater than y1, {obj["box"]}'

            data.append({
                'img_path': img_path,
                'objects': objects,
            })

    elif opt.eval_data == 'coco30k':
        from pycocotools.coco import COCO
        from pathlib import Path
        import pandas as pd
        
        uid_caption_path = Path('datasets/COCO/FID/uid_caption.csv')
        uid_caption_df = pd.read_csv(uid_caption_path)

        coco_dir = Path('datasets/COCO')
        coco_ann_dir = coco_dir / 'annotations'

        coco_instances_val2014_path = coco_ann_dir / 'instances_val2014.json'
        coco_captions_val2014_path = coco_ann_dir / 'captions_val2014.json'

        coco_captions = COCO(coco_captions_val2014_path)
        coco_instances = COCO(coco_instances_val2014_path)

        print("Mapping category ids to names...")
        cat_list = coco_instances.dataset['categories']
        cat2id = {cat_datum['name']: cat_datum['id'] for cat_datum in cat_list}
        catid2name = {v: k for k, v in cat2id.items()}

        coco30k_ids = uid_caption_df.uid.apply(lambda x: int(x.split('_')[0])).tolist()
        assert len(coco30k_ids) == 30000, f"Invalid coco30k_ids: {len(coco30k_ids)}"


        if opt.use_git_captions:
            git_caption_ann_path = coco_dir / 'coco_allbox_gitcaption.json'
            print('Using GIT captions - crop_caption')
            coco_allbox_gitcaption = json.load(open(git_caption_ann_path, 'r'))

        # for coco_id in coco30k_ids:
        for row in uid_caption_df.itertuples():
            # row: (index, uid, caption)
            # Pandas(Index=0, uid='346904_mscoco_0', caption='A bus driving down a road by a building.')

            img_id = int(row.uid.split('_')[0])

            # 1. text input (caption)
            cap_annIds = coco_captions.getAnnIds(imgIds=img_id)
            cap_anns = coco_captions.loadAnns(cap_annIds)
            captions = [ann['caption'] for ann in cap_anns]

            # 2. image information
            img_datum = coco_captions.loadImgs(img_id)[0]

            # 3. scene information
            box_annIds = coco_instances.getAnnIds(imgIds=img_id)
            box_anns = coco_instances.loadAnns(box_annIds)

            # if len(box_anns) == 0:
            #     # print(f'No scene elements in image {img_id}!')
            #     n_img_with_no_scene_elements += 1
            #     continue

            if opt.use_git_captions:
                git_caption_ann = coco_allbox_gitcaption[str(img_id)]

                boxid2gitcaption = {}
                for box in git_caption_ann['box']:
                    boxid2gitcaption[box['boxid']] = box['crop_caption']

            scene = []
            for box_ann in box_anns:
                #  {
                #   'segmentation: [....],
                #   'area': 5999.544500000001,
                #   'iscrowd': 0,
                #   'image_id': 324158,
                #   'bbox': [202.71, 96.55, 71.78, 153.14],
                #   'category_id': 1,
                #   'id': 2162813
                # }

                # bbox format: [x, y, w, h] -> [x1, y1, x2, y2]
                x, y, w, h = box_ann['bbox']
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                assert x1 >= 0 and y1 >= 0 and x2 <= img_datum['width'] and y2 <= img_datum['height']

                if opt.use_git_captions:
                    box_caption = boxid2gitcaption[box_ann['id']]
                else:
                    box_caption = catid2name[box_ann['category_id']]

                box = {
                    'text': box_caption,
                    # 'box': box_ann['bbox'],
                    'box': [x1, y1, x2, y2],
                }
                scene.append(box)

            img_path = coco_dir / 'images' / 'val2014' / img_datum['file_name']
            objects = scene

            caption = row.caption

            data.append({
                'img_path': img_path,
                'objects': objects,
                'caption': caption,
            })
    elif opt.eval_data == 'coco_val2017':
        from pycocotools.coco import COCO
        from pathlib import Path
        import pandas as pd

        coco_dir = Path('datasets/COCO')
        coco_ann_dir = coco_dir / 'annotations'

        coco_instances_val2017_path = coco_ann_dir / 'instances_val2017.json'
        coco_captions_val2017_path = coco_ann_dir / 'captions_val2017.json'

        coco_captions = COCO(coco_captions_val2017_path)
        coco_instances = COCO(coco_instances_val2017_path)

        print("Mapping category ids to names...")
        cat_list = coco_instances.dataset['categories']
        cat2id = {cat_datum['name']: cat_datum['id'] for cat_datum in cat_list}
        catid2name = {v: k for k, v in cat2id.items()}

        if opt.use_git_captions:
            git_caption_ann_path = coco_dir / 'coco_allbox_gitcaption.json'
            print('Using GIT captions - crop_caption')
            coco_allbox_gitcaption = json.load(open(git_caption_ann_path, 'r'))

        img_ids = list(coco_instances.getImgIds())
        assert len(img_ids) == 5000, f"Invalid img_ids: {len(img_ids)}"

        for img_id in coco_instances.getImgIds():
            # 1. text input (caption)
            cap_annIds = coco_captions.getAnnIds(imgIds=img_id)
            cap_anns = coco_captions.loadAnns(cap_annIds)
            captions = [ann['caption'] for ann in cap_anns]

            # 2. image information
            img_datum = coco_captions.loadImgs(img_id)[0]

            # 3. scene information
            box_annIds = coco_instances.getAnnIds(imgIds=img_id)
            box_anns = coco_instances.loadAnns(box_annIds)

            if opt.use_git_captions:
                git_caption_ann = coco_allbox_gitcaption[str(img_id)]

                boxid2gitcaption = {}
                for box in git_caption_ann['box']:
                    boxid2gitcaption[box['boxid']] = box['crop_caption']

            scene = []
            for box_ann in box_anns:
                #  {
                #   'segmentation: [....],
                #   'area': 5999.544500000001,
                #   'iscrowd': 0,
                #   'image_id': 324158,
                #   'bbox': [202.71, 96.55, 71.78, 153.14],
                #   'category_id': 1,
                #   'id': 2162813
                # }

                # bbox format: [x, y, w, h] -> [x1, y1, x2, y2]
                x, y, w, h = box_ann['bbox']
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                assert x1 >= 0 and y1 >= 0 and x2 <= img_datum['width'] and y2 <= img_datum['height']

                if opt.use_git_captions:
                    box_caption = boxid2gitcaption[box_ann['id']]
                else:
                    box_caption = catid2name[box_ann['category_id']]

                box = {
                    'text': box_caption,
                    # 'box': box_ann['bbox'],
                    'box': [x1, y1, x2, y2],
                }
                scene.append(box)

            img_path = coco_dir / 'images' / 'val2017' / img_datum['file_name']
            objects = scene

            # caption = row.caption
            caption = captions[0]

            data.append({
                'img_path': img_path,
                'objects': objects,
                'caption': caption,
            })




    else:
        print(f"Invalid eval_data: {opt.eval_data}")
        raise NotImplementedError

    if opt.gt_shuffle:
        new_data = []
        # shuffle img_path and objects

        all_img_paths = [datum['img_path'] for datum in data]
        all_objects = [datum['objects'] for datum in data]

        import random
        # random.shuffle(all_img_paths)
        random.shuffle(all_objects)

        for img_path, objects in zip(all_img_paths, all_objects):
            new_data.append({
                'img_path': img_path,
                'objects': objects,
            })
        
        data = new_data
        print('Shuffled data')        


    n_total_data = len(data)
    start_idx = n_total_data // world_size*rank
    end_idx = n_total_data // world_size*(rank+1) if rank!=world_size-1 else n_total_data
    chunked_data = data[start_idx:end_idx]

    print('==============================================')
    print(f"Data prepared  - global rank: {rank} / local rank: {local_rank} / world size: {world_size}")
    print(f'Start index: {start_idx} / End index: {end_idx}')
    print('==============================================')
    # time.sleep(1)
            

    size = 512
    print(f'Image size: {size}x{size}')

    # if opt.eval_data == 'clevr':
    if opt.eval_data in ['clevr', 'clevr320']:
        data_dump_dir = Path(opt.clevr_dump_dir)
    # elif opt.eval_data == 'layoutbench':
    elif opt.eval_data in ['layoutbench_v0', 'layoutbench_v1']:
        data_dump_dir = Path(opt.layoutbench_dump_dir) / opt.skill_split

    elif opt.eval_data == 'coco30k':
        data_dump_dir = Path(opt.coco30k_dump_dir)

    elif opt.eval_data == 'coco_val2017':
        data_dump_dir = Path(opt.coco_val2017_dump_dir)

        
    data_dump_dir.mkdir(exist_ok=True, parents=True)

    gt_img_dump_dir = data_dump_dir / f'gt_{size}x{size}'
    gt_img_dump_dir.mkdir(exist_ok=True, parents=True)
    print(f"GT images -> {gt_img_dump_dir}")

    gt_box_dump_dir = data_dump_dir / f'gt_{size}x{size}_boxes'
    gt_box_dump_dir.mkdir(exist_ok=True, parents=True)
    print(f"GT boxes -> {gt_box_dump_dir}")

    gen_img_dump_dir = data_dump_dir / f'{opt.name}'
    gen_img_dump_dir.mkdir(exist_ok=True, parents=True)
    print(f"Generated images -> {gen_img_dump_dir}")

    gen_box_dump_dir = data_dump_dir / f'{opt.name}_boxes'
    gen_box_dump_dir.mkdir(exist_ok=True, parents=True)
    print(f"Generated boxes -> {gen_box_dump_dir}")

    if opt.save_bbox_viz:
        gt_bbox_viz_dump_dir = data_dump_dir / f'gt_{size}x{size}_bbox_viz'
        gt_bbox_viz_dump_dir.mkdir(exist_ok=True, parents=True)
        print(f"GT bbox viz -> {gt_bbox_viz_dump_dir}")

        gen_bbox_viz_dump_dir = data_dump_dir / f'{opt.name}_bbox_viz'
        gen_bbox_viz_dump_dir.mkdir(exist_ok=True, parents=True)
        print(f"Generated bbox viz -> {gen_bbox_viz_dump_dir}")

    if opt.save_intermediate:
        assert 'iterinpaint' in opt.config, 'Intermediate images can only be saved for iterative inpainting'

        gen_intermediate_img_dump_dir = data_dump_dir / f'{opt.name}_iter'
        gen_intermediate_img_dump_dir.mkdir(exist_ok=True, parents=True)
        print(f"Generated intermediate images -> {gen_intermediate_img_dump_dir}")

    # Also save to a common directory for all skills - for global FID calculation
    if opt.eval_data in ['layoutbench_v0', 'layoutbench_v1']:
        allskill_data_dump_dir = Path(opt.layoutbench_dump_dir) / 'all'
        allskill_data_dump_dir.mkdir(exist_ok=True, parents=True)

        allskill_gt_img_dump_dir = allskill_data_dump_dir / f'gt_{size}x{size}'
        allskill_gt_img_dump_dir.mkdir(exist_ok=True, parents=True)
        print(f"GT images -> {allskill_gt_img_dump_dir}")

        allskill_gt_box_dump_dir = allskill_data_dump_dir / f'gt_{size}x{size}_boxes'
        allskill_gt_box_dump_dir.mkdir(exist_ok=True, parents=True)
        print(f"GT boxes -> {allskill_gt_box_dump_dir}")

        allskill_gen_img_dump_dir = allskill_data_dump_dir / f'{opt.name}'
        allskill_gen_img_dump_dir.mkdir(exist_ok=True, parents=True)
        print(f"Generated images -> {allskill_gen_img_dump_dir}")

        allskill_gen_box_dump_dir = allskill_data_dump_dir / f'{opt.name}_boxes'
        allskill_gen_box_dump_dir.mkdir(exist_ok=True, parents=True)
        print(f"Generated boxes -> {allskill_gen_box_dump_dir}")

        # if opt.save_bbox_viz:
        #     allskill_gt_bbox_viz_dump_dir = allskill_data_dump_dir / f'gt_{size}x{size}_bbox_viz'
        #     allskill_gt_bbox_viz_dump_dir.mkdir(exist_ok=True, parents=True)
        #     print(f"GT bbox viz -> {allskill_gt_bbox_viz_dump_dir}")

        #     allskill_gen_bbox_viz_dump_dir = allskill_data_dump_dir / f'{opt.name}_bbox_viz'
        #     allskill_gen_bbox_viz_dump_dir.mkdir(exist_ok=True, parents=True)
        #     print(f"Generated bbox viz -> {allskill_gen_bbox_viz_dump_dir}")

        # if opt.save_intermediate:
        #     assert 'iterinpaint' in opt.config, 'Intermediate images can only be saved for iterative inpainting'

        #     allskill_gen_intermediate_img_dump_dir = allskill_data_dump_dir / f'{opt.name}_iter'
        #     allskill_gen_intermediate_img_dump_dir.mkdir(exist_ok=True, parents=True)
        #     print(f"Generated intermediate images -> {allskill_gen_intermediate_img_dump_dir}")



    print('==============================================')
    print(f"Created Image Saving Directories  - global rank: {rank} / local rank: {local_rank} / world size: {world_size}")
    print('==============================================')
    # time.sleep(1)


    from ldm.gen_utils import sample_images, prepare_text, encode_scene, prepare_clip_tokenizer, sample_images_iterative_inpaint
    from ldm.viz_utils import plot_results, fig2img

    import ldm.data.transforms as T
    from PIL import Image
    
    transforms = T.Compose(
            [
                T.RandomResize([size]),
                T.CenterCrop((size, size)),
            ]
        )

    # tokenizer = prepare_clip_tokenizer(
    #     pretrained_model_name_or_path="openai/clip-vit-large-patch14",
    #     with_bbox=config.model.params.cond_stage_config.params.get('with_bbox', False),
    #     num_bins=config.model.params.cond_stage_config.params.get('num_bins', 1000),
    #     with_class_embedding=config.model.params.cond_stage_config.params.get('with_class_embedding', False),
    #     num_classes=config.model.params.cond_stage_config.params.get('num_classes', 48),
    #     max_length=config.model.params.cond_stage_config.params.get('extend_outputlen', 616),
    # )

    print('==============================================')
    print(f"Sampling starts  - global rank: {rank} / local rank: {local_rank} / world size: {world_size}")
    print('==============================================')
    # time.sleep(1)

    if world_size <= 8:
        disable = rank!=0
    else:
        disable = False

    def get_current_timestamp():
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for idx, datum in tqdm(enumerate(chunked_data),
        desc=f'Generating {len(chunked_data)} images (rank {rank} of {world_size}) - {get_current_timestamp()}',
        total=len(chunked_data),
        disable=disable):

        if world_size > 8 and not disable:
            if idx % 10 == 0:
                current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f'{current_timestamp} |  Rank {rank} / {world_size} Progress: {idx} / {len(chunked_data)}')
        
        img_path = datum['img_path']
        img_fname = Path(img_path).name
        img_stem = Path(img_path).stem

        gt_img = Image.open(img_path).convert('RGB')

        region_captions = [d['text'] for d in datum['objects']]
        bboxes = [d['box'] for d in datum['objects']]

        target = {
            # "image_id": datum['id'],
            "boxes": torch.tensor(bboxes, dtype=torch.float32).reshape(-1, 4),
            "box_caption": region_captions,
        }
        gt_img_transformed, target = transforms(gt_img, target)
        W, H = gt_img_transformed.size
        assert H == size and W == size, f"Image size mismatch: {H}x{W} vs {size}x{size}"

        assert len(target['boxes']) == len(target['box_caption']), f"Number of boxes and captions mismatch: {len(target['boxes'])} vs {len(target['box_caption'])}"

        objects = []
        for j in range(len(target['boxes'])):
            objects.append({
                'caption': target['box_caption'][j],
                'bbox': target['boxes'][j].tolist(),
            })

        out = encode_scene(objects, H=H, W=W,  src_bbox_format='xyxy', tgt_bbox_format='xyxy')

        text = out['text']
        box_captions = out['box_captions']
        boxes_normalized = out['boxes_normalized']
        boxes_unnormalized = out['boxes_unnormalized']
        
        prefix_global_caption = False
        global_caption = None
        # if opt.eval_data == 'coco30k':
        if 'caption' in datum:
            global_caption = datum['caption']
            prefix_global_caption = True

        prompts = [text]

        if not opt.gt_only:

            if 'iterinpaint' in opt.config:

                if opt.eval_data == 'coco30k':
                    background_instruction = 'Complete the image'
                else:
                    background_instruction = 'Add gray background'

                generated = sample_images_iterative_inpaint(
                    box_captions, boxes_unnormalized,
                    sampler, opt,
                    gen_batch_size=1,
                    size=size,
                    # paste=True,
                    paste=not opt.iterinpaint_nopaste,
                    box_generation_order=opt.box_generation_order,
                    global_caption=global_caption,
                    prefix_global_caption=prefix_global_caption,
                    background_instruction=background_instruction,
                    # verbose=False
                    verbose=True
                    )
                
                # 'context_imgs': context_imgs,
                # 'mask_imgs': mask_imgs,
                # 'prompts': prompts,
                # 'generated_images': generated_images,
                # 'final_image': generated_image,

                generated_img = generated['final_image']

                if opt.save_intermediate:
                    context_imgs = generated['context_imgs']
                    mask_imgs = generated['mask_imgs']
                    generated_images = generated['generated_images']

                    prompts = generated['prompts']

                    intermediate_img_dir = gen_intermediate_img_dump_dir / f'{img_stem}'
                    intermediate_img_dir.mkdir(exist_ok=True, parents=True)

                    for j, (context_img, mask_img, generated_image) in enumerate(zip(context_imgs, mask_imgs, generated_images)):

                        context_img_path = intermediate_img_dir / f'{j}_context.png'
                        mask_img_path = intermediate_img_dir / f'{j}_mask.png'
                        generated_image_path = intermediate_img_dir / f'{j}_generated.png'

                        # context_img.save(context_img_path)
                        # mask_img.save(mask_img_path)
                        # generated_image.save(generated_image_path)
                        image_save_retry(context_img, context_img_path)
                        image_save_retry(mask_img, mask_img_path)
                        image_save_retry(generated_image, generated_image_path)

                    # Save prompts 
                    prompts_path = intermediate_img_dir / 'prompts.txt'
                    with open(prompts_path, 'w') as f:
                        for prompt in prompts:
                            f.write(prompt + '\n')

            else:

                if prefix_global_caption:
                    EOS_token = '<|endoftext|>'
                    prompt = f'{global_caption} {EOS_token} {text}'
                    prompts = [prompt]
                generated_img = sample_images(
                    prompts, sampler, opt,
                    gen_batch_size=1, verbose=False)[0]

        if opt.save_gt:
            if 'jpg' in img_fname:
                gt_img_transformed = gt_img_transformed.convert('RGB')

            # Save GT image
            gt_path = gt_img_dump_dir / img_fname
            # gt_img_transformed.save(gt_path)
            image_save_retry(gt_img_transformed, gt_path)

            # Save allskills
            if opt.eval_data in ['layoutbench_v0', 'layoutbench_v1']:
                gt_path = allskill_gt_img_dump_dir / img_fname
                # gt_img.save(gt_path)
                image_save_retry(gt_img, gt_path)

            # Save GT boxes
            for j, box_caption in enumerate(box_captions):
                box_path = gt_box_dump_dir / f'{img_stem}_{j}_{box_caption}.png'

                box_unnormalized = boxes_unnormalized[j]

                box_img = gt_img_transformed.crop(box_unnormalized)

                if (box_unnormalized[0] < 0 or
                    box_unnormalized[1] < 0 or
                    box_unnormalized[2] > gt_img_transformed.size[0] or
                    box_unnormalized[3] > gt_img_transformed.size[1] or

                    box_unnormalized[2] - box_unnormalized[0] <= opt.min_obj_size or
                    box_unnormalized[3] - box_unnormalized[1] <= opt.min_obj_size or
                    
                    box_img.size[0] <= opt.min_obj_size or
                    box_img.size[1] <= opt.min_obj_size
                    ):
                    print(f'Warning: box {j} of {img_fname} is out of bounds - box: {box_unnormalized} img: {box_img.size}')
                    pass
                else:
                    # box_img.save(box_path)
                    image_save_retry(box_img, box_path)

            # Save allskills
            if opt.eval_data in ['layoutbench_v0', 'layoutbench_v1']:
                for j, box_caption in enumerate(box_captions):
                    box_path = allskill_gt_box_dump_dir / f'{img_stem}_{j}_{box_caption}.png'

                    box_unnormalized = boxes_unnormalized[j]

                    box_img = gt_img.crop(box_unnormalized)
                    # box_img.save(box_path)
                    image_save_retry(box_img, box_path)

            # Save GT bbox viz
            if opt.save_bbox_viz:
                fig = plot_results(
                    gt_img_transformed,
                    boxes_normalized,
                    box_captions,
                )
                gt_bbox_viz_img = fig2img(fig)
                gt_bbox_viz_path = gt_bbox_viz_dump_dir / img_fname
                if 'jpg' in img_fname:
                    gt_bbox_viz_img = gt_bbox_viz_img.convert('RGB')
                # gt_bbox_viz_img.save(gt_bbox_viz_path)
                image_save_retry(gt_bbox_viz_img, gt_bbox_viz_path)

        if not opt.gt_only:
            if 'jpg' in img_fname:
                generated_img = generated_img.convert('RGB')

            # Save generated image
            gen_path = gen_img_dump_dir / img_fname
            # generated_img.save(gen_path)
            image_save_retry(generated_img, gen_path)

            # Save allskills
            if opt.eval_data in ['layoutbench_v0', 'layoutbench_v1']:
                gen_path = allskill_gen_img_dump_dir / img_fname
                # generated_img.save(gen_path)
                image_save_retry(generated_img, gen_path)

            # Save generated boxes
            for j, box_caption in enumerate(box_captions):
                box_path = gen_box_dump_dir / f'{img_stem}_{j}_{box_caption}.png'

                box_unnormalized = boxes_unnormalized[j]

                box_img = generated_img.crop(box_unnormalized)

                if (box_unnormalized[0] < 0 or
                    box_unnormalized[1] < 0 or
                    box_unnormalized[2] > generated_img.size[0] or
                    box_unnormalized[3] > generated_img.size[1] or

                    box_unnormalized[2] - box_unnormalized[0] <= opt.min_obj_size or
                    box_unnormalized[3] - box_unnormalized[1] <= opt.min_obj_size or

                    box_img.size[0] <= opt.min_obj_size or
                    box_img.size[1] <= opt.min_obj_size
                    ):
                    print(f'Warning: box {j} of {img_fname} is out of bounds - box: {box_unnormalized} img: {box_img.size}')
                    pass
                else:
                    
                    # box_img.save(box_path)
                    image_save_retry(box_img, box_path)

            # Save allskills
            if opt.eval_data in ['layoutbench_v0', 'layoutbench_v1']:
                for j, box_caption in enumerate(box_captions):
                    box_path = allskill_gen_box_dump_dir / f'{img_stem}_{j}_{box_caption}.png'

                    box_unnormalized = boxes_unnormalized[j]

                    box_img = generated_img.crop(box_unnormalized)
                    # box_img.save(box_path)
                    image_save_retry(box_img, box_path)

            # Save generated bbox viz
            if opt.save_bbox_viz:
                fig = plot_results(
                    generated_img,
                    boxes_normalized,
                    box_captions,
                )
                gen_bbox_viz_img = fig2img(fig)
                gen_bbox_viz_path = gen_bbox_viz_dump_dir / img_fname
                if 'jpg' in img_fname:
                    gen_bbox_viz_img = gen_bbox_viz_img.convert('RGB')
                # gen_bbox_viz_img.save(gen_bbox_viz_path)
                image_save_retry(gen_bbox_viz_img, gen_bbox_viz_path)

    print('==============================================')
    print(f"Sampling Done!  - global rank: {rank} / local rank: {local_rank} / world size: {world_size}")
    print('==============================================')

        


if __name__ == "__main__":
    main()
