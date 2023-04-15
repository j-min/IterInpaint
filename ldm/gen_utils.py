import torch
from torch import autocast
from torch.utils.data import Dataset
import json
import pandas as pd
from PIL import Image
from PIL import ImageDraw
from pathlib import Path
import numpy as np
import random
import copy
from einops import repeat

from einops import rearrange
from tqdm import tqdm, trange
from contextlib import contextmanager, nullcontext
from pytorch_lightning import seed_everything


def prepare_clip_tokenizer(
    pretrained_model_name_or_path="openai/clip-vit-large-patch14",
    with_bbox=False,
    num_bins=1000,
    with_class_embedding=False,
    num_classes=48,
    max_length=616
    ):

    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path
    )

    if with_bbox:
        print(f'Updating {num_bins} tokens to tokenizer')
        for i in range(num_bins):
            token = f'<bin{str(i).zfill(3)}>'
            tokenizer.add_tokens(token)

    if with_class_embedding:
        print(f'Updating {num_classes} tokens to tokenizer')
        for i in range(num_classes):
            token = f'<class{str(i).zfill(3)}>'
            tokenizer.add_tokens(token)

    if tokenizer.model_max_length != max_length:
        print(f"Updating Tokenizer max length: {tokenizer.model_max_length} to {max_length}")
        tokenizer.model_max_length = max_length
    
    return tokenizer

def prepare_text(
    box_captions=[],
    box_normalized=[],
    global_caption=None,
    # image_resolution=512,
    text_reco=True,
    num_bins=1000,
    # tokenizer=None

    spatial_text=False,
    ):

    # Describe box shape in text
    if spatial_text:
        # box_descriptions = []
        # for box_sample_ii in range(len(box_captions)):
        #     box = box_normalized[box_sample_ii]
        #     box_caption = box_captions[box_sample_ii]
        #     box_description = prepare_spatial_description(box, box_caption)
        #     box_descriptions.append(box_description)
        # text = " ".join(box_descriptions)
        raise NotImplementedError

    # Describe box 
    else:

        box_captions_with_coords = []

        if isinstance(box_normalized, torch.Tensor):
            box_normalized = box_normalized.tolist()

        for box_sample_ii in range(len(box_captions)):

            box = box_normalized[box_sample_ii]
            box_caption = box_captions[box_sample_ii]

            # print(box_caption)

            # quantize into bins
            quant_x0 = int(round((box[0] * (num_bins - 1))))
            quant_y0 = int(round((box[1] * (num_bins - 1))))
            quant_x1 = int(round((box[2] * (num_bins - 1))))
            quant_y1 = int(round((box[3] * (num_bins - 1))))

            if text_reco:
                # ReCo format
                # Add SOS/EOS before/after regional caption
                SOS_token = '<|startoftext|>'
                EOS_token = '<|endoftext|>'
                box_captions_with_coords += [
                    f"<bin{str(quant_x0).zfill(3)}>",
                    f"<bin{str(quant_y0).zfill(3)}>",
                    f"<bin{str(quant_x1).zfill(3)}>",
                    f"<bin{str(quant_y1).zfill(3)}>",
                    SOS_token,
                    box_caption,
                    EOS_token
                ]

            else:
                box_captions_with_coords += [
                    f"<bin{str(quant_x0).zfill(3)}>",
                    f"<bin{str(quant_y0).zfill(3)}>",
                    f"<bin{str(quant_x1).zfill(3)}>",
                    f"<bin{str(quant_y1).zfill(3)}>",
                    box_caption
                ]

        text = " ".join(box_captions_with_coords)

    if global_caption is not None:
        # Global caption
        if text_reco:
            # ReCo format
            # Add SOS/EOS before/after regional caption
            # SOS_token = '<|startoftext|>'
            EOS_token = '<|endoftext|>'
            # global_caption = f"{SOS_token} {global_caption} {EOS_token}"

            # SOS token will be automatically added
            global_caption = f"{global_caption} {EOS_token}"

        text = f"{global_caption} {text}"

    return text


def encode_scene(obj_list, H=320, W=320, src_bbox_format='xywh', tgt_bbox_format='xyxy'):
    """Encode scene into text and bounding boxes
    Args:
        obj_list: list of dicts
            Each dict has keys:
                
                'color': str
                'material': str
                'shape': str
                or 
                'caption': str

                and

                'bbox': list of 4 floats (unnormalized)
                    [x0, y0, x1, y1] or [x0, y0, w, h]
    """
    box_captions = []
    for obj in obj_list:
        if 'caption' in obj:
            box_caption = obj['caption']
        else:
            box_caption = f"{obj['color']} {obj['material']} {obj['shape']}"
        box_captions += [box_caption]
    
    assert src_bbox_format in ['xywh', 'xyxy'], f"src_bbox_format must be 'xywh' or 'xyxy', not {src_bbox_format}"
    assert tgt_bbox_format in ['xywh', 'xyxy'], f"tgt_bbox_format must be 'xywh' or 'xyxy', not {tgt_bbox_format}"

    boxes_unnormalized = []
    boxes_normalized = []
    for obj in obj_list:
        if src_bbox_format == 'xywh':
            x0, y0, w, h = obj['bbox']
            x1 = x0 + w
            y1 = y0 + h
        elif src_bbox_format == 'xyxy':
            x0, y0, x1, y1 = obj['bbox']
            w = x1 - x0
            h = y1 - y0
        assert x1 > x0, f"x1={x1} <= x0={x0}"
        assert y1 > y0, f"y1={y1} <= y0={y0}"
        assert x1 <= W, f"x1={x1} > W={W}"
        assert y1 <= H, f"y1={y1} > H={H}"

        if tgt_bbox_format == 'xywh':
            bbox_unnormalized = [x0, y0, w, h]
            bbox_normalized = [x0 / W, y0 / H, w / W, h / H]

        elif tgt_bbox_format == 'xyxy':
            bbox_unnormalized = [x0, y0, x1, y1]
            bbox_normalized = [x0 / W, y0 / H, x1 / W, y1 / H]
            
        boxes_unnormalized += [bbox_unnormalized]
        boxes_normalized += [bbox_normalized]

    assert len(box_captions) == len(boxes_normalized), f"len(box_captions)={len(box_captions)} != len(boxes_normalized)={len(boxes_normalized)}"
        
        
    text = prepare_text(box_captions, boxes_normalized)
    
    out = {}
    out['text'] = text
    out['box_captions'] = box_captions
    out['boxes_normalized'] = boxes_normalized
    out['boxes_unnormalized'] = boxes_unnormalized
        
    return out

def grouper(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image).to(dtype=torch.float32)/127.5-1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
            "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
            "txt": num_samples * [txt],
            "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
            "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
            }
    return batch

def sample_images(prompts, sampler, opt, mask_img=None, context_img=None, gen_batch_size=1, verbose=True):
    """Generate images from a list of prompts."""

    # opt example
    # Namespace(C=4, H=512, W=512, ckpt='models/ldm/stable-diffusion-v1/model.ckpt',
    # config='configs/stable-diffusion/v1-inference-box.yaml',
    # ddim_eta=0.0, ddim_steps=50, embedding_path=None,
    # f=8, fixed_code=False, from_file=None, laion400m=False,
    # n_iter=1, n_rows=0, n_samples=1, outdir='outputs/', plms=True, precision='autocast',
    # prompt='a painting of a virus monster playing guitar', scale=2.0, seed=42, skip_grid=False, skip_save=False)

    # model.first_stage_model.float();

    model = sampler.model

    # h = 512
    # w = 512
    h = opt.H
    w = opt.W

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                # tic = time.time()
                all_samples = list()
                                
                for batch_text in grouper(prompts, gen_batch_size):
                    
                    current_bsz = len(batch_text)

                    # For classifier free quidance
                    uc = None
                    
                    if mask_img is not None:
                        assert context_img is not None, "context_img must be provided if mask_img is provided"

                        assert current_bsz == 1

                        image = context_img
                        mask = mask_img

                        batch = make_batch_sd(image, mask, txt=batch_text[0], device=model.device, num_samples=current_bsz)

                        
                        num_samples = current_bsz

                        c_cross = model.cond_stage_model.encode(batch["txt"])
     
                        c_cat = list()
                        for ck in model.concat_keys:
                            cc = batch[ck].float()
                            if ck != model.masked_image_key:
                                bchw = [num_samples, 4, h//8, w//8]
                                cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                            else:
                                cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
                            c_cat.append(cc)
                        c_cat = torch.cat(c_cat, dim=1)

                        # cond
                        cond={"c_concat": [c_cat], "c_crossattn": [c_cross]}

                        # uncond cond
                        # uc_cross = model.get_unconditional_conditioning(num_samples, "")
                        # uc_cross = model.get_learned_conditioning(current_bsz * [""])
                        uc_cross = model.cond_stage_model.encode(current_bsz * [""])
                        uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

                        uc = uc_full

                    else:
                        cond = model.cond_stage_model.encode(batch_text)

                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(current_bsz * [""])

                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    start_code = None
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                        conditioning=cond,
                                                        batch_size=current_bsz,
                                                        shape=shape,
                                                        verbose=verbose,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=start_code,
                                                        )

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    for x_sample in x_samples_ddim:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        x_sample_img = Image.fromarray(x_sample.astype(np.uint8))
                        all_samples += [x_sample_img]

    return all_samples

from ldm.task_utils import prepare_task

def sample_images_iterative_inpaint(
    box_captions,
    unnormalized_boxes,
    sampler, opt,
    
    gen_batch_size=1, verbose=True,
    size=512,
    
    paste=True,
    box_generation_order=None,
    prefix_global_caption=False,
    global_caption=None,
    background_instruction="Add gray background",

    context_img=None,
    ):

    # d = datum
    d = {
        'box_captions': box_captions,
        'unnormalized_boxes': unnormalized_boxes,
    }

    if box_generation_order is not None:
        assert box_generation_order in ['top-down', 'bottom-up', 'random'], "box_generation_order must be one of ['top-down', 'bottom-up', 'random']"

        # unnormalized_boxes: [x1, y1, x2, y2]

        box_indices = list(range(len(d['unnormalized_boxes'])))

        # top-down: sort by y1: ascending
        if box_generation_order == 'top-down':
            box_indices = sorted(box_indices, key=lambda x: d['unnormalized_boxes'][x][1])
            d['unnormalized_boxes'] = [d['unnormalized_boxes'][i] for i in box_indices]
            d['box_captions'] = [d['box_captions'][i] for i in box_indices]

        # bottom-up: sort by y2: descending
        elif box_generation_order == 'bottom-up':
            box_indices = sorted(box_indices, key=lambda x: d['unnormalized_boxes'][x][3], reverse=True)
            d['unnormalized_boxes'] = [d['unnormalized_boxes'][i] for i in box_indices]
            d['box_captions'] = [d['box_captions'][i] for i in box_indices]

        # random: shuffle
        elif box_generation_order == 'random':
            random.shuffle(box_indices)
            d['unnormalized_boxes'] = [d['unnormalized_boxes'][i] for i in box_indices]
            d['box_captions'] = [d['box_captions'][i] for i in box_indices]

        if verbose:
            print('Box generation order: ', box_generation_order)
    
    n_total_boxes = len(d['unnormalized_boxes'])

    context_imgs = []
    mask_imgs = []
    # masked_imgs = []
    generated_images = []
    prompts = []

    if context_img is None:
        context_img = Image.new('RGB', (size, size))
    assert context_img.size == (size, size), "context_img must be of size (size, size)"
    # context_draw = ImageDraw.Draw(context_img)
    if verbose:
        print('Initiailzed context image')

    background_mask_img = Image.new('L', (size, size))
    background_mask_draw = ImageDraw.Draw(background_mask_img)
    background_mask_draw.rectangle([(0, 0), background_mask_img.size], fill=255)

    for i in range(n_total_boxes):
        if verbose:
            print('Iter: ', i+1, 'total: ', n_total_boxes)

        # target_caption = d['box_captions'][i]
        # if verbose:
        #     print('Drawing ', target_caption)

        mask_img = Image.new('L', context_img.size)
        mask_draw = ImageDraw.Draw(mask_img)
        mask_draw.rectangle([(0, 0), mask_img.size], fill=0)

        box = d['unnormalized_boxes'][i]
        if type(box) == list:
            box = torch.tensor(box)
        mask_draw.rectangle(box.long().tolist(), fill=255)
        background_mask_draw.rectangle(box.long().tolist(), fill=0)

        mask_imgs.append(mask_img.copy())
        
        prompt = f"Add {d['box_captions'][i]}"

        if prefix_global_caption:
            EOS_token = '<|endoftext|>'
            prompt = f'{global_caption} {EOS_token} {prompt}'

        if verbose:
            print('prompt:', prompt)
        prompts += [prompt]

        context_imgs.append(context_img.copy())

        generated_image = sample_images(
            [prompt], sampler, opt,
            mask_img=mask_img, context_img=context_img,
            gen_batch_size=gen_batch_size, verbose=verbose
        )[0]

        # generated_image = pipe(
        #     prompt,
        #     context_img,
        #     mask_img,
        #     guidance_scale=guidance_scale).images[0]
        
        if paste:
            # context_img.paste(generated_image.crop(box.long().tolist()), box.long().tolist())
            src_box = box.long().tolist()

            # x1 -> x1 + 1
            # y1 -> y1 + 1
            paste_box = box.long().tolist()
            paste_box[0] -= 1
            paste_box[1] -= 1
            paste_box[2] += 1
            paste_box[3] += 1

            box_w = paste_box[2] - paste_box[0]
            box_h = paste_box[3] - paste_box[1]

            context_img.paste(generated_image.crop(src_box).resize((box_w, box_h)), paste_box)
            generated_images.append(context_img.copy())
        else:
            context_img = generated_image
            generated_images.append(context_img.copy())

    # if verbose:
    #     print('Fill background')

    mask_img = background_mask_img

    mask_imgs.append(mask_img)

    # prompt = 'Add gray background'
    prompt = background_instruction

    if prefix_global_caption:
        EOS_token = '<|endoftext|>'
        prompt = f'{global_caption} {EOS_token} {prompt}'

    if verbose:
        print('prompt:', prompt)
    prompts += [prompt]

    context_imgs.append(context_img.copy())

    # generated_image = pipe(
    #     prompt,
    #     context_img,
    #     mask_img,
    #     guidance_scale=guidance_scale).images[0]

    generated_image = sample_images(
        [prompt], sampler, opt,
        mask_img=mask_img, context_img=context_img,
        gen_batch_size=gen_batch_size, verbose=verbose
    )[0]

    generated_images.append(generated_image)

    assert len(context_imgs) == len(mask_imgs) == len(generated_images) == len(prompts), f"# context: {len(context_imgs)}, # mask: {len(mask_imgs)}, # generated: {len(generated_images)}, # prompts: {len(prompts)}"
    
    return {
        'context_imgs': context_imgs,
        'mask_imgs': mask_imgs,
        'prompts': prompts,
        'generated_images': generated_images,
        'final_image': generated_image,
    }

def remove_region_iterative_inpaint(
        context_img,
        unnormalized_boxes,
        sampler, opt, gen_batch_size=1,
        paste=True,
        box_multiplier=1.1,
        background_instruction='Add gray background',
        verbose=False):
    
    mask_img = Image.new('L', context_img.size)
    mask_draw = ImageDraw.Draw(mask_img)
    mask_draw.rectangle([(0, 0), mask_img.size], fill=0)

    d = {'unnormalized_boxes': unnormalized_boxes}

    mask_imgs = []
    generated_images = []
    context_imgs = []
    prompts = []

    for i in range(len(d['unnormalized_boxes'])):
        box = d['unnormalized_boxes'][i]
        if type(box) == list:
            box = torch.tensor(box)

        # enlarge xyxy box by box_multiplier
        box[0] = box[0] - (box[2] - box[0]) * (box_multiplier - 1) / 2
        box[1] = box[1] - (box[3] - box[1]) * (box_multiplier - 1) / 2
        box[2] = box[2] + (box[2] - box[0]) * (box_multiplier - 1) / 2
        box[3] = box[3] + (box[3] - box[1]) * (box_multiplier - 1) / 2

        mask_draw.rectangle(box.long().tolist(), fill=255)
        # background_mask_draw.rectangle(box.long().tolist(), fill=0)

        mask_imgs.append(mask_img.copy())
        context_imgs.append(context_img.copy())
    
        prompt = background_instruction
        prompts += [prompt]

        generated_image = sample_images(
            [prompt], sampler, opt,
            mask_img=mask_img, context_img=context_img,
            gen_batch_size=gen_batch_size, verbose=verbose
        )[0]

        if paste:
            # context_img.paste(generated_image.crop(box.long().tolist()), box.long().tolist())
            src_box = box.long().tolist()

            # x1 -> x1 + 1
            # y1 -> y1 + 1
            paste_box = box.long().tolist()
            paste_box[0] -= 1
            paste_box[1] -= 1
            paste_box[2] += 1
            paste_box[3] += 1

            box_w = paste_box[2] - paste_box[0]
            box_h = paste_box[3] - paste_box[1]

            context_img.paste(generated_image.crop(src_box).resize((box_w, box_h)), paste_box)
            generated_images.append(context_img.copy())
        else:
            pass
            context_img = generated_image
            generated_images.append(context_img.copy())

    generated_image = context_img

    return {
        'context_imgs': context_imgs,
        'mask_imgs': mask_imgs,
        'prompts': prompts,
        'generated_images': generated_images,
        'final_image': generated_image,
    }


def add_region_iterative_inpaint(
        context_img,
        box_captions,
        unnormalized_boxes,
        sampler, opt, gen_batch_size=1,
        paste=True,
        # box_multiplier=1.1,
        verbose=False):

    mask_img = Image.new('L', context_img.size)
    mask_draw = ImageDraw.Draw(mask_img)
    mask_draw.rectangle([(0, 0), mask_img.size], fill=0)

    d = {
        'unnormalized_boxes': unnormalized_boxes,
        'box_captions': box_captions,
    }

    mask_imgs = []
    generated_images = []
    context_imgs = []
    prompts = []

    for i in range(len(d['unnormalized_boxes'])):
        box = d['unnormalized_boxes'][i]
        if type(box) == list:
            box = torch.tensor(box)

        # # enlarge xyxy box by box_multiplier
        # box[0] = box[0] - (box[2] - box[0]) * (box_multiplier - 1) / 2
        # box[1] = box[1] - (box[3] - box[1]) * (box_multiplier - 1) / 2
        # box[2] = box[2] + (box[2] - box[0]) * (box_multiplier - 1) / 2
        # box[3] = box[3] + (box[3] - box[1]) * (box_multiplier - 1) / 2

        mask_draw.rectangle(box.long().tolist(), fill=255)
        # background_mask_draw.rectangle(box.long().tolist(), fill=0)

        mask_imgs.append(mask_img.copy())
        context_imgs.append(context_img.copy())

        # prompt = background_instruction
        # prompts += [prompt]

        prompt = f"Add {d['box_captions'][i]}"

        generated_image = sample_images(
            [prompt], sampler, opt,
            mask_img=mask_img.copy(), context_img=context_img.copy(),
            gen_batch_size=gen_batch_size, verbose=verbose
        )[0]

        if paste:
            # context_img.paste(generated_image.crop(box.long().tolist()), box.long().tolist())
            src_box = box.long().tolist()

            # x1 -> x1 + 1
            # y1 -> y1 + 1
            paste_box = box.long().tolist()
            paste_box[0] -= 1
            paste_box[1] -= 1
            paste_box[2] += 1
            paste_box[3] += 1

            box_w = paste_box[2] - paste_box[0]
            box_h = paste_box[3] - paste_box[1]

            context_img.paste(generated_image.crop(
                src_box).resize((box_w, box_h)), paste_box)
            generated_images.append(context_img.copy())
        else:
            pass
            context_img = generated_image
            generated_images.append(context_img.copy())

    generated_image = context_img

    return {
        'context_imgs': context_imgs,
        'mask_imgs': mask_imgs,
        'prompts': prompts,
        'generated_images': generated_images,
        'final_image': generated_image,
    }

def encode_from_custom_annotation(custom_annotations, size=512):
    #     custom_annotations = [
    #     {'x': 83, 'y': 335, 'width': 70, 'height': 69, 'label': 'blue metal cube'},
    #     {'x': 162, 'y': 302, 'width': 110, 'height': 138, 'label': 'blue metal cube'},
    #     {'x': 274, 'y': 250, 'width': 191, 'height': 234, 'label': 'blue metal cube'},
    #     {'x': 14, 'y': 18, 'width': 155, 'height': 205, 'label': 'blue metal cube'},
    #     {'x': 175, 'y': 79, 'width': 106, 'height': 119, 'label': 'blue metal cube'},
    #     {'x': 288, 'y': 111, 'width': 69, 'height': 63, 'label': 'blue metal cube'}
    # ]
    H, W = size, size

    objects = []
    for j in range(len(custom_annotations)):
        xyxy = [
            custom_annotations[j]['x'],
            custom_annotations[j]['y'],
            custom_annotations[j]['x'] + custom_annotations[j]['width'],
            custom_annotations[j]['y'] + custom_annotations[j]['height']]
        objects.append({
            'caption': custom_annotations[j]['label'],
            'bbox': xyxy,
        })

    out = encode_scene(objects, H=H, W=W,
                       src_bbox_format='xyxy', tgt_bbox_format='xyxy')

    return out


def inference_from_custom_annotation(custom_annotations, sampler, opt, global_caption=None, background_instruction='Add gray background'):

    out = encode_from_custom_annotation(custom_annotations, size=opt.H)
    text = out['text']
    box_captions = out['box_captions']
    # boxes_normalized = out['boxes_normalized']
    boxes_unnormalized = out['boxes_unnormalized']

    prefix_global_caption = global_caption is not None

    prompts = [text]

    if 'iterinpaint' in opt.config:

        generated = sample_images_iterative_inpaint(
            box_captions, boxes_unnormalized,
            sampler, opt,
            gen_batch_size=1,
            size=opt.H,
            # paste=True,
            paste=not opt.iterinpaint_nopaste,
            box_generation_order=opt.box_generation_order,
            background_instruction=background_instruction,
            global_caption=global_caption,
            prefix_global_caption=prefix_global_caption,
            verbose=False)
        return generated
    else:
        generated_img = sample_images(
            prompts, sampler, opt,
            gen_batch_size=1, verbose=False)[0]

        return generated_img
    



#### Below are for HF diffusers

def iterinpaint_sample_diffusers(pipe, datum, paste=True, verbose=False, guidance_scale=4.0, size=512, background_instruction='Add gray background'):
    d = datum

    d['unnormalized_boxes'] = d['boxes_unnormalized']
    
    n_total_boxes = len(d['unnormalized_boxes'])

    context_imgs = []
    mask_imgs = []
    # masked_imgs = []
    generated_images = []
    prompts = []

    context_img = Image.new('RGB', (size, size))
    # context_draw = ImageDraw.Draw(context_img)
    if verbose:
        print('Initiailzed context image')

    background_mask_img = Image.new('L', (size, size))
    background_mask_draw = ImageDraw.Draw(background_mask_img)
    background_mask_draw.rectangle([(0, 0), background_mask_img.size], fill=255)

    for i in range(n_total_boxes):
        if verbose:
            print('Iter: ', i+1, 'total: ', n_total_boxes)

        target_caption = d['box_captions'][i]
        if verbose:
            print('Drawing ', target_caption)

        mask_img = Image.new('L', context_img.size)
        mask_draw = ImageDraw.Draw(mask_img)
        mask_draw.rectangle([(0, 0), mask_img.size], fill=0)

        box = d['unnormalized_boxes'][i]
        if type(box) == list:
            box = torch.tensor(box) 
        mask_draw.rectangle(box.long().tolist(), fill=255)
        background_mask_draw.rectangle(box.long().tolist(), fill=0)

        mask_imgs.append(mask_img.copy())

        
        prompt = f"Add {d['box_captions'][i]}"

        if verbose:
            print('prompt:', prompt)
        prompts += [prompt]

        context_imgs.append(context_img.copy())

        generated_image = pipe(
            prompt,
            context_img,
            mask_img,
            guidance_scale=guidance_scale).images[0]
        
        if paste:
            # context_img.paste(generated_image.crop(box.long().tolist()), box.long().tolist())
            

            src_box = box.long().tolist()

            # x1 -> x1 + 1
            # y1 -> y1 + 1
            paste_box = box.long().tolist()
            paste_box[0] -= 1
            paste_box[1] -= 1
            paste_box[2] += 1
            paste_box[3] += 1

            box_w = paste_box[2] - paste_box[0]
            box_h = paste_box[3] - paste_box[1]

            context_img.paste(generated_image.crop(src_box).resize((box_w, box_h)), paste_box)
            generated_images.append(context_img.copy())
        else:
            context_img = generated_image
            generated_images.append(context_img.copy())

    if verbose:
        print('Fill background')

    mask_img = background_mask_img

    mask_imgs.append(mask_img)

    prompt = background_instruction

    if verbose:
        print('prompt:', prompt)
    prompts += [prompt]

    generated_image = pipe(
        prompt,
        context_img,
        mask_img,
        guidance_scale=guidance_scale).images[0]

    generated_images.append(generated_image)
    
    return {
        'context_imgs': context_imgs,
        'mask_imgs': mask_imgs,
        'prompts': prompts,
        'generated_images': generated_images,
    }