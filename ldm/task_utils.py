import torch
from torch.utils.data import Dataset
import json
import pandas as pd
from PIL import Image, ImageDraw

import ldm.data.transforms as T
import numpy as np
from pathlib import Path
import random


ALL_TASKS = [
    'add_object_given_background',
    'add_object_given_empty_background',
    'add_background_given_object',
]

def prepare_task(image, datum, task, verbose=False, prefix_plan=None, **kwargs):
    assert task in ALL_TASKS, f'Task {task} not in {ALL_TASKS}'

    if task == 'add_object_given_background':
        return prepare_add_object_given_background(image, datum, verbose=verbose, prefix_plan=prefix_plan)
    elif task == 'add_object_given_empty_background':
        return prepare_add_object_given_empty_backround(image, datum, verbose=verbose, prefix_plan=prefix_plan)
    elif task == 'add_background_given_object':
        return prepare_add_background_given_object(image, datum, verbose=verbose, prefix_plan=prefix_plan,
                                                   background_instruction=kwargs.get('background_instruction', None)
                                                   )

def prepare_add_object_given_background(image, datum, verbose=False, prefix_plan=None):
    """
    image: PIL.Image
    datum: dict - contains keys 'unnormalized_boxes' and 'box_captions'
    """
    task = 'add_object_given_background'
    if verbose:
        print('Task: ', task)
    assert 'unnormalized_boxes' in datum, 'unnormalized_boxes not in datum'
    assert 'box_captions' in datum, 'box_captions not in datum'

    d = datum

    mask_img = image.copy().convert('L')
    mask_draw = ImageDraw.Draw(mask_img)
    context_img = image.copy().convert('RGB')
    context_draw = ImageDraw.Draw(context_img)

    text_tokens = []

    mask_draw.rectangle([(0, 0), mask_img.size], fill=0)

    # sample the number of boxes to mask 
    n_total_boxes = len(datum['unnormalized_boxes'])
    # n_min_mask = 1
    # # n_max_mask = n_total_boxes
    # n_max_mask = 1
    # n_mask_objs = random.randint(n_min_mask, n_max_mask)

    n_mask_objs = 1
    # n_context_objs = random.randint(0, n_total_boxes - n_mask_objs)
    n_context_objs = n_total_boxes - n_mask_objs

    # sample the boxes to mask
    # mask_obj_indices = random.sample(range(len(datum['unnormalized_boxes'])), n_mask_objs)

    mask_obj_indices = list(range(n_mask_objs))

    # # sample the boxes to show
    # context_obj_indices = list(set(range(len(datum['unnormalized_boxes']))) - set(mask_obj_indices))

    if verbose:
        print('# total boxes: ', len(datum['unnormalized_boxes']))
        print('# boxes to mask: ', n_mask_objs)
        print('# boxes to show: ', n_context_objs)

    # Mask the boxes to be repainted
    for mask_obj_index in mask_obj_indices:
        box = d['unnormalized_boxes'][mask_obj_index]
        # Fill the object box with white - will be repainted
        mask_draw.rectangle(box.long().tolist(), fill=255)
        context_draw.rectangle(box.long().tolist(), fill=(0,0,0))

        box_caption = d['box_captions'][mask_obj_index]
        text_tokens += [
            f"Add {box_caption}"
        ]

    target_image = image
    text = ' '.join(text_tokens)

    return {
        'text': text,
        'target_image': target_image,
        'context_image': context_img,
        'mask_image': mask_img,
        'step_caption': box_caption,
    }

def prepare_add_object_given_empty_backround(image, datum, verbose=False, prefix_plan=None):
    """
    image: PIL.Image
    datum: dict - contains keys 'unnormalized_boxes' and 'box_captions'
    """
    task = 'add_object_given_empty_background'
    if verbose:
        print('Task: ', task)
        print("Inpaint objects given empty background")
        print('inpaint only 1 object')
        print("context: boxes to keep")
        print("inpaint: masked boxes")
        print('target: previous objects + added object + empty background')
    
    assert 'unnormalized_boxes' in datum, 'unnormalized_boxes not in datum'
    assert 'box_captions' in datum, 'box_captions not in datum'

    d = datum

    mask_img = image.copy().convert('L')
    mask_draw = ImageDraw.Draw(mask_img)
    context_img = image.copy().convert('RGB')
    context_draw = ImageDraw.Draw(context_img)

    mask_draw.rectangle([(0, 0), mask_img.size], fill=0)
    context_draw.rectangle([(0, 0), context_img.size], fill=(0,0,0))

    text_tokens = []

    # sample the number of boxes to mask 
    n_total_boxes = len(datum['unnormalized_boxes'])
    # n_min_mask = 1
    # # n_max_mask = n_total_boxes
    # n_max_mask = 1
    # n_mask_objs = random.randint(n_min_mask, n_max_mask)

    n_mask_objs = 1
    if n_total_boxes == 0:
        n_context_objs = 0
        target_image = Image.new('RGB', image.size)
        box_caption = ""
    else:
        n_context_objs = random.randint(0, n_total_boxes - n_mask_objs)

        # sample the boxes to mask
        # mask_obj_indices = random.sample(range(n_total_boxes), n_mask_objs)

        # sample the boxes to show
        # context_obj_indices = random.sample(set(range(n_total_boxes)) - set(mask_obj_indices), n_context_objs)

        context_obj_indices = list(range(n_context_objs))
        mask_obj_indices = [n_context_objs]

        if verbose:
            print('# total boxes: ', len(datum['unnormalized_boxes']))
            print('# boxes to mask: ', n_mask_objs)
            print('# boxes to show: ', n_context_objs)

        target_image = Image.new('RGB', image.size)

        # context boxes - to be preserved
        for context_obj_index in context_obj_indices:
            box = d['unnormalized_boxes'][context_obj_index]
            # mask_draw.rectangle(box.long().tolist(), fill=0)
            context_img.paste(image.crop(box.long().tolist()), box.long().tolist())

            target_image.paste(image.crop(box.long().tolist()), box.long().tolist())

        if prefix_plan:
            text_tokens += [
                f"Step {len(context_obj_indices)+1}:"
            ]

        # Mask the boxes to be repainted
        for mask_obj_index in mask_obj_indices:
            box = d['unnormalized_boxes'][mask_obj_index]
            # Fill the object box with white - will be repainted
            mask_draw.rectangle(box.long().tolist(), fill=255)
            # context_draw.rectangle(box.long().tolist(), fill=(0,0,0))

            box_caption = d['box_captions'][mask_obj_index]
            if prefix_plan:
                text_tokens += [
                    f"Add {box_caption} at {d['box_bins'][mask_obj_index]}",
                ]
            else:
                text_tokens += [
                    f"Add {box_caption}"
                ]

            target_image.paste(image.crop(box.long().tolist()), box.long().tolist())

    return {
        'text': ' '.join(text_tokens),
        'target_image': target_image,
        'context_image': context_img,
        'mask_image': mask_img,
        'box_caption': box_caption,
    }


def prepare_add_background_given_object(image, datum, verbose=False, prefix_plan=None, background_instruction="Add gray background"):
    """
    image: PIL.Image
    datum: dict - contains keys 'unnormalized_boxes' and 'box_captions'
    """

    task = 'add_background_given_object'
    if verbose:
        print('Task: ', task)
        print("Fill out background, given all objects")
        print("context: all boxes")
        print("inpaint: background")

    assert 'unnormalized_boxes' in datum, 'unnormalized_boxes not in datum'
    assert 'box_captions' in datum, 'box_captions not in datum'

    d = datum

    mask_img = image.copy().convert('L')
    mask_draw = ImageDraw.Draw(mask_img)
    context_img = image.copy().convert('RGB')
    context_draw = ImageDraw.Draw(context_img)

    text_tokens = []

    mask_draw.rectangle([(0, 0), mask_img.size], fill=255)
    context_draw.rectangle(
        [(0, 0), context_img.size], fill=(0, 0, 0))

    # context boxes - to be preserved - all boxes
    for keep_obj_index in range(len(datum['unnormalized_boxes'])):
        box = d['unnormalized_boxes'][keep_obj_index]
        mask_draw.rectangle(box.long().tolist(), fill=0)
        context_img.paste(image.crop(
            box.long().tolist()), box.long().tolist())

    target_image = image

    if prefix_plan:
        text_tokens += [
            f"Step {len(datum['unnormalized_boxes'])+1}:"
        ]

    text_tokens += [
        # "Add gray background"
        background_instruction
    ]

    return {
        'text': ' '.join(text_tokens),
        'target_image': target_image,
        'context_image': context_img,
        'mask_image': mask_img,
        'box_caption': 'gray background',
    }