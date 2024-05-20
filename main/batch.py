import logging
import math
import random

import numpy as np
import torch
from main import print_cuda_memory
from torchvision import transforms

IMAGENET_MEAN = 0.485, 0.456, 0.406
IMAGENET_STD = 0.229, 0.224, 0.225

from main.src.utils import load_video_data

logger = logging.getLogger()


def get_basic_augmentations():
    return transforms.Compose(
        [
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(
                degrees=(-5, 5),
                translate=(
                    0.05,
                    0.05,
                ),
                scale=(0.95, 1.05),
                shear=(-5, 5),
            ),
        ]
    )


def preprocess_video(video_key_data, augment=False):
    video, all_keypoints = video_key_data
    if augment:
        augmentation_transforms = get_basic_augmentations()
    else:
        augmentation_transforms = transforms.Compose([])

    def transform(frame):
        frame_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                augmentation_transforms,
            ]
        )
        return frame_transform(frame)

    transformed_frames = []
    transformed_keypoints = []
    for frame, keypoints in zip(video, all_keypoints):
        transformed_frames.append(transform(frame))
        transformed_keypoints.append(torch.from_numpy(keypoints))

    tensor_video = torch.stack(transformed_frames)
    tensor_keypoints = torch.stack(transformed_keypoints)

    return tensor_video, tensor_keypoints


class Batch:

    def __init__(
        self,
        torch_batch,
        txt_pad_index,
        sgn_dim,
        use_cuda: bool = False,
        is_train: bool = True,
    ):

        self.sequence = torch_batch.sequence

        self.sgn = [
            preprocess_video(
                load_video_data(elem, 10),
                augment=is_train,
            )
            for elem in torch_batch.sgn
        ]

        self.sgn_lengths = torch.tensor([len(entry[0]) for entry in self.sgn])

        self.sgn_dim = sgn_dim
        self.sgn_mask = None

        self.txt = None
        self.txt_mask = None
        self.txt_input = None
        self.txt_lengths = None

        self.gls = None
        self.gls_lengths = None

        self.num_txt_tokens = None
        self.num_gls_tokens = None
        self.use_cuda = use_cuda
        self.num_seqs = len(self.sgn)

        if hasattr(torch_batch, "txt"):
            txt, txt_lengths = torch_batch.txt
            self.txt_input = txt[:, :-1]
            self.txt_lengths = txt_lengths
            self.txt = txt[:, 1:]
            self.txt_mask = (self.txt_input != txt_pad_index).unsqueeze(1)
            self.num_txt_tokens = (self.txt != txt_pad_index).data.sum().item()

        if hasattr(torch_batch, "gls"):
            self.gls, self.gls_lengths = torch_batch.gls
            self.num_gls_tokens = self.gls_lengths.sum().detach().clone().numpy()

        if use_cuda:
            self._make_cuda()

    def _make_cuda(self):
        self.sgn = [(video.cuda(), keypoints.cuda()) for video, keypoints in self.sgn]

        if self.txt_input is not None:
            self.txt = self.txt.cuda()
            self.txt_mask = self.txt_mask.cuda()
            self.txt_input = self.txt_input.cuda()

    def sort_by_sgn_lengths(self):

        _, perm_index = self.sgn_lengths.sort(0, descending=True)
        rev_index = [0] * perm_index.size(0)
        for new_pos, old_pos in enumerate(perm_index.cpu().numpy()):
            rev_index[old_pos] = new_pos

        self.sgn = [self.sgn[i] for i in perm_index]
        self.sgn_lengths = self.sgn_lengths[perm_index]

        self.sequence = [self.sequence[pi] for pi in perm_index]

        if self.gls is not None:
            self.gls = self.gls[perm_index]
            self.gls_lengths = self.gls_lengths[perm_index]

        if self.txt is not None:
            self.txt = self.txt[perm_index]
            self.txt_mask = self.txt_mask[perm_index]
            self.txt_input = self.txt_input[perm_index]
            self.txt_lengths = self.txt_lengths[perm_index]

        if self.use_cuda:
            self._make_cuda()

        return rev_index
