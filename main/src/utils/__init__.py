import torch

from .checkpoint_utils import *
from .dataset import clean_text, load_video_data
from .decode import Decode
from .keypoints_utils import (
    calculate_average_midpoint,
    calculate_min_max,
    calculate_motion_vectors,
    get_ref_len,
    interpolate_missing_keypoints,
    normalize_and_scale_keypoints,
)
from .text_utils import lemmatize, remove_stopwords
from .tokenizer import TokenizerWrapper
from .validate_files import get_valid_segments
from .vocab import get_vocab


def to_cuda(elements):
    if torch.cuda.is_available():
        if type(elements) == tuple or type(elements) == list:
            return [x.cuda() for x in elements]
        return elements.cuda()
    return elements
