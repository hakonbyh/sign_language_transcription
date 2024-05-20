import glob
import logging
import os
import random
from math import ceil
from typing import List, Tuple

import cv2
import numpy as np
import torch
from main.src.constants import MAX_FRAME_COUNT, MIN_FRAME_COUNT
from sklearn.model_selection import train_test_split
from torchtext import data
from torchtext.data import Field, RawField
from tqdm import tqdm

logger = logging.getLogger(__name__)

from main.src.constants import FACE, LEFT_HAND, POSE, RIGHT_HAND
from main.src.utils import clean_text, lemmatize, remove_stopwords


def use_file(loaded_keypoints):
    total_frames = len(loaded_keypoints)

    keypoint_counts = {
        "pose": POSE,
        "face": FACE,
        "left_hand": LEFT_HAND,
        "right_hand": RIGHT_HAND,
    }

    for key, count in keypoint_counts.items():
        for kp_idx in range(count):
            known_frames = []
            known_values_x = []
            known_values_y = []
            for frame_idx in range(total_frames):
                keypoints_data = loaded_keypoints[frame_idx]["keypoints"].get(key, [])
                if keypoints_data and keypoints_data[kp_idx]:
                    known_frames.append(frame_idx)
                    known_values_x.append(keypoints_data[kp_idx][0])
                    known_values_y.append(keypoints_data[kp_idx][1])

            if len(known_frames) < 2:
                return False
    return True


def load_dataset_file(files_dir, desired_fps, random_seed=93):
    files = []

    videos_dir = os.path.join(files_dir, "videos")
    transcripts_dir = os.path.join(files_dir, "transcripts")

    video_paths = sorted(glob.glob(f"{videos_dir}/*.mp4"))

    for video_path in tqdm(video_paths, desc="Retrieving filepath"):
        video_capture = cv2.VideoCapture(video_path)

        fps = video_capture.get(cv2.CAP_PROP_FPS)
        skip_frames = max(1, int(fps / desired_fps))

        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        video_capture.release()

        effective_video_length = ceil(total_frames / skip_frames)

        if effective_video_length > MAX_FRAME_COUNT:
            logger.warning(
                f"Video {video_path} is too large to be processed with an effective length of {effective_video_length} frames. Skipping..."
            )
            continue
        elif effective_video_length < MIN_FRAME_COUNT:
            logger.warning(
                f"Video {video_path} is too small to be processed with an effective length of {effective_video_length} frames. Skipping..."
            )
            continue

        base_name = os.path.splitext(os.path.basename(video_path))[0]

        keypoints_dir = os.path.normpath(os.path.join(video_path, "../..", "keypoints"))

        keypoints_path = os.path.join(keypoints_dir, base_name + ".npy")
        loaded_keypoints = np.load(keypoints_path, allow_pickle=True)

        if not use_file(loaded_keypoints):
            logger.info("Could not interpolate. Skipping...")
            continue

        transcript_path = os.path.join(transcripts_dir, base_name + ".txt")

        with open(transcript_path, "r", encoding="utf-8") as file:
            transcript = file.read().strip()

        text = clean_text(transcript)

        lemmatized_sentence = lemmatize(text)
        glosses = remove_stopwords(lemmatized_sentence)

        video_dict = {
            "name": base_name,
            "gloss": glosses,
            "text": text,
            "sign": video_path,
            "sign_length": effective_video_length,
        }

        files.append(video_dict)

    if len(files) > 20000:
        logger.info("Sampling 20 000 files")
        files = random.sample(files, 20000)

    indices = list(range(len(files)))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, random_state=random_seed
    )
    random.seed(random_seed)
    train_val_indices = random.sample(train_indices, int(0.2 * len(indices)))

    train_files = [files[i] for i in train_indices]
    train_val_files = [files[i] for i in train_val_indices]
    val_files = [files[i] for i in val_indices]

    return train_files, train_val_files, val_files


class SignTranslationDataset(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        files: str,
        fields: Tuple[RawField, RawField, RawField, Field, Field, Field],
        keep_only=None,
        **kwargs,
    ):
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("sequence", fields[0]),
                ("sgn", fields[1]),
                ("sgn_len", fields[2]),
                ("gls", fields[3]),
                ("txt", fields[4]),
            ]

        samples = {}
        for s in files:
            seq_id = s["name"]
            if keep_only is not None:
                if seq_id not in keep_only:
                    continue
            if seq_id in samples:
                assert samples[seq_id]["name"] == s["name"]
                assert samples[seq_id]["gloss"] == s["gloss"]
                assert samples[seq_id]["text"] == s["text"]
                assert samples[seq_id]["sgn_length"] == s["sign_length"]
                samples[seq_id]["sign"] = torch.cat(
                    [samples[seq_id]["sign"].float(), s["sign"].float()], axis=1
                )
            else:
                samples[seq_id] = {
                    "name": s["name"],
                    "gloss": s["gloss"],
                    "text": s["text"],
                    "sign": s["sign"],
                    "sign_length": s["sign_length"],
                }

        examples = []
        for s in samples:
            sample = samples[s]
            examples.append(
                data.Example.fromlist(
                    [
                        sample["name"],
                        sample["sign"],
                        sample["sign_length"],
                        sample["gloss"].strip(),
                        sample["text"].strip(),
                    ],
                    fields,
                )
            )
        super().__init__(examples, fields, **kwargs)
