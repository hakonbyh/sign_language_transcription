import logging
import os
import re

import cv2
import numpy as np
import torch
from main.src.utils import (
    calculate_motion_vectors,
    get_ref_len,
    interpolate_missing_keypoints,
)
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\sæøåÆØÅ]", "", text)
    text = text.lower()
    return text


def load_video_data(video_path, desired_fps):
    video_capture = cv2.VideoCapture(video_path)

    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    skip_frames = max(1, int(fps / desired_fps))

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    keypoints_dir = os.path.normpath(os.path.join(video_path, "../..", "keypoints"))

    keypoints_path = os.path.join(keypoints_dir, base_name + ".npy")

    loaded_keypoints = np.load(keypoints_path, allow_pickle=True)

    fixed_keypoints = interpolate_missing_keypoints(loaded_keypoints)

    selected_keypoints = []
    frame_number = 0
    video = []
    while frame_number < total_frames:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        _, frame = video_capture.read()
        video.append(frame)
        selected_keypoints.append(fixed_keypoints[frame_number])
        frame_number += skip_frames

    video_capture.release()

    ref_len = get_ref_len(selected_keypoints, [0, -1])

    motion_vectors = []
    prev_pts = None
    for i in range(0, len(selected_keypoints)):
        all_keypoints = selected_keypoints[i]["keypoints"]
        current_pts = {
            key: all_keypoints[key]
            for key in all_keypoints
            if key in ["pose", "left_hand", "right_hand"]
        }
        if not prev_pts:
            motion_vector = [np.array([0, 0])] * sum(
                len(pts) for pts in current_pts.values()
            )
        else:
            motion_vector = calculate_motion_vectors(prev_pts, current_pts, ref_len)
        motion_vectors.append(np.array(motion_vector))
        prev_pts = current_pts

    flat_mvecs = [arr.flatten() for arr in motion_vectors]

    return video, flat_mvecs
