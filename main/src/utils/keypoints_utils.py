import logging

import numpy as np
from scipy import interpolate

from ..constants import (
    FACE,
    LEFT_HAND,
    LEFT_HIP,
    LEFT_SHOULDER,
    POSE,
    RIGHT_HAND,
    RIGHT_HIP,
    RIGHT_SHOULDER,
)

logger = logging.getLogger(__name__)


def calculate_motion_vectors(prev_keypoints, current_keypoints, ref_len):
    motion_vectors = []
    for key in current_keypoints.keys():
        for (prev_x, prev_y), (curr_x, curr_y) in zip(
            prev_keypoints[key], current_keypoints[key]
        ):
            motion_vectors.append(
                np.array(
                    [
                        100 * (curr_x - prev_x) / ref_len,
                        100 * (curr_y - prev_y) / ref_len,
                    ]
                )
            )
    return motion_vectors


def calculate_midpoint(left, right):
    return ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)


def calculate_min_max(loaded_keypoints):
    all_points = []
    for entry in loaded_keypoints:
        keypoints_data = entry["keypoints"]
        for key, points in keypoints_data.items():
            all_points.extend(points)
    min_x = min([x for x, y in all_points])
    max_x = max([x for x, y in all_points])
    min_y = min([y for x, y in all_points])
    max_y = max([y for x, y in all_points])
    return min_x, max_x, min_y, max_y


def calculate_average_midpoint(loaded_keypoints, frame_indices):
    midpoints = []
    for idx in frame_indices:
        keypoints_data = loaded_keypoints[idx]["keypoints"]
        midpoint = calculate_midpoint(
            keypoints_data["pose"][LEFT_HIP], keypoints_data["pose"][RIGHT_HIP]
        )
        midpoints.append(midpoint)
    avg_midpoint = (
        sum([x for x, y in midpoints]) / len(midpoints),
        sum([y for x, y in midpoints]) / len(midpoints),
    )
    return avg_midpoint


def get_ref_len(loaded_keypoints, frame_indices):
    ref_lens = []
    for idx in frame_indices:
        keypoints_data = loaded_keypoints[idx]["keypoints"]
        left_shoulder = keypoints_data["pose"][LEFT_SHOULDER]
        right_shoulder = keypoints_data["pose"][RIGHT_SHOULDER]

        ref_len = abs(right_shoulder[0] - left_shoulder[0])
        ref_lens.append(ref_len)
    avg_ref_len = sum([width for width in ref_lens]) / len(ref_lens)
    return avg_ref_len


def normalize_and_scale_keypoints(keypoints_list, ref_len, origin):
    normalized_keypoints = []

    origin_x, origin_y = origin

    for frame_data in keypoints_list:
        frame_keypoints = []

        for key, points in frame_data["keypoints"].items():
            if key in ["pose", "left_hand", "right_hand"]:
                for x, y in points:
                    shifted_x = x - origin_x
                    shifted_y = origin_y - y
                    norm_x = 100 * shifted_x / ref_len
                    norm_y = 100 * shifted_y / ref_len
                    frame_keypoints.append(np.array([norm_x, norm_y]))

        normalized_keypoints.append(np.array(frame_keypoints))

    return normalized_keypoints


def interpolate_missing_keypoints(loaded_keypoints):
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
                logger.warning("At least two points are needed for interpolation")
                continue

            f_interp_x = interpolate.interp1d(
                known_frames, known_values_x, kind="linear", fill_value="extrapolate"
            )
            f_interp_y = interpolate.interp1d(
                known_frames, known_values_y, kind="linear", fill_value="extrapolate"
            )

            for frame_idx in range(total_frames):
                if frame_idx not in known_frames:
                    if key not in loaded_keypoints[frame_idx]["keypoints"]:
                        loaded_keypoints[frame_idx]["keypoints"][key] = [None] * count
                    loaded_keypoints[frame_idx]["keypoints"][key][kp_idx] = (
                        f_interp_x(frame_idx),
                        f_interp_y(frame_idx),
                    )
    return loaded_keypoints
