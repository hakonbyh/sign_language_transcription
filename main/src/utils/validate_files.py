import glob
import logging
import os

import cv2
from tqdm import tqdm

from ..constants import MAX_FRAME_COUNT

logger = logging.getLogger(__name__)


def get_valid_segments(files_dir, desired_fps, tokenizer):
    valid_video_paths = []

    videos_dir = os.path.join(files_dir, "videos")
    transcripts_dir = os.path.join(files_dir, "transcripts")

    video_paths = sorted(glob.glob(f"{videos_dir}/*.mp4"))

    for video_path in tqdm(
        video_paths, total=len(video_paths), desc="Processing Videos"
    ):
        video_capture = cv2.VideoCapture(video_path)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        transcript_path = os.path.join(transcripts_dir, base_name + ".txt")

        if not video_capture.isOpened():
            logger.warning(f"Unable to open video file {video_path}. Skipping...")
            continue
        elif not os.path.exists(transcript_path):
            logger.warning(
                f"Unable to open transcript file {transcript_path}. Skipping..."
            )
            continue

        fps = video_capture.get(cv2.CAP_PROP_FPS)
        skip_frames = max(1, int(fps / desired_fps))

        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        effective_video_length = total_frames // skip_frames

        video_capture.release()

        if effective_video_length > MAX_FRAME_COUNT:
            logger.warning(
                f"Video {video_path} is too large to be processed with an effective length of {effective_video_length} frames. Skipping..."
            )
            continue

        with open(transcript_path, "r", encoding="utf-8") as file:
            transcript = file.read().strip()

        tokenized_transcript = tokenizer(transcript, return_tensors="pt")[
            "input_ids"
        ].squeeze(0)
        target_length = len(tokenized_transcript)

        if effective_video_length > target_length:
            valid_video_paths.append(video_path)
        else:
            logger.warning(
                f"Video {video_path} has effective length {effective_video_length} which is shorter than target length {target_length}."
            )

    no_files_removed = len(video_paths) - len(valid_video_paths)
    logger.info(f"Validation completed. Removed {no_files_removed} files.")

    return valid_video_paths
