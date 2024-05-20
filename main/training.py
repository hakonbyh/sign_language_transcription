import torch

torch.backends.cudnn.deterministic = True

import argparse
import os
import queue
import shutil
import time
import traceback
from typing import Dict, List

import numpy as np
import wandb
from main.batch import Batch
from main.builders import build_gradient_clipper, build_optimizer, build_scheduler
from main.data import load_data, make_data_iter
from main.helpers import (
    load_checkpoint,
    load_config,
    log_cfg,
    log_data_info,
    make_logger,
    make_model_dir,
    set_seed,
    symlink_update,
)
from main.loss import XentLoss
from main.metrics import wer_single
from main.model import SignModel, build_model
from main.prediction import test, validate_on_data
from main.vocabulary import SIL_TOKEN
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Dataset
from tqdm import tqdm

wandb.init(project="Master Thesis", entity="hakonbyh")


class TrainManager:
    def __init__(self, model: SignModel, config: dict) -> None:
        train_config = config["training"]

        self.model_dir = make_model_dir(
            train_config["model_dir"], overwrite=train_config.get("overwrite", False)
        )
        self.logger = make_logger(model_dir=self.model_dir)
        self.logger.info("LOGGER CREATED")
        self.logging_freq = train_config.get("logging_freq", 100)
        self.valid_report_file = "{}/validations.txt".format(self.model_dir)

        self.feature_size = config["data"]["feature_size"]
        self.dataset_version = config["data"].get("version", "phoenix_2014_trans")

        self.model = model
        self.txt_pad_index = self.model.txt_pad_index
        self.txt_bos_index = self.model.txt_bos_index
        self._log_parameters_list()
        self.do_recognition = (
            config["training"].get("recognition_loss_weight", 1.0) > 0.0
        )
        self.do_translation = (
            config["training"].get("translation_loss_weight", 1.0) > 0.0
        )

        if self.do_recognition:
            self._get_recognition_params(train_config=train_config)
        if self.do_translation:
            self._get_translation_params(train_config=train_config)

        self.last_best_lr = train_config.get("learning_rate", -1)
        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)
        self.clip_grad_fun = build_gradient_clipper(config=train_config)
        self.optimizer = build_optimizer(
            config=train_config, parameters=model.parameters()
        )
        self.batch_multiplier = train_config.get("batch_multiplier", 1)

        self.validation_freq = train_config.get("validation_freq", 100)
        self.num_valid_log = train_config.get("num_valid_log", 5)
        self.ckpt_queue = queue.Queue(maxsize=train_config.get("keep_last_ckpts", 5))
        self.eval_metric = train_config.get("eval_metric", "bleu")
        if self.eval_metric not in ["bleu", "chrf", "wer", "rouge"]:
            raise ValueError(
                "Invalid setting for 'eval_metric': {}".format(self.eval_metric)
            )
        self.early_stopping_metric = train_config.get(
            "early_stopping_metric", "eval_metric"
        )

        if self.early_stopping_metric in [
            "ppl",
            "translation_loss",
            "recognition_loss",
        ]:
            self.minimize_metric = True
        elif self.early_stopping_metric == "eval_metric":
            if self.eval_metric in ["bleu", "chrf", "rouge"]:
                assert self.do_translation
                self.minimize_metric = False
            else:
                self.minimize_metric = True
        else:
            raise ValueError(
                "Invalid setting for 'early_stopping_metric': {}".format(
                    self.early_stopping_metric
                )
            )

        self.frame_subsampling_ratio = config["data"].get(
            "frame_subsampling_ratio", None
        )
        self.random_frame_subsampling = config["data"].get(
            "random_frame_subsampling", None
        )
        self.random_frame_masking_ratio = config["data"].get(
            "random_frame_masking_ratio", None
        )

        self.scheduler, self.scheduler_step_at = build_scheduler(
            config=train_config,
            scheduler_mode="min" if self.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=config["model"]["encoder"]["hidden_size"],
        )

        self.level = config["data"]["level"]
        if self.level not in ["word", "bpe", "char"]:
            raise ValueError("Invalid segmentation level': {}".format(self.level))

        self.shuffle = train_config.get("shuffle", True)
        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"]
        self.batch_type = train_config.get("batch_type", "sentence")
        self.eval_batch_size = train_config.get("eval_batch_size", self.batch_size)
        self.eval_batch_type = train_config.get("eval_batch_type", self.batch_type)

        self.use_cuda = train_config["use_cuda"]
        if self.use_cuda:
            self.model.cuda()
            if self.do_translation:
                self.translation_loss_function.cuda()
            if self.do_recognition:
                self.recognition_loss_function.cuda()

        self.steps = 0
        self.stop = False
        self.total_txt_tokens = 0
        self.total_gls_tokens = 0
        self.best_ckpt_iteration = 0
        self.best_ckpt_score = np.inf if self.minimize_metric else -np.inf
        self.best_all_ckpt_scores = {}
        self.is_best = lambda score: (
            score < self.best_ckpt_score
            if self.minimize_metric
            else score > self.best_ckpt_score
        )

        if "load_model" in train_config.keys():
            model_load_path = train_config["load_model"]
            self.logger.info("Loading model from %s", model_load_path)
            reset_best_ckpt = train_config.get("reset_best_ckpt", False)
            reset_scheduler = train_config.get("reset_scheduler", False)
            reset_optimizer = train_config.get("reset_optimizer", False)
            self.init_from_checkpoint(
                model_load_path,
                reset_best_ckpt=reset_best_ckpt,
                reset_scheduler=reset_scheduler,
                reset_optimizer=reset_optimizer,
            )

        self.interval_ckpt_paths = []
        self.interval_ckpt_max = 3

    def _get_recognition_params(self, train_config) -> None:
        self.gls_silence_token = self.model.gls_vocab.stoi[SIL_TOKEN]
        assert self.gls_silence_token == 0

        self.recognition_loss_function = torch.nn.CTCLoss(
            blank=self.gls_silence_token, zero_infinity=True
        )
        self.recognition_loss_weight = train_config.get("recognition_loss_weight", 1.0)
        self.eval_recognition_beam_size = train_config.get(
            "eval_recognition_beam_size", 1
        )

    def _get_translation_params(self, train_config) -> None:
        self.label_smoothing = train_config.get("label_smoothing", 0.0)
        self.translation_loss_function = XentLoss(
            pad_index=self.txt_pad_index, smoothing=self.label_smoothing
        )
        self.translation_normalization_mode = train_config.get(
            "translation_normalization", "batch"
        )
        if self.translation_normalization_mode not in ["batch", "tokens"]:
            raise ValueError(
                "Invalid normalization {}.".format(self.translation_normalization_mode)
            )
        self.translation_loss_weight = train_config.get("translation_loss_weight", 1.0)
        self.eval_translation_beam_size = train_config.get(
            "eval_translation_beam_size", 1
        )
        self.eval_translation_beam_alpha = train_config.get(
            "eval_translation_beam_alpha", -1
        )
        self.translation_max_output_length = train_config.get(
            "translation_max_output_length", None
        )

    def _save_checkpoint(self) -> None:
        model_path = "{}/{}.ckpt".format(self.model_dir, self.steps)
        state = {
            "steps": self.steps,
            "total_txt_tokens": self.total_txt_tokens if self.do_translation else 0,
            "total_gls_tokens": self.total_gls_tokens if self.do_recognition else 0,
            "best_ckpt_score": self.best_ckpt_score,
            "best_all_ckpt_scores": self.best_all_ckpt_scores,
            "best_ckpt_iteration": self.best_ckpt_iteration,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
        }
        torch.save(state, model_path)
        if self.ckpt_queue.full():
            to_delete = self.ckpt_queue.get()
            try:
                os.remove(to_delete)
            except FileNotFoundError:
                self.logger.warning(
                    "Wanted to delete old checkpoint %s but " "file does not exist.",
                    to_delete,
                )

        self.ckpt_queue.put(model_path)

        symlink_update(
            "{}.ckpt".format(self.steps), "{}/best.ckpt".format(self.model_dir)
        )

    def _save_interval_checkpoint(self, epoch_no) -> None:
        model_path = "{}/{}.ckpt".format(self.model_dir, epoch_no)
        state = {
            "steps": self.steps,
            "total_txt_tokens": self.total_txt_tokens if self.do_translation else 0,
            "total_gls_tokens": self.total_gls_tokens if self.do_recognition else 0,
            "best_ckpt_score": self.best_ckpt_score,
            "best_all_ckpt_scores": self.best_all_ckpt_scores,
            "best_ckpt_iteration": self.best_ckpt_iteration,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
        }
        torch.save(state, model_path)

        self.interval_ckpt_paths.append(model_path)
        if len(self.interval_ckpt_paths) > self.interval_ckpt_max:
            oldest_ckpt = self.interval_ckpt_paths.pop(0)
            try:
                os.remove(oldest_ckpt)
            except FileNotFoundError:
                self.logger.warning(
                    "Wanted to delete old checkpoint %s but file does not exist.",
                    oldest_ckpt,
                )

    def init_from_checkpoint(
        self,
        path: str,
        reset_best_ckpt: bool = False,
        reset_scheduler: bool = False,
        reset_optimizer: bool = False,
    ) -> None:
        model_checkpoint = load_checkpoint(path=path, use_cuda=self.use_cuda)

        self.model.load_state_dict(model_checkpoint["model_state"])

        if not reset_optimizer:
            self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])
        else:
            self.logger.info("Reset optimizer.")

        if not reset_scheduler:
            if (
                model_checkpoint["scheduler_state"] is not None
                and self.scheduler is not None
            ):
                self.scheduler.load_state_dict(model_checkpoint["scheduler_state"])
        else:
            self.logger.info("Reset scheduler.")

        self.steps = model_checkpoint["steps"]
        self.total_txt_tokens = model_checkpoint["total_txt_tokens"]
        self.total_gls_tokens = model_checkpoint["total_gls_tokens"]

        if not reset_best_ckpt:
            self.best_ckpt_score = model_checkpoint["best_ckpt_score"]
            self.best_all_ckpt_scores = model_checkpoint["best_all_ckpt_scores"]
            self.best_ckpt_iteration = model_checkpoint["best_ckpt_iteration"]
        else:
            self.logger.info("Reset tracking of the best checkpoint.")

        if self.use_cuda:
            self.model.cuda()

    def train_and_validate(
        self, train_data: Dataset, train_val_data: Dataset, valid_data: Dataset
    ) -> None:
        train_iter = make_data_iter(
            train_data,
            batch_size=self.batch_size,
            batch_type=self.batch_type,
            train=True,
            shuffle=self.shuffle,
        )
        checkpoint_every_n_epochs = 1
        epoch_no = None
        self.logger.info("New version")
        for epoch_no in range(self.epochs):
            self.logger.info("EPOCH %d", epoch_no + 1)

            if self.scheduler is not None and self.scheduler_step_at == "epoch":
                self.scheduler.step(epoch=epoch_no)

            self.model.train()
            start = time.time()
            total_valid_duration = 0
            count = self.batch_multiplier - 1

            if self.do_recognition:
                processed_gls_tokens = self.total_gls_tokens
                epoch_recognition_loss = 0
            if self.do_translation:
                processed_txt_tokens = self.total_txt_tokens
                epoch_translation_loss = 0

            total_batches = len(train_data) // self.batch_size
            for batch in tqdm(
                iter(train_iter), total=total_batches, desc="Processing batch"
            ):
                batch = Batch(
                    torch_batch=batch,
                    txt_pad_index=self.txt_pad_index,
                    sgn_dim=self.feature_size,
                    use_cuda=self.use_cuda,
                    is_train=True,
                )
                update = count == 0

                try:
                    recognition_loss, translation_loss = self._train_batch(
                        batch, update=update
                    )
                except ValueError as e:
                    if "Expected more than 1 value per channel when training" in str(e):
                        self.logger.warning(f"Skipping batch due to ValueError: {e}")
                    else:
                        tb = traceback.format_exc()
                        self.logger.error(f"ValueError occurred: {tb}")
                    continue
                except Exception as e:
                    tb = traceback.format_exc()
                    self.logger.error(f"Unexpected error occurred: {tb}")
                    continue

                if self.do_recognition:
                    wandb.log({"train/train_recognition_loss": recognition_loss})
                    epoch_recognition_loss += recognition_loss.detach().cpu().numpy()

                if self.do_translation:
                    wandb.log({"train/train_translation_loss": translation_loss})
                    epoch_translation_loss += translation_loss.detach().cpu().numpy()

                count = self.batch_multiplier if update else count
                count -= 1

                if (
                    self.scheduler is not None
                    and self.scheduler_step_at == "step"
                    and update
                ):
                    self.scheduler.step()

                if self.steps % self.logging_freq == 0 and update:
                    elapsed = time.time() - start - total_valid_duration

                    log_out = "[Epoch: {:03d} Step: {:08d}] ".format(
                        epoch_no + 1,
                        self.steps,
                    )

                    if self.do_recognition:
                        elapsed_gls_tokens = (
                            self.total_gls_tokens - processed_gls_tokens
                        )
                        processed_gls_tokens = self.total_gls_tokens
                        log_out += "Batch Recognition Loss: {:10.6f} => ".format(
                            recognition_loss
                        )
                        log_out += "Gls Tokens per Sec: {:8.0f} || ".format(
                            elapsed_gls_tokens / elapsed
                        )
                    if self.do_translation:
                        elapsed_txt_tokens = (
                            self.total_txt_tokens - processed_txt_tokens
                        )
                        processed_txt_tokens = self.total_txt_tokens
                        log_out += "Batch Translation Loss: {:10.6f} => ".format(
                            translation_loss
                        )
                        log_out += "Txt Tokens per Sec: {:8.0f} || ".format(
                            elapsed_txt_tokens / elapsed
                        )
                    log_out += "Lr: {:.6f}".format(self.optimizer.param_groups[0]["lr"])
                    self.logger.info(log_out)
                    start = time.time()
                    total_valid_duration = 0

                if self.steps % self.validation_freq == 0 and update:
                    for eval_name, eval_data in (
                        ("train", train_val_data),
                        ("valid", valid_data),
                    ):
                        valid_start_time = time.time()
                        val_res = validate_on_data(
                            model=self.model,
                            data=eval_data,
                            batch_size=self.eval_batch_size,
                            use_cuda=self.use_cuda,
                            batch_type=self.eval_batch_type,
                            dataset_version=self.dataset_version,
                            sgn_dim=self.feature_size,
                            txt_pad_index=self.txt_pad_index,
                            do_recognition=self.do_recognition,
                            recognition_loss_function=(
                                self.recognition_loss_function
                                if self.do_recognition
                                else None
                            ),
                            recognition_loss_weight=(
                                self.recognition_loss_weight
                                if self.do_recognition
                                else None
                            ),
                            recognition_beam_size=(
                                self.eval_recognition_beam_size
                                if self.do_recognition
                                else None
                            ),
                            do_translation=self.do_translation,
                            translation_loss_function=(
                                self.translation_loss_function
                                if self.do_translation
                                else None
                            ),
                            translation_max_output_length=(
                                self.translation_max_output_length
                                if self.do_translation
                                else None
                            ),
                            level=self.level if self.do_translation else None,
                            translation_loss_weight=(
                                self.translation_loss_weight
                                if self.do_translation
                                else None
                            ),
                            translation_beam_size=(
                                self.eval_translation_beam_size
                                if self.do_translation
                                else None
                            ),
                            translation_beam_alpha=(
                                self.eval_translation_beam_alpha
                                if self.do_translation
                                else None
                            ),
                            frame_subsampling_ratio=self.frame_subsampling_ratio,
                            logger=self.logger,
                            val_name=eval_name,
                        )
                        self.model.train()

                        if self.do_recognition:

                            wandb.log(
                                {
                                    f"eval_{eval_name}/eval_{eval_name}_recognition_loss": val_res[
                                        "valid_recognition_loss"
                                    ],
                                }
                            )
                            wandb.log(
                                {
                                    f"eval_{eval_name}/wer": val_res["valid_scores"][
                                        "wer"
                                    ],
                                }
                            )
                            wandb.log(
                                {
                                    "eval_train/wer_score_del": val_res["valid_scores"][
                                        "wer_scores"
                                    ]["del_rate"],
                                    "eval_train/wer_score_ins": val_res["valid_scores"][
                                        "wer_scores"
                                    ]["ins_rate"],
                                    "eval_train/wer_score_sub": val_res["valid_scores"][
                                        "wer_scores"
                                    ]["sub_rate"],
                                }
                            )

                        if self.do_translation:
                            wandb.log(
                                {
                                    f"eval_{eval_name}/eval_{eval_name}_translation_loss": val_res[
                                        "valid_translation_loss"
                                    ],
                                }
                            )
                            wandb.log(
                                {
                                    f"eval_{eval_name}/eval_{eval_name}_ppl": val_res[
                                        "valid_ppl"
                                    ],
                                }
                            )

                            wandb.log(
                                {
                                    f"eval_{eval_name}/chrf": val_res["valid_scores"][
                                        "chrf"
                                    ],
                                }
                            )
                            wandb.log(
                                {
                                    f"eval_{eval_name}/rouge": val_res["valid_scores"][
                                        "rouge"
                                    ],
                                }
                            )
                            wandb.log(
                                {
                                    f"eval_{eval_name}/bleu": val_res["valid_scores"][
                                        "bleu"
                                    ],
                                }
                            )
                            wandb.log(
                                {
                                    f"eval_{eval_name}/bleu1": val_res["valid_scores"][
                                        "bleu_scores"
                                    ]["bleu1"],
                                    f"eval_{eval_name}/bleu2": val_res["valid_scores"][
                                        "bleu_scores"
                                    ]["bleu2"],
                                    f"eval_{eval_name}/bleu3": val_res["valid_scores"][
                                        "bleu_scores"
                                    ]["bleu3"],
                                    f"eval_{eval_name}/bleu4": val_res["valid_scores"][
                                        "bleu_scores"
                                    ]["bleu4"],
                                }
                            )

                        if eval_name == "valid":
                            if self.early_stopping_metric == "recognition_loss":
                                assert self.do_recognition
                                ckpt_score = val_res["valid_recognition_loss"]
                            elif self.early_stopping_metric == "translation_loss":
                                assert self.do_translation
                                ckpt_score = val_res["valid_translation_loss"]
                            elif self.early_stopping_metric in [
                                "ppl",
                                "perplexity",
                            ]:
                                assert self.do_translation
                                ckpt_score = val_res["valid_ppl"]
                            else:
                                ckpt_score = val_res["valid_scores"][self.eval_metric]

                            new_best = False
                            if self.is_best(ckpt_score):
                                self.best_ckpt_score = ckpt_score
                                self.best_all_ckpt_scores = val_res["valid_scores"]
                                self.best_ckpt_iteration = self.steps
                                self.logger.info(
                                    "Hooray! New best validation result [%s]!",
                                    self.early_stopping_metric,
                                )
                                if self.ckpt_queue.maxsize > 0:
                                    self.logger.info("Saving new checkpoint.")
                                    new_best = True
                                    self._save_checkpoint()

                            if (
                                self.scheduler is not None
                                and self.scheduler_step_at == "validation"
                            ):
                                prev_lr = self.scheduler.optimizer.param_groups[0]["lr"]
                                self.scheduler.step(ckpt_score)
                                now_lr = self.scheduler.optimizer.param_groups[0]["lr"]

                                if prev_lr != now_lr:
                                    if self.last_best_lr != prev_lr:
                                        self.stop = True

                            self._add_report(
                                valid_scores=val_res["valid_scores"],
                                valid_recognition_loss=(
                                    val_res["valid_recognition_loss"]
                                    if self.do_recognition
                                    else None
                                ),
                                valid_translation_loss=(
                                    val_res["valid_translation_loss"]
                                    if self.do_translation
                                    else None
                                ),
                                valid_ppl=(
                                    val_res["valid_ppl"]
                                    if self.do_translation
                                    else None
                                ),
                                eval_metric=self.eval_metric,
                                new_best=new_best,
                            )
                            valid_duration = time.time() - valid_start_time
                            total_valid_duration += valid_duration
                            self.logger.info(
                                "Validation result at epoch %3d, step %8d: duration: %.4fs\n\t"
                                "Recognition Beam Size: %d\t"
                                "Translation Beam Size: %d\t"
                                "Translation Beam Alpha: %d\n\t"
                                "Recognition Loss: %4.5f\t"
                                "Translation Loss: %4.5f\t"
                                "PPL: %4.5f\n\t"
                                "Eval Metric: %s\n\t"
                                "WER %3.2f\t(DEL: %3.2f,\tINS: %3.2f,\tSUB: %3.2f)\n\t"
                                "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
                                "CHRF %.2f\t"
                                "ROUGE %.2f",
                                epoch_no + 1,
                                self.steps,
                                valid_duration,
                                (
                                    self.eval_recognition_beam_size
                                    if self.do_recognition
                                    else -1
                                ),
                                (
                                    self.eval_translation_beam_size
                                    if self.do_translation
                                    else -1
                                ),
                                (
                                    self.eval_translation_beam_alpha
                                    if self.do_translation
                                    else -1
                                ),
                                (
                                    val_res["valid_recognition_loss"]
                                    if self.do_recognition
                                    else -1
                                ),
                                (
                                    val_res["valid_translation_loss"]
                                    if self.do_translation
                                    else -1
                                ),
                                val_res["valid_ppl"] if self.do_translation else -1,
                                self.eval_metric.upper(),
                                (
                                    val_res["valid_scores"]["wer"]
                                    if self.do_recognition
                                    else -1
                                ),
                                (
                                    val_res["valid_scores"]["wer_scores"]["del_rate"]
                                    if self.do_recognition
                                    else -1
                                ),
                                (
                                    val_res["valid_scores"]["wer_scores"]["ins_rate"]
                                    if self.do_recognition
                                    else -1
                                ),
                                (
                                    val_res["valid_scores"]["wer_scores"]["sub_rate"]
                                    if self.do_recognition
                                    else -1
                                ),
                                (
                                    val_res["valid_scores"]["bleu"]
                                    if self.do_translation
                                    else -1
                                ),
                                (
                                    val_res["valid_scores"]["bleu_scores"]["bleu1"]
                                    if self.do_translation
                                    else -1
                                ),
                                (
                                    val_res["valid_scores"]["bleu_scores"]["bleu2"]
                                    if self.do_translation
                                    else -1
                                ),
                                (
                                    val_res["valid_scores"]["bleu_scores"]["bleu3"]
                                    if self.do_translation
                                    else -1
                                ),
                                (
                                    val_res["valid_scores"]["bleu_scores"]["bleu4"]
                                    if self.do_translation
                                    else -1
                                ),
                                (
                                    val_res["valid_scores"]["chrf"]
                                    if self.do_translation
                                    else -1
                                ),
                                (
                                    val_res["valid_scores"]["rouge"]
                                    if self.do_translation
                                    else -1
                                ),
                            )

                            self._log_examples(
                                sequences=[s for s in valid_data.sequence],
                                gls_references=(
                                    val_res["gls_ref"] if self.do_recognition else None
                                ),
                                gls_hypotheses=(
                                    val_res["gls_hyp"] if self.do_recognition else None
                                ),
                                txt_references=(
                                    val_res["txt_ref"] if self.do_translation else None
                                ),
                                txt_hypotheses=(
                                    val_res["txt_hyp"] if self.do_translation else None
                                ),
                            )

                            valid_seq = [s for s in valid_data.sequence]
                            if self.do_recognition:
                                self._store_outputs(
                                    "dev.hyp.gls",
                                    valid_seq,
                                    val_res["gls_hyp"],
                                    "gls",
                                )
                                self._store_outputs(
                                    "references.dev.gls",
                                    valid_seq,
                                    val_res["gls_ref"],
                                )

                            if self.do_translation:
                                self._store_outputs(
                                    "dev.hyp.txt",
                                    valid_seq,
                                    val_res["txt_hyp"],
                                    "txt",
                                )
                                self._store_outputs(
                                    "references.dev.txt",
                                    valid_seq,
                                    val_res["txt_ref"],
                                )

                        if self.stop:
                            break

            if (epoch_no + 1) % checkpoint_every_n_epochs == 0:
                self.logger.info("Saving checkpoint at epoch %d", epoch_no + 1)
                self._save_interval_checkpoint(epoch_no + 1)
            if self.stop:
                if (
                    self.scheduler is not None
                    and self.scheduler_step_at == "validation"
                    and self.last_best_lr != prev_lr
                ):
                    self.logger.info(
                        "Training ended since there were no improvements in"
                        "the last learning rate step: %f",
                        prev_lr,
                    )
                else:
                    self.logger.info(
                        "Training ended since minimum lr %f was reached.",
                        self.learning_rate_min,
                    )
                break

            self.logger.info(
                "Epoch %3d: Total Training Recognition Loss %.2f "
                " Total Training Translation Loss %.2f ",
                epoch_no + 1,
                epoch_recognition_loss if self.do_recognition else -1,
                epoch_translation_loss if self.do_translation else -1,
            )
        else:
            self.logger.info("Training ended after %3d epochs.", epoch_no + 1)
        self.logger.info(
            "Best validation result at step %8d: %6.2f %s.",
            self.best_ckpt_iteration,
            self.best_ckpt_score,
            self.early_stopping_metric,
        )

        wandb.finish()

    def _train_batch(self, batch: Batch, update: bool = True) -> (Tensor, Tensor):

        recognition_loss, translation_loss = self.model.get_loss_for_batch(
            batch=batch,
            recognition_loss_function=(
                self.recognition_loss_function if self.do_recognition else None
            ),
            translation_loss_function=(
                self.translation_loss_function if self.do_translation else None
            ),
            recognition_loss_weight=(
                self.recognition_loss_weight if self.do_recognition else None
            ),
            translation_loss_weight=(
                self.translation_loss_weight if self.do_translation else None
            ),
        )

        if self.do_translation:
            if self.translation_normalization_mode == "batch":
                txt_normalization_factor = batch.num_seqs
            elif self.translation_normalization_mode == "tokens":
                txt_normalization_factor = batch.num_txt_tokens
            else:
                raise NotImplementedError("Only normalize by 'batch' or 'tokens'")

            normalized_translation_loss = translation_loss / (
                txt_normalization_factor * self.batch_multiplier
            )
        else:
            normalized_translation_loss = 0

        if self.do_recognition:
            normalized_recognition_loss = recognition_loss / self.batch_multiplier
        else:
            normalized_recognition_loss = 0

        total_loss = normalized_recognition_loss + normalized_translation_loss
        total_loss.backward()

        if self.clip_grad_fun is not None:
            self.clip_grad_fun(params=self.model.parameters())

        if update:
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.steps += 1

        if self.do_recognition:
            self.total_gls_tokens += batch.num_gls_tokens
        if self.do_translation:
            self.total_txt_tokens += batch.num_txt_tokens

        return normalized_recognition_loss, normalized_translation_loss

    def _add_report(
        self,
        valid_scores: Dict,
        valid_recognition_loss: float,
        valid_translation_loss: float,
        valid_ppl: float,
        eval_metric: str,
        new_best: bool = False,
    ) -> None:
        current_lr = -1
        for param_group in self.optimizer.param_groups:
            current_lr = param_group["lr"]

        if new_best:
            self.last_best_lr = current_lr

        if current_lr < self.learning_rate_min:
            self.stop = True

        with open(self.valid_report_file, "a", encoding="utf-8") as opened_file:
            opened_file.write(
                "Steps: {}\t"
                "Recognition Loss: {:.5f}\t"
                "Translation Loss: {:.5f}\t"
                "PPL: {:.5f}\t"
                "Eval Metric: {}\t"
                "WER {:.2f}\t(DEL: {:.2f},\tINS: {:.2f},\tSUB: {:.2f})\t"
                "BLEU-4 {:.2f}\t(BLEU-1: {:.2f},\tBLEU-2: {:.2f},\tBLEU-3: {:.2f},\tBLEU-4: {:.2f})\t"
                "CHRF {:.2f}\t"
                "ROUGE {:.2f}\t"
                "LR: {:.8f}\t{}\n".format(
                    self.steps,
                    valid_recognition_loss if self.do_recognition else -1,
                    valid_translation_loss if self.do_translation else -1,
                    valid_ppl if self.do_translation else -1,
                    eval_metric,
                    valid_scores["wer"] if self.do_recognition else -1,
                    (
                        valid_scores["wer_scores"]["del_rate"]
                        if self.do_recognition
                        else -1
                    ),
                    (
                        valid_scores["wer_scores"]["ins_rate"]
                        if self.do_recognition
                        else -1
                    ),
                    (
                        valid_scores["wer_scores"]["sub_rate"]
                        if self.do_recognition
                        else -1
                    ),
                    valid_scores["bleu"] if self.do_translation else -1,
                    valid_scores["bleu_scores"]["bleu1"] if self.do_translation else -1,
                    valid_scores["bleu_scores"]["bleu2"] if self.do_translation else -1,
                    valid_scores["bleu_scores"]["bleu3"] if self.do_translation else -1,
                    valid_scores["bleu_scores"]["bleu4"] if self.do_translation else -1,
                    valid_scores["chrf"] if self.do_translation else -1,
                    valid_scores["rouge"] if self.do_translation else -1,
                    current_lr,
                    "*" if new_best else "",
                )
            )

    def _log_parameters_list(self) -> None:
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info("Total params: %d", n_params)
        trainable_params = [
            n for (n, p) in self.model.named_parameters() if p.requires_grad
        ]
        self.logger.info("Trainable parameters: %s", sorted(trainable_params))
        assert trainable_params

    def _log_examples(
        self,
        sequences: List[str],
        gls_references: List[str],
        gls_hypotheses: List[str],
        txt_references: List[str],
        txt_hypotheses: List[str],
    ) -> None:
        if self.do_recognition:
            assert len(gls_references) == len(gls_hypotheses)
            num_sequences = len(gls_hypotheses)
        if self.do_translation:
            assert len(txt_references) == len(txt_hypotheses)
            num_sequences = len(txt_hypotheses)

        rand_idx = np.sort(np.random.permutation(num_sequences)[: self.num_valid_log])
        self.logger.info("Logging Recognition and Translation Outputs")
        self.logger.info("=" * 120)
        for ri in rand_idx:
            self.logger.info("Logging Sequence: %s", sequences[ri])
            if self.do_recognition:
                gls_res = wer_single(r=gls_references[ri], h=gls_hypotheses[ri])
                self.logger.info(
                    "\tGloss Reference :\t%s", gls_res["alignment_out"]["align_ref"]
                )
                self.logger.info(
                    "\tGloss Hypothesis:\t%s", gls_res["alignment_out"]["align_hyp"]
                )
                self.logger.info(
                    "\tGloss Alignment :\t%s", gls_res["alignment_out"]["alignment"]
                )
            if self.do_recognition and self.do_translation:
                self.logger.info("\t" + "-" * 116)
            if self.do_translation:
                txt_res = wer_single(r=txt_references[ri], h=txt_hypotheses[ri])
                self.logger.info(
                    "\tText Reference  :\t%s", txt_res["alignment_out"]["align_ref"]
                )
                self.logger.info(
                    "\tText Hypothesis :\t%s", txt_res["alignment_out"]["align_hyp"]
                )
                self.logger.info(
                    "\tText Alignment  :\t%s", txt_res["alignment_out"]["alignment"]
                )
            self.logger.info("=" * 120)

    def _store_outputs(
        self, tag: str, sequence_ids: List[str], hypotheses: List[str], sub_folder=None
    ) -> None:
        if sub_folder:
            out_folder = os.path.join(self.model_dir, sub_folder)
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            current_valid_output_file = "{}/{}.{}".format(out_folder, self.steps, tag)
        else:
            out_folder = self.model_dir
            current_valid_output_file = "{}/{}".format(out_folder, tag)

        with open(current_valid_output_file, "w", encoding="utf-8") as opened_file:
            for seq, hyp in zip(sequence_ids, hypotheses):
                opened_file.write("{}|{}\n".format(seq, hyp))


def collate_fn(batch):
    collated = {"video_segments": [], "keypoints": [], "targets": []}
    for video_segment, keypoints, target in batch:
        collated["video_segments"].append(video_segment.float())
        collated["keypoints"].append(keypoints.float())
        collated["targets"].append(target.long())
    return dict(collated)


def train(cfg_file: str) -> None:
    cfg = load_config(cfg_file)

    set_seed(seed=cfg["training"].get("random_seed", 42))

    train_data, train_val_data, dev_data, gls_vocab, txt_vocab = load_data(
        data_cfg=cfg["data"]
    )

    do_recognition = cfg["training"].get("recognition_loss_weight", 1.0) > 0.0
    do_translation = cfg["training"].get("translation_loss_weight", 1.0) > 0.0
    model = build_model(
        cfg=cfg["model"],
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        sgn_dim=cfg["data"]["feature_size"],
        do_recognition=do_recognition,
        do_translation=do_translation,
    )
    trainer = TrainManager(model=model, config=cfg)

    shutil.copy2(cfg_file, trainer.model_dir + "/config.yaml")

    log_cfg(cfg, trainer.logger)

    log_data_info(
        train_data=train_data,
        valid_data=dev_data,
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        logging_function=trainer.logger.info,
    )

    trainer.logger.info(str(model))

    gls_vocab_file = "{}/gls.vocab".format(cfg["training"]["model_dir"])
    gls_vocab.to_file(gls_vocab_file)
    txt_vocab_file = "{}/txt.vocab".format(cfg["training"]["model_dir"])
    txt_vocab.to_file(txt_vocab_file)

    trainer.train_and_validate(
        train_data=train_data, train_val_data=train_val_data, valid_data=dev_data
    )
    del train_data, train_val_data, dev_data

    ckpt = "{}/{}.ckpt".format(trainer.model_dir, trainer.best_ckpt_iteration)
    output_name = "best.IT_{:08d}".format(trainer.best_ckpt_iteration)
    output_path = os.path.join(trainer.model_dir, output_name)
    logger = trainer.logger
    del trainer
    test(cfg_file, ckpt=ckpt, output_path=output_path, logger=logger)
