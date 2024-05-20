import collections
import logging
import time
from pathlib import Path

import torch
from main.src.utils import (
    Decode,
    get_previous_checkpoints,
    load_best_checkpoint,
    save_checkpoint,
    to_cuda,
)
from tqdm import tqdm

from ..constants import SAVE_MODEL_DIRECTORY, VALIDATION_FREQUENCY


class Trainer:
    def __init__(
        self,
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        optimizer=None,
        slr_model=None,
        train_loader=None,
        val_loader=None,
        eval_metrics=None,
        blank_id=None,
        tokenizer=None,
        wandb=None,
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stop_count = early_stop_count
        self.epochs = epochs

        self.slr_model = slr_model

        self.tokenizer = tokenizer

        self.optimizer = optimizer(slr_model.parameters(), learning_rate)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.eval_metrics = eval_metrics

        self.wandb = wandb

        self.num_steps_per_val = len(self.train_loader) // VALIDATION_FREQUENCY

        self.global_step = 0
        self.start_time = time.time()

        self.train_history = {"train_loss": collections.OrderedDict()}

        self.validation_history = {"val_loss": collections.OrderedDict()}

        if eval_metrics:
            for metric_name in eval_metrics.keys():
                self.validation_history[metric_name] = collections.OrderedDict()

        no_gpus = torch.cuda.device_count()
        if no_gpus > 1:
            logging.info(f"Found multiple GPUs, using {no_gpus}.")
            self.slr_model = torch.nn.DataParallel(self.slr_model)

        self.slr_model = to_cuda(self.slr_model)

        self.decoder = Decode(tokenizer, blank_id)

        self.checkpoint_dir = Path(SAVE_MODEL_DIRECTORY)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def should_early_stop(self):
        val_loss = self.validation_history["val_loss"]
        if len(val_loss) < self.early_stop_count:
            return False
        relevant_loss = list(val_loss.values())[-self.early_stop_count :]
        first_loss = relevant_loss[0]
        if first_loss == min(relevant_loss):
            logging.info("Early stop criteria met")
            return True
        return False

    def train(self):
        self.slr_model.train()
        for epoch in range(self.epochs):
            logging.info(f"Starting training for epoch {epoch}...")
            self.epoch = epoch
            total_train_loss = 0.0
            num_batches = 0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")

            def should_validate_model():
                return self.global_step % self.num_steps_per_val == 0

            for batch in progress_bar:
                X_batch = batch["video_segments"]
                keypoints = batch["keypoints"]
                Y_batch = batch["targets"]
                loss = self.train_step(X_batch, Y_batch)
                total_train_loss += loss
                self.global_step += 1

                num_batches += 1
                progress_bar.set_postfix(loss=loss)

                if should_validate_model():

                    mean_train_loss = total_train_loss / num_batches
                    self.train_history["train_loss"][self.global_step] = mean_train_loss

                    if self.wandb:
                        self.wandb.log({"train_loss": mean_train_loss})

                    num_batches = 0
                    total_train_loss = 0.0

                    self.validation()
                    self.save_model()
                    if self.should_early_stop():
                        logging.info("Early stopping.")
                        return

                    logging.info(
                        f"Epoch {epoch+1}, average loss: {mean_train_loss:.4f}"
                    )

    def train_step(self, X_batch, Y_batch):
        X_batch = to_cuda(X_batch)
        Y_batch = to_cuda(Y_batch)

        log_probs = self.slr_model(X_batch, Y_batch)
        decoded_sentences = self.decoder.decode(log_probs)
        logging.info(decoded_sentences)
        predicted_tokens = [
            self.tokenizer.encode(sentence) for sentence in decoded_sentences
        ]
        predicted_tokens = to_cuda(torch.tensor(predicted_tokens))

        loss = self.slr_model.compute_ctc_loss(
            log_probs, predicted_tokens, Y_batch, alpha=10.0
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

    def validation(self):
        torch.cuda.empty_cache()
        logging.info("Starting validation...")
        self.slr_model.eval()
        total_val_loss = 0
        num_batches = 0
        validation_bar = tqdm(self.val_loader, desc="Validation")

        with torch.no_grad():
            ref_sentences = []
            hyp_sentences = []
            for batch in validation_bar:
                X_batch = to_cuda(batch["video_segments"])
                Y_batch = to_cuda(batch["targets"])

                log_probs = self.slr_model(X_batch)

                val_loss = self.slr_model.compute_ctc_loss(
                    log_probs, Y_batch, alpha=1.0
                )
                total_val_loss += val_loss.item()
                num_batches += 1
                ref_sentences.extend(self.decoder.decode_targets(Y_batch))
                hyp_sentences.extend(self.decoder.decode(log_probs))

        mean_val_loss = total_val_loss / num_batches
        logging.info(hyp_sentences)

        self.validation_history["val_loss"][self.global_step] = mean_val_loss
        if self.eval_metrics:
            for metric_name, metric_info in self.eval_metrics.items():
                metric_fn = metric_info["fn"]
                kwargs = metric_info["parameters"]
                self.validation_history[metric_name][self.global_step] = metric_fn(
                    ref_sentences, hyp_sentences, **kwargs
                )

        used_time = time.time() - self.start_time

        if self.wandb:
            metrics_to_log = {
                metric: values[self.global_step]
                for metric, values in self.validation_history.items()
            }
            self.wandb.log(metrics_to_log)

        logging.info(
            ", ".join(
                [
                    f"Epoch: {self.epoch:>1}",
                    f"Batches per seconds: {self.global_step / used_time:.2f}",
                    f"Global step: {self.global_step:>6}",
                    f"Validation Loss: {val_loss:.2f}",
                ]
            )
        )
        self.slr_model.train()
        torch.cuda.empty_cache()

    def save_model(self):
        logging.info("Saving model...")

        def is_best_model():
            val_loss = self.validation_history["val_loss"]
            validation_losses = list(val_loss.values())
            return validation_losses[-1] == min(validation_losses)

        state_dict = self.slr_model.state_dict()
        filepath = self.checkpoint_dir.joinpath(f"{self.global_step}.ckpt")

        save_checkpoint(state_dict, filepath, is_best_model())

    def load_best_model(self):
        state_dict = load_best_checkpoint(self.checkpoint_dir)
        if state_dict is None:
            logging.info(
                f"Could not load best checkpoint. Did not find under: {self.checkpoint_dir}"
            )
            return
        self.slr_model.load_state_dict(state_dict)
