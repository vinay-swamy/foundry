"""MPNN-specific training logging callbacks (kept separate from train_logging.py, which
imports rf3-specific loss helpers at module import time)."""
from beartype.typing import Any

from foundry.callbacks.callback import BaseCallback


class LogMPNNTrainingLossCallback(BaseCallback):
    """Log the scalar training loss to the fabric loggers every ``log_every_n`` steps.

    The MPNN ``training_step`` stores ``{"total_loss": ..., "loss_dict": ...}`` in
    ``_current_train_return``; the trainer passes it here as ``outputs``.
    """

    def __init__(self, log_every_n: int = 50):
        self.log_every_n = log_every_n

    def on_train_batch_end(
        self, trainer: Any, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        if not trainer.fabric.loggers or not trainer.fabric.is_global_zero:
            return
        step = trainer.state["global_step"]
        if step % self.log_every_n != 0:
            return
        loss = outputs.get("total_loss") if isinstance(outputs, dict) else None
        if loss is None:
            return
        trainer.fabric.log("train/loss", float(loss), step=step)
