import torch

from foundry.callbacks.mpnn_logging import LogMPNNTrainingLossCallback


class _FakeFabric:
    def __init__(self, has_loggers):
        self.loggers = [object()] if has_loggers else []
        self.is_global_zero = True
        self.calls = []

    def log(self, name, value, step=None):
        self.calls.append((name, value, step))


class _FakeTrainer:
    def __init__(self, step, has_loggers=True):
        self.fabric = _FakeFabric(has_loggers)
        self.state = {"global_step": step}


def test_logs_loss_on_cadence():
    cb = LogMPNNTrainingLossCallback(log_every_n=10)
    trainer = _FakeTrainer(step=20)
    cb.on_train_batch_end(trainer, {"total_loss": torch.tensor(2.5)}, batch=None, batch_idx=0)
    assert trainer.fabric.calls == [("train/loss", 2.5, 20)]


def test_skips_off_cadence():
    cb = LogMPNNTrainingLossCallback(log_every_n=10)
    trainer = _FakeTrainer(step=23)
    cb.on_train_batch_end(trainer, {"total_loss": torch.tensor(2.5)}, batch=None, batch_idx=0)
    assert trainer.fabric.calls == []


def test_noop_without_loggers():
    cb = LogMPNNTrainingLossCallback(log_every_n=1)
    trainer = _FakeTrainer(step=5, has_loggers=False)
    cb.on_train_batch_end(trainer, {"total_loss": torch.tensor(1.0)}, batch=None, batch_idx=0)
    assert trainer.fabric.calls == []
