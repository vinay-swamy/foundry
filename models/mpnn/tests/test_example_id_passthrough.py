"""The validation metrics must be joinable back to the manifest row they came from.

`example_id` has to survive three hops: the transform pipeline's final key subset, the
collator, and the trainer's call into the MetricManager (which reads it from
`extra_info["example_id"]`). Before this was fixed every hop dropped it, so every row
of `validation_output_all_epochs.csv` had an empty `example_id`.
"""

import torch
from mpnn.collate.feature_collator import FeatureCollator
from mpnn.pipelines.mpnn import build_mpnn_transform_pipeline
from mpnn.trainers.mpnn import metrics_extra_info_from_batch


def _example(example_id=None, n=2):
    ex = {"input_features": {"x": torch.zeros(n)}, "atom_array": f"aa-{example_id}"}
    if example_id is not None:
        ex["example_id"] = example_id
    return ex


def test_pipeline_minimal_return_keeps_example_id():
    pipeline = build_mpnn_transform_pipeline(
        model_type="protein_mpnn", is_inference=True, minimal_return=True
    )
    subset = pipeline.transforms[-1]
    out = subset.forward(
        {"input_features": {}, "atom_array": None, "example_id": "132l_A", "junk": 1}
    )
    assert out["example_id"] == "132l_A"
    assert "junk" not in out


def test_pipeline_subset_tolerates_missing_example_id():
    # The inference engine builds data without an id; that path must not break.
    pipeline = build_mpnn_transform_pipeline(
        model_type="protein_mpnn", is_inference=True, minimal_return=True
    )
    out = pipeline.transforms[-1].forward({"input_features": {}, "atom_array": None})
    assert "example_id" not in out


def test_collator_carries_example_ids_in_order():
    batch = FeatureCollator(default_padding={"x": 0})([_example("a"), _example("b")])
    assert batch["example_id"] == ["a", "b"]


def test_collator_omits_example_id_when_absent():
    batch = FeatureCollator(default_padding={"x": 0})([_example(), _example()])
    assert "example_id" not in batch


def test_metrics_extra_info_single_example():
    assert metrics_extra_info_from_batch({"example_id": ["132l_A"]}) == {
        "example_id": "132l_A"
    }


def test_metrics_extra_info_without_ids_is_empty():
    assert metrics_extra_info_from_batch({"input_features": {}}) == {}
