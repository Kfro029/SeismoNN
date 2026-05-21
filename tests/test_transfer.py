import torch
from torch import nn

from seismonn.models.masked_transformer import MaskedTraceReconstructionTransformer
from seismonn.models.transformer import TraceTransformerClassifier
from seismonn.training.transfer import (
    load_pretrained_encoder_weights,
    maybe_load_pretrained_encoder,
    select_compatible_encoder_weights,
)


def create_pretraining_model() -> MaskedTraceReconstructionTransformer:
    return MaskedTraceReconstructionTransformer(
        in_channels=2,
        time_steps=32,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.1,
        temporal_hidden_channels=16,
        max_receivers=16,
    )


def create_classifier_model() -> TraceTransformerClassifier:
    return TraceTransformerClassifier(
        in_channels=2,
        num_classes=3,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.1,
        temporal_hidden_channels=16,
        max_receivers=16,
    )


def fill_encoder_weights_with_constant(model: nn.Module, value: float) -> None:
    for name, parameter in model.named_parameters():
        if name.startswith(("temporal_encoder.", "encoder.")):
            parameter.data.fill_(value)


def test_select_compatible_encoder_weights_selects_only_encoder_keys():
    pretraining_model = create_pretraining_model()
    classifier_model = create_classifier_model()

    selected, summary = select_compatible_encoder_weights(
        pretrained_state_dict=pretraining_model.state_dict(),
        target_state_dict=classifier_model.state_dict(),
        prefixes=("temporal_encoder.", "encoder."),
    )

    assert len(selected) > 0
    assert summary["selected_key_count"] == len(selected)

    assert all(key.startswith(("temporal_encoder.", "encoder.")) for key in selected)

    assert "reconstruction_head.weight" not in selected
    assert "reconstruction_head.bias" not in selected


def test_load_pretrained_encoder_weights_loads_matching_weights(tmp_path):
    pretraining_model = create_pretraining_model()
    classifier_model = create_classifier_model()

    fill_encoder_weights_with_constant(pretraining_model, value=0.123)

    checkpoint_path = tmp_path / "pretrained.pt"

    torch.save(
        {
            "model_name": "masked_trace_transformer",
            "model_state_dict": pretraining_model.state_dict(),
            "model_config": {
                "name": "masked_trace_transformer",
                "d_model": 32,
            },
            "epoch": 1,
        },
        checkpoint_path,
    )

    summary = load_pretrained_encoder_weights(
        model=classifier_model,
        checkpoint_path=checkpoint_path,
        device=torch.device("cpu"),
    )

    assert summary["loaded_key_count"] > 0
    assert summary["checkpoint_model_name"] == "masked_trace_transformer"
    assert summary["checkpoint_epoch"] == 1

    classifier_state = classifier_model.state_dict()

    assert torch.allclose(
        classifier_state["temporal_encoder.0.weight"],
        torch.full_like(classifier_state["temporal_encoder.0.weight"], 0.123),
    )


def test_maybe_load_pretrained_encoder_returns_none_when_disabled(tmp_path):
    classifier_model = create_classifier_model()

    result = maybe_load_pretrained_encoder(
        model=classifier_model,
        pretrained_config={"enabled": False},
        project_root=tmp_path,
        device=torch.device("cpu"),
    )

    assert result is None
