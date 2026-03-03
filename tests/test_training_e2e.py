"""End-to-end training integration tests for full Presto pipeline."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F


def test_e2e_trainer_with_presto_dataset_and_dataloader(tmp_path):
    from presto.data import (
        PrestoDataset,
        create_dataloader,
        generate_synthetic_binding_data,
        generate_synthetic_elution_data,
        generate_synthetic_tcr_data,
        generate_synthetic_mhc_sequences,
    )
    from presto.models.presto import Presto
    from presto.training.trainer import Trainer

    alleles = ["HLA-A*02:01", "HLA-B*07:02"]
    dataset = PrestoDataset(
        binding_records=generate_synthetic_binding_data(12, alleles),
        elution_records=generate_synthetic_elution_data(8, alleles),
        tcr_records=generate_synthetic_tcr_data(8, alleles),
        mhc_sequences=generate_synthetic_mhc_sequences(alleles),
    )
    dataloader = create_dataloader(dataset, batch_size=4, shuffle=False)

    model = Presto(d_model=64, n_layers=1, n_heads=4)
    trainer = Trainer(model, lr=1e-3, device="cpu")

    losses = []
    for step, batch in enumerate(dataloader):
        train_batch = {
            "pep_tok": batch.pep_tok,
            "mhc_a_tok": batch.mhc_a_tok,
            "mhc_b_tok": batch.mhc_b_tok,
            "mhc_class": batch.mhc_class,
        }
        if batch.bind_target is not None:
            train_batch["bind_target"] = batch.bind_target
            train_batch["bind_qual"] = batch.bind_qual
        if batch.tcr_a_tok is not None:
            train_batch["tcr_a_tok"] = batch.tcr_a_tok
        if batch.tcr_b_tok is not None:
            train_batch["tcr_b_tok"] = batch.tcr_b_tok
        if batch.tcell_label is not None:
            train_batch["tcell_label"] = batch.tcell_label
        if batch.elution_label is not None:
            train_batch["elution_label"] = batch.elution_label

        loss = trainer.train_step(train_batch)
        losses.append(loss.item())
        if step >= 2:
            break

    assert losses
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)

    ckpt = tmp_path / "e2e_checkpoint.pt"
    trainer.save_checkpoint(ckpt)
    assert ckpt.exists()


def test_e2e_train_synthetic_script_run(tmp_path):
    from presto.scripts.train_synthetic import run

    checkpoint = tmp_path / "synthetic.ckpt"
    data_dir = tmp_path / "synthetic_data"
    args = SimpleNamespace(
        epochs=1,
        batch_size=4,
        lr=1e-3,
        d_model=64,
        n_layers=1,
        n_heads=4,
        n_binding=12,
        n_elution=8,
        n_tcr=8,
        data_dir=str(data_dir),
        checkpoint=str(checkpoint),
        seed=7,
    )

    run(args)
    assert checkpoint.exists()


def test_e2e_tiny_full_model_training_is_finite_and_improves():
    from presto.data.collate import PrestoSample, PrestoCollator
    from presto.models.presto import Presto
    from presto.training.losses import censor_aware_loss
    from presto.training.trainer import Trainer

    def _as_train_batch(collated_batch):
        batch = {
            "pep_tok": collated_batch.pep_tok,
            "mhc_a_tok": collated_batch.mhc_a_tok,
            "mhc_b_tok": collated_batch.mhc_b_tok,
            "mhc_class": collated_batch.mhc_class,
        }
        if collated_batch.targets:
            batch["targets"] = collated_batch.targets
        if collated_batch.target_masks:
            batch["target_masks"] = collated_batch.target_masks
        if collated_batch.target_quals:
            batch["target_quals"] = collated_batch.target_quals
        optional = [
            "flank_n_tok",
            "flank_c_tok",
            "tcr_a_tok",
            "tcr_b_tok",
            "tcell_context",
            "bind_target",
            "bind_qual",
            "bind_mask",
            "kon_target",
            "kon_mask",
            "koff_target",
            "koff_mask",
            "t_half_target",
            "t_half_mask",
            "tm_target",
            "tm_mask",
            "tcell_label",
            "tcell_mask",
            "elution_label",
            "elution_mask",
            "processing_label",
            "processing_mask",
        ]
        for key in optional:
            value = getattr(collated_batch, key)
            if value is not None:
                batch[key] = value
        return batch

    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
        return float((values * mask).sum().item() / (mask.sum().item() + 1e-8))

    def _supervised_metrics(model: Presto, batch: dict) -> dict:
        model.eval()
        with torch.no_grad():
            outputs = model(
                pep_tok=batch["pep_tok"],
                mhc_a_tok=batch["mhc_a_tok"],
                mhc_b_tok=batch["mhc_b_tok"],
                mhc_class="I",
                tcr_a_tok=batch.get("tcr_a_tok"),
                tcr_b_tok=batch.get("tcr_b_tok"),
                flank_n_tok=batch.get("flank_n_tok"),
                flank_c_tok=batch.get("flank_c_tok"),
            )
        bind = censor_aware_loss(
            outputs["assays"]["KD_nM"].squeeze(-1),
            batch["bind_target"].squeeze(-1),
            batch["bind_qual"].squeeze(-1),
            reduction="none",
        )
        t_half = (outputs["assays"]["t_half"].squeeze(-1) - batch["t_half_target"].squeeze(-1)) ** 2
        tm = (outputs["assays"]["Tm"].squeeze(-1) - batch["tm_target"].squeeze(-1)) ** 2
        tcell = F.binary_cross_entropy_with_logits(
            outputs["tcell_logit"].squeeze(-1),
            batch["tcell_label"],
            reduction="none",
        )
        elution = F.binary_cross_entropy_with_logits(
            outputs["elution_logit"].squeeze(-1),
            batch["elution_label"],
            reduction="none",
        )
        processing = F.binary_cross_entropy_with_logits(
            outputs["processing_logit"].squeeze(-1),
            batch["processing_label"],
            reduction="none",
        )
        return {
            "bind": _masked_mean(bind, batch["bind_mask"]),
            "t_half": _masked_mean(t_half, batch["t_half_mask"]),
            "tm": _masked_mean(tm, batch["tm_mask"]),
            "tcell": _masked_mean(tcell, batch["tcell_mask"]),
            "elution": _masked_mean(elution, batch["elution_mask"]),
            "processing": _masked_mean(processing, batch["processing_mask"]),
        }

    torch.manual_seed(7)

    samples = []
    for i in range(12):
        positive = i % 2 == 0
        samples.append(
            PrestoSample(
                peptide="SIINFEKL" if positive else "GILGFVFTL",
                flank_n="AAAAAA",
                flank_c="GGGGGG",
                mhc_a="MAVMAPRTLLLLLSGALALTQTWAG",
                mhc_b="MSRSVALAVLALLSLSGLEA",
                mhc_class="I",
                tcr_a="CAVRDTNTNAGKSTF",
                tcr_b="CASSLGQDTQYF",
                bind_value=2.0 if positive else 5.8,
                bind_qual=0,
                t_half=4.0 if positive else 0.5,  # hours
                tm=62.0 if positive else 42.0,  # Celsius
                tcell_label=1.0 if positive else 0.0,
                elution_label=1.0 if positive else 0.0,
                processing_label=1.0 if positive else 0.0,
                sample_id=f"s{i}",
            )
        )

    collator = PrestoCollator(max_pep_len=15, max_mhc_len=64, max_tcr_len=32, max_flank_len=10)
    collated = collator(samples)
    batch = _as_train_batch(collated)

    model = Presto(d_model=64, n_layers=1, n_heads=4)
    trainer = Trainer(model=model, lr=5e-3, device="cpu")

    before = _supervised_metrics(trainer.model, batch)
    for _ in range(20):
        loss = trainer.train_step(batch)
        assert torch.isfinite(loss)
    after = _supervised_metrics(trainer.model, batch)

    # The aggregate supervised objective should improve, with most heads improving.
    before_total = sum(before.values()) / len(before)
    after_total = sum(after.values()) / len(after)
    assert after_total <= before_total + 1e-5
    improved_heads = sum(1 for key in before if after[key] <= before[key] + 1e-5)
    assert improved_heads >= 4

    # Forward pass should stay finite and output expected keys.
    trainer.model.eval()
    with torch.no_grad():
        full_outputs = trainer.model(
            pep_tok=batch["pep_tok"],
            mhc_a_tok=batch["mhc_a_tok"],
            mhc_b_tok=batch["mhc_b_tok"],
            mhc_class="I",
            tcr_a_tok=batch["tcr_a_tok"],
            tcr_b_tok=batch["tcr_b_tok"],
            flank_n_tok=batch["flank_n_tok"],
            flank_c_tok=batch["flank_c_tok"],
        )
        core_outputs = trainer.model(
            pep_tok=batch["pep_tok"],
            mhc_a_tok=batch["mhc_a_tok"],
            mhc_b_tok=batch["mhc_b_tok"],
            mhc_class="I",
            flank_n_tok=batch["flank_n_tok"],
            flank_c_tok=batch["flank_c_tok"],
        )

    expected_core = {
        "processing_logit",
        "processing_class1_logit",
        "processing_class2_logit",
        "mhc_class_logits",
        "mhc_class_probs",
        "mhc_is_class1_prob",
        "mhc_is_class2_prob",
        "pmhc_vec",
        "binding_latents",
        "binding_logit",
        "presentation_logit",
        "presentation_class1_logit",
        "presentation_class2_logit",
        "core_start_logit",
        "recognition_cd8_logit",
        "recognition_cd4_logit",
        "immunogenicity_cd8_logit",
        "immunogenicity_cd4_logit",
        "ms_logit",
        "assays",
        "elution_logit",
        "recognition_repertoire_logit",
        "immunogenicity_logit",
        "tcell_logit",
        "tcell_context_logits",
    }
    assert expected_core.issubset(full_outputs.keys())
    assert expected_core.issubset(core_outputs.keys())
    for key in ("tcr_vec", "match_logit", "cell_type_logits"):
        assert key not in full_outputs
        assert key not in core_outputs

    for outputs in (full_outputs, core_outputs):
        for value in outputs.values():
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all()
            elif isinstance(value, dict):
                for nested in value.values():
                    assert torch.isfinite(nested).all()
