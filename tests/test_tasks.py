"""Tests for canonical task definitions and balancing helpers."""

import pytest
import torch
import torch.nn as nn

from presto.training.tasks import (
    TASK_REGISTRY,
    BindingTask,
    ElutionTask,
    ImmunogenicityTask,
    MHC_CHAIN_TYPE_MAP,
    MHCPairingTask,
    MHCChainTypeTask,
    ProcessingTask,
    RandomBalancer,
    SpeciesTask,
    SPECIES_MAP,
    StaticBalancer,
    Task,
    TaskSpec,
    TaskTracker,
    TcellAssayTask,
    TcrEvidenceTask,
    UncertaintyBalancer,
    get_task,
    register_task,
    route_sample,
)


class TestTaskRegistry:
    def test_default_tasks_registered(self):
        expected = {
            "mhc_chain_type",
            "species",
            "mhc_pairing",
            "binding",
            "elution",
            "processing",
            "stability",
            "tcr_evidence",
            "immunogenicity",
            "tcell_assay",
        }
        assert expected.issubset(TASK_REGISTRY.keys())
        assert "tcr_pairing" not in TASK_REGISTRY
        assert "tcr_pmhc" not in TASK_REGISTRY
        assert "receptor_chain_type" not in TASK_REGISTRY

    def test_get_task_valid(self):
        task = get_task("binding")
        assert isinstance(task, BindingTask)
        assert task.name == "binding"

    def test_get_task_invalid(self):
        with pytest.raises(ValueError, match="Unknown task"):
            get_task("nonexistent_task")

    def test_register_custom_task(self):
        class CustomTask(Task):
            name = "custom_test"
            spec = TaskSpec(required_fields={"x"})

            def compute_loss(self, outputs, batch, mask=None):
                return torch.tensor(0.0)

            def compute_metrics(self, outputs, batch):
                return {}

        task = CustomTask()
        register_task(task)
        assert get_task("custom_test") is task
        del TASK_REGISTRY["custom_test"]


class TestSampleRouting:
    def test_route_binding_sample(self):
        sample = {
            "pep_seq": "SIINFEKL",
            "mhc_allele": "HLA-A*02:01",
            "bind_value": 5.0,
        }
        tasks = set(route_sample(sample))
        assert "binding" in tasks
        assert "mhc_chain_type" in tasks
        assert "species" in tasks
        assert "tcr_evidence" in tasks
        assert "immunogenicity" in tasks

    def test_route_tcr_evidence_sample(self):
        sample = {
            "pep_seq": "SIINFEKL",
            "mhc_allele": "HLA-A*02:01",
            "tcr_evidence_label": 1.0,
        }
        tasks = set(route_sample(sample))
        assert "tcr_evidence" in tasks
        assert "binding" not in tasks

    def test_route_tcell_sample(self):
        sample = {
            "pep_seq": "SIINFEKL",
            "mhc_allele": "HLA-A*02:01",
            "tcell_label": 1.0,
        }
        tasks = set(route_sample(sample))
        assert "tcell_assay" in tasks
        assert "tcr_evidence" in tasks

    def test_task_accepts_method(self):
        task = get_task("binding")
        assert task.accepts({"pep_seq": "X", "mhc_allele": "Y", "bind_value": 1.0})
        assert not task.accepts({"pep_seq": "X"})


class TestLabelDerivation:
    def test_mhc_chain_type_derivation(self):
        task = MHCChainTypeTask()
        assert task.derive_label({"mhc_allele": "HLA-A*02:01"}) == MHC_CHAIN_TYPE_MAP["HLA-A"]
        assert task.derive_label({"mhc_allele": "HLA-DRB1*01:01"}) == MHC_CHAIN_TYPE_MAP["HLA-DRB"]
        assert task.derive_label({"mhc_allele": "H-2-Kb"}) == MHC_CHAIN_TYPE_MAP["H-2-K"]

    def test_species_derivation(self):
        task = SpeciesTask()
        assert task.derive_label({"mhc_allele": "HLA-A*02:01"}) == SPECIES_MAP["HLA"]
        assert task.derive_label({"mhc_allele": "H-2-Kb"}) == SPECIES_MAP["H-2"]
        assert task.derive_label({"mhc_allele": "Mamu-A*01"}) == SPECIES_MAP["Mamu"]

    def test_mhc_pairing_derivation(self):
        task = MHCPairingTask()
        assert task.derive_label(
            {"mhc_a_allele": "HLA-A*02:01", "mhc_b_allele": "B2M"}
        ) == 1
        assert task.derive_label(
            {"mhc_a_allele": "HLA-DRA*01:01", "mhc_b_allele": "HLA-DRB1*01:01"}
        ) == 1
        assert task.derive_label(
            {"mhc_a_allele": "HLA-DRA*01:01", "mhc_b_allele": "HLA-DQB1*06:02"}
        ) == 0


class TestNegativeGeneration:
    def test_mhc_pairing_negatives(self):
        task = MHCPairingTask()
        positives = [
            {"mhc_a_seq": "ALPHA1", "mhc_b_seq": "BETA1", "pairing_label": 1},
            {"mhc_a_seq": "ALPHA2", "mhc_b_seq": "BETA2", "pairing_label": 1},
        ]
        negatives = task.generate_negatives(positives, positives, n_per_positive=2)
        assert len(negatives) == 4
        assert all(neg["pairing_label"] == 0 for neg in negatives)

    def test_elution_negatives_keep_mhc(self):
        task = ElutionTask()
        positives = [{"pep_seq": "SIINFEKL", "mhc_allele": "HLA-A*02:01", "elution_label": 1}]
        all_samples = positives + [
            {"pep_seq": "NLVPMVATV", "mhc_allele": "HLA-A*02:01", "elution_label": 1}
        ]
        negatives = task.generate_negatives(positives, all_samples, n_per_positive=3)
        assert len(negatives) == 3
        assert all(neg["mhc_allele"] == "HLA-A*02:01" for neg in negatives)
        assert all(neg["elution_label"] == 0 for neg in negatives)


class TestLossesAndMetrics:
    def test_binding_censor_aware_loss(self):
        task = BindingTask()
        outputs = {"binding_pred": torch.tensor([[1.0], [2.0], [3.0]])}
        batch = {
            "bind_target": torch.tensor([[1.0], [2.0], [2.0]]),
            "bind_qual": torch.tensor([[0], [-1], [1]]),
        }
        loss = task.compute_loss(outputs, batch)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_tcr_evidence_loss_and_metrics(self):
        task = TcrEvidenceTask()
        outputs = {"tcr_evidence_logit": torch.tensor([[3.0], [-3.0]])}
        batch = {"tcr_evidence_label": torch.tensor([1.0, 0.0])}
        loss = task.compute_loss(outputs, batch)
        metrics = task.compute_metrics(outputs, batch)
        assert torch.isfinite(loss)
        assert "auroc" in metrics
        assert metrics["auroc"] == pytest.approx(1.0, rel=1e-5)

    def test_immunogenicity_loss(self):
        task = ImmunogenicityTask()
        outputs = {"immunogenicity_logit": torch.tensor([[2.0], [-2.0]])}
        batch = {"immunogenicity_label": torch.tensor([1.0, 0.0])}
        loss = task.compute_loss(outputs, batch)
        assert torch.isfinite(loss)

    def test_tcell_assay_loss(self):
        task = TcellAssayTask()
        outputs = {"tcell_logit": torch.tensor([[2.0], [-2.0]])}
        batch = {"tcell_label": torch.tensor([1.0, 0.0])}
        loss = task.compute_loss(outputs, batch)
        assert torch.isfinite(loss)

    def test_processing_accuracy_metric(self):
        task = ProcessingTask()
        outputs = {"processing_logit": torch.tensor([[10.0], [-10.0]])}
        batch = {"processing_label": torch.tensor([1, 0])}
        metrics = task.compute_metrics(outputs, batch)
        assert metrics["accuracy"] == pytest.approx(1.0, rel=1e-5)

    def test_mhc_chain_accuracy_metric(self):
        task = MHCChainTypeTask()
        outputs = {
            "mhc_chain_logits": torch.tensor(
                [
                    [10.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 10.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 10.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [10.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            )
        }
        batch = {"mhc_chain_type_label": torch.tensor([0, 1, 2, 3])}
        metrics = task.compute_metrics(outputs, batch)
        assert metrics["accuracy"] == pytest.approx(0.75, rel=1e-5)


class TestTaskBalancing:
    def test_static_balancer(self):
        balancer = StaticBalancer({"task1": 1.0, "task2": 0.5, "task3": 2.0})
        result = balancer.get_weights(
            {"task1": torch.tensor(1.0), "task2": torch.tensor(1.0), "task3": torch.tensor(1.0)}
        )
        assert result["task1"] == 1.0
        assert result["task2"] == 0.5
        assert result["task3"] == 2.0

    def test_random_balancer(self):
        balancer = RandomBalancer()
        weights = balancer.get_weights(
            {"task1": torch.tensor(1.0), "task2": torch.tensor(1.0), "task3": torch.tensor(1.0)}
        )
        assert sum(weights.values()) == pytest.approx(3.0, rel=0.1)

    def test_uncertainty_balancer_init(self):
        balancer = UncertaintyBalancer(["binding", "elution", "tcell_assay"])
        assert len(balancer.log_vars) == 3
        for name in ["binding", "elution", "tcell_assay"]:
            assert isinstance(balancer.log_vars[name], nn.Parameter)

    def test_uncertainty_balancer_weights(self):
        balancer = UncertaintyBalancer(["task1", "task2"])
        with torch.no_grad():
            balancer.log_vars["task1"].fill_(0.0)
            balancer.log_vars["task2"].fill_(-1.0)
        weights = balancer.get_weights({"task1": torch.tensor(1.0), "task2": torch.tensor(1.0)})
        assert weights["task1"] == pytest.approx(1.0, rel=1e-5)
        assert weights["task2"] == pytest.approx(2.718, rel=0.01)

    def test_uncertainty_balancer_total_loss(self):
        balancer = UncertaintyBalancer(["task1", "task2"])
        with torch.no_grad():
            balancer.log_vars["task1"].fill_(0.0)
            balancer.log_vars["task2"].fill_(0.0)
        total = balancer.compute_total_loss(
            {"task1": torch.tensor(1.0), "task2": torch.tensor(2.0)}
        )
        assert total.item() == pytest.approx(3.0, rel=1e-5)


class TestTaskTracker:
    def test_tracker_init(self):
        tracker = TaskTracker(["task1", "task2", "task3"])
        assert tracker.steps == {"task1": 0, "task2": 0, "task3": 0}

    def test_tracker_update(self):
        tracker = TaskTracker(["task1", "task2"])
        tracker.update("task1", 0.5)
        tracker.update("task1", 0.4)
        tracker.update("task2", 0.6)
        assert tracker.steps["task1"] == 2
        assert tracker.losses["task1"] == [0.5, 0.4]

    def test_tracker_least_trained(self):
        tracker = TaskTracker(["task1", "task2", "task3"])
        tracker.update("task1", 0.5)
        tracker.update("task1", 0.5)
        tracker.update("task2", 0.5)
        least = tracker.get_least_trained(2)
        assert "task3" in least
        assert "task2" in least
        assert "task1" not in least

    def test_tracker_all_trained(self):
        tracker = TaskTracker(["task1", "task2"])
        assert not tracker.all_trained(min_steps=1)
        tracker.update("task1", 0.5)
        assert not tracker.all_trained(min_steps=1)
        tracker.update("task2", 0.5)
        assert tracker.all_trained(min_steps=1)
        assert not tracker.all_trained(min_steps=2)

    def test_tracker_avg_loss(self):
        tracker = TaskTracker(["task1"])
        for i in range(10):
            tracker.update("task1", float(i))
        assert tracker.get_avg_loss("task1") == pytest.approx(4.5, rel=1e-5)
        assert tracker.get_avg_loss("task1", window=5) == pytest.approx(7.0, rel=1e-5)

    def test_tracker_summary(self):
        tracker = TaskTracker(["task1", "task2"])
        tracker.update("task1", 0.5)
        summary = tracker.summary()
        assert summary["steps"]["task1"] == 1
        assert "avg_losses" in summary
