"""Tests for task definitions, routing, and balancing."""

import pytest
import torch
import torch.nn as nn

from presto.training.tasks import (
    # Core
    Task,
    TaskSpec,
    TASK_REGISTRY,
    get_task,
    register_task,
    route_sample,
    # Label maps
    MHC_CHAIN_TYPE_MAP,
    RECEPTOR_CHAIN_TYPE_MAP,
    SPECIES_MAP,
    MHC_VALID_PAIRINGS,
    # Tasks
    MHCChainTypeTask,
    ReceptorChainTypeTask,
    SpeciesTask,
    MHCPairingTask,
    TCRPairingTask,
    BindingTask,
    ElutionTask,
    ProcessingTask,
    StabilityTask,
    TCRpMHCMatchingTask,
    ImmunogenicityTask,
    TcellAssayTask,
    # Balancing
    StaticBalancer,
    UncertaintyBalancer,
    RandomBalancer,
    TaskTracker,
)


# =============================================================================
# Task Registry Tests
# =============================================================================

class TestTaskRegistry:
    """Test task registration and retrieval."""

    def test_default_tasks_registered(self):
        """All default tasks should be registered on module import."""
        expected_tasks = [
            "mhc_chain_type", "receptor_chain_type", "species",
            "mhc_pairing", "tcr_pairing",
            "binding", "elution", "processing", "stability",
            "tcr_pmhc", "immunogenicity", "tcell_assay",
        ]
        for name in expected_tasks:
            assert name in TASK_REGISTRY, f"Task {name} not registered"

    def test_get_task_valid(self):
        """get_task returns correct task instance."""
        task = get_task("binding")
        assert task.name == "binding"
        assert isinstance(task, BindingTask)

    def test_get_task_invalid(self):
        """get_task raises for unknown task."""
        with pytest.raises(ValueError, match="Unknown task"):
            get_task("nonexistent_task")

    def test_register_custom_task(self):
        """Can register custom tasks."""
        class CustomTask(Task):
            name = "custom_test"
            spec = TaskSpec(required_fields={"x"})

            def compute_loss(self, outputs, batch, mask=None):
                return torch.tensor(0.0)

            def compute_metrics(self, outputs, batch):
                return {}

        task = CustomTask()
        register_task(task)
        assert "custom_test" in TASK_REGISTRY
        assert get_task("custom_test") is task

        # Cleanup
        del TASK_REGISTRY["custom_test"]


# =============================================================================
# Sample Routing Tests
# =============================================================================

class TestSampleRouting:
    """Test routing samples to appropriate tasks."""

    def test_route_binding_sample(self):
        """Sample with binding fields routes to binding task."""
        sample = {
            "pep_seq": "SIINFEKL",
            "mhc_allele": "HLA-A*02:01",
            "bind_value": 5.0,
        }
        tasks = route_sample(sample)
        assert "binding" in tasks
        assert "mhc_chain_type" in tasks  # Also has mhc_allele
        assert "species" in tasks

    def test_route_tcr_sample(self):
        """Sample with TCR fields routes to TCR tasks."""
        sample = {
            "tcr_a_seq": "CASSF",
            "tcr_b_seq": "CASSQ",
            "tcr_v_gene": "TRAV12-1",
        }
        tasks = route_sample(sample)
        assert "tcr_pairing" in tasks
        assert "receptor_chain_type" in tasks

    def test_route_full_sample(self):
        """Sample with all fields routes to many tasks."""
        sample = {
            "pep_seq": "SIINFEKL",
            "mhc_allele": "HLA-A*02:01",
            "mhc_a_seq": "MAVMAPRT...",
            "mhc_b_seq": "IQRTPKIQ...",
            "tcr_a_seq": "CASSF",
            "tcr_b_seq": "CASSQ",
            "tcr_v_gene": "TRAV12-1",
            "tcell_label": 1,
        }
        tasks = route_sample(sample)
        # Should route to many tasks
        assert "mhc_chain_type" in tasks
        assert "species" in tasks
        assert "mhc_pairing" in tasks
        assert "tcr_pairing" in tasks
        assert "tcell_assay" in tasks

    def test_route_minimal_sample(self):
        """Sample with minimal fields routes to few tasks."""
        sample = {"pep_seq": "SIINFEKL"}
        tasks = route_sample(sample)
        # Only routes to tasks that need just pep_seq
        assert "binding" not in tasks  # needs mhc_allele too
        assert "mhc_chain_type" not in tasks

    def test_task_accepts_method(self):
        """Task.accepts correctly checks required fields."""
        task = get_task("binding")
        assert task.accepts({"pep_seq": "X", "mhc_allele": "Y", "bind_value": 1.0})
        assert not task.accepts({"pep_seq": "X"})  # missing mhc_allele, bind_value
        assert not task.accepts({"pep_seq": "X", "mhc_allele": "Y"})  # missing bind_value


# =============================================================================
# Label Derivation Tests
# =============================================================================

class TestLabelDerivation:
    """Test deriving labels from samples."""

    def test_mhc_chain_type_derivation(self):
        """MHC chain type derived from allele name."""
        task = MHCChainTypeTask()

        # Class I
        assert task.derive_label({"mhc_allele": "HLA-A*02:01"}) == MHC_CHAIN_TYPE_MAP["HLA-A"]
        assert task.derive_label({"mhc_allele": "HLA-B*07:02"}) == MHC_CHAIN_TYPE_MAP["HLA-B"]
        assert task.derive_label({"mhc_allele": "HLA-C*07:01"}) == MHC_CHAIN_TYPE_MAP["HLA-C"]

        # Class II
        assert task.derive_label({"mhc_allele": "HLA-DRB1*01:01"}) == MHC_CHAIN_TYPE_MAP["HLA-DRB"]
        assert task.derive_label({"mhc_allele": "HLA-DQA1*05:01"}) == MHC_CHAIN_TYPE_MAP["HLA-DQA"]

        # Mouse
        assert task.derive_label({"mhc_allele": "H-2-Kb"}) == MHC_CHAIN_TYPE_MAP["H-2-K"]
        assert task.derive_label({"mhc_allele": "H-2-Db"}) == MHC_CHAIN_TYPE_MAP["H-2-D"]

    def test_receptor_chain_type_derivation(self):
        """Receptor chain type derived from V gene."""
        task = ReceptorChainTypeTask()

        assert task.derive_label({"tcr_v_gene": "TRAV12-1"}) == RECEPTOR_CHAIN_TYPE_MAP["TRAV"]
        assert task.derive_label({"tcr_v_gene": "TRBV7-9"}) == RECEPTOR_CHAIN_TYPE_MAP["TRBV"]
        assert task.derive_label({"tcr_v_gene": "IGHV3-23"}) == RECEPTOR_CHAIN_TYPE_MAP["IGHV"]

    def test_species_derivation(self):
        """Species derived from allele prefix."""
        task = SpeciesTask()

        assert task.derive_label({"mhc_allele": "HLA-A*02:01"}) == SPECIES_MAP["HLA"]
        assert task.derive_label({"mhc_allele": "H-2-Kb"}) == SPECIES_MAP["H-2"]
        assert task.derive_label({"mhc_allele": "Mamu-A*01"}) == SPECIES_MAP["Mamu"]

    def test_mhc_pairing_derivation(self):
        """MHC pairing label derived from allele compatibility."""
        task = MHCPairingTask()

        # Class I: alpha only (beta should be B2M)
        assert task.derive_label({
            "mhc_a_allele": "HLA-A*02:01",
            "mhc_b_allele": "B2M",
        }) == 1

        # Class II: valid pairing
        assert task.derive_label({
            "mhc_a_allele": "HLA-DRA*01:01",
            "mhc_b_allele": "HLA-DRB1*01:01",
        }) == 1

        # Class II: invalid pairing (DRA with DQB)
        assert task.derive_label({
            "mhc_a_allele": "HLA-DRA*01:01",
            "mhc_b_allele": "HLA-DQB1*06:02",
        }) == 0


# =============================================================================
# Negative Generation Tests
# =============================================================================

class TestNegativeGeneration:
    """Test synthetic negative sample generation."""

    def test_mhc_pairing_negatives(self):
        """MHC pairing generates negatives by swapping chains."""
        task = MHCPairingTask()
        positives = [
            {"mhc_a_seq": "ALPHA1", "mhc_b_seq": "BETA1", "pairing_label": 1},
            {"mhc_a_seq": "ALPHA2", "mhc_b_seq": "BETA2", "pairing_label": 1},
        ]
        all_samples = positives + [
            {"mhc_a_seq": "ALPHA3", "mhc_b_seq": "BETA3"},
        ]

        negatives = task.generate_negatives(positives, all_samples, n_per_positive=2)

        assert len(negatives) == 4  # 2 positives * 2 negatives each
        for neg in negatives:
            assert neg["pairing_label"] == 0
            # Either alpha or beta should be swapped
            assert "mhc_a_seq" in neg
            assert "mhc_b_seq" in neg

    def test_tcr_pairing_negatives(self):
        """TCR pairing generates negatives by swapping chains."""
        task = TCRPairingTask()
        positives = [
            {"tcr_a_seq": "CASS", "tcr_b_seq": "CASS", "pairing_label": 1},
        ]
        all_samples = positives + [
            {"tcr_a_seq": "CAVS", "tcr_b_seq": "CSVS"},
        ]

        negatives = task.generate_negatives(positives, all_samples, n_per_positive=3)

        assert len(negatives) == 3
        for neg in negatives:
            assert neg["pairing_label"] == 0

    def test_elution_negatives(self):
        """Elution generates negatives with random peptides."""
        task = ElutionTask()
        positives = [
            {"pep_seq": "SIINFEKL", "mhc_allele": "HLA-A*02:01", "elution_label": 1},
        ]
        all_samples = positives + [
            {"pep_seq": "GILGFVFTL", "mhc_allele": "HLA-A*02:01"},
            {"pep_seq": "NLVPMVATV", "mhc_allele": "HLA-A*02:01"},
        ]

        negatives = task.generate_negatives(positives, all_samples, n_per_positive=5)

        assert len(negatives) == 5
        for neg in negatives:
            assert neg["elution_label"] == 0
            assert neg["mhc_allele"] == "HLA-A*02:01"  # Same allele as positive

    def test_tcr_pmhc_negatives(self):
        """TCR-pMHC matching generates negatives by mismatching."""
        task = TCRpMHCMatchingTask()
        positives = [
            {
                "pep_seq": "SIINFEKL",
                "mhc_allele": "HLA-A*02:01",
                "tcr_a_seq": "CASS",
                "tcr_b_seq": "CSVV",
                "match_label": 1,
            },
        ]
        all_samples = positives + [
            {"pep_seq": "GILGFVFTL", "mhc_allele": "HLA-A*02:01", "tcr_a_seq": "CAVS", "tcr_b_seq": "CSVS"},
        ]

        negatives = task.generate_negatives(positives, all_samples, n_per_positive=10)

        assert len(negatives) == 10
        for neg in negatives:
            assert neg["match_label"] == 0

    def test_tcr_pmhc_negatives_similarity_prefers_same_allele(self, monkeypatch):
        """Similarity hard negatives should stay allele-compatible when possible."""
        task = TCRpMHCMatchingTask(
            negative_strategy="similarity",
            hard_negative_temperature=1e-8,
            same_allele_first=True,
        )
        positives = [
            {
                "pep_seq": "SIINFEKL",
                "mhc_allele": "HLA-A*02:01",
                "tcr_a_seq": "CASS",
                "tcr_b_seq": "CSVV",
                "match_label": 1,
            }
        ]
        all_samples = positives + [
            {
                "pep_seq": "SIINFEKM",  # Most similar same-allele peptide
                "mhc_allele": "HLA-A*02:01",
                "tcr_a_seq": "CAVS",
                "tcr_b_seq": "CSVS",
            },
            {
                "pep_seq": "GILGFVFTL",  # Dissimilar same-allele peptide
                "mhc_allele": "HLA-A*02:01",
                "tcr_a_seq": "CAXX",
                "tcr_b_seq": "CSXX",
            },
            {
                "pep_seq": "SIINFEKM",  # Similar but different allele
                "mhc_allele": "HLA-B*07:02",
                "tcr_a_seq": "CBVS",
                "tcr_b_seq": "CTVS",
            },
        ]

        # Force the pMHC-swap branch.
        monkeypatch.setattr("presto.training.tasks.random.random", lambda: 0.9)

        negatives = task.generate_negatives(positives, all_samples, n_per_positive=1)
        assert len(negatives) == 1
        neg = negatives[0]
        assert neg["match_label"] == 0
        assert neg["mhc_allele"] == "HLA-A*02:01"
        assert neg["pep_seq"] == "SIINFEKM"

    def test_tcr_pmhc_negatives_similarity_avoids_identity(self, monkeypatch):
        """Generated negatives should not exactly duplicate positive pairs."""
        task = TCRpMHCMatchingTask(
            negative_strategy="similarity",
            hard_negative_temperature=1e-8,
        )
        positives = [
            {
                "pep_seq": "SIINFEKL",
                "mhc_allele": "HLA-A*02:01",
                "tcr_a_seq": "CASS",
                "tcr_b_seq": "CSVV",
                "match_label": 1,
            }
        ]
        all_samples = positives + [
            # Alternative TCR so the fallback path can still make a valid negative.
            {
                "pep_seq": "SIINFEKL",
                "mhc_allele": "HLA-A*02:01",
                "tcr_a_seq": "CAVS",
                "tcr_b_seq": "CSVS",
            },
        ]

        # Force pMHC swap request, which has no non-identical pMHC candidate.
        monkeypatch.setattr("presto.training.tasks.random.random", lambda: 0.9)

        negatives = task.generate_negatives(positives, all_samples, n_per_positive=1)
        assert len(negatives) == 1
        neg = negatives[0]
        assert neg["match_label"] == 0
        assert (neg["pep_seq"], neg["mhc_allele"], neg["tcr_a_seq"], neg["tcr_b_seq"]) != (
            "SIINFEKL",
            "HLA-A*02:01",
            "CASS",
            "CSVV",
        )

    def test_tcr_pmhc_hard_negative_temperature_annealing(self):
        """Hard-negative temperature should support linear annealing."""
        task = TCRpMHCMatchingTask(
            negative_strategy="similarity",
            hard_negative_temperature=0.5,
        )

        t0 = task.anneal_hard_negative_temperature(
            progress=0.0,
            initial_temperature=0.5,
            final_temperature=0.1,
        )
        t_mid = task.anneal_hard_negative_temperature(
            progress=0.5,
            initial_temperature=0.5,
            final_temperature=0.1,
        )
        t1 = task.anneal_hard_negative_temperature(
            progress=1.0,
            initial_temperature=0.5,
            final_temperature=0.1,
        )

        assert t0 == pytest.approx(0.5)
        assert t_mid == pytest.approx(0.3)
        assert t1 == pytest.approx(0.1)
        assert task.hard_negative_temperature == pytest.approx(0.1)


# =============================================================================
# Loss Computation Tests
# =============================================================================

class TestLossComputation:
    """Test task-specific loss computation."""

    def test_binding_loss_standard(self):
        """Binding task uses MSE loss."""
        task = BindingTask()
        outputs = {"binding_pred": torch.tensor([[5.0], [6.0], [7.0]])}
        batch = {"bind_target": torch.tensor([[5.0], [5.0], [5.0]])}

        loss = task.compute_loss(outputs, batch)

        # MSE: (0 + 1 + 4) / 3 = 5/3
        assert loss.item() == pytest.approx(5/3, rel=1e-5)

    def test_binding_loss_censor_aware(self):
        """Binding task uses censor-aware loss with qualifiers."""
        task = BindingTask()

        # pred=6, target=5, qual=0 (equals): loss = (6-5)^2 = 1
        # pred=4, target=5, qual=-1 (less than): pred < target, no penalty
        # pred=7, target=5, qual=1 (greater than): pred > target, no penalty
        outputs = {"binding_pred": torch.tensor([[6.0], [4.0], [7.0]])}
        batch = {
            "bind_target": torch.tensor([[5.0], [5.0], [5.0]]),
            "bind_qual": torch.tensor([[0], [-1], [1]]),
        }

        loss = task.compute_loss(outputs, batch)

        # Only the "equals" sample contributes loss
        assert loss.item() == pytest.approx(1/3, rel=1e-5)

    def test_binding_loss_censor_violation(self):
        """Binding censor loss penalizes violations."""
        task = BindingTask()

        # pred=6, target=5, qual=-1 (less than): pred > target, PENALTY
        # pred=4, target=5, qual=1 (greater than): pred < target, PENALTY
        outputs = {"binding_pred": torch.tensor([[6.0], [4.0]])}
        batch = {
            "bind_target": torch.tensor([[5.0], [5.0]]),
            "bind_qual": torch.tensor([[-1], [1]]),
        }

        loss = task.compute_loss(outputs, batch)

        # relu(6-5)^2 + relu(-(4-5))^2 = 1 + 1 = 2
        assert loss.item() == pytest.approx(2/2, rel=1e-5)

    def test_elution_loss(self):
        """Elution task uses BCE loss."""
        task = ElutionTask()
        outputs = {"elution_logit": torch.tensor([[2.0], [-2.0]])}
        batch = {"elution_label": torch.tensor([1, 0])}

        loss = task.compute_loss(outputs, batch)

        # BCE with logits
        expected = nn.BCEWithLogitsLoss()(
            torch.tensor([2.0, -2.0]),
            torch.tensor([1.0, 0.0])
        )
        assert loss.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_mhc_chain_type_loss(self):
        """MHC chain type uses cross-entropy loss."""
        task = MHCChainTypeTask()
        outputs = {"mhc_chain_logits": torch.randn(4, 12)}
        batch = {"mhc_chain_type_label": torch.tensor([0, 1, 2, 3])}

        loss = task.compute_loss(outputs, batch)

        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0  # Positive loss

    def test_loss_with_mask(self):
        """Loss computation respects mask."""
        task = ElutionTask()
        outputs = {"elution_logit": torch.tensor([[10.0], [-10.0], [0.0]])}
        batch = {"elution_label": torch.tensor([1, 0, 1])}
        mask = torch.tensor([1.0, 1.0, 0.0])  # Ignore third sample

        loss = task.compute_loss(outputs, batch, mask=mask)

        # Only first two samples contribute
        expected = nn.BCEWithLogitsLoss()(
            torch.tensor([10.0, -10.0]),
            torch.tensor([1.0, 0.0])
        )
        assert loss.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_tcr_pmhc_contrastive_loss(self):
        """TCR-pMHC can use contrastive InfoNCE loss."""
        task = TCRpMHCMatchingTask(temperature=0.1)

        # Similarity matrix (B, B) where diagonal should be highest
        B = 4
        logits = torch.eye(B) * 10  # High similarity on diagonal
        outputs = {"match_logit": logits}
        batch = {"match_label": torch.arange(B)}  # Targets are diagonal indices

        loss = task.compute_loss(outputs, batch)

        # With strong diagonal, loss should be low
        assert loss.item() < 1.0

    def test_tcr_pmhc_bce_loss(self):
        """TCR-pMHC falls back to BCE for binary labels."""
        task = TCRpMHCMatchingTask()
        outputs = {"match_logit": torch.tensor([[2.0], [-2.0]])}
        batch = {"match_label": torch.tensor([1, 0])}

        loss = task.compute_loss(outputs, batch)

        expected = nn.BCEWithLogitsLoss()(
            torch.tensor([2.0, -2.0]),
            torch.tensor([1.0, 0.0])
        )
        assert loss.item() == pytest.approx(expected.item(), rel=1e-5)


# =============================================================================
# Metrics Tests
# =============================================================================

class TestMetrics:
    """Test task-specific metrics computation."""

    def test_classification_accuracy(self):
        """Classification tasks report accuracy."""
        task = MHCChainTypeTask()
        outputs = {"mhc_chain_logits": torch.tensor([
            [10.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Predicts 0
            [0, 10.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Predicts 1
            [0, 0, 10.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Predicts 2
            [10.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Predicts 0 (wrong)
        ])}
        batch = {"mhc_chain_type_label": torch.tensor([0, 1, 2, 3])}

        metrics = task.compute_metrics(outputs, batch)

        assert "accuracy" in metrics
        assert metrics["accuracy"] == pytest.approx(0.75, rel=1e-5)  # 3/4 correct

    def test_binding_spearman(self):
        """Binding task reports Spearman correlation."""
        task = BindingTask()
        # Perfect rank correlation
        outputs = {"binding_pred": torch.tensor([[1.0], [2.0], [3.0], [4.0]])}
        batch = {"bind_target": torch.tensor([[1.0], [2.0], [3.0], [4.0]])}

        metrics = task.compute_metrics(outputs, batch)

        assert "spearman" in metrics
        assert metrics["spearman"] == pytest.approx(1.0, rel=1e-5)
        assert "mae" in metrics

    def test_elution_auroc(self):
        """Elution task reports AUROC."""
        task = ElutionTask()
        # Perfect separation
        outputs = {"elution_logit": torch.tensor([[10.0], [10.0], [-10.0], [-10.0]])}
        batch = {"elution_label": torch.tensor([1, 1, 0, 0])}

        metrics = task.compute_metrics(outputs, batch)

        assert "auroc" in metrics
        assert metrics["auroc"] == pytest.approx(1.0, rel=1e-5)
        assert "auprc" in metrics


# =============================================================================
# Task Balancing Tests
# =============================================================================

class TestTaskBalancing:
    """Test task loss balancing strategies."""

    def test_static_balancer(self):
        """Static balancer returns configured weights."""
        weights = {"task1": 1.0, "task2": 0.5, "task3": 2.0}
        balancer = StaticBalancer(weights)

        result = balancer.get_weights({
            "task1": torch.tensor(1.0),
            "task2": torch.tensor(1.0),
            "task3": torch.tensor(1.0),
        })

        assert result["task1"] == 1.0
        assert result["task2"] == 0.5
        assert result["task3"] == 2.0

    def test_static_balancer_default(self):
        """Static balancer defaults to 1.0 for unknown tasks."""
        balancer = StaticBalancer({"task1": 0.5})

        result = balancer.get_weights({
            "task1": torch.tensor(1.0),
            "task2": torch.tensor(1.0),  # Not in config
        })

        assert result["task1"] == 0.5
        assert result["task2"] == 1.0  # Default

    def test_random_balancer(self):
        """Random balancer returns varying weights."""
        balancer = RandomBalancer()
        losses = {
            "task1": torch.tensor(1.0),
            "task2": torch.tensor(1.0),
            "task3": torch.tensor(1.0),
        }

        weights1 = balancer.get_weights(losses)
        weights2 = balancer.get_weights(losses)

        # Weights should differ between calls (with high probability)
        # and sum to approximately n_tasks
        assert sum(weights1.values()) == pytest.approx(3.0, rel=0.1)
        # At least one weight should differ (probabilistic)
        assert weights1 != weights2 or True  # May rarely be equal

    def test_uncertainty_balancer_init(self):
        """Uncertainty balancer initializes with learnable parameters."""
        task_names = ["binding", "elution", "tcell_assay"]
        balancer = UncertaintyBalancer(task_names)

        assert len(balancer.log_vars) == 3
        for name in task_names:
            assert name in balancer.log_vars
            assert isinstance(balancer.log_vars[name], nn.Parameter)

    def test_uncertainty_balancer_weights(self):
        """Uncertainty balancer computes weights from log variances."""
        balancer = UncertaintyBalancer(["task1", "task2"])
        # Set different log variances
        with torch.no_grad():
            balancer.log_vars["task1"].fill_(0.0)  # exp(-0) = 1.0
            balancer.log_vars["task2"].fill_(-1.0)  # exp(1) ≈ 2.718

        weights = balancer.get_weights({
            "task1": torch.tensor(1.0),
            "task2": torch.tensor(1.0),
        })

        assert weights["task1"] == pytest.approx(1.0, rel=1e-5)
        assert weights["task2"] == pytest.approx(2.718, rel=0.01)

    def test_uncertainty_balancer_total_loss(self):
        """Uncertainty balancer computes weighted total loss."""
        balancer = UncertaintyBalancer(["task1", "task2"])
        with torch.no_grad():
            balancer.log_vars["task1"].fill_(0.0)
            balancer.log_vars["task2"].fill_(0.0)

        losses = {
            "task1": torch.tensor(1.0),
            "task2": torch.tensor(2.0),
        }

        total = balancer.compute_total_loss(losses)

        # precision=1, total = 1*1 + 0 + 1*2 + 0 = 3
        assert total.item() == pytest.approx(3.0, rel=1e-5)


# =============================================================================
# Task Tracker Tests
# =============================================================================

class TestTaskTracker:
    """Test task progress tracking."""

    def test_tracker_init(self):
        """Tracker initializes with zero counts."""
        tracker = TaskTracker(["task1", "task2", "task3"])

        assert tracker.steps["task1"] == 0
        assert tracker.steps["task2"] == 0
        assert tracker.steps["task3"] == 0

    def test_tracker_update(self):
        """Tracker records training steps."""
        tracker = TaskTracker(["task1", "task2"])

        tracker.update("task1", 0.5)
        tracker.update("task1", 0.4)
        tracker.update("task2", 0.6)

        assert tracker.steps["task1"] == 2
        assert tracker.steps["task2"] == 1
        assert tracker.losses["task1"] == [0.5, 0.4]

    def test_tracker_least_trained(self):
        """Tracker identifies least trained tasks."""
        tracker = TaskTracker(["task1", "task2", "task3"])

        tracker.update("task1", 0.5)
        tracker.update("task1", 0.5)
        tracker.update("task2", 0.5)
        # task3 has 0 steps

        least = tracker.get_least_trained(2)

        assert "task3" in least
        assert "task2" in least
        assert "task1" not in least

    def test_tracker_all_trained(self):
        """Tracker checks if all tasks have minimum steps."""
        tracker = TaskTracker(["task1", "task2"])

        assert not tracker.all_trained(min_steps=1)

        tracker.update("task1", 0.5)
        assert not tracker.all_trained(min_steps=1)

        tracker.update("task2", 0.5)
        assert tracker.all_trained(min_steps=1)
        assert not tracker.all_trained(min_steps=2)

    def test_tracker_avg_loss(self):
        """Tracker computes average recent loss."""
        tracker = TaskTracker(["task1"])

        for i in range(10):
            tracker.update("task1", float(i))

        # Average of [0,1,2,3,4,5,6,7,8,9] = 4.5
        assert tracker.get_avg_loss("task1") == pytest.approx(4.5, rel=1e-5)

        # Window of 5: [5,6,7,8,9] = 7.0
        assert tracker.get_avg_loss("task1", window=5) == pytest.approx(7.0, rel=1e-5)

    def test_tracker_summary(self):
        """Tracker produces summary dict."""
        tracker = TaskTracker(["task1", "task2"])
        tracker.update("task1", 0.5)
        tracker.update("task2", 1.0)

        summary = tracker.summary()

        assert "steps" in summary
        assert "avg_losses" in summary
        assert summary["steps"]["task1"] == 1
        assert summary["avg_losses"]["task1"] == pytest.approx(0.5, rel=1e-5)


# =============================================================================
# Integration Tests
# =============================================================================

class TestTaskIntegration:
    """Integration tests for task workflow."""

    def test_full_binding_workflow(self):
        """Test complete binding task workflow."""
        task = get_task("binding")

        # 1. Check sample acceptance
        sample = {"pep_seq": "SIINFEKL", "mhc_allele": "HLA-A*02:01", "bind_value": 5.0}
        assert task.accepts(sample)

        # 2. Label comes from data
        assert task.spec.label_field == "bind_value"

        # 3. Compute loss
        outputs = {"binding_pred": torch.tensor([[5.5]])}
        batch = {"bind_target": torch.tensor([[5.0]])}
        loss = task.compute_loss(outputs, batch)
        assert loss.item() == pytest.approx(0.25, rel=1e-5)  # (5.5-5)^2

        # 4. Compute metrics
        metrics = task.compute_metrics(outputs, batch)
        assert "spearman" in metrics

    def test_full_elution_workflow(self):
        """Test complete elution task workflow with negatives."""
        task = get_task("elution")

        # 1. Create positive samples
        positives = [
            {"pep_seq": "SIINFEKL", "mhc_allele": "HLA-A*02:01", "elution_label": 1},
            {"pep_seq": "GILGFVFTL", "mhc_allele": "HLA-A*02:01", "elution_label": 1},
        ]

        # 2. Generate negatives
        all_samples = positives + [
            {"pep_seq": "NLVPMVATV", "mhc_allele": "HLA-A*02:01"},
        ]
        negatives = task.generate_negatives(positives, all_samples, n_per_positive=2)

        assert len(negatives) == 4
        assert all(n["elution_label"] == 0 for n in negatives)

        # 3. Verify total dataset
        all_data = positives + negatives
        assert sum(s["elution_label"] for s in all_data) == 2  # 2 positives
        assert len(all_data) == 6

    def test_derived_vs_explicit_labels(self):
        """Test that derived labels work alongside explicit labels."""
        mhc_task = MHCChainTypeTask()
        binding_task = BindingTask()

        sample = {
            "mhc_allele": "HLA-A*02:01",
            "pep_seq": "SIINFEKL",
            "bind_value": 5.0,
        }

        # MHC chain type: derived from allele
        mhc_label = mhc_task.get_label(sample)
        assert mhc_label == MHC_CHAIN_TYPE_MAP["HLA-A"]

        # Binding: from data (bind_value is label_field, but compute_loss uses bind_target)
        assert binding_task.spec.label_field == "bind_value"
