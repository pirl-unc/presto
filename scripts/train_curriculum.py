#!/usr/bin/env python
"""Curriculum learning training script for PRESTO.

Implements staged training following the biological hierarchy:
1. Chain classification (independent, abundant data)
2. Pairing prediction (MHC chain pairing, TCR chain pairing)
3. pMHC binding (IEDB data, processing, stability)
4. pMHC:TCR matching (VDJdb, IEDB T-cell)

Usage:
    python -m presto.scripts.train_curriculum --config curriculum.yaml
    python -m presto.scripts.train_curriculum --stage CHAIN_CLASSIFICATION
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from presto.models import (
    SequenceEncoder,
    MHCChainClassifier,
    ChainAttributeClassifier,
    MHCPairingPredictor,
    TCRPairingPredictor,
    HardNegativeMiner,
    Presto,
)
from presto.training import (
    CurriculumStage,
    CurriculumScheduler,
    StageConfig,
    DEFAULT_CURRICULUM,
    sample_hard_negatives,
    get_batch_config,
)
from presto.training.losses import UncertaintyWeighting
from presto.data.vocab import (
    AA_TO_IDX,
    SPECIES_TO_IDX,
    MHC_CHAIN_TO_IDX,
    CHAIN_TO_IDX,
    CELL_TO_IDX,
)


# =============================================================================
# Synthetic Data Generation for Curriculum
# =============================================================================

def generate_random_sequence(length: int) -> str:
    """Generate a random amino acid sequence."""
    aas = "ACDEFGHIKLMNPQRSTVWY"
    return "".join(random.choice(aas) for _ in range(length))


def tokenize(seq: str) -> torch.Tensor:
    """Tokenize a sequence."""
    tokens = [AA_TO_IDX.get(aa, AA_TO_IDX["<UNK>"]) for aa in seq]
    return torch.tensor(tokens, dtype=torch.long)


@dataclass
class ChainClassificationSample:
    """Sample for chain classification task."""
    sequence: str
    species: str
    chain_type: str
    is_mhc: bool
    is_pseudo: bool = False
    phenotype: Optional[str] = None  # For TCR/BCR chains


@dataclass
class PairingSample:
    """Sample for pairing prediction task."""
    seq_a: str
    seq_b: str
    chain_type_a: str
    chain_type_b: str
    is_valid_pair: bool
    species: str


def generate_chain_classification_data(
    n_samples: int,
    include_mhc: bool = True,
    include_receptor: bool = True,
) -> List[ChainClassificationSample]:
    """Generate synthetic chain classification samples."""
    samples = []
    species_list = ["human", "mouse"]

    # MHC chain types and typical lengths
    mhc_types = {
        "MHC_I_ALPHA": (260, 290),
        "B2M": (95, 105),
        "MHC_II_ALPHA": (180, 220),
        "MHC_II_BETA": (200, 240),
    }

    # Receptor chain types and typical CDR3 lengths
    receptor_types = {
        "TRA": (100, 130),
        "TRB": (100, 130),
        "TRG": (90, 120),
        "TRD": (90, 120),
        "IGH": (110, 150),
        "IGK": (100, 120),
        "IGL": (100, 120),
    }

    # Phenotype mapping
    phenotypes = {
        "TRA": "ab_T", "TRB": "ab_T",
        "TRG": "gd_T", "TRD": "gd_T",
        "IGH": "B_cell", "IGK": "B_cell", "IGL": "B_cell",
    }

    for _ in range(n_samples):
        species = random.choice(species_list)

        if include_mhc and include_receptor:
            is_mhc = random.random() < 0.5
        elif include_mhc:
            is_mhc = True
        else:
            is_mhc = False

        if is_mhc:
            chain_type = random.choice(list(mhc_types.keys()))
            len_range = mhc_types[chain_type]
            length = random.randint(*len_range)
            is_pseudo = random.random() < 0.1  # 10% pseudosequences
            if is_pseudo:
                length = 34  # NetMHCpan pseudosequence length
                chain_type = chain_type + "_PSEUDO" if not chain_type.endswith("_PSEUDO") else chain_type

            samples.append(ChainClassificationSample(
                sequence=generate_random_sequence(length),
                species=species,
                chain_type=chain_type,
                is_mhc=True,
                is_pseudo=is_pseudo,
            ))
        else:
            chain_type = random.choice(list(receptor_types.keys()))
            len_range = receptor_types[chain_type]
            length = random.randint(*len_range)
            phenotype = phenotypes[chain_type]

            samples.append(ChainClassificationSample(
                sequence=generate_random_sequence(length),
                species=species,
                chain_type=chain_type,
                is_mhc=False,
                phenotype=phenotype,
            ))

    return samples


def generate_pairing_data(n_samples: int) -> List[PairingSample]:
    """Generate synthetic pairing samples (positive and negative)."""
    samples = []
    species_list = ["human", "mouse"]

    # Valid MHC pairs
    valid_mhc_pairs = [
        ("MHC_I_ALPHA", "B2M"),
        ("MHC_II_ALPHA", "MHC_II_BETA"),
    ]

    # Valid TCR pairs
    valid_tcr_pairs = [
        ("TRA", "TRB"),
        ("TRG", "TRD"),
    ]

    # Valid BCR pairs
    valid_bcr_pairs = [
        ("IGH", "IGK"),
        ("IGH", "IGL"),
    ]

    all_valid_pairs = valid_mhc_pairs + valid_tcr_pairs + valid_bcr_pairs

    # Chain lengths
    chain_lengths = {
        "MHC_I_ALPHA": 275, "B2M": 100,
        "MHC_II_ALPHA": 200, "MHC_II_BETA": 220,
        "TRA": 115, "TRB": 115, "TRG": 105, "TRD": 105,
        "IGH": 130, "IGK": 110, "IGL": 110,
    }

    for _ in range(n_samples):
        species = random.choice(species_list)
        is_positive = random.random() < 0.5

        if is_positive:
            # Valid pair
            pair = random.choice(all_valid_pairs)
            chain_type_a, chain_type_b = pair
        else:
            # Invalid pair (cross-type or cross-family)
            all_chains = list(chain_lengths.keys())
            chain_type_a = random.choice(all_chains)
            chain_type_b = random.choice(all_chains)
            # Make sure it's actually invalid
            while (chain_type_a, chain_type_b) in all_valid_pairs or (chain_type_b, chain_type_a) in all_valid_pairs:
                chain_type_b = random.choice(all_chains)

        samples.append(PairingSample(
            seq_a=generate_random_sequence(chain_lengths.get(chain_type_a, 100)),
            seq_b=generate_random_sequence(chain_lengths.get(chain_type_b, 100)),
            chain_type_a=chain_type_a,
            chain_type_b=chain_type_b,
            is_valid_pair=is_positive,
            species=species,
        ))

    return samples


# =============================================================================
# Datasets for Each Stage
# =============================================================================

class ChainClassificationDataset(Dataset):
    """Dataset for chain classification task."""

    def __init__(self, samples: List[ChainClassificationSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = tokenize(sample.sequence)

        return {
            "tokens": tokens,
            "species": SPECIES_TO_IDX.get(sample.species, 0),
            "chain_type": MHC_CHAIN_TO_IDX.get(sample.chain_type, 0) if sample.is_mhc else CHAIN_TO_IDX.get(sample.chain_type, 0),
            "is_mhc": sample.is_mhc,
            "is_pseudo": sample.is_pseudo,
            "phenotype": CELL_TO_IDX.get(sample.phenotype, 0) if sample.phenotype else 0,
        }


class PairingDataset(Dataset):
    """Dataset for pairing prediction task."""

    def __init__(self, samples: List[PairingSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens_a = tokenize(sample.seq_a)
        tokens_b = tokenize(sample.seq_b)

        # Determine if MHC or receptor pairing
        mhc_types = {"MHC_I_ALPHA", "B2M", "MHC_II_ALPHA", "MHC_II_BETA"}
        is_mhc = sample.chain_type_a in mhc_types

        if is_mhc:
            chain_type_a = MHC_CHAIN_TO_IDX.get(sample.chain_type_a, 0)
            chain_type_b = MHC_CHAIN_TO_IDX.get(sample.chain_type_b, 0)
        else:
            chain_type_a = CHAIN_TO_IDX.get(sample.chain_type_a, 0)
            chain_type_b = CHAIN_TO_IDX.get(sample.chain_type_b, 0)

        return {
            "tokens_a": tokens_a,
            "tokens_b": tokens_b,
            "chain_type_a": chain_type_a,
            "chain_type_b": chain_type_b,
            "is_valid": float(sample.is_valid_pair),
            "species": SPECIES_TO_IDX.get(sample.species, 0),
            "is_mhc": is_mhc,
        }


def collate_chain_classification(batch):
    """Collate chain classification samples."""
    # Pad sequences
    max_len = max(item["tokens"].shape[0] for item in batch)
    tokens = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, item in enumerate(batch):
        tokens[i, :item["tokens"].shape[0]] = item["tokens"]

    return {
        "tokens": tokens,
        "species": torch.tensor([item["species"] for item in batch]),
        "chain_type": torch.tensor([item["chain_type"] for item in batch]),
        "is_mhc": torch.tensor([item["is_mhc"] for item in batch]),
        "is_pseudo": torch.tensor([item["is_pseudo"] for item in batch]),
        "phenotype": torch.tensor([item["phenotype"] for item in batch]),
    }


def collate_pairing(batch):
    """Collate pairing samples."""
    # Pad sequences
    max_len_a = max(item["tokens_a"].shape[0] for item in batch)
    max_len_b = max(item["tokens_b"].shape[0] for item in batch)

    tokens_a = torch.zeros(len(batch), max_len_a, dtype=torch.long)
    tokens_b = torch.zeros(len(batch), max_len_b, dtype=torch.long)

    for i, item in enumerate(batch):
        tokens_a[i, :item["tokens_a"].shape[0]] = item["tokens_a"]
        tokens_b[i, :item["tokens_b"].shape[0]] = item["tokens_b"]

    return {
        "tokens_a": tokens_a,
        "tokens_b": tokens_b,
        "chain_type_a": torch.tensor([item["chain_type_a"] for item in batch]),
        "chain_type_b": torch.tensor([item["chain_type_b"] for item in batch]),
        "is_valid": torch.tensor([item["is_valid"] for item in batch]),
        "species": torch.tensor([item["species"] for item in batch]),
        "is_mhc": torch.tensor([item["is_mhc"] for item in batch]),
    }


# =============================================================================
# Stage-Specific Training
# =============================================================================

class CurriculumTrainer:
    """Trainer that implements curriculum learning."""

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        device: str = "cpu",
        curriculum: List[StageConfig] = None,
    ):
        self.d_model = d_model
        self.device = device
        self.curriculum = curriculum or DEFAULT_CURRICULUM
        self.scheduler = CurriculumScheduler(self.curriculum)

        # Create models for each stage
        self.mhc_chain_classifier = MHCChainClassifier(
            d_model=d_model, n_layers=n_layers // 2, n_heads=n_heads // 2
        ).to(device)

        self.receptor_chain_classifier = ChainAttributeClassifier(
            d_model=d_model, n_layers=n_layers // 2, n_heads=n_heads // 2
        ).to(device)

        self.mhc_pairing_predictor = MHCPairingPredictor(d_model=d_model).to(device)
        self.tcr_pairing_predictor = TCRPairingPredictor(d_model=d_model).to(device)

        # Full model (initialized later, uses encoders from classifiers)
        self.presto = None

        # Hard negative miner
        self.hard_neg_miner = HardNegativeMiner(temperature=0.1)

        # Optimizers (will be created per stage)
        self.optimizer = None
        self.uncertainty_weighting = None

    def _get_stage_parameters(self) -> List[nn.Parameter]:
        """Get parameters to optimize for current stage."""
        stage = self.scheduler.current_stage.stage

        if stage == CurriculumStage.CHAIN_CLASSIFICATION:
            params = list(self.mhc_chain_classifier.parameters()) + \
                     list(self.receptor_chain_classifier.parameters())

        elif stage == CurriculumStage.PAIRING:
            params = list(self.mhc_pairing_predictor.parameters()) + \
                     list(self.tcr_pairing_predictor.parameters())
            # Also keep training classifiers to prevent forgetting
            params += list(self.mhc_chain_classifier.parameters())
            params += list(self.receptor_chain_classifier.parameters())

        elif stage == CurriculumStage.PMHC_BINDING:
            if self.presto is None:
                self._init_full_model()
            params = list(self.presto.parameters())

        else:  # FULL
            if self.presto is None:
                self._init_full_model()
            params = list(self.presto.parameters())

        return params

    def _init_full_model(self):
        """Initialize full Presto model, transferring learned weights."""
        self.presto = Presto(
            d_model=self.d_model,
            n_layers=4,
            n_heads=8,
        ).to(self.device)

        # Transfer encoder weights from chain classifiers
        # (In practice, you'd want to share encoders more directly)

    def _setup_optimizer(self):
        """Setup optimizer for current stage."""
        params = self._get_stage_parameters()
        lr = self.scheduler.get_learning_rate()
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

        # Uncertainty weighting for multi-task
        n_tasks = len(self.scheduler.get_active_tasks())
        self.uncertainty_weighting = UncertaintyWeighting(n_tasks=n_tasks).to(self.device)
        self.optimizer.add_param_group({"params": self.uncertainty_weighting.parameters()})

    def train_stage_1(self, dataloader: DataLoader) -> float:
        """Train chain classification (Stage 1)."""
        self.mhc_chain_classifier.train()
        self.receptor_chain_classifier.train()

        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()

            losses = []

            # MHC samples
            mhc_mask = batch["is_mhc"]
            if mhc_mask.any():
                mhc_tokens = batch["tokens"][mhc_mask]
                mhc_logits = self.mhc_chain_classifier(mhc_tokens)

                # Species loss
                species_loss = F.cross_entropy(
                    mhc_logits["species_logits"],
                    batch["species"][mhc_mask]
                )
                # Chain type loss
                chain_loss = F.cross_entropy(
                    mhc_logits["chain_logits"],
                    batch["chain_type"][mhc_mask]
                )
                # Pseudo detection loss
                pseudo_loss = F.binary_cross_entropy_with_logits(
                    mhc_logits["pseudo_logit"].squeeze(-1),
                    batch["is_pseudo"][mhc_mask].float()
                )
                losses.extend([species_loss, chain_loss, pseudo_loss])

            # Receptor samples
            receptor_mask = ~batch["is_mhc"]
            if receptor_mask.any():
                receptor_tokens = batch["tokens"][receptor_mask]
                receptor_logits = self.receptor_chain_classifier(receptor_tokens)

                # Species loss
                species_loss = F.cross_entropy(
                    receptor_logits["species_logits"],
                    batch["species"][receptor_mask]
                )
                # Chain type loss
                chain_loss = F.cross_entropy(
                    receptor_logits["chain_logits"],
                    batch["chain_type"][receptor_mask]
                )
                # Phenotype loss
                phenotype_loss = F.cross_entropy(
                    receptor_logits["phenotype_logits"],
                    batch["phenotype"][receptor_mask]
                )
                losses.extend([species_loss, chain_loss, phenotype_loss])

            if losses:
                # Simple averaging - uncertainty weighting can be added later
                loss = sum(losses) / len(losses)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._get_stage_parameters(), max_norm=1.0
                )
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def train_stage_2(self, dataloader: DataLoader) -> float:
        """Train pairing prediction (Stage 2)."""
        self.mhc_pairing_predictor.train()
        self.tcr_pairing_predictor.train()
        self.mhc_chain_classifier.encoder.eval()  # Use frozen encoder
        self.receptor_chain_classifier.encoder.eval()

        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()

            losses = []

            # MHC pairing samples
            mhc_mask = batch["is_mhc"]
            if mhc_mask.any():
                tokens_a = batch["tokens_a"][mhc_mask]
                tokens_b = batch["tokens_b"][mhc_mask]

                # Get embeddings from classifier encoder
                with torch.no_grad():
                    z_a, _ = self.mhc_chain_classifier.encoder(tokens_a)
                    z_b, _ = self.mhc_chain_classifier.encoder(tokens_b)

                # Predict pairing
                pairing_logit = self.mhc_pairing_predictor(
                    z_a, z_b,
                    chain_type_a=batch["chain_type_a"][mhc_mask],
                    chain_type_b=batch["chain_type_b"][mhc_mask],
                )
                pairing_loss = F.binary_cross_entropy_with_logits(
                    pairing_logit.squeeze(-1),
                    batch["is_valid"][mhc_mask]
                )
                losses.append(pairing_loss)

            # TCR/BCR pairing samples
            receptor_mask = ~batch["is_mhc"]
            if receptor_mask.any():
                tokens_a = batch["tokens_a"][receptor_mask]
                tokens_b = batch["tokens_b"][receptor_mask]

                # Get embeddings
                with torch.no_grad():
                    z_a, _ = self.receptor_chain_classifier.encoder(tokens_a)
                    z_b, _ = self.receptor_chain_classifier.encoder(tokens_b)

                # Predict pairing
                pairing_logit = self.tcr_pairing_predictor(
                    z_a, z_b,
                    chain_type_a=batch["chain_type_a"][receptor_mask],
                    chain_type_b=batch["chain_type_b"][receptor_mask],
                )
                pairing_loss = F.binary_cross_entropy_with_logits(
                    pairing_logit.squeeze(-1),
                    batch["is_valid"][receptor_mask]
                )
                losses.append(pairing_loss)

            if losses:
                loss = sum(losses) / len(losses)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._get_stage_parameters(), max_norm=1.0
                )
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def train_epoch(self, dataloaders: Dict[str, DataLoader]) -> float:
        """Train one epoch for current stage."""
        stage = self.scheduler.current_stage.stage

        if stage == CurriculumStage.CHAIN_CLASSIFICATION:
            return self.train_stage_1(dataloaders["chain_classification"])

        elif stage == CurriculumStage.PAIRING:
            return self.train_stage_2(dataloaders["pairing"])

        else:
            # Stages 3 and 4 use full Presto model
            # Would use similar pattern with appropriate dataloaders
            return 0.0

    def step_epoch(self, epoch_loss: float) -> bool:
        """Step to next epoch, potentially advancing stage.

        Returns True if stage advanced.
        """
        advanced = self.scheduler.step_epoch(epoch_loss)
        if advanced:
            self._setup_optimizer()
            print(f"  -> Advanced to stage: {self.scheduler.current_stage.stage.name}")
        return advanced


def run(args: argparse.Namespace) -> None:
    """Run curriculum training with parsed arguments."""

    # Setup
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate data
    print("Generating synthetic data...")
    chain_samples = generate_chain_classification_data(args.n_samples)
    pairing_samples = generate_pairing_data(args.n_samples)

    # Create datasets and dataloaders
    chain_dataset = ChainClassificationDataset(chain_samples)
    pairing_dataset = PairingDataset(pairing_samples)

    dataloaders = {
        "chain_classification": DataLoader(
            chain_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_chain_classification,
        ),
        "pairing": DataLoader(
            pairing_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_pairing,
        ),
    }

    # Create trainer
    trainer = CurriculumTrainer(
        d_model=args.d_model,
        device=device,
    )
    trainer._setup_optimizer()

    # Training loop
    print(f"\nStarting curriculum training...")
    print(f"Initial stage: {trainer.scheduler.current_stage.stage.name}")

    for epoch in range(args.epochs):
        stage = trainer.scheduler.current_stage.stage

        loss = trainer.train_epoch(dataloaders)
        advanced = trainer.step_epoch(loss)

        print(f"Epoch {epoch+1}/{args.epochs} [{stage.name}]: loss={loss:.4f}")

        if advanced:
            if trainer.scheduler.is_final_stage:
                print("Reached final stage!")

    print("\nTraining complete!")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Curriculum training for PRESTO")
    parser.add_argument("--epochs", type=int, default=40, help="Total epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n_samples", type=int, default=1000, help="Samples per task")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device")
    args = parser.parse_args(argv)

    run(args)


if __name__ == "__main__":
    main()
