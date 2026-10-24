from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class DataConfig:
    corpus: str = "toy-grammar"
    seq_len: int = 32
    n_train: int = 4000
    n_val: int = 500
    split_seed: int = 0
    min_givens: int = 36
    max_givens: int = 45
    extra_train_seeds: list[int] = field(default_factory=list)
    extra_train_per_seed: int = 1_000_000
    extra_train_unique: bool = False
    extra_train_keep_unique: bool = False


@dataclass
class CodecConfig:
    dim: int = 128
    n_head: int = 4
    enc_layers: int = 2
    dec_layers: int = 2
    dropout: float = 0.1
    levels: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    codebook_sizes: list[int] = field(default_factory=lambda: [128] * 6)
    decay: float = 0.95
    epsilon: float = 1e-5
    commitment: float = 0.25
    quant_resi_ratio: float = 0.5
    restart_threshold: float = 0.05
    fine_drop_prob: float = 0.5
    fine_drop_min_keep: int = 1
    pre_quant_groups: int = 0
    learned_l0: bool = False


@dataclass
class GeneratorConfig:
    n_embd: int = 256
    n_layer: int = 6
    n_head: int = 4
    dropout: float = 0.1
    prefix_len: int = 1
    prefix_vocab: int = 0
    rope_base: float = 10000.0
    attn_pattern: str = "block_diagonal"
    corrupt_first: float = 0.0
    corrupt_last: float = 0.0
    corrupt_source: str = "uniform"


@dataclass
class TrainConfig:
    seed: int = 0
    codec_steps: int = 3000
    codec_lr: float = 1e-3
    generator_steps: int = 4000
    generator_lr: float = 6e-4
    batch_size: int = 128
    warmup: int = 100
    weight_decay: float = 0.01
    log_every: int = 250
    compile: bool = False
    max_minutes: float = 0.0


@dataclass
class Config:
    name: str = "toy-grammar"
    out_dir: str = "artifacts/toy-grammar"
    codec_checkpoint: str | None = None
    data: DataConfig = field(default_factory=DataConfig)
    codec: CodecConfig = field(default_factory=CodecConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        sections = {f.name: f.type for f in fields(cls)}
        kwargs = {}
        for key, value in d.items():
            if key not in sections:
                raise ValueError(f"unknown config key {key!r}")
            if isinstance(value, dict):
                sub = {"data": DataConfig, "codec": CodecConfig, "generator": GeneratorConfig,
                       "train": TrainConfig}[key]
                known = {f.name for f in fields(sub)}
                unknown = set(value) - known
                if unknown:
                    raise ValueError(f"unknown keys in {key}: {sorted(unknown)}")
                kwargs[key] = sub(**value)
            else:
                kwargs[key] = value
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> Config:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    def validate(self) -> None:
        c = self.codec
        if len(c.levels) != len(c.codebook_sizes):
            raise ValueError("codec.levels and codec.codebook_sizes must have the same length")
        if c.levels != sorted(c.levels) or len(set(c.levels)) != len(c.levels):
            raise ValueError("codec.levels must be strictly increasing")
        if c.levels[-1] != self.data.seq_len:
            raise ValueError("the last level length must equal data.seq_len")
        if self.generator.attn_pattern not in ("block_diagonal", "block_causal"):
            raise ValueError("generator.attn_pattern must be block_diagonal or block_causal")
        g = self.generator
        if not (0.0 <= g.corrupt_first <= 1.0 and 0.0 <= g.corrupt_last <= 1.0):
            raise ValueError("generator corruption probabilities must be in [0, 1]")
        if g.corrupt_source not in ("uniform", "model"):
            raise ValueError("generator.corrupt_source must be uniform or model")
        if c.dim % c.n_head or self.generator.n_embd % self.generator.n_head:
            raise ValueError("embedding sizes must be divisible by head counts")
