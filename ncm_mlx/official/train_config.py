"""Configuration for training the released TinyStories architecture family.

One JSON file describes the data window, the codec (tokenizer), the generator,
the autoregressive baseline and the three optimisation stages. Unknown
keys fail on load. The codec and generator sections map one to one onto
``spec.CodecSpec`` and ``spec.GeneratorSpec``, so trained checkpoints load
through the same ``official`` inference path as the released weights.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from ncm_mlx.official import spec as specs

PRECISIONS = ("float32", "bfloat16")
OPTIMIZERS = ("adamw", "muon")
CORRUPTION_MODES = ("uniform", "neighbor")
OFFICIAL_TS_LEVELS = [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]


@dataclass
class TextData:
    seq_len: int = 128
    raw_dir: str = "artifacts/tinystories/raw"
    cache_root: str = "artifacts/tinystories"
    bpe: str = "artifacts/official/tinystories/mlx/bpe.json"
    limit: int = 0
    val_rows: int = 2048
    unigram_rows: int = 200_000
    prefix_len: int = 0


@dataclass
class TextCodec:
    dim: int = 256
    n_head: int = 4
    levels: list[int] = field(default_factory=lambda: list(OFFICIAL_TS_LEVELS))
    codebook_sizes: list[int] = field(default_factory=lambda: [2048] * 12)
    l0_strides: list[int] = field(default_factory=lambda: [4, 4, 4, 2])
    group_norm_groups: int = 8
    phi_ratio: float = 0.5
    dropout: float = 0.3
    decay: float = 0.95
    epsilon: float = 1e-5
    commitment: float = 0.25
    fine_drop_prob: float = 0.5
    fine_drop_min_keep: int = 1
    restart_threshold: float = 0.0


@dataclass
class TextGenerator:
    n_layer: int = 6
    n_head: int = 6
    dim: int = 384
    rope_base: float = 100000.0
    dropout: float = 0.1
    corrupt_first: float = 0.75
    corrupt_last: float = 0.25
    loss_alpha: float = 0.0
    corruption_mode: str = "uniform"
    corruption_neighbors: int = 8
    corruption_rate_jitter: float = 0.0


STAGE_OPTIONAL = {"optimizer": "adamw", "muon_lr": 0.02, "muon_momentum": 0.95,
                  "muon_weight_decay": 0.0, "muon_ns_steps": 5}


@dataclass
class Stage:
    steps: int = 1000
    batch_size: int = 128
    lr: float = 3e-4
    lr_floor: float = 3e-5
    warmup: int = 100
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    max_minutes: float = 0.0
    optimizer: str = "adamw"
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.0
    muon_ns_steps: int = 5


@dataclass
class TextTrain:
    seed: int = 0
    log_every: int = 100
    eval_every: int = 500
    compile: bool = True
    precision: str = "float32"
    codec: Stage = field(default_factory=lambda: Stage(lr=5e-4, lr_floor=5e-5, weight_decay=0.01))
    generator: Stage = field(default_factory=Stage)
    ar: Stage = field(default_factory=Stage)


_SECTIONS = {"data": TextData, "codec": TextCodec, "generator": TextGenerator, "train": TextTrain}


def _build(cls, value: dict, where: str):
    known = {f.name: f for f in fields(cls)}
    unknown = set(value) - set(known)
    if unknown:
        raise ValueError(f"unknown keys in {where}: {sorted(unknown)}")
    kwargs = {}
    for k, v in value.items():
        if cls is TextTrain and k in ("codec", "generator", "ar"):
            kwargs[k] = _build(Stage, v, f"{where}.{k}")
        else:
            kwargs[k] = v
    return cls(**kwargs)


@dataclass
class TextConfig:
    name: str = "tinystories-text"
    out_dir: str = "artifacts/tinystories-text"
    data: TextData = field(default_factory=TextData)
    codec: TextCodec = field(default_factory=TextCodec)
    generator: TextGenerator = field(default_factory=TextGenerator)
    train: TextTrain = field(default_factory=TextTrain)

    def to_dict(self) -> dict:
        d = asdict(self)
        if not self.data.prefix_len:
            del d["data"]["prefix_len"]
        if self.train.precision == "float32":
            del d["train"]["precision"]
        for name in ("codec", "generator", "ar"):
            stage = d["train"][name]
            if stage["optimizer"] == "adamw":
                for key in STAGE_OPTIONAL:
                    del stage[key]
        gen = d["generator"]
        if gen["corruption_mode"] == "uniform":
            del gen["corruption_neighbors"]
        if gen["corruption_rate_jitter"] == 0:
            del gen["corruption_rate_jitter"]
        if gen["corruption_mode"] == "uniform":
            del gen["corruption_mode"]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TextConfig:
        kwargs = {}
        for key, value in d.items():
            if key in _SECTIONS:
                kwargs[key] = _build(_SECTIONS[key], value, key)
            elif key in ("name", "out_dir"):
                kwargs[key] = value
            else:
                raise ValueError(f"unknown config key {key!r}")
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> TextConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def validate(self) -> None:
        c, g = self.codec, self.generator
        if self.data.prefix_len < 0:
            raise ValueError("data.prefix_len must be 0 (unconditional) or a positive token count")
        if len(c.levels) != len(c.codebook_sizes) or len(c.levels) < 2:
            raise ValueError("codec.levels and codec.codebook_sizes need the same length (>= 2)")
        if c.levels != sorted(set(c.levels)) or c.levels[-1] != self.data.seq_len:
            raise ValueError("codec.levels must increase strictly and end at data.seq_len")
        prod = 1
        for s in c.l0_strides:
            prod *= s
        if prod * c.levels[0] != self.data.seq_len:
            raise ValueError("codec.l0_strides must map data.seq_len to the first level")
        if c.dim % c.n_head or c.dim % c.group_norm_groups or g.dim % g.n_head:
            raise ValueError("widths must be divisible by head and group counts")
        if (g.dim // g.n_head) % 2:
            raise ValueError("generator head width must be even for RoPE")
        for p in (c.dropout, g.dropout, c.fine_drop_prob, g.corrupt_first, g.corrupt_last):
            if not 0.0 <= p < 1.0:
                raise ValueError("dropout and corruption probabilities must be in [0, 1)")
        if not 1 <= c.fine_drop_min_keep <= len(c.levels):
            raise ValueError("codec.fine_drop_min_keep must be a level count")
        if self.train.precision not in PRECISIONS:
            raise ValueError(f"train.precision must be one of {list(PRECISIONS)}")
        if g.corruption_mode not in CORRUPTION_MODES:
            raise ValueError(f"generator.corruption_mode must be one of {list(CORRUPTION_MODES)}")
        if not 0.0 <= g.corruption_rate_jitter < 1.0:
            raise ValueError("generator.corruption_rate_jitter must be in [0, 1)")
        if g.corruption_mode == "neighbor" and not 1 <= g.corruption_neighbors < min(c.codebook_sizes):
            raise ValueError("generator.corruption_neighbors must be in [1, smallest codebook size)")
        for name in ("codec", "generator", "ar"):
            st = getattr(self.train, name)
            if st.optimizer not in OPTIMIZERS:
                raise ValueError(f"train.{name}.optimizer must be one of {list(OPTIMIZERS)}")
            if st.muon_lr <= 0 or not 0 <= st.muon_momentum < 1 or st.muon_ns_steps < 1:
                raise ValueError(f"train.{name}: invalid muon_lr, muon_momentum or muon_ns_steps")
            if st.steps < 1 or st.batch_size < 1 or not 0 <= st.warmup < st.steps or st.lr_floor > st.lr:
                raise ValueError(f"train.{name}: invalid steps, batch, warmup or learning rates")

    def codec_spec(self) -> specs.CodecSpec:
        c = self.codec
        return specs.CodecSpec(vocab_size=8192, seq_len=self.data.seq_len, dim=c.dim, n_head=c.n_head,
                               levels=tuple(c.levels), codebook_sizes=tuple(c.codebook_sizes),
                               l0_strides=tuple(c.l0_strides), group_norm_groups=c.group_norm_groups,
                               phi_ratio=c.phi_ratio, encoder_positional=True)

    def generator_spec(self) -> specs.GeneratorSpec:
        g, P = self.generator, self.data.prefix_len
        return specs.GeneratorSpec(n_layer=g.n_layer, n_head=g.n_head, dim=g.dim, prefix_len=P or 1,
                                   rope_base=g.rope_base, query_level0=P > 0, tied_heads=True,
                                   prefix_pad_id=1)

    def task_spec(self) -> specs.TaskSpec:
        return specs.TaskSpec(name=self.name, remote_dir="", tokenizer="bpe", codec=self.codec_spec(),
                              generator=self.generator_spec(), files={}, temperature=1.0, eos_id=0, pad_id=1)
