"""Released NCM checkpoints supported by the MLX port.

Each spec restates, as plain data, the hyperparameters of one upstream
``config.yaml`` that affect inference. The converter refuses a downloaded
``config.yaml`` whose SHA-256 differs from the pinned file, so a spec cannot
silently drift from the checkpoint it describes.

Checkpoints: https://huggingface.co/hazyresearch/ncm-checkpoints (MIT,
Copyright (c) 2025 Hermann Kumbong). Their configurations and weights are not
redistributed here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

REPO = "hazyresearch/ncm-checkpoints"
REVISION = "d342ce8aaea58830eef3b8ad473664673e41f692"
UPSTREAM_CODE = "https://github.com/HazyResearch/ncm"
UPSTREAM_CODE_REVISION = "c70274273f489abc73832b2b98584a8483b6b203"


@dataclass(frozen=True)
class CodecSpec:
    vocab_size: int
    seq_len: int
    dim: int
    n_head: int
    levels: tuple[int, ...]
    codebook_sizes: tuple[int, ...]
    l0_strides: tuple[int, ...]
    group_norm_groups: int = 8
    phi_ratio: float = 0.5
    encoder_positional: bool = True


@dataclass(frozen=True)
class GeneratorSpec:
    n_layer: int
    n_head: int
    dim: int
    prefix_len: int
    rope_base: float
    query_level0: bool
    tied_heads: bool
    prefix_pad_id: int | None = None


@dataclass(frozen=True)
class TaskSpec:
    name: str
    remote_dir: str
    tokenizer: str
    codec: CodecSpec
    generator: GeneratorSpec
    files: dict[str, str] = field(default_factory=dict)
    temperature: float = 1.0
    eos_id: int | None = None
    pad_id: int | None = None

    def to_json(self) -> dict:
        return asdict(self)


_TS_LEVELS = (1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128)
_SUDOKU_LEVELS = (1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 89)

TASKS: dict[str, TaskSpec] = {
    "tinystories": TaskSpec(
        name="tinystories",
        remote_dir="tinystories",
        tokenizer="bpe",
        codec=CodecSpec(vocab_size=8192, seq_len=128, dim=384, n_head=6, levels=_TS_LEVELS,
                        codebook_sizes=(4096,) * 12, l0_strides=(4, 4, 4, 2)),
        generator=GeneratorSpec(n_layer=12, n_head=12, dim=768, prefix_len=1, rope_base=100000.0,
                                query_level0=False, tied_heads=True, prefix_pad_id=1),
        files={
            "config.yaml": "4009baccad5a159bfc12cd48f3da17c22c4a281b86f71f2fc70b7c1bbb62004b",
            "tokenizer.pt": "76e26003719db32dc3d542518951c286be07d6a5881f29ad5152d299764b0f81",
            "generator.pt": "0acaf4f795390f9f7ccc8f84047867619a1d00e291681e299b855d7e563be6ff",
            "bpe_vocab_8192.pickle": "fb176b4c68108e4b5ef61b9ff2af3731600124e3e160970a6bd78cd0a5d3ff6c",
            "bpe_merges_8192.pickle": "0d4cab13f2e7b684b75eaa290a52e596ba87744ef13d943cf3a50a5a8e080ec1",
        },
        temperature=1.0,
        eos_id=0,
        pad_id=1,
    ),
}


def _sudoku(difficulty: str, config: str, tokenizer: str, generator: str) -> TaskSpec:
    """The three Sudoku releases share one architecture; their configs differ only in training dropout."""
    return TaskSpec(
        name=f"sudoku-{difficulty}",
        remote_dir=f"sudoku/{difficulty}",
        tokenizer="sudoku",
        codec=CodecSpec(vocab_size=12, seq_len=89, dim=256, n_head=4, levels=_SUDOKU_LEVELS,
                        codebook_sizes=(16,) * 12, l0_strides=(89,)),
        generator=GeneratorSpec(n_layer=8, n_head=8, dim=512, prefix_len=91, rope_base=10000.0,
                                query_level0=True, tied_heads=False),
        files={"config.yaml": config, "tokenizer.pt": tokenizer, "generator.pt": generator},
        temperature=0.0,
    )


TASKS.update({t.name: t for t in (
    _sudoku("easy", "68d74fe310e909ca1892338258a2d7143293e685ec1c4747265d9134013d1605",
            "3f9c87e99fd49900929cd650181f55bf00530280416992b11ddc4ea869b39d68",
            "1cefebcd1eae0e210b7e991fdce0e86964d0e897e94a3b432d1008ab92d9926e"),
    _sudoku("medium", "a399820c7adbfd0c6ad6e443a63feb7fe7d35d066fb2981b79440af0b237beec",
            "aca5523a0c477a6f142bfd6c2ba157353c01b1e3f072bb0f75e34b3b75ebc211",
            "8336b53fa3dcbddd9375374b8b7409feba2c2ae3b5a1fc5167ecd7852b26ae17"),
    _sudoku("hard", "8d58c49e6fe52e60191e06df4061937d5bb84f97c5ffbcd93ed9830c32f8df2f",
            "3bc6aecad8e15820f9d9bb09750e3c59bdf91ca953f1f11dbd109aaed1d80c0b",
            "9e0bfe44abc1ca3930b7a63cd56cd0b9564b95e29edfe12dd0903207b89df8b8"),
)})


def get(name: str) -> TaskSpec:
    if name not in TASKS:
        raise SystemExit(f"unknown official task {name!r}; choose one of {', '.join(TASKS)}")
    return TASKS[name]


def from_json(d: dict) -> TaskSpec:
    codec = CodecSpec(**{k: tuple(v) if isinstance(v, list) else v for k, v in d["codec"].items()})
    gen = GeneratorSpec(**d["generator"])
    return TaskSpec(**{**d, "codec": codec, "generator": gen})
