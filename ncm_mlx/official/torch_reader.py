"""Read PyTorch zip checkpoints into NumPy arrays without PyTorch.

A ``.pt`` file written by ``torch.save`` is a zip archive holding a pickle
(``data.pkl``) and one raw buffer per tensor storage (``data/<key>``). The
pickle names a handful of PyTorch globals to rebuild tensors. This reader maps
an explicit allow-list of those globals to NumPy constructors and refuses every
other global, so a modified checkpoint cannot run code during loading.
"""

from __future__ import annotations

import collections
import pickle
import zipfile
from pathlib import Path

import numpy as np

_STORAGE_DTYPES = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


class _StorageType:
    def __init__(self, name: str):
        self.name = name
        self.dtype = np.dtype(_STORAGE_DTYPES[name])


def _rebuild_tensor(storage, offset, size, stride, requires_grad=False, hooks=None, metadata=None):
    del requires_grad, hooks, metadata
    size, stride = tuple(int(s) for s in size), tuple(int(s) for s in stride)
    last = offset + sum((n - 1) * s for n, s in zip(size, stride)) if all(size) else offset
    if offset < 0 or (all(size) and last >= storage.size):
        raise pickle.UnpicklingError("tensor view exceeds its storage")
    if not size:
        return np.array(storage[offset])
    view = np.lib.stride_tricks.as_strided(
        storage[offset:], shape=size, strides=tuple(s * storage.itemsize for s in stride), writeable=False)
    return np.array(view)


class _Unpickler(pickle.Unpickler):
    def __init__(self, file, archive: zipfile.ZipFile, root: str):
        super().__init__(file)
        self._archive = archive
        self._root = root
        self._storages: dict[str, np.ndarray] = {}

    def find_class(self, module: str, name: str):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor
        if module == "torch" and name in _STORAGE_DTYPES:
            return _StorageType(name)
        raise pickle.UnpicklingError(f"refusing to load global {module}.{name}")

    def persistent_load(self, pid):
        if not (isinstance(pid, tuple) and len(pid) == 5 and pid[0] == "storage"):
            raise pickle.UnpicklingError(f"unsupported persistent id {pid!r}")
        _, kind, key, _location, numel = pid
        if not isinstance(kind, _StorageType):
            raise pickle.UnpicklingError("storage type is not an allowed PyTorch storage")
        key = str(key)
        if key not in self._storages:
            raw = self._archive.read(f"{self._root}/data/{key}")
            arr = np.frombuffer(raw, dtype=kind.dtype.newbyteorder("<"))
            if arr.size < int(numel):
                raise pickle.UnpicklingError(f"storage {key} is shorter than declared")
            self._storages[key] = arr[: int(numel)]
        return self._storages[key]


def load(path: str | Path) -> dict:
    """Load a ``torch.save`` zip checkpoint as nested dicts of NumPy arrays."""
    with zipfile.ZipFile(path) as archive:
        pickles = [n for n in archive.namelist() if n.endswith("/data.pkl") and n.count("/") == 1]
        if len(pickles) != 1:
            raise ValueError(f"{path}: expected one data.pkl in a PyTorch zip checkpoint")
        root = pickles[0].split("/")[0]
        order = f"{root}/byteorder"
        if order in archive.namelist() and archive.read(order).strip() != b"little":
            raise ValueError(f"{path}: only little-endian checkpoints are supported")
        with archive.open(pickles[0]) as f:
            return _Unpickler(f, archive, root).load()
