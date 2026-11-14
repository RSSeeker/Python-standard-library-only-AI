"""GGUF v3 读写（教学简化版，只支持 F32 张量 + 基本元数据）。

格式要点（小端）：

- header：magic ``b"GGUF"``、version=3、tensor_count、metadata_kv_count；
- metadata KV：key(str: u64 长度 + UTF-8)、value_type(u32) + value；
  支持类型：UINT32=4、UINT64=10、STRING=8；
- tensor info：name、n_dims(u32)、dims(u64×n)、ggml_type(u32，0=F32)、offset(u64)；
- 数据段从 32 字节对齐处开始，offset 相对数据段起点；张量按行主序存 f32。

本实现与 llama.cpp 的 GGUF 在 header/元数据层面兼容，但只覆盖 F32 张量与
本库 MLP 需要的子集；写入时把权重矩阵的 dims 记为 ``[in, out]``（权重形状
``[out][in]``），bias 记 ``[out]``。
"""

import io
import os
import struct

import AI

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGML_TYPE_F32 = 0
_ALIGNMENT = 32

_TYPE_UINT32 = 4
_TYPE_UINT64 = 10
_TYPE_STRING = 8


def _gguf_check_path(path, name="path"):
    if not isinstance(path, str) or not path.strip():
        raise ValueError("GGUF: %s 必须是非空字符串" % name)
    return path


def _write_string(f, text):
    data = text.encode("utf-8")
    f.write(struct.pack("<Q", len(data)))
    f.write(data)


def _read_string(f):
    (n,) = struct.unpack("<Q", f.read(8))
    if n > 1 << 24:
        raise ValueError("GGUF: 字符串长度异常（%d）" % n)
    raw = f.read(n)
    if len(raw) != n:
        raise ValueError("GGUF: 文件截断（字符串）")
    return raw.decode("utf-8")


def _flatten(tensor, dims):
    if len(dims) == 1:
        if not isinstance(tensor, (list, tuple)) or len(tensor) != dims[0]:
            raise ValueError("GGUF: 一维张量长度不符")
        return [float(v) for v in tensor]
    if not isinstance(tensor, (list, tuple)) or len(tensor) != dims[1]:
        raise ValueError("GGUF: 张量行数不符")
    out = []
    for row in tensor:
        if not isinstance(row, (list, tuple)) or len(row) != dims[0]:
            raise ValueError("GGUF: 张量列数不符")
        out.extend(float(v) for v in row)
    return out


def _unflatten(flat, dims):
    if len(dims) == 1:
        return list(flat)
    cols = dims[0]
    return [list(flat[i * cols:(i + 1) * cols]) for i in range(dims[1])]


def save_gguf(tensors, path, metadata=None):
    """把 ``{name: nested list}`` 写成 GGUF 文件；dims 从嵌套 list 形状推断。"""
    _gguf_check_path(path)
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("GGUF: tensors 必须是非空 dict")
    metadata = {} if metadata is None else metadata
    if not isinstance(metadata, dict):
        raise ValueError("GGUF: metadata 必须是 dict")
    infos = []
    offsets = []
    payload = bytearray()
    for name, value in tensors.items():
        if not isinstance(name, str) or not name:
            raise ValueError("GGUF: 张量名必须是非空字符串")
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
            dims = [len(value[0]), len(value)]
        elif isinstance(value, (list, tuple)):
            dims = [len(value)]
        else:
            raise ValueError("GGUF: 张量 %r 必须是嵌套 list" % name)
        flat = _flatten(value, dims)
        infos.append((name, dims))
        offsets.append(len(payload))
        payload.extend(struct.pack("<%df" % len(flat), *flat))
        while len(payload) % _ALIGNMENT:
            payload.append(0)
    with io.open(path, "wb") as f:
        f.write(GGUF_MAGIC)
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(infos)))
        f.write(struct.pack("<Q", len(metadata)))
        for key, value in metadata.items():
            _write_string(f, str(key))
            if isinstance(value, bool):
                raise ValueError("GGUF: metadata 不支持 bool")
            if isinstance(value, int) and 0 <= value <= 0xFFFFFFFF:
                f.write(struct.pack("<I", _TYPE_UINT32))
                f.write(struct.pack("<I", value))
            elif isinstance(value, int):
                f.write(struct.pack("<I", _TYPE_UINT64))
                f.write(struct.pack("<Q", value))
            elif isinstance(value, str):
                f.write(struct.pack("<I", _TYPE_STRING))
                _write_string(f, value)
            else:
                raise ValueError("GGUF: metadata 值只支持 int/str")
        for (name, dims), offset in zip(infos, offsets):
            _write_string(f, name)
            f.write(struct.pack("<I", len(dims)))
            for d in dims:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", GGML_TYPE_F32))
            f.write(struct.pack("<Q", offset))
        pos = f.tell()
        pad = (-pos) % _ALIGNMENT
        f.write(b"\x00" * pad)
        f.write(bytes(payload))
    return path


def load_gguf(path):
    """读 GGUF 文件，返回 ``{"metadata", "tensors", "dims"}``。"""
    _gguf_check_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError("GGUF: 文件不存在: %s" % path)
    with io.open(path, "rb") as f:
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError("GGUF: magic 不是 b'GGUF'")
        (version,) = struct.unpack("<I", f.read(4))
        if version != GGUF_VERSION:
            raise ValueError("GGUF: 只支持 version=3，收到 %d" % version)
        (tensor_count,) = struct.unpack("<Q", f.read(8))
        (kv_count,) = struct.unpack("<Q", f.read(8))
        metadata = {}
        for _ in range(kv_count):
            key = _read_string(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            if vtype == _TYPE_UINT32:
                (value,) = struct.unpack("<I", f.read(4))
            elif vtype == _TYPE_UINT64:
                (value,) = struct.unpack("<Q", f.read(8))
            elif vtype == _TYPE_STRING:
                value = _read_string(f)
            else:
                raise ValueError("GGUF: 不支持的元数据类型 %d" % vtype)
            metadata[key] = value
        infos = []
        for _ in range(tensor_count):
            name = _read_string(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            if n_dims not in (1, 2):
                raise ValueError("GGUF: 只支持 1/2 维张量，收到 %d 维" % n_dims)
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            (dtype,) = struct.unpack("<I", f.read(4))
            if dtype != GGML_TYPE_F32:
                raise ValueError("GGUF: 只支持 F32 张量，收到类型 %d" % dtype)
            (offset,) = struct.unpack("<Q", f.read(8))
            infos.append((name, dims, offset))
        pos = f.tell()
        data_start = pos + ((-pos) % _ALIGNMENT)
        f.seek(data_start)
        tensors = {}
        dims_map = {}
        for name, dims, offset in infos:
            count = 1
            for d in dims:
                count *= d
            f.seek(data_start + offset)
            raw = f.read(count * 4)
            if len(raw) != count * 4:
                raise ValueError("GGUF: 文件截断（张量 %r）" % name)
            flat = list(struct.unpack("<%df" % count, raw))
            tensors[name] = _unflatten(flat, dims)
            dims_map[name] = dims
    return {"metadata": metadata, "tensors": tensors, "dims": dims_map}


def model_to_gguf_tensors(model):
    """把 MLP 权重转成 GGUF 张量 dict（名字同 HF 适配层）。"""
    if not isinstance(model, AI.MLP):
        raise TypeError("GGUF: model 必须是 AI.MLP")
    tensors = {}
    for i, layer in enumerate(model.layers):
        tensors["model.layers.%d.weight" % i] = [row[:] for row in layer.weights]
        tensors["model.layers.%d.bias" % i] = layer.biases[:]
    return tensors
