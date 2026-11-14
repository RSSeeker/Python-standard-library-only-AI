"""HF 目录约定兼容层：config.json / tokenizer.json / model.safetensors。

只覆盖「单机、无依赖」的教学版约定，不追求与 transformers 完全一致：

- ``config.json``：标准字段（``architectures``/``model_type``/``hidden_size``/
  ``num_hidden_layers``/``activation_function``/``framework``）+ 自定义
  ``ai_layers``（每层 in/out/activation），用于精确重建 MLP；
- ``model.safetensors``：张量名采用 ``model.layers.<i>.weight`` /
  ``model.layers.<i>.bias``（与 PyTorch 命名习惯一致），底层用 AI.safetensors_save；
- ``tokenizer.json``：可选的 ``{"type": "wordlevel", "words": [...], "word2id": {...}}``。
"""

import io
import json
import os

import AI


def _hf_check_dir(directory, name="directory"):
    if not isinstance(directory, str) or not directory.strip():
        raise ValueError("HF: %s 必须是非空字符串" % name)
    return directory


def _hf_model_layers(model):
    if not isinstance(model, AI.MLP):
        raise TypeError("HF: model 必须是 AI.MLP")
    layers = []
    for layer in model.layers:
        layers.append({
            "in_features": int(layer.fan_in),
            "out_features": int(layer.fan_out),
            "activation": layer.activation_name,
        })
    return layers


def _hf_config(model, model_type="mlp", num_labels=None):
    layers = _hf_model_layers(model)
    return {
        "architectures": ["MLPForClassification" if (num_labels or 0) > 1
                          else "MLPForRegression"],
        "model_type": model_type,
        "framework": "ai-py",
        "hidden_size": layers[0]["out_features"],
        "num_hidden_layers": len(layers),
        "activation_function": layers[0]["activation"],
        "ai_layers": layers,
        "ai_num_labels": num_labels,
    }


def save_hf_directory(model, directory, tokenizer=None, model_type="mlp",
                      num_labels=None):
    """把 MLP 保存成 HF 风格目录，返回写出的文件列表。"""
    _hf_check_dir(directory)
    os.makedirs(directory, exist_ok=True)
    config = _hf_config(model, model_type=model_type, num_labels=num_labels)
    tensors = {}
    for i, layer in enumerate(model.layers):
        tensors["model.layers.%d.weight" % i] = [row[:] for row in layer.weights]
        tensors["model.layers.%d.bias" % i] = layer.biases[:]
    written = []
    config_path = os.path.join(directory, "config.json")
    with io.open(config_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(config, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write("\n")
    written.append(config_path)
    weights_path = os.path.join(directory, "model.safetensors")
    AI.safetensors_save(tensors, weights_path)
    written.append(weights_path)
    if tokenizer is not None:
        if not isinstance(tokenizer, (list, tuple)) or not tokenizer:
            raise ValueError("HF: tokenizer 必须是词表 list/tuple")
        words = [str(w) for w in tokenizer]
        tok = {"type": "wordlevel", "words": words,
               "word2id": {w: i for i, w in enumerate(words)}}
        tok_path = os.path.join(directory, "tokenizer.json")
        with io.open(tok_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(tok, f, ensure_ascii=False, sort_keys=True, indent=2)
            f.write("\n")
        written.append(tok_path)
    return written


def _hf_read_json(path):
    with io.open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("HF: %s 顶层必须是 dict" % os.path.basename(path))
    return data


def load_hf_directory(directory):
    """读取 HF 风格目录，返回 ``{"config", "model", "tokenizer"}``。"""
    _hf_check_dir(directory)
    if not os.path.isdir(directory):
        raise FileNotFoundError("HF: 目录不存在: %s" % directory)
    config_path = os.path.join(directory, "config.json")
    weights_path = os.path.join(directory, "model.safetensors")
    if not os.path.exists(config_path):
        raise FileNotFoundError("HF: 缺少 config.json: %s" % directory)
    if not os.path.exists(weights_path):
        raise FileNotFoundError("HF: 缺少 model.safetensors: %s" % directory)
    config = _hf_read_json(config_path)
    layers = config.get("ai_layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("HF: config.json 缺少 ai_layers（非本库写出的目录）")
    sizes = [int(layers[0]["in_features"])]
    acts = []
    for layer in layers:
        if not isinstance(layer, dict):
            raise ValueError("HF: ai_layers 每项必须是 dict")
        sizes.append(int(layer["out_features"]))
        acts.append(str(layer.get("activation", "linear")))
    model = AI.MLP(sizes, acts)
    tensors = AI.safetensors_load(weights_path)
    for i, layer in enumerate(model.layers):
        key_w = "model.layers.%d.weight" % i
        key_b = "model.layers.%d.bias" % i
        if key_w not in tensors or key_b not in tensors:
            raise ValueError("HF: model.safetensors 缺少张量 %r/%r" % (key_w, key_b))
        layer.weights = [list(row) for row in tensors[key_w]]
        layer.biases = list(tensors[key_b])
    tokenizer = None
    tok_path = os.path.join(directory, "tokenizer.json")
    if os.path.exists(tok_path):
        tokenizer = _hf_read_json(tok_path)
    return {"config": config, "model": model, "tokenizer": tokenizer}
