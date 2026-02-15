import random
import math
import collections
import pickle
import pathlib

class Point:
    def __init__(self, weight_list, bias, location, is_output=False):
        self.weight_list = weight_list
        self.bias = bias
        self.location = location
        self.history = []
        self.z = 0.0
        self.is_output = is_output

    def activation(self, inputs):
        self.history = inputs
        if not self.weight_list:
            return inputs[0]
        self.z = sum(w * x for w, x in zip(self.weight_list, inputs)) + self.bias
        if self.is_output:
            return self.z
        return self.z if self.z > 0 else 0.01 * self.z

    def derivative(self):
        if self.is_output:
            return 1.0
        return 1.0 if self.z > 0 else 0.01

def build_network(layer_sizes):
    network = []
    for layer_idx, layer_size in enumerate(layer_sizes):
        layer = []
        fan_in = layer_sizes[layer_idx - 1] if layer_idx else 1
        for node_idx in range(layer_size):
            if layer_idx == 0:
                weights = []
            else:
                std = math.sqrt(2.0 / fan_in)
                weights = [random.gauss(0, std) for _ in range(fan_in)]
            bias = 0.0
            is_output = (layer_idx == len(layer_sizes) - 1)
            layer.append(Point(weights, bias, (layer_idx, node_idx), is_output))
        network.append(layer)
    return network

def forward(network, inputs):
    current = inputs
    for layer in network:
        current = [node.activation(current) for node in layer]
    return current

def save_network_pickle(network, filename="network.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(network, f)
    print(f"网络已保存至二进制文件: {filename}")

def load_network_pickle(filename="network.pkl"):
    if not pathlib.Path(filename).exists():
        print(f"错误：文件 {filename} 不存在")
        return None
    with open(filename, "rb") as f:
        network = pickle.load(f)
    print(f"从 {filename} 成功加载网络")
    return network

def train(network, data, epochs, lr=1e-3, batch_size=2, patience=50):
    print("训练开始")
    m_w, v_w = collections.defaultdict(float), collections.defaultdict(float)
    m_b, v_b = collections.defaultdict(float), collections.defaultdict(float)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0
    best_loss, cnt = float('inf'), 0
    items = list(data.items())

    for epoch in range(1, epochs + 1):
        random.shuffle(items)
        epoch_total_loss = 0.0
        for k in range(0, len(items), batch_size):
            batch = items[k : k + batch_size]
            batch_grads_w, batch_grads_b = collections.defaultdict(float), collections.defaultdict(float)

            for x_tuple, y_tuple in batch:
                x, y = list(x_tuple), list(y_tuple)
                pred = forward(network, x)
                sample_loss = sum((p - t) ** 2 for p, t in zip(pred, y)) / len(y)
                epoch_total_loss += sample_loss
                deltas = [2 * (p - t) / len(y) * network[-1][i].derivative() for i, (p, t) in enumerate(zip(pred, y))]

                for layer_idx in range(len(network) - 1, 0, -1):
                    curr, prev = network[layer_idx], network[layer_idx - 1]
                    next_deltas = [0.0 for _ in prev]
                    for j, node in enumerate(curr):
                        batch_grads_b[(layer_idx, j)] += deltas[j]
                        for i, w in enumerate(node.weight_list):
                            batch_grads_w[(layer_idx, j, i)] += deltas[j] * prev[i].activation(prev[i].history)
                            next_deltas[i] += deltas[j] * w * prev[i].derivative()
                    deltas = next_deltas

            step += 1
            # 更新 Bias
            for (l, n), gb in batch_grads_b.items():
                gb /= len(batch)
                m_b[(l,n)] = beta1 * m_b[(l,n)] + (1-beta1)*gb
                v_b[(l,n)] = beta2 * v_b[(l,n)] + (1-beta2)*(gb**2)
                m_hat, v_hat = m_b[(l,n)]/(1-beta1**step), v_b[(l,n)]/(1-beta2**step)
                network[l][n].bias -= lr * m_hat / (math.sqrt(v_hat) + eps)
            # 更新 Weights
            for (l, n, i), gw in batch_grads_w.items():
                gw /= len(batch)
                m_w[(l,n,i)] = beta1 * m_w[(l,n,i)] + (1-beta1)*gw
                v_w[(l,n,i)] = beta2 * v_w[(l,n,i)] + (1-beta2)*(gw**2)
                m_hat, v_hat = m_w[(l,n,i)]/(1-beta1**step), v_w[(l,n,i)]/(1-beta2**step)
                network[l][n].weight_list[i] -= lr * m_hat / (math.sqrt(v_hat) + eps)

        avg_loss = epoch_total_loss / len(data)
        print(f"\rEpoch {epoch}/{epochs} | MSE: {avg_loss}", end="")
        if avg_loss < best_loss: best_loss, cnt = avg_loss, 0
        else: cnt += 1
        if cnt >= patience: break
    print("\n训练完成")

def main():
    # 训练
    net = build_network([5, 6, 2])
    train_data = {
        (0.2, 0.4, 0.6, 0.8, 1.0): (0.04, 0.08),
        (0.4, 0.6, 0.8, 1.0, 1.2): (0.16, 0.24),
        (0.6, 0.8, 1.0, 1.2, 1.4): (0.36, 0.48)
    }
    train(net, train_data, epochs=2000, lr=0.01)
    
    # 保存 (Pickle)
    save_network_pickle(net, "model.pkl")
    
    # 彻底删除原对象
    del net
    
    # 加载 (Pickle)
    loaded_net = load_network_pickle("model.pkl")
    
    if loaded_net:
        # 测试加载后的网络
        test_in = [0.4, 0.6, 0.8, 1.0, 1.2]
        target_output = [0.16, 0.24]
        pred = forward(loaded_net, test_in)
        print(f"验证输入: {test_in}")
        print(f"目标结果: {target_output}")
        print(f"加载后的网络预测: {pred}")
"""
提供以下函数:
build_network(layer_sizes)
forward(network, inputs)
train(network, data, epochs, ...)
save_network_pickle(network, filename)
load_network_pickle(filename)
"""
if __name__ == '__main__':
    main()
