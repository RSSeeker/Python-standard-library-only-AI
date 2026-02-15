import math
import random
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
        # 隐藏层使用 LeakyReLU，输出层保持线性（后面统一接 Softmax）
        if self.is_output:
            return self.z
        return self.z if self.z > 0 else 0.01 * self.z

    def derivative(self):
        if self.is_output:
            return 1.0
        return 1.0 if self.z > 0 else 0.01

def softmax(logits, temperature=1.0):
    # 引入温度调节：T越小越保守，T越大越随机
    logits = [x / temperature for x in logits]
    max_val = max(logits)
    exps = [math.exp(x - max_val) for x in logits]
    sum_exps = sum(exps)
    return [x / sum_exps for x in exps]

def build_network(layer_sizes):
    network = []
    for layer_idx, layer_size in enumerate(layer_sizes):
        layer = []
        fan_in = layer_sizes[layer_idx - 1] if layer_idx else 1
        for node_idx in range(layer_size):
            if layer_idx == 0:
                weights = []
            else:
                # He 初始化（适配 LeakyReLU）
                std = math.sqrt(2.0 / fan_in)
                weights = [random.gauss(0, std) for _ in range(fan_in)]
            bias = 0.0
            is_output = (layer_idx == len(layer_sizes) - 1)
            layer.append(Point(weights, bias, (layer_idx, node_idx), is_output))
        network.append(layer)
    return network

def forward(network, inputs, temperature=1.0):
    current = inputs
    for layer in network:
        current = [node.activation(current) for node in layer]
    return softmax(current, temperature)

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

class TextProcessor:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode_one_hot(self, char):
        vec = [0.0] * self.vocab_size
        if char in self.char_to_idx:
            vec[self.char_to_idx[char]] = 1.0
        return vec

    def prepare_data(self, text, window_size):
        data = []
        for i in range(len(text) - window_size):
            inp_chars = text[i : i + window_size]
            target_char = text[i + window_size]
            
            inp_vec = []
            for c in inp_chars:
                inp_vec.extend(self.encode_one_hot(c))
            
            target_vec = self.encode_one_hot(target_char)
            data.append((inp_vec, target_vec))
        return data

def train_text_ai(network, train_data, epochs=100, lr=0.01):
    m_w, v_w = collections.defaultdict(float), collections.defaultdict(float)
    m_b, v_b = collections.defaultdict(float), collections.defaultdict(float)
    beta1, beta2, eps, step = 0.9, 0.999, 1e-8, 0
    
    for epoch in range(1, epochs + 1):
        random.shuffle(train_data)
        total_loss = 0.0
        
        for x, y in train_data:
            probs = forward(network, x)
            # 交叉熵损失
            total_loss -= sum(t * math.log(p + 1e-10) for p, t in zip(probs, y))
            
            # 反向传播梯度
            deltas = [p - t for p, t in zip(probs, y)]
            step += 1
            
            for layer_idx in range(len(network) - 1, 0, -1):
                curr_layer, prev_layer = network[layer_idx], network[layer_idx - 1]
                next_deltas = [0.0 for _ in prev_layer]
                
                for j, node in enumerate(curr_layer):
                    # Adam 更新 Bias
                    gb = deltas[j]
                    m_b[(layer_idx, j)] = beta1 * m_b[(layer_idx, j)] + (1 - beta1) * gb
                    v_b[(layer_idx, j)] = beta2 * v_b[(layer_idx, j)] + (1 - beta2) * (gb**2)
                    mb_hat = m_b[(layer_idx, j)] / (1 - beta1**step)
                    vb_hat = v_b[(layer_idx, j)] / (1 - beta2**step)
                    node.bias -= lr * mb_hat / (math.sqrt(vb_hat) + eps)
                    
                    for i, w in enumerate(node.weight_list):
                        # 获取前一层激活值
                        prev_node = prev_layer[i]
                        input_val = prev_node.activation(prev_node.history)
                        
                        gw = deltas[j] * input_val
                        m_w[(layer_idx, j, i)] = beta1 * m_w[(layer_idx, j, i)] + (1 - beta1) * gw
                        v_w[(layer_idx, j, i)] = beta2 * v_w[(layer_idx, j, i)] + (1 - beta2) * (gw**2)
                        mw_hat = m_w[(layer_idx, j, i)] / (1 - beta1**step)
                        vw_hat = v_w[(layer_idx, j, i)] / (1 - beta2**step)
                        node.weight_list[i] -= lr * mw_hat / (math.sqrt(vw_hat) + eps)
                        
                        next_deltas[i] += deltas[j] * w * prev_node.derivative()
                deltas = next_deltas
                
        print(f"Epoch {epoch}/{epochs} | Avg Loss: {total_loss / len(train_data)}")

def sample_index(probs):
    """根据概率分布进行随机抽样"""
    r = random.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r < cumulative:
            return i
    return len(probs) - 1

def generate_text(network, tp, start_str, length=15, window_size=3, temp=1.0):
    generated = start_str
    for _ in range(length):
        # 截取最后 window_size 个字符作为输入
        context = generated[-window_size:]
        # 如果长度不足，向左填充第一个字符
        if len(context) < window_size:
            context = context.rjust(window_size, tp.chars[0])
            
        test_vec = []
        for c in context:
            test_vec.extend(tp.encode_one_hot(c))
        
        # 得到带温度调节的概率
        probs = forward(network, test_vec, temperature=temp)
        # 随机抽样获取下一个字
        next_idx = sample_index(probs)
        generated += tp.idx_to_char[next_idx]
    return generated

def main():
    # 语料库（长文本）
    text = "君不见黄河之水天上来奔流到海不复回君不见高堂明镜悲白发朝如青丝暮成雪人生得意须尽欢莫使金樽空对月"

    # 记忆窗口
    window_size = 3
    tp = TextProcessor(text)
    dataset = tp.prepare_data(text, window_size)

    # 构建网络：[输入, 128, 词表大小]
    input_dim = window_size * tp.vocab_size
    net = build_network([input_dim, 128, tp.vocab_size])

    print(f"词表大小: {tp.vocab_size} | 窗口大小: {window_size}")
    print("--- 正在强化记忆训练 ---")

    # 训练
    train_text_ai(net, dataset, epochs=150, lr=0.005)
    # 保存 (Pickle)
    save_network_pickle(net, "model.pkl")

    # 彻底删除原对象
    del net

    # 加载 (Pickle)
    loaded_net = load_network_pickle("model.pkl")

    if loaded_net:
        # 多种温度生成对比
        print("\n--- 预测结果展示 ---")
        start_key = "君不见"
        
        for t in [0.2, 0.7, 1.2]:
            result = generate_text(loaded_net, tp, start_key, length=20, window_size=window_size, temp=t)
            print(f"温度 {t:.1f} (控制创造力): {result}")
'''
提供以下函数：
build_network(layer_sizes)
forward(network, inputs, temperature=1.0)
softmax(logits, temperature=1.0)
train_text_ai(network, train_data, epochs=100, lr=0.01)
TextProcessor.encode_one_hot(self, char)
TextProcessor.prepare_data(self, text, window_size)
generate_text(network, tp, start_str, length=15, window_size=3, temp=1.0)
sample_index(probs)
save_network_pickle(network, filename)
load_network_pickle(filename)
'''
if __name__ == "__main__":
    main()
