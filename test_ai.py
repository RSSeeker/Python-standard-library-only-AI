import pytest
import os
import math
from AI import Point, build_network, forward, save_network_pickle, load_network_pickle, train

# --- 1. 神经元逻辑测试 ---

def test_leaky_relu():
    """验证 LeakyReLU 激活函数在正负值下的表现"""
    # 模拟一个简单的神经元
    p = Point(weight_list=[2.0], bias=0.5, location=(1, 0))
    # 正数输入: 2.0 * 1.0 + 0.5 = 2.5
    assert p.activation([1.0]) == 2.5
    # 负数输入: 2.0 * (-1.0) + 0.5 = -1.5 -> LeakyReLU: -1.5 * 0.01 = -0.015
    assert p.activation([-1.0]) == pytest.approx(-0.015)

def test_derivative():
    """验证导数计算是否正确"""
    p = Point(weight_list=[1.0], bias=0.0, location=(1, 0), is_output=False)
    p.activation([1.0]) # z > 0
    assert p.derivative() == 1.0
    p.activation([-1.0]) # z < 0
    assert p.derivative() == 0.01

# --- 2. 网络构建与前向传播测试 ---

def test_network_shapes():
    """验证 build_network 生成的层数和神经元数量"""
    layer_sizes = [5, 10, 2]
    net = build_network(layer_sizes)
    assert len(net) == 3
    assert len(net[0]) == 5
    assert len(net[1]) == 10
    assert len(net[2]) == 2
    # 验证权重连接：第二层神经元的权重列表长度应等于第一层的数量
    assert len(net[1][0].weight_list) == 5

def test_forward_output_range():
    """验证前向传播输出的维度是否正确"""
    net = build_network([3, 4, 1])
    output = forward(net, [0.5, 0.1, -0.2])
    assert isinstance(output, list)
    assert len(output) == 1

# --- 3. 模型持久化测试 ---

def test_save_load_consistency(tmp_path):
    """验证模型保存后再加载，其权重和偏置保持不变"""
    net = build_network([2, 3, 1])
    file_path = tmp_path / "model.pkl"
    
    # 修改一个特定的权重以便后续验证
    net[1][0].weight_list[0] = 0.12345
    net[1][0].bias = 0.6789
    
    save_network_pickle(net, str(file_path))
    loaded_net = load_network_pickle(str(file_path))
    
    assert loaded_net[1][0].weight_list[0] == 0.12345
    assert loaded_net[1][0].bias == 0.6789
    assert len(loaded_net) == len(net)

# --- 4. 训练流程测试 ---

def test_train_loss_reduction():
    """冒烟测试：确保训练过程可以跑通，且对于简单规律 Loss 能够下降"""
    net = build_network([1, 4, 1])
    # 简单的线性关系：y = 2x
    data = {
        (0.1,): (0.2,),
        (0.5,): (1.0,),
        (0.9,): (1.8,)
    }
    
    # 记录训练前的 Loss
    def get_mse(n, d):
        total = 0
        for x, y in d.items():
            p = forward(n, list(x))
            total += sum((a - b)**2 for a, b in zip(p, y))
        return total / len(d)
    
    initial_loss = get_mse(net, data)
    # 进行短期高强度训练
    train(net, data, epochs=50, lr=0.1, batch_size=1)
    final_loss = get_mse(net, data)
    
    # 验证训练后 Loss 有所改善
    assert final_loss < initial_loss
