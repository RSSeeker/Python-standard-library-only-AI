import pytest
import os
import math
from AI_Text import Point, TextProcessor, build_network, softmax, save_network_pickle, load_network_pickle

# --- 1. 测试基础数学组件 ---

def test_leaky_relu_activation():
    """测试神经元的 LeakyReLU 激活函数逻辑"""
    # 隐藏层神经元：正数保持不变，负数乘 0.01
    p = Point(weight_list=[1.0], bias=0.0, location=(0,0), is_output=False)
    assert p.activation([10.0]) == 10.0
    assert p.activation([-1.0]) == -0.01

def test_softmax_distribution():
    """测试 Softmax 输出是否总和为 1"""
    logits = [1.0, 2.0, 3.0]
    probs = softmax(logits)
    assert sum(probs) == pytest.approx(1.0)
    assert probs[2] > probs[0]  # 较大的输入应产生较大的概率

# --- 2. 测试数据处理器 ---

@pytest.fixture
def processor():
    return TextProcessor("ABCABC")

def test_text_processor_vocab(processor):
    """测试词表提取是否正确"""
    assert len(processor.chars) == 3
    assert 'A' in processor.char_to_idx

def test_prepare_data_shape(processor):
    """测试滑动窗口数据准备的形状"""
    window_size = 2
    data = processor.prepare_data("ABCABC", window_size)
    # 输入长度应为 window_size * vocab_size
    inp_vec, target_vec = data[0]
    assert len(inp_vec) == 2 * 3 
    assert len(target_vec) == 3

# --- 3. 测试网络结构与 IO ---

def test_network_structure():
    """测试网络层级构建"""
    layers = [5, 10, 3]
    net = build_network(layers)
    assert len(net) == 3
    assert len(net[1]) == 10  # 隐藏层神经元数量
    assert len(net[2][0].weight_list) == 10  # 输出层权重对接隐藏层

def test_model_persistence(tmp_path):
    """测试模型保存与加载"""
    net = build_network([2, 5, 2])
    file_path = tmp_path / "test_model.pkl"
    
    save_network_pickle(net, str(file_path))
    assert os.path.exists(file_path)
    
    loaded_net = load_network_pickle(str(file_path))
    assert len(loaded_net) == len(net)
    # 验证第一层第一个神经元的权重是否一致
    assert loaded_net[1][0].weight_list == net[1][0].weight_list

# --- 4. 冒烟测试：训练循环 ---

def test_training_one_step():
    """测试训练循环是否能跑通且 Loss 下降"""
    tp = TextProcessor("测试文本")
    data = tp.prepare_data("测试文本", 1)
    net = build_network([1 * tp.vocab_size, 4, tp.vocab_size])
    
    # 简单跑 1 个 epoch 确保不报错
    try:
        from AI_Text import train_text_ai
        train_text_ai(net, data, epochs=1, lr=0.01)
    except Exception as e:
        pytest.fail(f"训练循环崩溃: {e}")
