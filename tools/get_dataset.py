from datasets import Dataset

def load_custom_dataset(jsonl_file_path):
    """加载 JSONL 格式的数据（每行一个 JSON 对象）"""
    import json
    
    data = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"JSON 解析错误: {e}, 行内容: {line[:100]}...")
                    continue
    
    print(f"succeed {len(data)} data loaded")
    
    # 转换为 HuggingFace Dataset 格式
    return Dataset.from_dict({
        'main_caption': [item.get('main_caption', '') for item in data],
        'winner_location': [item.get('winner_location', '') for item in data],
        'loser_location': [item.get('loser_location', '') for item in data],
        'beats': [item.get('beats', [[], []]) for item in data],
        'chords': [item.get('chords', []) for item in data],
        'chords_time': [item.get('chords_time', []) for item in data],
        'bpm': [item.get('bpm', 0) for item in data],
        'key': [item.get('key', []) for item in data],
        'prompt_idx': [item.get('prompt_idx', -1) for item in data],
        'dataset': [item.get('dataset', '') for item in data],
        'location': [item.get('location', '') for item in data],  # 原始音频位置（如果需要）
    })



def load_eval_dataset(jsonl_file_path, max_samples=-1):
    """加载验证/测试集 JSONL（单音频格式）"""
    import json
    data = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}")
                    continue
    if max_samples > 0 and max_samples < len(data):
        data = data[:max_samples]
    print(f"succeed {len(data)} val/test data loaded")
    return Dataset.from_dict({
        'main_caption': [item.get('main_caption', '') for item in data],
        'location': [item.get('location', '') for item in data],
        'beats': [item.get('beats', [[], []]) for item in data],
        'chords': [item.get('chords', []) for item in data],
        'chords_time': [item.get('chords_time', []) for item in data],
        'bpm': [item.get('bpm', 0) for item in data],
        'key': [item.get('key', []) for item in data],
    })