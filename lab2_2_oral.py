"""
任务目标：
1. **压缩感知 NAS（Compression-Aware NAS）**:
   - 在每次 NAS 试验中，构建和训练模型后，应用 Mase 的 `CompressionPipeline` 进行 **量化和剪枝**。
   - 进一步训练（post-training），以弥补压缩带来的精度损失。
   - 采用在任务 1（NAS 任务）中表现最好的采样器（TPESampler）进行搜索。
   
2. **对比三种方法的搜索效果**：
   - **NAS（未压缩）**：任务 1 仅搜索最佳超参数，不应用压缩。
   - **压缩感知 NAS（无后续训练）**：在搜索过程中执行压缩，但不进行后续训练。
   - **压缩感知 NAS（有后续训练）**：在执行压缩后，进行后续训练以恢复模型性能。

3. **绘制对比曲线**：
   - X 轴：搜索轮次数。
   - Y 轴：到当前试验次数为止的最高精度。
   - 对比三种方法，评估压缩感知搜索的影响。
"""

import torch
import json
import optuna
import matplotlib.pyplot as plt
import os
import copy  # 用于深拷贝

from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoConfig
from chop.pipelines import CompressionPipeline
from chop import MaseGraph
from chop.nn.modules import Identity
from chop.tools.utils import deepsetattr
from chop.tools import get_tokenized_dataset, get_trainer
from optuna.samplers import TPESampler
import torch.nn as nn

# 定义一个 Identity 模块（用于替换全连接层）
class MyIdentity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x

# -----------------------------
# 1. 加载任务 1 的 NAS 结果（未压缩 NAS）
# -----------------------------
with open("nas_results.json", "r") as f:
    nas_results = json.load(f)

# 采用任务 1 中表现最好的采样器（TPESampler）
best_sampler = TPESampler()

# -----------------------------
# 2. 检查 GPU
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# 3. 加载 IMDb 数据集
# -----------------------------
checkpoint = "prajjwal1/bert-tiny"  # 预训练模型检查点
tokenizer_checkpoint = "bert-base-uncased"  # 分词器检查点
dataset_name = "imdb"

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

# -----------------------------
# 4. 定义搜索空间（超参数与线性层替换策略）
# -----------------------------
search_space = {
    "num_layers": [2, 4, 8],
    "num_heads": [2, 4, 8, 16],
    "hidden_size": [128, 192, 256, 384, 512],
    "intermediate_size": [512, 768, 1024, 1536, 2048],
    "linear_layer_choice": [
        "linear",    # 使用标准 nn.Linear
        "identity",  # 替换为 MyIdentity（恒等映射）
    ],
}

# -----------------------------
# 5. CompressionPipeline 配置（量化与剪枝）
# -----------------------------
compression_pipeline = CompressionPipeline()

quantization_config = {
    "by": "type",
    "default": {
        "config": {"name": None},
    },
    "linear": {
        "config": {
            "name": "integer",
            "data_in_width": 16, "data_in_frac_width": 8,
            "weight_width": 16, "weight_frac_width": 8,
            "bias_width": 16, "bias_frac_width": 8,
        }
    },
}

pruning_config = {
    "weight": {
        "sparsity": 0.5,
        "method": "l1-norm",
        "scope": "local",
    },
    "activation": {
        "sparsity": 0.5,
        "method": "l1-norm",
        "scope": "local",
    },
}

# -----------------------------
# 6. 目标函数（压缩感知 NAS）
# -----------------------------
def objective(trial):
    # (a) 选择超参数
    config = AutoConfig.from_pretrained(checkpoint)
    for param in ["num_layers", "num_heads", "hidden_size", "intermediate_size"]:
        idx = trial.suggest_int(param, 0, len(search_space[param]) - 1)
        setattr(config, param, search_space[param][idx])

    # 确保 hidden_size 可以被 num_heads 整除
    if config.hidden_size % config.num_heads != 0:
        raise optuna.exceptions.TrialPruned("hidden_size must be divisible by num_heads")

    # (b) 选择线性层替换策略
    linear_choice_idx = trial.suggest_int("linear_layer_choice", 0, len(search_space["linear_layer_choice"]) - 1)
    linear_choice_str = search_space["linear_layer_choice"][linear_choice_idx]

    # (c) 构建模型
    model = AutoModelForSequenceClassification.from_config(config)
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Linear) and (layer.in_features == layer.out_features):
            if linear_choice_str == "identity":
                deepsetattr(model, name, MyIdentity())
    
    model.to(device)

    # (d) 训练 1 轮
    trainer = get_trainer(
        model=model, tokenized_dataset=dataset, tokenizer=tokenizer, evaluate_metric="accuracy", num_train_epochs=1
    )
    trainer.train()

    # (e) 进行压缩（量化 + 剪枝）
    model.cpu()  
    mg = MaseGraph(model, hf_input_names=["input_ids", "attention_mask", "labels", "token_type_ids"])
    mg, _ = compression_pipeline(
        mg,
        pass_args={"quantize_transform_pass": copy.deepcopy(quantization_config), "prune_transform_pass": copy.deepcopy(pruning_config)},
    )
    compressed_model = mg.model

    # (f) 直接评估（无后续训练）
    compressed_model.to(device)
    trainer_no_post = get_trainer(model=compressed_model, tokenized_dataset=dataset, tokenizer=tokenizer, evaluate_metric="accuracy", num_train_epochs=0)
    score_no_post = trainer_no_post.evaluate()["eval_accuracy"]

    # (g) 进行后续训练并评估
    model_for_post = copy.deepcopy(compressed_model)
    model_for_post.to(device)
    trainer_with_post = get_trainer(model=model_for_post, tokenized_dataset=dataset, tokenizer=tokenizer, evaluate_metric="accuracy", num_train_epochs=1)
    trainer_with_post.train()
    score_with_post = trainer_with_post.evaluate()["eval_accuracy"]

    # 存储结果
    trial.set_user_attr("score_no_post", score_no_post)
    trial.set_user_attr("score_with_post", score_with_post)

    return score_with_post

# -----------------------------
# 7. 运行压缩感知 NAS 并保存结果
# -----------------------------
study = optuna.create_study(direction="maximize", sampler=best_sampler)
study.optimize(objective, n_trials=30)

results_with_post = [trial.user_attrs["score_with_post"] for trial in study.trials]
results_no_post = [trial.user_attrs["score_no_post"] for trial in study.trials]

with open("compression_results.json", "w") as f:
    json.dump({"with_post": results_with_post, "no_post": results_no_post}, f, indent=2)

# -----------------------------
# 8. 绘制对比曲线
# -----------------------------
plt.figure(figsize=(10, 6))

# (A) NAS（未压缩）
tpe_vals = nas_results["TPESampler"]
plt.plot(range(1, len(tpe_vals) + 1), [max(tpe_vals[:i]) for i in range(1, len(tpe_vals) + 1)], label="NAS (No Compression)", marker='o')

# (B) 压缩感知 NAS（无后续训练）
plt.plot(range(1, len(results_no_post) + 1), [max(results_no_post[:i]) for i in range(1, len(results_no_post) + 1)], label="Compression-Aware NAS (No Post-Training)", marker='o')

# (C) 压缩感知 NAS（有后续训练）
plt.plot(range(1, len(results_with_post) + 1), [max(results_with_post[:i]) for i in range(1, len(results_with_post) + 1)], label="Compression-Aware NAS (With Post-Training)", marker='o')

plt.xlabel("Number of Trials")
plt.ylabel("Max Achieved Accuracy")
plt.title("Comparison: NAS vs Compression-Aware NAS")
plt.legend()
plt.grid(True)
plt.savefig("plots/lab2_t2.png")
plt.show()
