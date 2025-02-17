"""
任务目标：
1. **层级独立量化搜索（Layer-Wise Quantization Search）**：
   - 在 Tutorial 6 代码中，所有 `LinearInteger` 层使用相同的整数位宽和小数位宽，这可能不是最佳选择。
   - 由于不同层对量化的敏感度不同，目标是让每一层独立选择适合的 `width` 和 `fractional width`，提高整体量化模型的性能。
   - 让 `width` 选择范围为 `[8, 16, 32]`，`fractional width` 选择范围为 `[2, 4, 8]`。

2. **运行 Optuna 进行超参数搜索**：
   - 为每一层选择 `LinearInteger` 还是标准 `torch.nn.Linear`。
   - 为 `LinearInteger` 层独立分配 `width` 和 `fractional width`。
   - 评估 IMDb 数据集上的准确率。

3. **绘制搜索过程曲线**：
   - X 轴：搜索轮次数。
   - Y 轴：到当前试验次数为止的最高准确率。
"""

import torch
import optuna
import matplotlib.pyplot as plt
from transformers import AutoModel
from chop.tools import get_tokenized_dataset, get_trainer
from chop.tools.utils import deepsetattr
from chop.nn.quantized.modules.linear import LinearInteger
from copy import deepcopy
from pathlib import Path
import dill  # 用于加载模型

# -----------------------------
# 1. 设备选择
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# 2. 加载 IMDb 数据集
# -----------------------------
dataset_name = "imdb"
tokenizer_checkpoint = "bert-base-uncased"

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

# -----------------------------
# 3. 加载最佳预训练模型（从任务 5 保存的最佳模型）
# -----------------------------
checkpoint = "prajjwal1/bert-tiny"

# 从 pickle 文件加载模型（假设任务 5 选择的最佳模型）
with open(f"{Path.home()}/tutorial_5_best_model.pkl", "rb") as f:
    model = dill.load(f)

# -----------------------------
# 4. 定义搜索空间（量化参数）
# -----------------------------
search_space = {
    "linear_layer_choices": [torch.nn.Linear, LinearInteger],  # 选择标准全连接层或量化层
    "width_choices": [8, 16, 32],  # 选择整数位宽
    "fractional_width_choices": [2, 4, 8],  # 选择小数位宽
}

# -----------------------------
# 5. 训练函数（构造带独立量化层的模型）
# -----------------------------
def construct_model(trial):
    """
    依据 Optuna 试验参数构造模型：
    - 遍历所有线性层，独立决定是否替换为 `LinearInteger`。
    - 如果替换为 `LinearInteger`，则独立选择 `width` 和 `fractional width`。
    """
    trial_model = deepcopy(model)  # 深拷贝模型，避免修改原始模型

    for name, layer in trial_model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            # 选择是否使用 Integer 量化层
            new_layer_cls = trial.suggest_categorical(f"{name}_type", search_space["linear_layer_choices"])
            if new_layer_cls == torch.nn.Linear:
                continue  # 继续使用标准 `torch.nn.Linear`
            
            # 采样独立的量化精度参数
            width = trial.suggest_categorical(f"{name}_width", search_space["width_choices"])
            frac_width = trial.suggest_categorical(f"{name}_frac_width", search_space["fractional_width_choices"])

            # 创建新的 `LinearInteger` 层，并匹配原始 `Linear` 层的参数
            kwargs = {
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "config": {
                    "data_in_width": width,
                    "data_in_frac_width": frac_width,
                    "weight_width": width,
                    "weight_frac_width": frac_width,
                    "bias_width": width,
                    "bias_frac_width": frac_width,
                },
            }

            new_layer = new_layer_cls(**kwargs)  # 初始化新的 `LinearInteger` 层
            new_layer.weight.data = layer.weight.data  # 复制权重
            deepsetattr(trial_model, name, new_layer)  # 替换原模型的 `Linear` 层

    return trial_model.to(device)

# -----------------------------
# 6. 目标函数（评估模型性能）
# -----------------------------
def objective(trial):
    """
    目标函数：
    - 依据 `trial` 选择不同层的量化配置，构造模型。
    - 在 IMDb 数据集上训练 1 轮，并评估分类准确率。
    """
    model = construct_model(trial)  # 构造模型

    # 使用 `get_trainer` 训练并评估
    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,  # 只训练 1 轮
    )
    trainer.train()
    eval_results = trainer.evaluate()

    trial.set_user_attr("model", model)  # 记录当前最佳模型
    return eval_results["eval_accuracy"]  # 返回准确率作为优化目标

# -----------------------------
# 7. 运行 Optuna 搜索（随机采样）
# -----------------------------
sampler = optuna.samplers.RandomSampler()  # 采用随机搜索
study = optuna.create_study(direction="maximize", study_name="bert-int-quantization", sampler=sampler)

n_trials = 20  # 运行 20 次搜索试验
study.optimize(objective, n_trials=n_trials)

# -----------------------------
# 8. 绘制搜索过程曲线
# -----------------------------
# 提取最佳准确率曲线
trials = range(1, n_trials + 1)
best_acc = [max([t.value for t in study.trials[:i]]) for i in trials]

plt.figure(figsize=(10, 6))
plt.plot(trials, best_acc, label="Best Accuracy", marker="o")
plt.xlabel("Number of Trials")  # X 轴：搜索轮次数
plt.ylabel("Max Achieved Accuracy")  # Y 轴：最高准确率
plt.title("Integer Quantization Search: Accuracy vs Trials")  # 标题
plt.legend()  # 添加图例
plt.grid(True)  # 显示网格

# 保存图表
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/lab3_t1.png")
plt.show()
