"""
任务目标：
1. 采用 Optuna 进行神经架构搜索（NAS），探索 BERT 模型的超参数和层选择。
2. 采用两种不同的采样方法：
   - **GridSampler**（网格搜索）：遍历所有可能的参数组合。
   - **TPESampler**（贝叶斯优化）：使用 Tree-structured Parzen Estimator (TPE) 进行智能搜索。
3. 评估不同模型配置在 IMDb 数据集上的分类准确率。
4. 记录不同搜索方法的最优精度，并绘制搜索次数 vs. 最优精度曲线，以比较 GridSampler 和 TPESampler 的搜索效率。
"""

import torch
import torch.nn as nn
import optuna
import matplotlib.pyplot as plt
import os

from transformers import AutoConfig, AutoModelForSequenceClassification
from chop.tools.utils import deepsetattr
from chop.nn.modules import Identity
from chop.tools import get_tokenized_dataset, get_trainer

# 引入 GridSampler 和 TPESampler 进行搜索
from optuna.samplers import GridSampler, TPESampler

# 检查是否有可用的 GPU，如果有则使用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# 1. 准备 IMDb 数据集
# -----------------------------
checkpoint = "prajjwal1/bert-tiny"  # 预训练模型检查点
tokenizer_checkpoint = "bert-base-uncased"  # 分词器检查点
dataset_name = "imdb"  # 数据集名称

# 加载 IMDb 数据集并获取分词器
dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

# -----------------------------
# 2. 定义超参数搜索空间
# -----------------------------
search_space = {
    "num_layers": [2, 4, 8],  # Transformer 层数
    "num_heads": [2, 4, 8, 16],  # Self-Attention 头数
    "hidden_size": [128, 192, 256, 384, 512],  # 隐藏层维度
    "intermediate_size": [512, 768, 1024, 1536, 2048],  # FFN 中间层维度
    "linear_layer_choice": [  # 替换线性层的方式
        nn.Linear,
        Identity,
    ],
}

# -----------------------------
# 3. 构建模型的函数
# -----------------------------
def construct_model(trial):
    """ 根据 Optuna 试验参数构造模型 """
    config = AutoConfig.from_pretrained(checkpoint)  # 从预训练配置加载基础参数

    # 选择 Transformer 相关超参数
    for param in ["num_layers", "num_heads", "hidden_size", "intermediate_size"]:
        chosen_idx = trial.suggest_int(param, 0, len(search_space[param]) - 1)
        setattr(config, param, search_space[param][chosen_idx])

    # 选择线性层的替换方式
    chosen_linear_idx = trial.suggest_int("linear_layer_choice", 0, len(search_space["linear_layer_choice"]) - 1)
    chosen_linear_cls = search_space["linear_layer_choice"][chosen_linear_idx]

    # 根据配置初始化模型
    trial_model = AutoModelForSequenceClassification.from_config(config)

    # 遍历模型的所有 nn.Linear 层，并根据选择的方式进行替换
    for name, layer in trial_model.named_modules():
        if isinstance(layer, nn.Linear) and layer.in_features == layer.out_features:
            if chosen_linear_cls == nn.Linear:
                continue  # 保持原样
            elif chosen_linear_cls == Identity:
                new_layer = Identity()  # 替换为恒等映射
                deepsetattr(trial_model, name, new_layer)
            else:
                raise ValueError(f"Unknown layer type: {chosen_linear_cls}")

    return trial_model.to(device)  # 将模型移动到 GPU（如果可用）

# -----------------------------
# 4. 定义优化目标（objective 函数）
# -----------------------------
def objective(trial):
    """ 训练并评估模型，返回 IMDb 数据集上的分类准确率 """
    model = construct_model(trial)

    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,  # 只训练 1 轮（减少计算开销）
    )

    trainer.train()
    eval_results = trainer.evaluate()
    trial.set_user_attr("model", model)  # 记录最佳模型

    return eval_results["eval_accuracy"]

# -----------------------------
# 5. 运行 GridSampler 和 TPESampler
# -----------------------------

# (A) **使用 GridSampler 进行网格搜索**
print("Running NAS with GridSampler")

# 1) 构造 GridSampler 的搜索空间
grid_sampler = GridSampler({
    param: list(range(len(vals))) for param, vals in search_space.items()
})

study_grid = optuna.create_study(direction="maximize", sampler=grid_sampler)

# 2) 运行 30 次试验（由于搜索空间较大，可以调整 n_trials 以减少计算量）
study_grid.optimize(objective, n_trials=30)

# (B) **使用 TPESampler 进行贝叶斯优化**
print("Running NAS with TPESampler")
tpe_sampler = TPESampler()
study_tpe = optuna.create_study(direction="maximize", sampler=tpe_sampler)
study_tpe.optimize(objective, n_trials=30)

# -----------------------------
# 6. 保存实验结果
# -----------------------------
os.makedirs("results", exist_ok=True)
torch.save(
    {"grid_trials": study_grid.trials, "tpe_trials": study_tpe.trials},
    "results/new_lab2_1_results.pt",
)

# -----------------------------
# 7. 绘制搜索对比曲线
# -----------------------------
def plot_results(study, label):
    """ 绘制 Optuna 试验过程的最优精度变化曲线 """
    trials = study.trials
    max_acc = []  # 记录到当前搜索轮次为止的最高准确率
    best_so_far = 0  # 追踪最优精度

    for t in trials:
        if t.value is not None:
            best_so_far = max(best_so_far, t.value)
        max_acc.append(best_so_far)

    plt.plot(range(len(max_acc)), max_acc, label=label, marker='o')  # 绘制曲线

plt.figure(figsize=(10, 6))

# 绘制 GridSampler 和 TPESampler 的搜索效果
plot_results(study_grid, "GridSampler")
plot_results(study_tpe, "TPESampler")

plt.xlabel("Number of Trials")  # X 轴：搜索轮次
plt.ylabel("Max Achieved Accuracy")  # Y 轴：最高精度
plt.title("Comparison of NAS Strategies: Grid vs TPE")  # 图表标题
plt.legend()  # 添加图例
plt.grid(True)  # 显示网格

# 保存图表
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/lab2_t1.png")
plt.show()
