"""
任务目标：
1. 选取在任务 1 中获得最高精度的模型（QAT 处理后，固定点宽度 16）。
2. 对该模型进行剪枝实验，稀疏率范围为 0.1 到 0.9。
3. 采用两种不同的剪枝方法：
   - 随机剪枝（Random Pruning）
   - L1 范数剪枝（L1-Norm Pruning）
4. 在 IMDb 数据集上评估不同剪枝率下的模型精度。
5. 记录剪枝后的精度，并绘制剪枝率与模型精度的关系曲线。
"""

import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification
from chop import MaseGraph
import chop.passes as passes
from chop.tools import get_tokenized_dataset, get_trainer
import os
from pathlib import Path

# 检查是否有可用的 GPU，如果有则使用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义剪枝实验的稀疏率范围（从 0.1 到 0.9）
sparsity_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 加载 IMDb 数据集，并获取分词器
checkpoint = "prajjwal1/bert-tiny"  # 预训练 BERT 模型检查点
tokenizer_checkpoint = "bert-base-uncased"  # 分词器检查点
dataset_name = "imdb"  # 数据集名称

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

# 加载预训练的 BERT 模型
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
model.config.problem_type = "single_label_classification"  # 设置问题类型为单标签分类

# 初始化 MaseGraph（用于剪枝和量化）
mg = MaseGraph(
    model,
    hf_input_names=["input_ids", "attention_mask", "labels"],  # 指定输入张量名称
)

# 运行元数据分析以优化模型
mg, _ = passes.init_metadata_analysis_pass(mg)
mg, _ = passes.add_common_metadata_analysis_pass(mg)

# 从任务 1 选取最佳模型（QAT 处理后，固定点宽度 16）
mg = MaseGraph.from_checkpoint(f"models/qat_model_width_16")

# 剪枝配置函数，返回剪枝参数
def get_pruning_config(sparsity, method="l1-norm", scope="local"):
    return {
        "weight": {
            "sparsity": sparsity,  # 设置权重的稀疏率
            "method": method,  # 剪枝方法（Random 或 L1-Norm）
            "scope": scope,  # 剪枝范围（local 表示局部剪枝）
        },
        "activation": {
            "sparsity": sparsity,  # 设置激活值的稀疏率
            "method": method,  # 剪枝方法
            "scope": scope,  # 剪枝范围
        },
    }

# 评估模型精度的函数
def evaluate_model(model):
    trainer = get_trainer(
        model=model,  # 训练器会自动处理数据加载和设备放置
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",  # 评估指标设为准确率
        num_train_epochs=5,  # 进行 5 轮训练，以提高剪枝后模型的性能
    )
    trainer.train()  # 训练剪枝后的模型
    eval_results = trainer.evaluate()  # 运行评估
    return eval_results["eval_accuracy"]  # 返回评估得到的准确率

# 初始化结果存储列表
random_accuracies = []  # 存储随机剪枝的精度
l1_norm_accuracies = []  # 存储 L1-Norm 剪枝的精度

# 遍历不同稀疏率进行剪枝实验
for sparsity in sparsity_levels:
    print(f"\nProcessing sparsity level: {sparsity}")

    # 随机剪枝（Random Pruning）
    random_pruning_config = get_pruning_config(sparsity, method="random")  # 配置剪枝参数
    mg_random, _ = passes.prune_transform_pass(mg, pass_args=random_pruning_config)  # 应用剪枝
    random_accuracy = evaluate_model(mg_random.model)  # 评估剪枝后模型的精度
    random_accuracies.append(random_accuracy)  # 记录精度

    # L1 范数剪枝（L1-Norm Pruning）
    l1_pruning_config = get_pruning_config(sparsity, method="l1-norm")  # 配置剪枝参数
    mg_l1, _ = passes.prune_transform_pass(mg, pass_args=l1_pruning_config)  # 应用剪枝
    l1_accuracy = evaluate_model(mg_l1.model)  # 评估剪枝后模型的精度
    l1_norm_accuracies.append(l1_accuracy)  # 记录精度

    # 创建存储剪枝模型的文件夹
    os.makedirs("models_lab1_2", exist_ok=True)

    # 保存剪枝后的模型（state_dict 方式）
    torch.save(mg_random.model.state_dict(), f"models_lab1_2/random_pruned_sparsity_{sparsity}.pt")
    torch.save(mg_l1.model.state_dict(), f"models_lab1_2/l1_pruned_sparsity_{sparsity}.pt")

# 绘制不同剪枝方法在不同稀疏率下的精度曲线
plt.figure(figsize=(10, 6))
plt.plot(sparsity_levels, random_accuracies, label="Random Pruning", marker="o")  # 随机剪枝
plt.plot(sparsity_levels, l1_norm_accuracies, label="L1-Norm Pruning", marker="o")  # L1 范数剪枝
plt.xlabel("Sparsity")  # X 轴标签：稀疏率
plt.ylabel("Accuracy")  # Y 轴标签：准确率
plt.title("Accuracy vs Sparsity for Pruning Strategies")  # 图表标题
plt.legend()  # 添加图例
plt.grid(True)  # 显示网格

# 保存绘制的图像
plt.savefig("plots/lab1_t2.png")
plt.show()
