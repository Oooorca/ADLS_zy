import torch
import matplotlib.pyplot as plt
from chop.tools import get_tokenized_dataset, get_trainer
import chop.passes as passes
from transformers import AutoModelForSequenceClassification
from chop import MaseGraph
import os
from pathlib import Path

#加载 IMDb 数据集 并使用 BERT 预训练模型进行文本分类。
#初始化 MaseGraph 并进行元数据分析 以便后续的量化优化。
#遍历不同的固定点位宽（4, 8, 16, 24, 32），对 Linear 层进行量化：
#PTQ（后训练量化）：应用固定点量化后直接评估模型性能。
#QAT（量化感知训练）：在量化后进行微调训练，再次评估模型性能。
#记录不同位宽下的模型准确率 并保存量化后的模型。
#绘制 PTQ 和 QAT 的准确率曲线 以观察量化对模型性能的影响。

# 检查是否有可用的 GPU 设备，如果有则使用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义一组固定点的位宽值范围
fixed_point_widths = [4, 8, 16, 24, 32]  # 设定不同的位宽值

# 用于存储 PTQ（后训练量化）和 QAT（量化感知训练）的精度结果
ptq_accuracies = []
qat_accuracies = []

# 创建存储模型和图表的文件夹（如果不存在则自动创建）
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# 加载 IMDb 数据集，并获取对应的分词器
checkpoint = "prajjwal1/bert-tiny"  # 预训练模型的检查点
tokenizer_checkpoint = "bert-base-uncased"  # 分词器的检查点
dataset_name = "imdb"  # 数据集名称

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

# 加载预训练的 BERT 模型
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
model.config.problem_type = "single_label_classification"  # 设置问题类型为单标签分类

# 初始化 MaseGraph（模型量化和优化工具）
mg = MaseGraph(
    model,
    hf_input_names=["input_ids", "attention_mask", "labels"],  # 指定输入张量名称
)

# 运行元数据分析以优化模型
mg, _ = passes.init_metadata_analysis_pass(mg)
mg, _ = passes.add_common_metadata_analysis_pass(mg)

# 从检查点加载已有的 MaseGraph
mg = MaseGraph.from_checkpoint(f"{Path.home()}/tutorial_2_lora")

# 将模型移动到 GPU（如果可用）
mg.model.to(device)

# 定义模型评估函数，计算模型的准确率
def evaluate_model(model):
    trainer = get_trainer(
        model=model,  # 训练器会自动处理数据加载和设备放置
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",  # 评估指标设为准确率
    )
    eval_results = trainer.evaluate()  # 运行评估
    return eval_results['eval_accuracy']  # 返回评估得到的准确率

# 遍历不同的固定点位宽进行量化实验
for width in fixed_point_widths:
    print(f"\nProcessing fixed-point width: {width}")

    # 定义量化配置，根据不同的层类型应用不同的量化参数
    quantization_config = {
        "by": "type",
        "default": {
            "config": {
                "name": None,  # 默认不进行量化
            }
        },
        "linear": {  # 对 Linear 层进行量化
            "config": {
                "name": "integer",
                "data_in_width": width,  # 输入数据位宽
                "data_in_frac_width": width // 2,  # 输入数据的小数部分位宽
                "weight_width": width,  # 权重的位宽
                "weight_frac_width": width // 2,  # 权重的小数部分位宽
                "bias_width": width,  # 偏置的位宽
                "bias_frac_width": width // 2,  # 偏置的小数部分位宽
            }
        },
    }

    # 应用量化转换
    mg, _ = passes.quantize_transform_pass(
        mg,
        pass_args=quantization_config,
    )

    # 确保量化后的模型在 GPU 上
    mg.model.to(device)

    # 评估 PTQ（后训练量化）模型的精度
    ptq_accuracy = evaluate_model(mg.model)
    ptq_accuracies.append(ptq_accuracy)  # 记录 PTQ 的精度

    # 保存 PTQ 量化后的模型
    mg.export(f"models/ptq_model_width_{width}")

    # 运行 QAT（量化感知训练）
    trainer = get_trainer(
        model=mg.model,  # 训练时使用量化后的模型
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
    )
    trainer.train()  # 进行量化感知训练

    # 评估 QAT 模型的精度
    qat_accuracy = evaluate_model(mg.model)
    qat_accuracies.append(qat_accuracy)  # 记录 QAT 的精度

    # 保存 QAT 量化后的模型
    mg.export(f"models/qat_model_width_{width}")

# 绘制 PTQ 和 QAT 在不同固定点位宽下的准确率曲线
plt.figure(figsize=(10, 6))
plt.plot(fixed_point_widths, ptq_accuracies, label='PTQ', marker='o')  # 绘制 PTQ 结果
plt.plot(fixed_point_widths, qat_accuracies, label='QAT', marker='o')  # 绘制 QAT 结果
plt.xlabel('Fixed Point Width')  # X 轴标签：固定点位宽
plt.ylabel('Accuracy')  # Y 轴标签：准确率
plt.title('Accuracy vs Fixed Point Width for PTQ and QAT')  # 标题
plt.legend()  # 添加图例
plt.grid(True)  # 显示网格

# 保存绘制的图像
plt.savefig("plots/t1_1.png")
plt.close()
