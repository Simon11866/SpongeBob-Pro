# Zero 🧽

> 从零训练的中文大语言模型（LLM）完整开源工程：**预训练 → SFT → GRPO 强化学习 → 推理与评测**，全流程可复现。

SpongeBob-Pro 是一个约 **0.1B 参数**的 decoder-only Transformer 中文语言模型训练项目，使用 **PyTorch 从零实现**（非依赖现有大模型权重），涵盖了一个现代 LLM 从数据到上线所需的完整训练链路：

- 自研 **15k 中英双语 BPE Tokenizer**
- 类 LLaMA 架构：RMSNorm / RoPE / SwiGLU / Grouped Query Attention (GQA) / Flash Attention / 权重共享
- **预训练**（单卡 + 多卡 DDP 两种脚本）
- **SFT** 指令微调（只计算 assistant 部分 loss）
- **GRPO** 强化学习（格式奖励 + DeepSeek Judge 评分）
- **C3 / XCOPA / mini_bench** 多维评测 + DeepSeek Judge 生成式评测
- 交互式对话推理脚本

---

## ✨ 特性

| 模块 | 说明 |
|------|------|
| 🪄 架构 | Decoder-only Transformer，GQA、RoPE、SwiGLU、RMSNorm、Flash Attention |
| 🔤 Tokenizer | 15k 词表，ByteLevel BPE，中英文双语，支持 Chat Template |
| 🚀 预训练 | 支持 DDP 多卡训练、Warmup + Cosine LR、Grad Clip、Checkpoint 续训 |
| 💬 SFT | 多轮对话，只对 assistant 计算 loss，支持 DeepSeek Judge 评测 |
| 🧮 GRPO | 组相对策略优化，格式检查 + Judge 奖励，支持参考模型 KL 约束 |
| 📊 评测 | C3 / XCOPA 多选题准确率 + mini_bench 生成式多维评分 |
| 🎙 推理 | 流式对话，支持单轮 / 多轮，预训练文本续写 |

---

## 📁 项目结构

```
SpongeBob-Pro
├── model/
│   ├── config.py                 # 模型配置（SpongeBobConfig）
│   └── model_spongebob_pro.py    # 模型实现（RMSNorm/RoPE/GQA/SwiGLU...）
├── tokenizer_15k/                # 训练好的 15k 中英双语 BPE Tokenizer
├── dataset/
│   ├── preprocess_data.py        # 预训练数据预处理（jsonl → .bin）
│   ├── pretrain_dataset.py       # 预训练数据集（内存映射加载 .bin）
│   ├── sft_dataset.py            # SFT 数据集（只算 assistant loss）
│   └── grpo_dataset.py           # GRPO 数据集（仅 prompt）
├── train/
│   ├── pretrain.py               # 预训练（多卡 DDP）
│   ├── pretrain_without_ddp.py   # 预训练（单卡）
│   ├── train_sft.py              # SFT 指令微调
│   ├── train_grpo.py             # GRPO 强化学习
│   ├── train_tokenizer.py        # 训练 BPE Tokenizer
│   └── utils.py                  # 训练工具（LR 调度/DDP/采样器）
├── benchmark/
│   ├── evaluator.py              # C3 / XCOPA 多选题评测
│   ├── mini_bench/               # mini_bench 生成式评测 + DeepSeek Judge
│   └── *.jsonl                   # 评测数据
├── data/                         # 预处理后的数据元信息（.meta）
├── out_grpo/                     # GRPO 训练日志示例
└── eval.py                       # 交互式对话推理脚本
```

---

## 🚀 快速开始

### 环境依赖

```bash
# 建议 Python 3.10+
pip install torch transformers accelerate numpy tqdm datasets tokenizers openai
```

### 1. 训练 Tokenizer（可选，已提供训练好的 tokenizer_15k）

```bash
python train/train_tokenizer.py
```

### 2. 数据预处理（预训练数据 jsonl → .bin）

```bash
python dataset/preprocess_data.py \
  --input   /path/to/pretrain.jsonl \
  --output  /path/to/pretrain_data \
  --tokenizer ./tokenizer_15k \
  --seq_len 512
```

### 3. 预训练

**单卡：**
```bash
python train/pretrain_without_ddp.py \
  --data_path /path/to/pretrain_data.bin \
  --save_dir  ./pretrain_out/exp_1 \
  --batch_size 128 --learning_rate 1e-3
```

**多卡 DDP：**
```bash
torchrun --nproc_per_node=4 train/pretrain.py \
  --data_path /path/to/pretrain_data.bin \
  --save_dir  ./pretrain_out/exp_1 \
  --batch_size 128 --learning_rate 1e-3
```

### 4. SFT 指令微调

```bash
python train/train_sft.py \
  --data_path   ./data_raw/mini/sft.jsonl \
  --tokenizer_path ./tokenizer_15k \
  --from_weight ./pretrain_out/.../pretrain_768.pth \
  --save_dir ./out_sft/exp_1 \
  --learning_rate 2e-5
```

### 5. GRPO 强化学习

```bash
export DEEPSEEK_API_KEY="your-deepseek-key"

python train/train_grpo.py \
  --data_path   ./benchmark/mini_bench/100miniSponge.jsonl \
  --tokenizer_path ./tokenizer_15k \
  --sft_model_path ./out_sft/.../sft_768.pth \
  --judge_api_key "$DEEPSEEK_API_KEY" \
  --save_dir ./out_grpo/exp_1
```

### 6. 推理对话

```bash
python eval.py \
  --model_path ./out_sft/.../sft_768.pth \
  --tokenizer_path ./tokenizer_15k \
  --model_type sft --multi_turn
```

> 权重文件名中含 `pretrain` 时自动切换为文本续写模式，含 `sft` 时为对话模式。

---

## 🧠 模型架构

- **基础配置（约 0.1B）**：`hidden_size=768, layers=12, heads=12, KV heads=4 (GQA), intermediate=2048, vocab=15000`
- **位置编码**：RoPE（旋转位置编码），`rope_theta=10000`
- **激活函数**：SwiGLU（`silu`）
- **归一化**：RMSNorm（Pre-Norm 残差结构）
- **注意力**：Grouped Query Attention + Flash Attention（`scaled_dot_product_attention`）
- **权重绑定**：`lm_head` 与 `embed_tokens` 共享权重
- **KV Cache**：推理时加速

---

## 📊 评测

### C3 / XCOPA（多选题准确率）

```bash
python benchmark/test_xcopa_improved.py   # XCOPA 示例
```

### mini_bench + DeepSeek Judge（生成式多维评分）

```bash
export DEEPSEEK_API_KEY="your-deepseek-key"
python benchmark/mini_bench/run_test.py --max_prompts 5
```

评测维度：`fluency`（流畅度）、`factuality`（事实性）、`instruction_following`（指令遵循）。

---

## 🙏 致谢与说明

- 训练全程使用 **SwanLab** 进行实验追踪（代码中已配置）
- GRPO / Judge 评分依赖 **DeepSeek API**，需自行申请 API Key
- 预训练 / SFT 数据请使用自有合规数据；`benchmark/` 内置评测数据集来自公开基准 C3 / XCOPA

---

## ⚠️ 注意事项

- 代码中涉及 `/apdcephfs_qy4/...` 等路径为本项目训练环境的绝对路径，实际运行请替换为你的本地路径
- SwanLab API Key、DeepSeek API Key 等敏感信息请通过环境变量或命令行传入，**勿提交到公开仓库**

---

## 📄 License

本项目仅供学习与研究使用。
