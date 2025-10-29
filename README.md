# GUI Knowledge Bench 

> 🧭 Hierarchical Evaluation Toolkit for Multimodal GUI Understanding  
> 📊 Evaluation Data: [Hugging Face Dataset](https://huggingface.co/datasets/KendrickShi/GUI-Knowledge-Bench)  
> 🕸️ Website: [web](https://kendrick-stein.github.io/GUI-Knowledge-Bench-web/)  
> 🌍 Switch between **English** / **中文** by expanding the sections below.

---

<details open>
<summary><b>English Version 🇬🇧</b></summary>

## 🧩 Introduction

**GUI Knowledge Bench Runner** is a Python toolkit to evaluate multimodal GUI understanding across three knowledge types:

- **Interface Perception**: state information, layout semantics, widget functions  
- **Interaction Prediction**: action type and parameters, action effect (resulting screenshot)  
- **Instruction Understanding**: task completion verification and task planning  

It loads a YAML model configuration, constructs multimodal prompts (images + text), routes requests to the selected model/API, and saves results and logs for reproducible benchmarking.

---

## ⚙️ Features

- Structured tasks: Interface Perception, Interaction Prediction, Instruction Understanding  
- Multimodal prompt construction with optional visual annotations and knowledge prompts  
- OpenAI-compatible engine routing (Qwen compatible mode, generic proxies)  
- Concurrent execution with per-run logs and consolidated results  
- Visual annotation and action drawing utilities  
- Config-driven ablations and reproducible runs  

---

## 📁 Repository Structure

| Path | Description |
|------|--------------|
| `.env` | Environment variables (API keys, dataset paths) |
| `Inference/` | Core runner and tools |
| `inference_advanced.py` | Main entry: load config, build prompts, call model, save outputs |
| `tools/load_config_tools.py` | Read YAML, enumerate test files, filter by TestingScope |
| `tools/message_gen_tools.py` | Build multimodal messages and ablations |
| `tools/draw_picture.py` | Visual annotation and GUI action drawing |
| `configs/ModelConfigList/ConfigExamples/*.yaml` | Example model configs |
| `log/{timestamp}/` | Auto-created logs per run |
| `KnowledgeBench/` | Benchmark JSON files (optional local copies) |
| `FINAL_RESULT/` | Results output |

---

## 🧰 Requirements and Installation

**Python 3.9+**

```bash
pip install pyyaml python-dotenv openai pydantic pillow opencv-python numpy ray
# For dataset download
pip install huggingface_hub datasets git-lfs
````

> Note: `git-lfs` also requires system-level installation.

---

## 📦 Data Preparation

**Dataset:** [KendrickShi/GUI-Knowledge-Bench](https://huggingface.co/datasets/KendrickShi/GUI-Knowledge-Bench)

After download:

* Set `KnowledgeBenchDir` in your YAML to the directory that contains the benchmark question JSON files.
* Set `dataset_base_dir` in `.env` to a JSON list of base directories containing images.

**Download methods:**

**1️⃣ git-lfs (recommended):**

```bash
git lfs install
git clone https://huggingface.co/datasets/KendrickShi/GUI-Knowledge-Bench datasets/GUI-Knowledge-Bench
```

**2️⃣ huggingface_hub (Python):**

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="KendrickShi/GUI-Knowledge-Bench",
    repo_type="dataset",
    local_dir="datasets/GUI-Knowledge-Bench",
    local_dir_use_symlinks=False
)
```

**3️⃣ datasets library:**

```python
from datasets import load_dataset
ds = load_dataset("KendrickShi/GUI-Knowledge-Bench", split="train")
```

---

## 🌍 Environment Variables (.env)

| Variable           | Description                                                            |
| ------------------ | ---------------------------------------------------------------------- |
| `deerapi_key`      | API key for an OpenAI-compatible proxy (e.g., DeerAPI)                 |
| `deerapi_url`      | Base URL, e.g. `https://api.deerapi.com/v1/`                           |
| `qwen_api_key`     | API key for Qwen (OpenAI-compatible mode)                              |
| `qwen_url`         | Base URL, e.g. `https://dashscope.aliyuncs.com/compatible-mode/v1`     |
| `dataset_base_dir` | JSON list of base directories, e.g. `["datasets/GUI-Knowledge-Bench"]` |

---

## 🧾 Configuration (YAML)

**Required fields:**

* `KnowledgeBenchDir`: Directory containing benchmark JSON files
* `ModelName`: Model ID (e.g., `gpt-5-chat-latest`, `qwen2.5-vl-7b-instruct`)
* `Engine`: Provider routing key (see Engine mapping)
* `ResultsDir`: Output directory
* `thinking`: Whether to include a thought field
* `MaxWorker`: Max concurrent tasks
* `kwargs`: Model call parameters (`max_tokens`, `temperature`, etc.)
* `ablation_options`: `{visual_prompt: bool, knowledge_prompt: bool}`
* `TestingScope`: Which subtasks to run

---

## 🧮 Engine and ModelName Mapping

Supported engines:

* **Engine: qwen** → uses `qwen_api_key + qwen_url` (compatible mode)
* **Engine: deerapi_gpt** → uses `deerapi_key + deerapi_url`

Special cases in `inference_advanced.py`:

```python
["grok-4-0709", "claude-sonnet-4-20250514", "gemini-2.5-pro-preview-06-05"]
```

---

## 🚀 Quick Start Examples

**YAML Example**

```yaml
KnowledgeBenchDir: datasets/GUI-Knowledge-Bench/questions
ModelName: gpt-5-chat-latest
Engine: deerapi_gpt
ResultsDir: FINAL_RESULT
thinking: false
MaxWorker: 4
kwargs:
  max_tokens: 1024
  temperature: 0.2
  top_p: 0.9
  timeout: 120
ablation_options:
  visual_prompt: true
  knowledge_prompt: true
TestingScope:
  - InterfacePerception
  - InteractionPrediction
  - InstructionUnderstanding
```

`.env`

```bash
dataset_base_dir=["datasets/GUI-Knowledge-Bench"]
```

---

## 📊 Run and Outputs

```bash
python Inference/inference_advanced.py \
  --yaml_dir Inference/configs/ModelConfigList/ConfigExamples/openai_gpt5_chat_model_config.yaml
```

**Outputs:**

* Results → `FINAL_RESULT/.../ModelName/*.json`
* Logs → `Inference/log/{timestamp}/`
* Includes: `config.yaml`, `config.json`, `error.jsonl`, `failed_jsons/`

---

## 🧠 Troubleshooting

* **Image not found** → Check `.env` paths
* **Rate limit or timeout** → Tune `kwargs.timeout`, `MaxWorker`
* **Tool schema failures** → Ensure schema matches model capabilities

---

## 🔧 How to Extend (Engines and Clients)

* Add new branch in `client_init(engine_type)` in `inference_advanced.py`
* Add per-ModelName handling in `eval_message()` if needed
* Keep schema consistent with target SDK

---

## 📝 Notes

* Dataset license: **apache-2.0** (check before redistribution)
* Folder layout may vary; adjust `KnowledgeBenchDir` and `dataset_base_dir` accordingly.

</details>

---

<details>
<summary><b>中文版本 🇨🇳</b></summary>

## 🧩 简介

**GUI Knowledge Bench Runner** 是一个用于评估 GUI 多模态理解能力的 Python 工具包，覆盖三类知识任务：

* **界面理解**：状态信息、布局语义、控件功能
* **交互预测**：动作类型与参数、动作效果（结果截图）
* **指令理解**：任务完成判定与任务规划

工具包通过读取 YAML 配置生成多模态提示（图片 + 文本），调用选定模型/API，并保存结果与日志，以实现可复现的评测。

---

## ⚙️ 功能特性

* 任务结构清晰：界面理解、交互预测、指令理解
* 多模态提示构造，可选视觉标注与知识提示
* OpenAI 兼容引擎路由（通义/Qwen 模式、通用代理）
* 并发执行与日志汇总
* 可视化标注与动作绘制工具
* 基于配置的消融实验与可复现运行

---

## 📁 仓库结构

| 路径                                              | 描述                               |
| ----------------------------------------------- | -------------------------------- |
| `.env`                                          | 环境变量（API Key、数据集路径）              |
| `Inference/`                                    | 核心评测与工具                          |
| `inference_advanced.py`                         | 主入口：加载配置、构造提示、调用模型、保存结果          |
| `tools/load_config_tools.py`                    | 读取 YAML、枚举测试文件、按 TestingScope 过滤 |
| `tools/message_gen_tools.py`                    | 构造多模态消息与消融                       |
| `tools/draw_picture.py`                         | 可视化标注与动作绘制                       |
| `configs/ModelConfigList/ConfigExamples/*.yaml` | 模型配置示例                           |
| `log/{timestamp}/`                              | 每次运行自动生成日志                       |
| `KnowledgeBench/`                               | 评测题目 JSON 文件                     |
| `FINAL_RESULT/`                                 | 结果输出目录                           |

---

## 🧰 安装依赖

**Python 3.9+**

```bash
pip install pyyaml python-dotenv openai pydantic pillow opencv-python numpy ray
# 数据集下载相关
pip install huggingface_hub datasets git-lfs
```

> 注意：`git-lfs` 需系统级安装。

---

## 📦 数据准备

**数据集：** [KendrickShi/GUI-Knowledge-Bench](https://huggingface.co/datasets/KendrickShi/GUI-Knowledge-Bench)

下载完成后：

* 在 YAML 中设置 `KnowledgeBenchDir` 指向 JSON 测试文件目录
* 在 `.env` 中设置 `dataset_base_dir` 为包含图片的目录列表

**下载方式：**

**1️⃣ git-lfs（推荐）**

```bash
git lfs install
git clone https://huggingface.co/datasets/KendrickShi/GUI-Knowledge-Bench datasets/GUI-Knowledge-Bench
```

**2️⃣ huggingface_hub（Python）**

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="KendrickShi/GUI-Knowledge-Bench",
    repo_type="dataset",
    local_dir="datasets/GUI-Knowledge-Bench",
    local_dir_use_symlinks=False
)
```

**3️⃣ datasets 库**

```python
from datasets import load_dataset
ds = load_dataset("KendrickShi/GUI-Knowledge-Bench", split="train")
```

---

## 🌍 环境变量 (.env)

| 变量                 | 说明                                                           |
| ------------------ | ------------------------------------------------------------ |
| `deerapi_key`      | OpenAI 兼容代理 API Key（如 DeerAPI）                               |
| `deerapi_url`      | 基础 URL，如 `https://api.deerapi.com/v1/`                       |
| `qwen_api_key`     | Qwen（通义）API Key                                              |
| `qwen_url`         | 基础 URL，如 `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `dataset_base_dir` | 图片根目录 JSON 列表                                                |

---

## 🧾 配置文件 (YAML)

**必填字段：**

* `KnowledgeBenchDir`：评测题目 JSON 所在目录
* `ModelName`：模型名称（如 `gpt-5-chat-latest`）
* `Engine`：服务提供者路由键
* `ResultsDir`：输出目录
* `thinking`：是否输出思考字段
* `MaxWorker`：最大并发数
* `kwargs`：调用参数（`max_tokens`, `temperature` 等）
* `ablation_options`：是否启用视觉提示与知识提示
* `TestingScope`：要评测的子任务

---

## 🧮 引擎与模型映射

当前支持：

* **Engine: qwen** → 使用 `qwen_api_key + qwen_url`
* **Engine: deerapi_gpt** → 使用 `deerapi_key + deerapi_url`

特殊处理模型：

```python
["grok-4-0709", "claude-sonnet-4-20250514", "gemini-2.5-pro-preview-06-05"]
```

---

## 🚀 快速开始示例

**YAML 示例：**

```yaml
KnowledgeBenchDir: datasets/GUI-Knowledge-Bench/questions
ModelName: qwen2.5-vl-7b-instruct
Engine: qwen
ResultsDir: FINAL_RESULT
thinking: false
MaxWorker: 4
kwargs:
  max_tokens: 1024
  temperature: 0.2
  top_p: 0.9
  timeout: 120
ablation_options:
  visual_prompt: true
  knowledge_prompt: true
TestingScope:
  - InterfacePerception
  - InteractionPrediction
  - InstructionUnderstanding
```

`.env` 示例：

```bash
dataset_base_dir=["datasets/GUI-Knowledge-Bench"]
```

---

## 📊 运行与输出

```bash
python Inference/inference_advanced.py --yaml_dir Inference/configs/ModelConfigList/ConfigExamples/openai_gpt5_chat_model_config.yaml
```

输出：

* 结果：`FINAL_RESULT/.../ModelName/*.json`
* 日志：`Inference/log/{timestamp}/`
* 包含：`config.yaml`、`config.json`、`error.jsonl`、`failed_jsons/`

---

## 🧠 常见问题

* 图片找不到 → 检查 `.env` 中路径
* 速率限制或超时 → 调整 `timeout` 与 `MaxWorker`
* 工具调用错误 → 检查 schema 是否与模型能力匹配

---

## 🔧 扩展方法

* 在 `inference_advanced.py` 的 `client_init()` 中添加新分支
* 若需特殊处理模型，可在 `eval_message()` 添加逻辑
* 确保消息与工具 schema 与 SDK 一致

---

## 📝 备注

* 数据集许可证：**apache-2.0**
* 文件层级可能变化，请确认路径设置正确。

</details>
