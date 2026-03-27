# CUDA Task 使用说明

`cuda_task` 目录用于放置 `math_agent` 的 CUDA Kernel 优化任务。

当前目录下有两类内容：

- 具体任务目录，例如：
  - `case1_sinkhorn`
  - `case2_moe`
- 一套位于 `cuda_task` 根目录的公共模板文件

一个典型任务目录通常包含：

- `initial_program.py`：初始实现
- `task_prompt.txt`：任务目标和约束
- `eval_program_with_profile.py`：正式评测脚本
- `task_config.yaml`：常规 PES 配置
- `task_config_codex.yaml`：Codex fallback 配置
- `task_config_switch.yaml`：双模型切换配置
- `run.sh`：常规 PES 入口
- `run_with_codex.sh`：Codex 模式入口
- `run_with_llm_switch.sh`：双模型切换模式入口
- `codex_agent_seed_gen.py`：Codex seed 生成脚本

本文只以 `case1_sinkhorn` 为例说明。`case2_moe` 和后续任务目录可以按同样方式使用。

## 前置条件

- 已安装 LoongFlow 运行依赖
- 推荐在 `loongflow_v2` 环境中运行
- 如果要使用 Codex：
  - 已安装 Codex CLI
  - `codex login status` 正常
  - 当前机器可以访问 Codex 后端

## 三种运行模式

### 1. 常规 PES

不使用 Codex，也不切换模型，直接从 `initial_program.py` 开始迭代。

```bash
cd /Users/machenyu01/Downloads/LoongFlow/agents/math_agent/cuda_task/case1_sinkhorn
bash ./run.sh
```

如果你已经有一个自定义初始文件，也可以指定：

```bash
cd /Users/machenyu01/Downloads/LoongFlow/agents/math_agent/cuda_task/case1_sinkhorn
INITIAL_FILE=your_seed.py bash ./run.sh
```

### 2. Codex 模式

这条模式仍然保留常规 PES 主链，但允许两种 Codex 介入方式：

- 在开始前先让 Codex 生成 seed
- 在后续迭代阶段，当本轮优化不明显时触发 Codex fallback

入口：

```bash
cd /Users/machenyu01/Downloads/LoongFlow/agents/math_agent/cuda_task/case1_sinkhorn
bash ./run_with_codex.sh
```

开关：

- `USE_CODEX_SEED_BOOTSTRAP=1`
  - 先用 Codex 生成 seed，再进入 PES
- `USE_CODEX_SEED_BOOTSTRAP=0`
  - 不生成 seed，直接进入带 Codex fallback 的 PES

例如：

```bash
cd /Users/machenyu01/Downloads/LoongFlow/agents/math_agent/cuda_task/case1_sinkhorn
USE_CODEX_SEED_BOOTSTRAP=1 bash ./run_with_codex.sh
```

```bash
cd /Users/machenyu01/Downloads/LoongFlow/agents/math_agent/cuda_task/case1_sinkhorn
USE_CODEX_SEED_BOOTSTRAP=0 bash ./run_with_codex.sh
```

Codex seed 只有在评测成功时才会被采用；否则会自动回退到 `initial_program.py`。

### 3. 双模型切换模式

这条模式不使用 Codex，而是在常规 PES 的基础上按上一轮分数提升情况在两套 LLM 配置之间切换。

当前规则是：

- 第 1 轮固定用 `qianfan`
- 从第 2 轮开始：
  - 当以下**三个条件同时满足**时，用 `qianfan`（faster_provider）：
    1. 当前迭代次数 >= `iteration_threshold`（默认为 2）
    2. 子代分数 - 父代分数 > `score_improvement_threshold`（默认为 1.0）
    3. speedup >= `speedup_lower_threshold`（默认为 1.5）
  - 否则用 `chatanywhere`（slower_provider）

入口：

```bash
cd /Users/machenyu01/Downloads/LoongFlow/agents/math_agent/cuda_task/case1_sinkhorn
bash ./run_with_llm_switch.sh
```

## Codex 产物说明

启用 Codex seed bootstrap 后，相关产物会写入：

- `output_with_codex/bootstrap/`

常见文件包括：

- `codex_seed.py`：最终选中的 seed 文件
- `codex_seed_eval.json`：最终 seed 的评测结果
- `codex_seed_report.json`：结构化报告
- `codex_seed_run.log`：精简日志
- `codex_seed_debug.log`：详细调试日志

如果设置了多次尝试，还可能出现：

- `codex_seed.attempt_N.py`
- `codex_seed.attempt_N.eval.json`

## 常见问题

### 1. 找不到 Codex CLI

检查：

```bash
command -v codex
codex --version
```

如果 `codex` 不在 `PATH` 中，可以显式指定路径：

```bash
python codex_agent_seed_gen.py --codex-bin /path/to/codex
```

### 2. Codex CLI 未登录

检查：

```bash
codex login status
```

然后登录：

```bash
codex login
```

### 3. 当前机器无法访问 Codex/OpenAI

如果当前机器网络不通，不建议在本机直接跑 Codex seed。
更稳的方式是：

1. 在可用机器上先生成 `codex_seed.py`
2. 再把它带回目标机器
3. 用常规 PES 入口运行：

```bash
INITIAL_FILE=codex_seed.py bash ./run.sh
```

## 其他说明

- `run.sh`：纯 LoongFlow 常规 PES
- `run_with_codex.sh`：Codex 模式
- `run_with_llm_switch.sh`：双模型切换模式
- 当前以 `case1_sinkhorn` 为例，`case2_moe` 可按同样方式组织和使用
