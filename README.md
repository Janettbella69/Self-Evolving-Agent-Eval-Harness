# Agent Eval 

通用的 LLM Agent 评测平台：把"这个 Agent 到底好不好"从一句感觉，变成**可测量、可校准、可回归**的工程问题。

被测 Agent 只需暴露一个 HTTP API——平台把它当黑盒，负责其余的一切：数据集管理、批量实验、trace 采集、分层打分（规则 → 指标 → LLM Judge）、人工标注界面、Judge 校准闭环、回归门禁。

AZORA 是第一个完整接入案例（见 `examples/azora/` 与 `datasets/`）。

## 为什么需要它

Agent 产品的迭代困境：改了 prompt / 换了模型 / 加了工具，**你怎么知道变好了还是变坏了？**

- 手动试几个 query —— 样本太小，且你只会试自己想得到的
- 直接上 LLM-as-a-Judge —— 未经校准的 judge 只是把"感觉"外包给了另一个模型
- 只看线上指标 —— 滞后、混杂，无法归因到某次改动

这个平台实现的是完整的评测闭环：

```
trace 采集 → 错误分析(开放编码→taxonomy) → 人工标注(gold) → 分层自动打分
     ↑                                                            │
     └──────────── 回归门禁 ← Judge 校准(TPR/TNR + few-shot) ←────┘
```

## 架构

```
┌─────────────────────────────┐        ┌──────────────────────────┐
│   被测 Agent（黑盒）          │        │   Eval Platform (:8100)   │
│   PRODUCT_API_URL (:8001)   │◄───────│                          │
│                             │  HTTP  │  runner/    批量实验执行   │
│   评测专用隔离实例            │        │  graders/   L0/L1/L2 打分 │
│   (docker-compose.eval.yml, │        │  storage/   SQLite 存储   │
│    固定资源配额,消除基础设施   │        │  api/       REST + WS    │
│    噪声)                     │        │  cli.py     实验 CLI      │
└─────────────────────────────┘        └────────────┬─────────────┘
                                                    │
                                       ┌────────────▼─────────────┐
                                       │  标注前端 (frontend/)     │
                                       │  Trace 查看 / Pass-Fail   │
                                       │  标注 / 开放编码 / 实验对比 │
                                       └──────────────────────────┘
```

## 分层评测体系（L0 → L1 → L2）

便宜、确定性的检查先跑；昂贵、需要判断力的检查后跑。失败在哪一层，就归因到哪一层（`graders/failure_funnel.py`）。

| 层 | 模块 | 检查内容 | 成本 |
|----|------|---------|------|
| L0 | `l0_structure.py` / `l0_constraints.py` | 输出结构完整性、硬约束（预算、必备条件）是否满足 | 免费，纯规则 |
| L1 | `l1_metrics.py` | 可计算指标（数量、覆盖度、来源多样性等） | 免费，纯代码 |
| L2 | `l2_judge.py` / `llm_graders.py` | 需要判断力的维度：groundedness、actionability、rubric 合规、trap 检测 | LLM 调用 |

配套：`error_taxonomy.py`（失效模式分类）、`composite.py`（加权合成总分）、`scoring.py`（阈值判定）。

## Judge 校准闭环（核心差异点）

**未校准的 judge 不可信任。** 平台把校准做成了可重复的工作流：

1. `scripts/sample_for_annotation.py` — 分层采样待标注 trace
2. `ANNOTATION_GUIDE.md` + 标注前端 — 人工标注 gold labels（Pass/Fail 判据显式成文，标注人是人不是模型）
3. `scripts/validate_evaluator.py` — judge 判定 vs 人工 gold，输出 TPR/TNR 对齐报告
4. `scripts/extract_few_shots.py` — 从分歧样本提炼 few-shot 示例（`calibration/few_shots/`）
5. `scripts/calibration_loop.py` — 注入 few-shots 重跑，迭代直到对齐达标

工程保证：

- **Judge 必须强于被测模型**（`config.py`：GRADING_MODEL 默认取更强模型——弱模型抓不住强模型级别的错误，同级模型会共享盲区）
- **Judge prompt 版本哈希**（`JUDGE_PROMPT_VERSION`）——每次实验记录 judge 版本，分数变化可归因："是 Agent 变了，还是尺子变了？"
- **通过阈值由人工标注校准得出**，并随标注量增长定期重校准，不拍脑袋

## 使用指南

按使用旅程组织。所有命令在仓库根目录执行，评测后端需已启动（见 Quickstart）。

### 1. 定义数据集

数据集是一个 JSON 文件（真实示例见 `datasets/`）：

```json
{
  "name": "My Agent Baseline V1",
  "description": "First regression suite for my agent",
  "cases": [
    {
      "key": "noise_cancel_budget",
      "query": "best noise cancelling headphones under $400 with 30 hour battery",
      "type": "constrained",
      "constraints": {"price_max": 400, "must_have": ["noise cancelling", "30+ hour battery"]}
    }
  ]
}
```

- `key` 在数据集内唯一；`type` 是你自定义的 case 分类（后续按类型分析）；`constraints` 供 L0 约束 grader 消费；有参考答案的 case 可加 `golden_data`
- 导入：把文件放进 `datasets/` 后执行 `python cli.py import --dataset <文件名>`（不带 `.json`）
- 数据集带版本号：case 变更自动 +1，每次实验记录所用版本，结果可追溯

### 2. 跑实验、读结果

```bash
python cli.py run --dataset "My Agent Baseline V1" --tag v1.0 --trials 2 --concurrency 2
python cli.py list                                # 历史实验
python cli.py summary --experiment-id 1           # 汇总：分数、通过情况、各 grader 表现
python cli.py regression --latest --threshold 5   # 最近两次对比，掉分超阈值 → exit 1
```

实时进度看 UI 实验页（WebSocket 推送）。每条 trace 记录 prompt 版本、模型、judge prompt 版本哈希——分数变化可归因到"Agent 变了"还是"尺子变了"。

### 3. 人工标注（建立 gold labels）

```bash
python scripts/sample_for_annotation.py --experiment-id 1 --count 50   # 采样（--seed 可复现）
```

然后在 `frontend/` 的 Trace 页（`/traces/:id`）逐条判 Pass/Fail、打开放编码（open codes）。判据写在 `ANNOTATION_GUIDE.md`——换领域先重写它。标注落库到 `human_pass` / `human_scores` 字段。

### 4. 校准 judge（拿到可信的 TPR/TNR）

```bash
python scripts/calibration_loop.py --create-splits        # train/dev/test 划分（一次性）
python scripts/validate_evaluator.py --per-grader         # judge vs 人工 gold 对齐报告
python scripts/extract_few_shots.py                       # 从 train 分歧样本提炼 few-shots
python scripts/calibration_loop.py --run-dev --notes "v2 few-shots"   # dev 集重打分
# …迭代到对齐达标后…
python scripts/calibration_loop.py --run-test             # test 集一次性终评（防过拟合）
python scripts/calibration_loop.py --history              # 校准历史
```

### 5. 接入 CI

```bash
EVAL_DATASET="My Agent Baseline V1" scripts/run_eval.sh regression --threshold 5
```

模式：`smoke`（2 case 冒烟）/ `regression`（回归门禁）/ `capability`（全量 ×N trials）。退出码 0=通过、1=回归、2=平台或被测系统不可达，可直接做流水线门禁。支持消融实验（选择性关闭 Agent 组件，定位能力来源）。

### 6. 导入生产 trace

```bash
python scripts/export_langfuse.py --days 30 --limit 200 --tag my-agent --grade
```

需要 `.env` 中的 LangFuse 密钥；`--grade` 表示导入后立即打分。脚本用的是 langfuse 2.x 的 `fetch_traces` 接口，`requirements.txt` 已锁 `<3`。

### 环境变量参考

| 变量 | 作用 | 默认 |
|---|---|---|
| `PRODUCT_API_URL` | 被测 Agent 的 API 地址 | `http://localhost:8001` |
| `EVAL_API_KEY` | 与被测系统的共享密钥 | 空 |
| `ANTHROPIC_API_KEY` | L2 judge 调用所需 | 空 |
| `GRADING_MODEL` | judge 模型（必须强于被测模型） | `openai/gpt-5.4`（需经网关） |
| `ANTHROPIC_BASE_URL` | Anthropic 兼容网关（如 OpenRouter）。默认 judge 模型只能经网关访问；用纯 Anthropic key 时把 `GRADING_MODEL` 改成 Claude 模型 | 空 |
| `ANTHROPIC_AUTH_TOKEN` | 网关要求时替代 `ANTHROPIC_API_KEY` | 空 |
| `GRADING_FALLBACK_MODEL` | judge 主模型失败时的备用模型 | `minimax/minimax-m2.7-highspeed` |
| `USER_SIMULATOR_MODEL` | 替用户回答被测 Agent 追问的模型 | `minimax/minimax-m2.7` |
| `JUDGE_ENABLED` | 是否启用 L2 judge | 关 |
| `JUDGE_PRESET` | judge 是否使用 Claude Code 全工具预设（更慢，可联网核实） | 关 |
| `PASS_THRESHOLD` | 通过阈值（应由校准得出，勿拍脑袋） | `60` |
| `PRODUCT_CODE_PATH` | 被测 Agent 源码路径，preset judge 深检用（可选） | 空 |
| `LANGFUSE_SECRET_KEY` / `LANGFUSE_PUBLIC_KEY` | 生产 trace 导入 | 空 |
| `EVAL_DATASET` | `run_eval.sh` 使用的数据集名 | `Legacy Baseline V1` |
| `EVAL_CORS_ORIGINS` | 评测后端 CORS 放行来源 | `localhost:5200` |

## 接入你自己的 Agent

领域耦合集中在三层，替换它们即可接入新 Agent：

1. **端点**：让你的 Agent 暴露一个任务 API，设置 `PRODUCT_API_URL` + `EVAL_API_KEY`（`backend/.env.example`）
2. **数据集**：按 `datasets/*.json` 的 schema 编写 cases（`key` / `query` / `type` / `constraints`，可选 `golden_data`）
3. **标注指南 + graders**：参照 `ANNOTATION_GUIDE.md` 写出你的领域 Pass/Fail 判据；替换/增补 L0 约束检查和 L2 judge 维度

平台层（runner / storage / 标注前端 / 校准脚本 / CLI）与领域无关，无需改动。

## Quickstart

前置要求：**Python 3.10+**（3.11 实测验证）、**Node 18+**。L2 judge 通过 Claude Agent SDK 启动 Claude Code 子进程，`requirements.txt` 锁定的 SDK 版本自带 CLI，不用单独安装；只做 import / UI 浏览 / 人工标注则连 judge 也不需要。

**1. 启动评测后端（端口 8100）**

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend
cp .env.example .env    # 至少填 PRODUCT_API_URL；要用 L2 judge 再填 ANTHROPIC_API_KEY
python main.py          # 起来后 http://localhost:8100/docs 可见全部 API
```

**2. 导入示例数据集、跑通 CLI（这一步不需要任何 API key）**

```bash
python cli.py import --dataset legacy_v1    # 在仓库根目录执行，导入 20 个示例 case
python cli.py list
```

**3. 启动标注/实验 UI**

```bash
cd frontend
npm install && npm run dev                  # http://localhost:5200，/api 与 /ws 已代理到 8100
```

`frontend-v2/` 是新版 UI 的静态原型，没有接任何接口，不能用来标注（见其 README）。

**4. 跑一个实验（这一步需要被测 Agent 在线）**

```bash
python cli.py run --dataset "Legacy Baseline V1" --tag first-run
python cli.py summary --experiment-id 1
python cli.py regression --latest --threshold 5
```

依赖边界：`run` 需要 `PRODUCT_API_URL` 指向一个真实运行的被测 Agent；L2 judge 打分需要 `ANTHROPIC_API_KEY`。两者都没有时，import / list / UI 浏览 / 人工标注仍然全部可用。被测系统的隔离实例部署示例见 `examples/azora/docker-compose.eval.yml`。

## 目录结构

```
backend/          # FastAPI 评测后端：api/ graders/ runner/ storage/ agent/
frontend/         # 标注与实验 UI（可用：Pass/Fail 标注、开放编码、失效漏斗、分数趋势、实验对比）
frontend-v2/      # 新版 UI 静态原型（未接后端，见其 README）
calibration/      # Judge 校准 few-shots（groundedness / actionability / rubric / trap）
datasets/         # 数据集（AZORA 案例：deepshop_v1 / legacy_v1）
scripts/          # 采样、标注、校准、回归、LangFuse 导出等工作流脚本
examples/azora/   # 第一个接入案例的部署配置
l2-codex/         # 跨厂商 L2 judge 实验（Codex SDK）
cli.py            # 实验 CLI
ANNOTATION_GUIDE.md  # 标注指南（AZORA 案例实例，可作为你的领域模板）
```

## 现状与边界

- 平台架构与被测系统解耦（HTTP 黑盒），但**第一案例是购物研究场景**：部分 grader（如 `search_quality.py`、`retrieval_grader.py`）和数据集带有该领域印记，接入新领域时按上文三层替换。
- 校准数值（阈值、TPR/TNR）以你自己数据上 `validate_evaluator.py` 的输出为准——不要沿用案例中的数字。
- 生产 trace 导入目前对接 LangFuse（`scripts/export_langfuse.py`）。

## 方法论参考

评测方法论对齐 [Hamel Husain 的 evals 体系](https://hamel.dev/blog/posts/evals/)：错误分析先行、人工 gold 为锚、judge 必须校准、指标服务于迭代而非汇报。

与 [openai/evals](https://github.com/openai/evals) 定位互补：它是**基准 registry + 模板化跑分**（YAML 注册 + JSONL 数据 + `oaieval` CLI），适合横向比较模型能力，其 model-graded judge 为 zero-shot、校准手段只有 meta-eval 一致率（简单 accuracy，无 TPR/TNR、无数据划分）；本平台是**单一 Agent 产品的迭代评测系统**——judge 校准做到 TPR/TNR + train/dev/test 划分 + few-shot 迭代，并自带标注 UI、回归门禁与生产 trace 回流。
