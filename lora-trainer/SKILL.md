---
name: lora-trainer
description: "Prepare and launch LoRA, LoKr, or T-LoRA training through lora-scripts-next (SD-Trainer), Anima-first. Use when the user asks to train a character, style, or concept LoRA, choose parameters or repeats and epochs, start a run, or monitor training on Anima, SD1.5, SDXL, or Flux. Organize and caption the dataset, run dataset-doctor, choose a documented starting preset from image count and VRAM, show one confirmation card, then call the mikazuki API and monitor its log."
---

# lora-trainer — LoRA 训练编排入口

把「训练一个 LoRA」的意图，变成一次受控的训练：准备数据 → 体检 → 组装配置 → **用户确认** → 通过 trainer API 开训 → 监看 log。本 skill 默认 **Anima 优先**，也支持 SD1.5 / SDXL / Flux。

> 训练耗时、占显存、难回退。**在 `POST /api/run` 之前必须把组装好的配置摊给用户、明确确认**；未确认不开训。

底层 = `lora-scripts-next`（SD-Trainer）的 `mikazuki` FastAPI（`http://127.0.0.1:28000`，路由前缀 `/api`）。完整契约见 `../references/trainer-api.md`。

## 适用范围（用户用的不是 SD-Trainer 时怎么办）

本 skill 的**开训与打标只针对 SD-Trainer** 的 mikazuki API——`trainer-api.md` 的契约只对它成立。**绝不要**把这套 `/api/run` body 发给别的训练器（kohya_ss GUI、OneTrainer、ai-toolkit…），它们的 API 完全不同。用户在用别的训练器时，按可移植性分层处理：

1. **体检照做**：`dataset-doctor` 与 `fix_dataset.py` 是训练器无关的（标准 kohya `<repeats>_<concept>` + sidecar caption 约定），完整可用，全程离线。
2. **参数知识可借用**：`anima-params.md` / `presets.md` 的键就是 kohya sd-scripts 词汇（`network_dim`、`unet_lr`、repeats/epochs 步数预算）。可以据此给参数建议，或替用户生成一份 kohya 风格 TOML / 命令行，让他们在自己的训练器里手动跑。
3. **不自动开训**：明确告知「自动开训只支持 SD-Trainer」，给两条路——装 SD-Trainer（见 README），或拿第 2 点生成的配置自己跑。

`/api/version` 连不上 ≠ 中止一切：数据整理、体检、修复全部离线可做；开训需要 trainer 在线，打标则按底模需要已配置的 sd-image-sorter 或 trainer 端点。

## 两条路径

- **快速通道（默认）**：用户只给**一个图片文件夹**。结构整理、打标、体检、修复、选参全部自动；用户只看一张「确认卡」。一般用户走这条。
- **进阶通道**：用户自己点名 preset / dim / LR / 优化器等参数时走「进阶流程」，参数知识见 `../references/`。

两条路都**必须**过 `dataset-doctor` 闸门，**必须**确认后才开训。

## 快速通道（只需要一个图片文件夹）

**0. 连 trainer、读显卡。** `GET /api/version` 确认在跑（连不上 → 让用户启动 `run_gui.bat`，可建议在输入框输入 `! run_gui.bat` 直接在会话里运行）。再 `GET /api/graphic_cards` 自动读显存——**不要问用户显存多大**。

**1. 收集最少输入。** 必需：图片文件夹路径、练**单角色/概念**还是**画风**。单角色再明确 `1girl` / `1boy` / `1other`;画风 trigger 必须用 `@` 前缀。其余自动推：

- **trigger**（角色/概念才需要）：从概念/文件夹名生成候选——小写、去空格；若是常见词，做数字变体避免撞概念（`mychar` → `mych4r`）。写进确认卡让用户改。
- **output_name** = `<concept>-anima-v1`；版本号冲突就 +1。

**2. 自动准备数据。** 每步先跑 dry-run 把计划摊给用户，**确认一次后**加 `--apply` 执行（fixer 不删文件，原件进 `_quarantine/`）：

- 散图、没有 `<repeats>_<concept>` 结构 → `fix_dataset.py organize --repeats <R> --concept <name>`（R 见第 3 步）。
- 没 caption且目标是 Anima → 调 `../lora-pipeline/scripts/tag_dataset.py --dataset-dir <concept-dir> --trigger <T> --subject-tag <count>`;画风传 `--style`。快速通道与完整 pipeline 必须使用同一套分段 caption、模型阈值和语义审查。其他底模才用 `POST /api/interrogate`。
- 跑 `doctor.py --trigger <T> --epochs <N>` 体检：FAIL/WARN 的每个 issue code 都有对应建议（见 `../dataset-doctor/SKILL.md` 修复手册）。把要做的修复**一次列全、统一确认**，再逐条执行；重检到 PASS，或让用户明确接受剩余 WARN。

**3. 自动选参。** 不要让用户做数学：

- `repeats = clamp(round(150 / 图片数), 1, 10)`，`epochs = 10`（画风 12）→ 约 1500 步仅作首轮预算。doctor 去重后重算实际步数，并通过固定 seed 快照比较决定是否早停。
- preset（`../references/presets.md`）：显存 ≥16GB → preset 1（角色）/ 2（画风）；≤12GB → preset 3（LoKr + `blocks_to_swap=16`；8GB 改 24）。

**4. 确认卡（强制）。** 用人话摊牌，等用户回「确认」：

```
📋 训练确认卡
- 练什么：角色 LoRA「zkz」 · trigger: zkz（要换就说）
- 数据：32 张图 × 5 重复 × 10 轮 = 1600 步
- 显卡：RTX 4070 12GB → 用省显存配置（LoKr）
- 输出：./output/zkz-anima-v1/，每 2 轮存一个文件
- 底模：Anima（dim32/alpha16 · bf16 · unet_lr 2e-5）
回复「确认」开训；想改哪一项直接说。
```

**5. 开训 + 监看。** 同进阶流程第 5–6 步。

## 进阶流程

**1. 收集输入。** `train_data_dir`（`<repeats>_<concept>` 的父目录）、trigger word、`output_name`、模型类型、目标（角色/画风/概念）。缺就问。

**2. 闸门：先体检。** 调 `dataset-doctor` 跑 `doctor.py --trigger <TRIGGER> --epochs <N> --batch-size <B>`：FAIL → **停**，先修（用 `fix_dataset.py` 对应命令）再重检；WARN → 摊给用户确认是否继续；PASS → 继续。

**3. 组装 config。** 从 `../references/presets.md` 选模板，填 `<TRAIN_DATA_DIR>` / `<OUTPUT_NAME>`，用 doctor 的 `effective_images` 估算总步数：

```
steps_per_epoch = ceil(effective_images / train_batch_size)
total_steps     = steps_per_epoch × max_train_epochs   # 角色首轮检查区间约 1000–2500
```

偏离区间就调 repeats/epochs。Anima 官方起点是 rank 32、`unet_lr=2e-5`;训练器服务端默认 `5e-5` 不是官方推荐。其余关键项：`mixed_precision=bf16`、`gradient_checkpointing=true`、`attn_mode` 留空自动、Windows `max_data_loader_n_workers=0`。caption 一律用单行 `.txt`;不要发送无效的 `prefer_json_caption`。

**4. 摊牌 + 确认（强制）。** 列出关键项（模型类型、底模、`train_data_dir`、`network_dim/alpha`、`unet_lr`、epochs、`total_steps`、`output_dir`、显存预估），**明确征得同意**再开训。

**5. 开训。** `POST /api/run`，body = 组装好的 flat config（JSON）。用 PowerShell `Invoke-RestMethod`（示例见 `trainer-api.md`）。返回 `status: fail` → 把 `message` 给用户并修正；`success` → 记下 `data.task_id` 与 `data.train_log_stream_url`。

**6. 监看。** 轮询 `GET /api/train/log/tail/{task_id}?limit=240`（或让用户开 `data.train_log_url` / 6008 监控面板）。盯：`loss=nan`（多见于 fp16 → 改 bf16 或换优化器）、OOM（加 `gradient_checkpointing` / `blocks_to_swap` / 降分辨率）、报错退出。

**读 loss 曲线和预览图。** 开训前记下发出 `POST /api/run` 的 unix 时间 `<LAUNCH_TS>`，监看时每隔几分钟（或每个 epoch）跑一次：

```
python scripts/task_insights.py --task-id <task_id> --out ./insights --output-dir <output_dir> --output-name <output_name> --logging-dir <logging_dir> --since <LAUNCH_TS> --base-model <pretrained_model_name_or_path>
```

- 新版 trainer 直接回答 `GET /api/tasks/{task_id}/metrics` 与 `/previews`（已按本次任务的起止时间过滤，同名重训不会混进上一次的图和曲线）。脚本优先用它们，输出里 `source` = `trainer_api`。
- 老版 trainer（这些路由返回 404 Not Found），或 trainer 重启后 task_id 失效：脚本自动退回扫描 `<output_dir>/sample` 和 TensorBoard 的 `logging_dir`，`source` = `folder_scan`，`fallback_because` 写明原因。退回扫描时**必须**带 `--since`，否则同名旧图和旧曲线会混进来。
- 看 `loss_summary`：`has_nan_or_inf=true` → 按上面的 nan 处理。loss 噪声大，只当辅助：`last` 明显高于 `min` 或后半段回升，只说明「值得试更早的快照」，不是定论。预览图落在 `--out` 目录，每次只看新增的几张。
- `loss_reason` / `previews_reason` 非空表示没拿到数据（没开 `enable_preview`、没装 tensorboard、还没到第一次出图……），照实转述，不要编数据。

## 训练完成后（别漏这步）

1. 产物在 `output_dir` 下：按 `save_every_n_epochs` 会有多个 epoch 快照（`<name>-000002.safetensors` …）+ 最终档。
   **每次训练结束都要报告 base**：`task_insights.py` 的输出里 `lora_base` 读的是产物 safetensors 文件头（只读头，不加载权重；也可单独跑 `python scripts/lora_base_info.py <lora> --base-model <pretrained_model_name_or_path>`）。按 key 里最大的 `blocks_<n>`：只有 0–27 → **28 块（Anima 2B / 1.0）**；出现 28–39 → **40 块（2.9B）**。同时读底模文件头的块数并对比，`compatible=false` 或 `warnings` 非空时把原话（中英文都有）转给用户：28 块 LoRA 放到 2.9B 上要先做 key 重映射（如 ComfyUI-Anima-28to40-Lora-Stack 节点），否则会作用到错误的层；40 块 LoRA 在 2B 上无法正确加载。`notes` 里的「只找到 0–27 块」说明按 key 布局无法区分「从不碰 28–39 块的原生 40 块 LoRA」，照实转述。最终报告里写明：产物是 28 块还是 40 块、配置的底模路径和它的块数。命名沿用 `<concept>-anima-v<N>`，该约定没有位置放块数标签，所以不改文件名，base 只写进 `lora_base`。读不了文件头（`error`）就如实说「未能确认」，不要猜。
2. **怎么挑**：复制至少一个早期和最终快照到 ComfyUI，用 `validate.py` 在相同 seed 下与 strength 0 baseline 比较。选择能表达目标、同时仍响应姿势/主体/背景变化的快照。trainer 自带的预览图和 loss（`scripts/task_insights.py`）只做粗筛辅助，不替代 `validate.py` 的同 seed 对比。
3. 身份弱时先比较权重 0.8–1.1，再决定是否加练；僵脸、固定姿势或内容污染时用更早快照。

## 模型与适配器分支

**模型类型**（决定 `model_train_type` 与底模，见 `trainer-api.md` 映射表）：默认 → **Anima**（`anima-lora`，`networks.lora_anima`）；用户点名 SDXL / SD1.5 / Flux → `sdxl-lora` / `sd-lora` / `flux-lora`；全量微调 → `anima-finetune` / `sdxl-finetune`（Anima ~24GB，先提醒）。

**适配器** `lora_type`（见 `anima-params.md`）：默认 `lora`；显存紧 / 小文件强风格 → `lokr`；timestep-aware → `tlora`（`network_dim=32`）。

## 读取导航

| 需要处理的事 | 读取 / 执行 |
| --- | --- |
| 开训前体检数据集 / caption | `../dataset-doctor/scripts/doctor.py` |
| 一行命令修数据集问题（dry-run → --apply） | `../dataset-doctor/scripts/fix_dataset.py` |
| API 端点、`/api/run` body、SSE、打标、显卡查询 | `../references/trainer-api.md` |
| 读本次训练的 loss 曲线 / 预览图（优先 trainer API，老版自动退回扫描文件夹） | `scripts/task_insights.py` |
| 判断产物 LoRA 是 28 块（2B / 1.0）还是 40 块（2.9B），并对比底模 | `scripts/lora_base_info.py`（或 `task_insights.py` 输出里的 `lora_base`） |
| 参数含义、默认值、VRAM、步数预算 | `../references/anima-params.md` |
| 起手配置模板（角色/画风/低显存） | `../references/presets.md` |
| caption 写法、trigger 取法 | `../references/caption-guide.md` |

## 自检

1. 目标训练器是 SD-Trainer；不是的话**没有**把 `/api/run` 契约发给别的训练器，而是走「适用范围」的分层处理。
2. trainer 已在 28000 响应（`/api/version`）；显存来自 `/api/graphic_cards`，没问用户。
3. 已跑 `dataset-doctor`；FAIL 没放行，WARN 已让用户确认。
4. 所有 `fix_dataset.py` 修复都先 dry-run、经用户确认才 `--apply`；没有手写删除/改写命令。
5. `model_train_type` 与底模 / `network_module` 一致（Anima=`networks.lora_anima`，LoKr=`lycoris.kohya`，T-LoRA=`networks.tlora_anima`）。
6. 角色训练 trigger 一致率达标；默认 `cache_text_encoder_outputs=true`、`shuffle_caption=false`。若显式开启 shuffle，必须用足够的 `keep_tokens` 或 `keep_tokens_separator` 保留完整固定前缀，而不是假设 trigger 在首位。
7. 估算了 `total_steps`；~1000–2500 只作首轮检查区间，偏离时已说明依据而非硬拦截。
8. **已出确认卡并取得开训确认**；未确认不 POST。
9. Anima 用 `bf16`；`Automagic`/`CAME` 会被服务端强制 bf16。
10. 读 loss / 预览用 `task_insights.py`；退回文件夹扫描时带了 `--since`，并如实转述 `loss_reason` / `previews_reason`。

## 可验证案例（快速通道）

用户：「用 `D:/data/mychar` 这堆图练个角色 LoRA」（文件夹里是 30 张散图，没 caption）。

1. `/api/version` OK；`/api/graphic_cards` → 4070 12GB。
2. 唯一一问：已说是角色 → 不再问。trigger 候选 `mych4r`（mychar 是常见词，做数字变体）。
3. 准备数据：`fix_dataset.py organize "D:/data/mychar" --repeats 5 --concept mych4r`（dry-run 摊牌 → 确认 → `--apply`）→ `tag_dataset.py --dataset-dir ... --trigger mych4r --subject-tag 1girl` → 复查语义审查 → `doctor.py --trigger mych4r --epochs 10` → PASS 或剩余 WARN 已接受。
4. 自动选参：30 张 → repeats=5、epochs=10 → 约 1500 步首轮预算；12GB → preset 3。
5. 确认卡 → 用户回「确认」→ `POST /api/run` → 监看 → 报告 `./output/mych4r-anima-v1/*.safetensors`，并提示先试最后一个快照。

## 执行交接

- 数据集 / caption 体检与修复 → `dataset-doctor`（体检 `doctor.py`，修复 `fix_dataset.py`）。
- 端点 / 请求体 / SSE / 打标细节 → `../references/trainer-api.md`；参数 → `../references/anima-params.md`；模板 → `../references/presets.md`；caption → `../references/caption-guide.md`。
- 本 skill 负责准备、组装与（确认后）开训 + 监看；不绕过体检闸门，不在未确认时启动训练或改动用户文件。
