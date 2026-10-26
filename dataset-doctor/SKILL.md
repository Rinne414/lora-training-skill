---
name: dataset-doctor
description: "Audit and repair a kohya-style LoRA dataset before training. Use when the user asks to check, validate, inspect, fix, clean, 体检, or 修 a dataset, captions, or tags. Inspect image count, repeats and step budget, resolution and aspect buckets, duplicates, corrupt or non-RGB images, missing or empty captions, trigger consistency, and tag-frequency hygiene. Return a structural PASS, WARN, or FAIL report and apply only confirmed fix_dataset.py actions, dry-run first with displaced originals quarantined."
---

# dataset-doctor — LoRA 数据集 / caption 体检 + 修复

在训练前对 LoRA 数据集与标注做**结构/卫生预检**，给出 **PASS / WARN / FAIL** 结论和按优先级排序的修复建议；PASS 不代表图像与 caption 的语义已经人工验证。确认后用 `fix_dataset.py` 一行命令执行修复。**体检只读**；修复**默认 dry-run**，加 `--apply` 才动文件，且**从不删除**——被移除的文件全部进数据集内的 `_quarantine/`（体检与训练都会忽略它），随时可还原。任何 `--apply` 都必须先得到用户确认。

底层针对 `lora-scripts-next`（SD-Trainer）的数据集约定：`train_data_dir` 下放 `<repeats>_<concept>` 子文件夹（如 `7_zkz` = 重复 7 次、概念 `zkz`），caption 是与图同名的 sidecar `.txt`——**单行**、逗号分隔、可在末尾用「`. `」接自然语言（已验证格式见 `../references/caption-guide.md`）。Anima `.json` caption 仅是 UI 宣称的可选项，训练端 v2.7.0 未发现读取实现，不要依赖。

## 默认职责

接收一个数据集路径，运行体检脚本，解读 JSON/报告，向用户说明问题与修复顺序；用户确认后用 `fix_dataset.py` 执行修复并重检。trainer 本身**只检查目录是否存在、有没有图**，不检查质量——质量这块由本 skill 负责，二者互补。

## 适用范围（不限 SD-Trainer）

体检与修复是**训练器无关**的：检查对象是标准 kohya 数据集约定（`<repeats>_<concept>` 文件夹 + 同名 sidecar caption），sd-scripts / kohya_ss / OneTrainer 等训练器通用；脚本全程离线、不碰 GPU、不需要 trainer 在线。用户打算在**任何**训练器上练 LoRA，体检都照做。只有两处例外：

- **WD14 打标**：Anima 使用 `../lora-pipeline/scripts/tag_dataset.py`，与完整 pipeline 保持一致；SD1.5/SDXL/Flux 可走 SD-Trainer 的 `POST /api/interrogate`。
- **Anima `.json` caption 校验**只对已经持有 `.json` 标注的数据集有意义，且 `.json` 是否真被训练端读取**未经验证**（v2.7.0 本地代码中无读取实现）——doctor 默认要求 `.txt`；仅在明确使用会读取 JSON 的其他训练器时传 `--prefer-json`。

## 触发与分支

有「训练前检查数据集 / caption / tag 质量」意图时触发：体检、检查数据集、看看标注、tag 有没有问题、能不能开训、trigger 一致吗、有没有重复图、步数够不够。

- 只是要**自动打标**（还没标注）→ Anima 走 `tag_dataset.py`;其他底模走 `/api/interrogate`;打完再回来体检。
- 要**开始训练 / 调参** → 交给 `lora-trainer`；它会在开训前调用本 skill 作为闸门。
- 要查 caption 怎么写 / trigger 怎么取 → 读 `../references/caption-guide.md`。

## 读取导航

| 需要处理的事 | 读取 / 执行 |
| --- | --- |
| 跑一次完整体检（默认） | `scripts/doctor.py`（见「运行体检」） |
| 修复体检发现的问题（dry-run → 确认 → `--apply`） | `scripts/fix_dataset.py`（见「修复手册」） |
| 只看图像层面问题 | `scripts/scan_dataset.py` |
| 只看 caption / tag 问题 | `scripts/check_captions.py` |
| caption 格式、trigger、JSON 结构 | `../references/caption-guide.md` |
| 打标 / 重打标 API | `../references/trainer-api.md` |
| 训练参数与步数预算 | `../references/anima-params.md` |

## 运行体检

脚本依赖 **Pillow**（纯 stdlib + Pillow，无需 numpy）。用一个装了 Pillow 的 Python 运行——trainer 自带的 embedded Python 最稳妥：

- 本机示例：`C:\SD-Trainer\python_embeded\python.exe`
- 通用：任何 `python`（缺 Pillow 时 `pip install pillow`）

主入口 `doctor.py`（同时跑 scan + caption 并给总结论）：

```powershell
& "<PYTHON>" "<SKILL_DIR>/scripts/doctor.py" "<TRAIN_DATA_DIR>" `
    --trigger <TRIGGER> --epochs <N> --batch-size <B> --target-reso 1024,1024 --json
```

- `<TRAIN_DATA_DIR>`：`<repeats>_<concept>` 的**父目录**（也可直接指向某个概念子文件夹）。
- `--trigger`：预期的触发词；不给则脚本会**推断**一个候选。
- `--epochs/--batch-size`：用于算总步数预算（可选但推荐）。
- 输出：`--json`（机器读，含 `verdict` / `summary` / `recommendations` / `scan` / `captions`）、`--report`（人读 Markdown）、默认两者都打印。
- 退出码：PASS/WARN = 0，FAIL = 2。

需要分别看时用 `scan_dataset.py` / `check_captions.py`，参数同名。

## 解读结论

- **FAIL（critical）**：有图无法解码（会让训练崩）、根本没有图。**必须先修，禁止开训。**
- **WARN（high/medium）**：缺 caption、**多行 caption（训练端只读第一行）**、trigger 不一致、重复图、非 RGB、低于训练分辨率、过/欠标、artifact tag、JSON 结构错、**人数标签自相矛盾**（`1girl` 与 `2girls`、`solo` 与 `multiple girls`）、**过期的训练缓存**（改过的图 / caption 仍会按旧缓存训练）。**可训，但先把建议过一遍并向用户确认。**
- **PASS（low/info 或无）**：可以训。

按 `recommendations`（已按严重度排序、去重）逐条处理。`scan.issues` / `captions.issues` 里每条都带 `code`、`message`、`fix`、`items`（示例文件，最多 12 个）。

## 修复手册（对应 issue code）

`fix_dataset.py` 与 `doctor.py` 用同一个 Python 运行。**所有命令默认 dry-run**（只打印计划），把计划摊给用户、确认后**同一条命令加 `--apply`** 执行。被移走的文件进 `<dataset>/_quarantine/`，不删除。

```powershell
& "<PYTHON>" "<SKILL_DIR>/scripts/fix_dataset.py" <command> "<TRAIN_DATA_DIR>" [选项] [--apply]
```

| code | 修复命令 |
| --- | --- |
| `no_concept_folders` | `fix_dataset.py organize <dir> --repeats <R> --concept <name>` |
| `corrupt_images` | `fix_dataset.py quarantine-corrupt <dir>` |
| `exact_duplicates` | `fix_dataset.py dedupe <dir>`（保留分辨率最高的一张） |
| `near_duplicates` | `fix_dataset.py dedupe <dir> --near` |
| `non_rgb_mode` | `fix_dataset.py to-rgb <dir>`（原件备份进 `_quarantine/`） |
| `trigger_inconsistent` | `fix_dataset.py add-trigger <dir> --trigger <T>`（.txt 按 Anima 分段顺序插入；.json 写 `character`） |
| `artifact_tags` / `duplicate_tags` | `fix_dataset.py strip-tags <dir>`（移除 watermark/signature 等来源噪声;准确的逐图 quality 标签不是 artifact） |
| `multiline_caption` | 训练端只读 `.txt` 第一行，后续行**静默丢失**。手动把每个 caption 合并成单行：tag 在前、`. ` 接自然语言（格式见 `../references/caption-guide.md`） |
| `ubiquitous_tags` | 先核对哪个是有意的 trigger，其余用 `strip-tags --tags "..."` 删 |
| `missing_captions` / `empty_captions` | Anima：`tag_dataset.py --dataset-dir <concept-dir> ...`;其他底模：`POST /api/interrogate` + `additional_tags=<TRIGGER>` |
| `below_target_resolution` / `tiny_images` | 换更高分辨率源；`bucket_no_upscale` 下小图细节受限（无自动修复） |
| `subject_count_conflict` | `fix_dataset.py fix-subject-count <dir>`：同类人数标签留较大的（`1girl`+`2girls` → `2girls`；`1girl`+`multiple girls` → `multiple girls`），多人图去掉 `solo`；`solo focus` 不动。caption 就地改写（没有备份），计划里逐条列出会删哪些 tag；改完文本编码器缓存会过期，见下一行 |
| `stale_cache` | `fix_dataset.py quarantine-cache <dir>`：只把过期 / 孤立的缓存文件移进 `_quarantine/`（kohya 的 `*.npz`、musubi 风格的 `*_<arch>.safetensors`（读文件头里的 `architecture` 确认是缓存）、`*_latent_cache/` 文件夹），图片和 caption 不动；训练器下次会重建。按文件修改时间比较：图片比缓存新 → 旧 latent；caption 比文本编码器缓存（`*_te`）新 → 旧 caption；找不到同名图 → 孤立。如果数据集是用不保留修改时间的方式复制来的，先看清单再确认 |
| `mixed_repeats` | 核对各概念「有效图片数」比例是否符合预期（无需修复） |

`--apply` 之前 `fix_dataset.py` 会先问 trainer（`GET /api/datasets/in-use`）这个文件夹是不是正被排队或运行中的训练使用：是 → 退出码 3 并打印 STOP，什么都不改；先问用户，同意后同一条命令加 `--allow-in-use`。dry-run 不问；trainer 没开或太旧只打一行提示、照常继续（`--trainer-url` 可改地址，默认 `http://127.0.0.1:28000`）。

打标 / 修复后，**重新跑一次 `doctor.py`** 确认问题已清。误修了想还原 → 把 `_quarantine/` 里的文件移回原位即可。

## 自检

1. 路径指向的是 `train_data_dir`（`<repeats>_<concept>` 的父目录），不是散图目录；散图会触发 `no_concept_folders`。
2. 用了带 Pillow 的 Python；脚本能正常输出 JSON。
3. 已知 trigger 时传了 `--trigger`，并核对 `trigger.presence_pct`；Anima trigger 按分段顺序出现，不强制首位。
4. 解读了 `verdict` 并按 `recommendations` 顺序给建议。
5. 任何 `fix_dataset.py --apply` / 打标操作**已先 dry-run 摊牌并取得用户确认**；从不手写删除命令，从不直接删文件（一律走 `_quarantine/`）。
6. 修复后重跑体检确认。
7. 改过图片或 caption 之后（包括 `fix-subject-count` / `strip-tags` / `add-trigger`）再开训，先看体检有没有 `stale_cache`；有就走 `quarantine-cache`，否则会按旧缓存训练。

## 可验证案例

用户：训练前帮我看看 `D:/data/mychar`（trigger 是 `mych4r`，打算 10 epoch）。

```powershell
& "C:\SD-Trainer\python_embeded\python.exe" `
  "<SKILL_DIR>/scripts/doctor.py" "D:/data/mychar" --trigger mych4r --epochs 10 --report
```

读 `verdict`：若 WARN 且 `missing_captions`+`exact_duplicates`，则：① Anima 用 `tag_dataset.py` 给缺标的图打标并生成语义审查；② `fix_dataset.py dedupe "D:/data/mychar"` dry-run 列出重复组 → 用户确认 → 加 `--apply`（被移除的进 `_quarantine/`）；③ 修完重跑体检；④ PASS 或用户明确接受剩余 WARN 后，把数据集 + 建议的 repeats/epochs 交给 `lora-trainer`。

## 执行交接

- 体检与修复在本 skill；**实际开训、组装 config、`POST /api/run`、监看 SSE log 交给 `lora-trainer`。**
- 打标执行细节（端点、字段、轮询）见 `../references/trainer-api.md`；caption 写法见 `../references/caption-guide.md`。
- 本 skill 不自行启动训练；修复一律 dry-run 先行、确认后 `--apply`、移除走 `_quarantine/` 不删除。
