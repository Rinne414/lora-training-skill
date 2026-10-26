---
name: lora-pipeline
description: "Build an Anima LoRA from a single-character or style name, from Danbooru collection through curation, section-ordered WD14 captions, dataset checks, confirmed training, fixed-seed checkpoint validation, and a staged Civitai draft. Use for requests such as make a LoRA of X, collect images and train a character LoRA, train and upload to Civitai, or build a dataset from scratch and publish. Delegate training to lora-trainer and structural checks to dataset-doctor; never train without confirmation and never auto-publish."
---

# lora-pipeline — 从「角色名字」到「Civitai 草稿」的端到端流水线

把「帮我做一个 X 的 LoRA」变成一条受控流水线:**收图 → 精修 → 打标/复查 → 体检 → 训练 → 验证 → 打包上传(停在 Draft)**。本 skill 是**总指挥**——真正的训练交给 [`lora-trainer`](../lora-trainer/SKILL.md),质量交给 [`dataset-doctor`](../dataset-doctor/SKILL.md),自己负责把两端(收集/精修/打标 和 验证/发布)接起来。默认 **Anima 优先**。

> **两道人类闸门,不可移动:**
> 1. **训练前必须出「确认卡」** 并取得明确同意(继承 `lora-trainer` 的规则)——训练慢、占显存、难回退。
> 2. **绝不自动发布。** 全流程可以自动到 **Draft(草稿)**为止;点 Publish 是公开行为、会被索引,**必须由用户亲自点**。没有用户当场明说「发布」,绝不传 `--publish`。

## 依赖的本地工具(环境相关)

这些是外部本地工具；表中路径/端口是惯例默认值，脚本都可用 flag/env 覆盖。契约细节见两份参考文档。

| 工具 | 用途 | 默认位置 |
|---|---|---|
| DanbooruDownload | 按 tag 下载图 | `C:\tools\DanbooruDownload`(env `DANBOORU_DL_DIR`) |
| sd-image-sorter | WD14 打标 FastAPI | [Rinne414/sd-image-sorter](https://github.com/Rinne414/sd-image-sorter),`:8487`,兼容版本见 `collect-and-tag.md` |
| SD-Trainer | 训练后端 mikazuki API | `http://127.0.0.1:28000`,`run_gui.bat` 启动 |
| ComfyUI | 验证出图 | `http://127.0.0.1:8188`(env `COMFY_URL`) |
| civitai-uploader | 填 Civitai 上传向导 | `<repo>\tools\civitai-uploader`(安装见其 README) |

`<PY>` = 任意带 Pillow 的 Python 3.10+(如 `<DanbooruDownload>\.venv\Scripts\python.exe`)。管线脚本在 `scripts/`,纯 stdlib + Pillow。

## 工作目录布局

一个 `--work <dir>` 项目根,各阶段写入子目录:

```
<work>/
  raw/                       collect.py    → danbooru 图 + .txt sidecar
  curation_manifest.json     curate.py     → keep/drop 名单
  dataset/<R>_<concept>/     build_dataset.py → 精修后的 RGB 图（kohya 结构）
  dataset/<R>_<concept>/*.txt tag_dataset.py  → WD14 Anima 分段 caption
  dataset/<R>_<concept>.tag-audit.json        → 特征修剪 + 二次语义审查
  output/<name>/             lora-trainer  → safetensors 快照
  validation/                validate.py   → ComfyUI 样图
  civitai_upload/            make_civitai_pack.py → MODEL_CARD + *.civitai.json + samples/
```

## 七个阶段(精确步骤)

```
0  输入:角色/概念「名字」(必需)。目标:单角色|画风(用户没说才问);单角色还要明确 1girl|1boy|1other。
   推导:danbooru tag(默认 name_(series))、trigger(小写去空格,常见词做变体 mychar→mych4r)、
        output_name = <concept>-anima-v1、work 目录。
1  COLLECT  scripts/collect.py --tag "<danbooru tag>" --work <work> --limit 200 --apply
            → 建议 raw/ ≥30 张；不足时提示扩大来源或检查多样性，但不以数量单独硬拦截。见 collect-and-tag.md。
2  CURATE   scripts/curate.py --work <work> --subject-tag <1girl|1boy|1other>
            画风改传 --style → 丢错主体/多人/漫画/参考图/太小/坏图
   BUILD    scripts/build_dataset.py --work <work> --concept <c>  → dataset/<R>_<c>/(自动定 repeats)
3  TAG      scripts/tag_dataset.py --dataset-dir <work>/dataset/<R>_<c> --trigger <t>
            --subject-tag <1girl|1boy|1other>;画风传 --style 且 trigger 必须以 @ 开头
            → Anima 正确打标:干净 tagger(eva02/swinv2,不用 pixai 单跑、绝不用 toriigate 当 tagger)
              → `content_mode=template` + `preset_id=anima_tags_only`(官方段落顺序、@artist、
              safety 词表 questionable→nsfw、单行、未评分图片不伪造质量标签)→ **caption 悖论修剪**:
              把 ≥0.9 出现率的不变身份特征加进 blacklist,并把候选与决定写入 tag-audit.json。
              重打标后再报告错主体/多人语义警告;只警告,由用户复查,不自动删除。
            BUILD 与 TAG 都会先问 trainer(`GET /api/datasets/in-use`)这个文件夹是不是正被排队或运行中的训练使用:
            是 → 退出码 3 并打印 STOP,先问用户;同意后同一条命令加 `--allow-in-use`。trainer 没开或太旧只打一行提示,照常继续。
4  DOCTOR   ../dataset-doctor/scripts/doctor.py "<work>/dataset" --trigger <t> --epochs 10 --report
            → 按 dataset-doctor 修复到 PASS(FAIL 挡下,WARN 让用户接受)。
5  TRAIN    交给 lora-trainer:以约 1500 步作首轮预算并核对 doctor 实际步数
            →【确认卡】→ POST /api/run → 监看 log。
            train_data_dir = "<work>/dataset";output_dir = "<work>/output/<name>"。
6  VALIDATE 把至少一个早期与最终快照拷进 ComfyUI/models/loras/ → scripts/validate.py
            --work <work> --lora <early> <final> --trigger <t> --target character --subject-tag <count>
            → 同 seed 比 baseline/各快照;身份或画风成立且姿势/内容仍可变才通过。画风用 --target style。
7  PUBLISH  scripts/make_civitai_pack.py --work <work> --concept <c> --trigger <t> --name <name>
                --lora-file <...safetensors> --display "<title>" --series "<series>"
            → civitai_upload/。一次性 login.bat 后:
              <civitai-uploader>\upload.bat "<work>\civitai_upload\<name>.civitai.json"
            → 填满 4 步向导、传文件+样图、**停在 Draft**。报告草稿 URL,让用户点 Publish。
```

阶段 1–4、6–7a 只动数据、可离线/本地;只有打标(2)、训练(5)、验证(6)、上传(7b)需要对应服务在线。任一服务连不上 ≠ 中止:先把能离线做的做完,再提示用户启动缺的服务(可建议在输入框 `! run.bat` / `! run_gui.bat` 直接在会话里起)。

## 委派导航

| 阶段 | 读取 / 执行 |
|---|---|
| 1–3 收集 / 精修 / 打标 的契约与坑 | [`../references/collect-and-tag.md`](../references/collect-and-tag.md) + `scripts/collect.py`·`curate.py`·`build_dataset.py`·`tag_dataset.py` |
| 4 体检 + 一行修复 | [`../dataset-doctor/SKILL.md`](../dataset-doctor/SKILL.md)(`doctor.py` / `fix_dataset.py`) |
| 5 选参 · 确认卡 · 开训 · 监看 | [`../lora-trainer/SKILL.md`](../lora-trainer/SKILL.md) + `../references/{trainer-api,anima-params,presets}.md` |
| caption 格式 / 触发词 / 官方推荐质量词 | [`../references/caption-guide.md`](../references/caption-guide.md) |
| 6–7 验证出图 + Civitai 上传的契约与坑 | [`../references/validate-and-publish.md`](../references/validate-and-publish.md) + `scripts/validate.py`·`make_civitai_pack.py` |

## 质量词

Anima 官方 tag 顺序把质量/元信息放**最前**。本流程针对 Anima base 的**出图/样图 prompt**
使用官方前缀 `masterpiece, best quality, score_7, safe`;Anima-Aesthetic 不需要 `score_*`。
训练 caption 可使用 human quality 或 `score_9` 到 `score_1`,但必须来自逐图评分;没评分就省略。
段落顺序、@artist、safety 词表、单行、以及角色的
caption 悖论修剪,全部由 `anima` preset + trait-candidates 处理——细节与「为什么」见
[`caption-guide.md`](../references/caption-guide.md) 的 *Anima tagging craft* 一节。

## 自检

1. 入口是「名字」时:已推导 danbooru tag / trigger / output_name / work,没让用户做这些机械推导。
2. 顺序是 **收集 → 精修 → 建库 → 打标 → 体检**(对**建好的** concept 文件夹打标,不是对 raw)。
3. 打标用 `content_mode="template"` + Anima preset;阈值绑定 tagger;未评分时 `quality_override=""`;已复查 tag-audit 的语义警告与 trait prune 决定。
4. 过了 `dataset-doctor` 闸门:FAIL 没放行,WARN 已让用户接受;触发词一致率达标。
5. **训练前出了确认卡并取得同意**;约 1500 步只是首轮预算,不是统一最佳值。
6. 发布前用同 seed 比过 baseline、早期与最终快照;base_model 用精确名 `Anima`。
7. **停在 Draft**:没有用户当场明说「发布」,绝不 `--publish`。报告了草稿 URL。

## 可验证案例(端到端)

用户:「帮我做一个 `my_character_(my_series)` 的角色 LoRA,用 Anima,收图打标训练完自动传 Civitai,我只点发布。」

1. 推导:tag=`my_character_(my_series)`、trigger=`mychar`、name=`mychar-anima-v1`、work=`D:/work/mychar`。
2. `collect.py --apply` → raw/ 126 张 → `curate.py --subject-tag 1girl` → `build_dataset.py --concept mychar` → `dataset/5_mychar/`。
3. `tag_dataset.py --dataset-dir ... --trigger mychar --subject-tag 1girl` → 复查 tag-audit → `doctor.py` → 修到 PASS/WARN 已接受。
4. 委派 `lora-trainer`:以约 1500 步作首轮预算 →【确认卡】→ 用户「确认」→ 开训 → 监看。
5. 拷贝早期与最终快照 → `validate.py --lora <early> <final> --target character --subject-tag 1girl` → 与 baseline 比较后选快照。
6. `make_civitai_pack.py ...` → `civitai_upload/` → `upload.bat *.civitai.json` → **Draft**(LoRA/Character、tags、base_model Anima、trigger mychar、文件、样图都已填)→ 报告草稿 URL,等用户点 Publish。

## 执行交接

- 本 skill 只做**编排**:串起收集/精修/打标 与 验证/发布,并守住训练确认与人工发布边界。
- 训练细节 → `lora-trainer`;数据质量 → `dataset-doctor`;工具契约与坑 → 两份 `references/*.md`。
