# Trainer API contract — lora-scripts-next / SD-Trainer (mikazuki)

Authoritative notes for driving the **lora-scripts-next** trainer (a.k.a.
*Next Trainer* / SD-Trainer, the `mikazuki` FastAPI backend) programmatically.
Verified against SD-Trainer **v2.7.0**.

> The trainer must be running. Launch it with `run_gui.bat` (portable) or
> `python gui.py`. The backend then serves on **`http://127.0.0.1:28000`** and
> all JSON routes live under the **`/api`** prefix
> (`mikazuki/app/application.py`: `app.include_router(api_router, prefix="/api")`).

Default ports (see `gui.py`):

| Service | Port | Purpose |
|---|---|---|
| GUI / API | `28000` | WebUI + `/api/*` |
| TensorBoard | `6006` | loss/LR curves |
| Train Monitor | `6008` | GPU stats, samples, logs dashboard |

---

## 1. Start a training run — `POST /api/run`

Body = a **flat JSON config object** (one big dict of sd-scripts/GUI keys).
The server (`mikazuki/app/api.py::create_toml_file`) will:

1. fix types + merge `*_custom` arg tables,
2. pop `model_train_type` (default `sd-lora`) and `gpu_ids`,
3. apply model-specific server defaults (Anima: `unet_lr` 5e-5, bf16, `attn_mode` auto, …),
4. validate `train_data_dir` exists & has images, and validate the model path,
5. write `config/autosave/<timestamp>.toml` (+ `-dataset.toml`, `-promopt.txt`),
6. launch `accelerate launch <trainer> --config_file <toml>` as a background task.

`model_train_type` → trainer script (`trainer_mapping`):

| `model_train_type` | script | notes |
|---|---|---|
| `anima-lora` / `sd3-lora` | `scripts/dev/anima_train_network.py` | **Anima LoRA (this suite's focus)** |
| `anima-finetune` | `scripts/dev/anima_train.py` | full DiT finetune (~24 GB) |
| `sd-lora` | `scripts/stable/train_network.py` | SD1.5 |
| `sdxl-lora` | `scripts/stable/sdxl_train_network.py` | SDXL |
| `flux-lora` | `scripts/dev/flux_train_network.py` | Flux |
| `sdxl-finetune` / `sd-dreambooth` | `sdxl_train.py` / `train_db.py` | finetune/dreambooth |
| `anima-lora-fast` | (plugin runtime) | optional Fast plugin; gated by install state |

Minimum useful Anima LoRA body (full key reference: `anima-params.md`):

```jsonc
{
  "model_train_type": "anima-lora",
  "lora_type": "lora",
  "pretrained_model_name_or_path": "./sd-models/anima/anima-base-v1.0.safetensors",
  "vae": "./sd-models/anima/qwen_image_vae.safetensors",
  "qwen3": "./sd-models/anima/qwen_3_06b_base.safetensors",
  "train_data_dir": "./train/myproject",          // parent of <repeats>_<concept>
  "resolution": "1024,1024",
  "output_name": "my-anima-lora",
  "output_dir": "./output/my-anima-lora",
  "max_train_epochs": 10,
  "save_every_n_epochs": 2,
  "network_dim": 32,
  "network_alpha": 16,
  "unet_lr": 2e-5,
  "optimizer_type": "AdamW8bit",
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "enable_preview": false
  // gpu_ids: ["0"]   // optional; sets CUDA_VISIBLE_DEVICES
}
```

The server itself defaults Anima LoRA to rank 16 / `5e-5`; this example deliberately
overrides it with the Anima model-card starting point, rank 32 / `2e-5`. See
`presets.md` for complete suite configurations.

**Response** (`mikazuki/process.py::run_train`) — note the ready-made stream URL:

```jsonc
{
  "status": "success",
  "message": "Training started / 训练开始 ID: <task_id>",
  "data": {
    "task_id": "<uuid>",
    "train_log_path": "/train-log",
    "train_log_stream": "/api/train/log/stream/<task_id>",
    "train_log_url": "http://127.0.0.1:28000/train-log?task_id=<task_id>",
    "train_log_stream_url": "http://127.0.0.1:28000/api/train/log/stream/<task_id>"
  }
}
```

Failure path returns `{"status": "fail", "message": "..."}` (e.g. bad data dir,
missing model). Always check `status`.

PowerShell example:

```powershell
$body = Get-Content config.json -Raw
$r = Invoke-RestMethod -Uri http://127.0.0.1:28000/api/run -Method Post `
       -ContentType 'application/json' -Body $body
$r.data.task_id
```

---

## 2. Live training logs (SSE) — `GET /api/train/log/stream/{task_id}`

Server-Sent Events; one JSON per event: `{"text": "..."}` then `{"done": true}`.
Use the `train_log_stream_url` from the `/api/run` response directly.

Lightweight polling alternatives:

- `GET /api/train/log/tail/{task_id}?limit=240` → recent lines (`data.lines`, `data.done`).
- `GET /api/train/tasks` and `GET /api/tasks` → running/known tasks (`tm.dump()`).
- `GET /api/tasks/terminate/{task_id}` → stop a run.

> `task_id` is only valid for the **current** server session.

### 2b. Per-task loss, previews and config (newer trainers only)

Trainers that ship `mikazuki/utils/task_insights.py` (lora-scripts-next main; the
story-next fork from branch `feat/task-insights-api`) answer these. Older trainers
return `404 {"detail":"Not Found"}` for them; use the folder-scan fallback below.

| Route | Returns (`data`) |
|---|---|
| `GET /api/tasks/{task_id}/metrics` | `tags`: `loss/average`, `loss/current`, `loss/epoch_average`, `lr/unet` as `[{step, value}]` (about 500 points per tag, evenly thinned); `progress`: `{step, total_steps, epoch, total_epochs, percent}` parsed from stdout |
| `GET /api/tasks/{task_id}/previews` | `images`: `[{name, epoch, step, mtime, url, thumb_url}]`, oldest first; `preview_enabled` |
| `GET /api/tasks/{task_id}/previews/{filename}` | the image file; add `?thumb=1` for a 384px JPEG |
| `GET /api/tasks/{task_id}/config` | `config` (the run's TOML as JSON), `train_type`, `output_name` |

- Everything is filtered to that task's own run window (`started_at` .. `finished_at`
  in the task metadata), so a re-run with the **same `output_name`** and folders never shows the
  earlier run's images or curve.
- No data yet is HTTP **200** with an empty result plus `reason` / `reason_text`
  (`output_dir_missing`, `no_preview_in_window`, `previews_not_configured`,
  `logging_dir_missing`, `tensorboard_not_installed`, `no_run_in_window`, ...). Only an unknown
  `task_id` (restarted server, or pruned from the bounded finished-task history) is `404 "Unknown task_id..."`.
- `{filename}` must be a name from the `previews` list; anything else (`..`, separators,
  absolute paths) is a 404.

`lora-trainer/scripts/task_insights.py` wraps this: it asks these routes first and, when the
trainer is older or forgot the task, scans `<output_dir>/sample` and the TensorBoard
`logging_dir` instead (pass `--since <unix time of POST /api/run>` so an earlier same-name run
is left out). Its JSON output says which one it used (`source`) and why (`fallback_because`).

`lora-trainer/scripts/lora_base_info.py` (also run by `task_insights.py` when it is given `--output-dir`
and `--output-name`; its result is the `lora_base` key) reads the finished LoRA's safetensors header
and says which Anima it was trained for. Anima 2.9B inserts 12 blocks (at 2, 5, 8, 11, 14, 17, 21, 24,
27, 30, 33, 36) into the 28-block model, so the highest `blocks_<n>` in the tensor names decides:
only 0-27 means 28 blocks (2B / 1.0); any 28-39 means 40 blocks (2.9B). With `--base-model` it reads the
configured checkpoint's header (`net.blocks.<n>`) the same way and compares. A 28-block LoRA on a 2.9B
model needs a key remap (for example the ComfyUI-Anima-28to40-Lora-Stack node); a 40-block LoRA does not
load correctly on 2B. A native 40-block LoRA that never touches blocks 28-39 cannot be told apart from a
28-block one by its keys; the report says so.

`lora-trainer/scripts/failure_catalogue.py` reads `GET /api/train/log/tail/{task_id}?limit=2000` while a run
goes (`data.lines` is the last `limit` lines, `data.total` the count of all lines so far, `data.done`
the end flag; it keeps `total` in a state file to read only the new lines) and matches the known
failures in its rule table: CUDA OOM, NaN loss, missing model or caption files, 0 images, `num_repeats`
below 1, a folder without a repeat count, unsupported train type, paging file too small (`os error 1455`),
DLL load failed, disk full, port in use, xformers / flash-attn missing, no CUDA, host RAM, damaged
images. Each finding carries a one-line message and a concrete fix in English and Chinese; a category is
reported at most once per 60 seconds, and `final_findings` scans the tail once the run is done. The tail
route answers `404 "Unknown task_id"` after a trainer restart (reported as `task_unknown_to_trainer`).
Its rules match what the trainer prints, never CLI flag names or config echoes (`--nan_guard`,
`xformers = true`, `num_repeats: 1`); the console wraps long messages over many short lines, so the
matcher joins up to six lines. The same script scans any text: `--text "<message>"` for the failure
`message` of `POST /api/run` or a queue entry's `error`, `--file <log>` for a log file.

### 2c. Dataset in use — `409 dataset_in_use` and `GET /api/datasets/in-use` (newer trainers only)

Trainers that ship `mikazuki/utils/dataset_inuse.py` (the story-next fork, branch
`feat/dataset-inuse-guard`) refuse to change a dataset while a **queued or running** training
still reads it. Finished, failed and cancelled tasks do not count; a paused queue entry does not
count either (pausing is how a user frees a dataset for editing). Folders are matched as
absolute, symlink-resolved, case-insensitive (Windows) paths, subfolders included, over
`train_data_dir`, `reg_data_dir`, `source_image_dir`, `resized_image_dir` and the folders a
`dataset_config` TOML names; `data/cat` never matches `data/cats`.

Guarded writes: `POST /api/dataset-editor/caption`, `/batch`, `/undo`, `/redo` (only the folders
of the files actually touched) and `POST /api/interrogate` (its folder; with
`batch_input_recursive` also parents of a training folder). Reading is never blocked
(`/dataset-editor/scan`, `/image`, `/history`, `/api/tagger/status`, ...).

A refused write is **HTTP 409** and changes nothing:

```jsonc
{
  "status": "fail",
  "reason": "dataset_in_use",
  "operation": "caption_write",        // batch_edit | undo | redo | interrogate
  "message": "...",                    // = detail = reason_text, English / 中文
  "reason_text": "This dataset is used by a queued or running training task: \"night-lora\" (running). Captions changed mid-run are only partly picked up, and cached latents and text-encoder outputs will not see the change. ... / 该数据集正被排队或运行中的训练任务使用：...",
  "path": "D:/data/cat/5_cat",
  "tasks": [{"task_id": "...", "entry_id": null, "name": "night-lora", "output_name": "night-lora",
             "status": "running",   // queued | running | terminating
             "source": "task",      // task | queue (a queue entry that has not started yet)
             "matched_dir": "D:/data/cat", "relation": "inside"}],   // same | inside | contains
  "force_hint": "Repeat the same request with force=true (query or body field) to proceed."
}
```

Not a hard block: repeat the **same request** with `?force=true` (or `"force": true` in the JSON
body) and it goes through; the trainer logs `dataset-in-use override`. Only do that after the user
agreed. Older trainers never answer 409 here.

`GET /api/datasets/in-use?path=<folder>[&recursive=false]` answers without writing anything:
`data = {path, recursive, in_use, tasks[]}` (same `tasks` objects). `recursive` (default `true`)
also reports a task whose training folder sits *inside* `path`; with `false` only a task whose folder
contains `path` (or equals it) counts. A missing folder is fine (`in_use: false`); an empty `path` is 400.
An older trainer returns `404 {"detail":"Not Found"}`: treat that as "unknown", not "free".

`lora-pipeline/scripts/in_use_guard.py` wraps the check for scripts that rewrite a dataset folder
on disk (they bypass the trainer's guard): `build_dataset.py`, `tag_dataset.py` and
`dataset-doctor/scripts/fix_dataset.py --apply` call it first and exit **3** with a
`STOP:` message unless `--allow-in-use` is given (ask the user first). A trainer that is down or
too old only prints a note and the script carries on. Standalone: `python in_use_guard.py --path
"<folder>"` prints JSON and exits 3 when in use.

### 2d. Engines, queue and `/api/run` in newer trainers

The story-next fork (branches `feat/engine-state` and later) adds routes that the skill reads through
`lora-trainer/scripts/trainer_state.py`; an older trainer answers `404 {"detail":"Not Found"}` for all of
them and the script reports "older build, skipped" instead of failing.

| Route | Use |
|---|---|
| `GET /api/engines` | per train type: `state` (`ready`, `broken`, `not_installed`, `unknown`, ...), `reason_zh/en`, `hint_zh/en` |
| `POST /api/engines/diagnose` | body `{train_type, config}`; deeper `issues` per engine (also a standing Windows + NVIDIA warning `windows_sysmem_fallback`) and `config_issues` for model files that do not exist |
| `GET /api/queue` | `entries[]` (`status`, `task_id`, `interrupted`), `active`, `running_task_id`, and after a restart `restored = {queued, interrupted, waiting, message_zh, message_en}` |
| `POST /api/queue/start` | starts a restored queue. After a restart the trainer holds queued entries (`restored.waiting = true`) and never starts the GPU by itself: ask the user first |
| `POST /api/run` | now answers `{"queued": true, "entry_id": ...}` (no `task_id` yet; the entry gets one in `GET /api/queue` once it starts). If the name was taken it is renamed `name_2`, `name_3`, ... and the response carries `output_name_renamed = {from, to}`: use the new name for the LoRA file, previews and the result record |
| `POST /api/plugins/anima-lora/repair` | a **full reinstall** of the Anima Fast environment (can re-download gigabytes). The skill never calls it without asking the user, and `trainer_state.py` has no command for it |

Entries with `interrupted = true` were training when the trainer restarted. They are marked failed and are
never re-queued automatically; the skill lists them and leaves "Requeue" to the user.

---

## 3. Auto-caption (WD14 tagger) — `POST /api/interrogate`

Generates sidecar `.txt` captions for every image in a folder. Body =
`TaggerInterrogateRequest` (`mikazuki/app/models.py`):

| field | default | meaning |
|---|---|---|
| `path` | *(required)* | folder of images to tag |
| `interrogator_model` | `wd14-convnextv2-v2` | must be in `available_interrogators` |
| `threshold` | `0.35` | general-tag confidence cutoff |
| `character_threshold` | `0.6` | character-tag confidence cutoff |
| `additional_tags` | `""` | comma list prepended to every caption (good for a **trigger word**) |
| `exclude_tags` | `""` | comma list removed from every caption |
| `add_rating_tag` | `false` | append safe/sensitive/nsfw rating tag |
| `replace_underscore` | `true` | `long_hair` → `long hair` |
| `batch_input_recursive` | `false` | recurse into subfolders |
| `batch_output_action_on_conflict` | `ignore` | `ignore` / `copy` / `append` / `prepend` for existing captions |
| `force` | `false` | also accepted as `?force=true`; only needed after a `409 dataset_in_use` (section 2c) |

Returns immediately (`"打标任务已提交"`); it runs as a background task. Poll:

- `GET /api/tagger/status` → full progress snapshot.
- `GET /api/tagger/download-status` → model download phase.
- `POST /api/tagger/cancel`, `POST /api/tagger/reset`.
- `POST /api/tagger/prefetch` (`{interrogator_model}`) → pre-download a model.

`409`-style busy guard: only one tag/download job at a time
(`tagger_progress.is_busy()`).

PowerShell example for the trainer's generic tagger (prepend an additional tag):

```powershell
$body = @{ path = "D:/data/myproject/5_zkz"; additional_tags = "zkz"; threshold = 0.35 } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:28000/api/interrogate -Method Post `
  -ContentType 'application/json' -Body $body
```

> WD14 models live under `tagger-models/wd14/<model-key>/` (default
> `wd14-convnextv2-v2`, ~400 MB, bundled in the portable build).

---

## 4. Discovery / housekeeping endpoints

| Method · Route | Purpose |
|---|---|
| `GET /api/version` | trainer version |
| `GET /api/graphic_cards` | available GPUs (`data.cards`); `status:"pending"` until probed |
| `GET /api/presets` | preset templates (`config/presets/*.toml`) |
| `GET /api/schemas/all` | full parameter schema (per model type) |
| `GET /api/config/saved_params` | last-used params |
| `GET /api/get_files?pick_type=model-file\|model-saved-file\|train-dir` | list models / outputs / train dirs |

---

## 5. No-GUI / CLI path (advanced)

Standard Anima Kohya training is **meant to go through the WebUI/API**. For
headless flows the trainer also ships:

- `scripts/cli/train_by_toml.ps1` / `.sh` — legacy SD1.5/SDXL/Flux via a TOML.
- `scripts/cli/train_anima_fast_by_toml.bat` / `.sh` — the optional Anima **Fast** plugin.

These call `accelerate launch <trainer> --config_file <toml>` themselves. Prefer
`POST /api/run` when the GUI is up — it validates inputs and gives you a
`task_id` + live log stream for free.
