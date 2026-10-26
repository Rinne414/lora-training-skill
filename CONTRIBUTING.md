# Contributing

Contributions welcome. This is a small, opinionated repo — the guidelines below exist
mostly so a PR doesn't get bounced for something mechanical.

## Ground rules

**No machine-local paths.** Every path in committed code must be a placeholder
(`D:/data/mychar`, `C:\SD-Trainer`) or come from a flag or env var. If you need your
own paths, put them in `krea2-pipeline/scripts/env.bat` — it's gitignored.

**No personal data.** No real character/project names from your own work, no Civitai
model IDs, no account handles, no API keys or tokens. There are no secrets in this
repo and it should stay that way.

**Don't name models after real artists.** Applies to docs and examples too. Use
neutral, style-descriptive names.

## Running the tests

Install the contributor dependencies, then run all four `unittest` suites:

```powershell
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python -m unittest discover -s dataset-doctor/tests -v
python -m unittest discover -s lora-pipeline/tests -v
python -m unittest discover -s lora-trainer/tests -v
```

The tests are offline and need no GPU. They must pass before a PR.

Runtime scripts need Python 3.10+ and Pillow. Package validation also uses PyYAML;
`requirements-dev.txt` installs all contributor dependencies.

## Code conventions

- **Stdlib + Pillow only** for `dataset-doctor`. It has to run under the trainer's
  embedded Python without pip installs. New dependencies need a good reason.
- **Dry-run by default.** Anything that modifies a dataset prints a plan and does
  nothing until `--apply`. Removals move files to `_quarantine/` — never delete.
- **Small files.** Prefer splitting over growing a module past ~400 lines.
- **Handle errors explicitly.** No bare `except: pass`. A silent failure in a dataset
  tool means someone trains on broken data.

## Editing skills

`SKILL.md` files are instructions an agent executes, not prose for humans. When
changing one:

- Keep the two human gates intact — **never train without a confirm card, never
  auto-publish.** These are the point of the project, not friction to optimize away.
- Skills cross-reference `references/` by relative path. Keep the directory layout.
- If you change a script's flags, update the `SKILL.md` that calls it *and* `GUIDE.md`.

## Docs

- `docs/WALKTHROUGH.md` — the teaching path for newcomers, long-form.
- `GUIDE.md` — reference and agent contract, EN + 简体中文. If you change the English
  half, the Chinese half should follow; flag it in the PR if you can't.
- `references/*.md` — verified facts the skills cite. Don't add speculation here; if
  something is untested, say so explicitly.

## Pull requests

- One concern per PR.
- Conventional commit prefixes: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- Say what you actually tested. "Trained a 30-image character LoRA on a 12 GB card" is
  far more useful than "should work".
- Reporting a bug? Include GPU + VRAM, the trainer version, and the doctor's output.

## What's likely to be accepted

Bug fixes, new doctor checks, presets verified on real hardware, non-NVIDIA or Linux
support, better error messages, and doc corrections — especially if you hit the
problem yourself.

Large architectural rewrites are best discussed in an issue first.
