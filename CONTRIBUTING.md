# Contributing to AquaScope

Thanks for your interest! AquaScope is a personal project, but PRs and issues
are welcome — especially around the laptop / non-Jetson code path, since
that's the bit that benefits most from being battle-tested on hardware I
don't own.

## Quick start

```bash
git clone https://github.com/sandheepp/aquascope.git
cd aquascope
bash scripts/run_local.sh                          # webcam mode
bash scripts/run_local.sh --video your_clip.mp4    # demo on any video file
```

Open [http://localhost:8080](http://localhost:8080) once it boots.

## What to work on

Good first issues:

- **Platform fixes** for macOS / Windows. The Linux path is well exercised;
  the others are best-effort. If `run_local.sh` blows up on your machine,
  open an issue with the traceback.
- **Sample dataset / sample video** under [samples/](samples/) so the README
  flow works without a real aquarium.
- **Docs** — the [docs/](docs/) folder is intentionally short. Add concept
  explainers, screenshots, or walkthroughs.
- **Dashboard polish** — accessibility, mobile responsiveness, dark mode
  consistency.

Bigger swings (open an issue first to discuss):

- Distillation pipeline (offload bigger models on a desktop GPU into the
  Jetson engine).
- Multi-camera support.
- Alternative trackers (DeepSORT, etc.) behind the existing ByteTrack interface.

## Code style

- **No new dependencies** without a clear reason.
- **No comments** for things the code already says — only `# Why:` style
  notes when behavior is non-obvious.
- Keep functions small and named after what they do, not how they do it.
- Match the existing file headers (one-line module docstring, then a blank
  line) so `grep` over the codebase stays consistent.

## Submitting a PR

1. Branch off `main`.
2. Run `python -m py_compile app/*.py training/*.py` locally — the CI will
   do the same.
3. Open the PR against `main` with a one-paragraph "what + why".
4. If the change touches the dashboard, include a screenshot or short clip
   in the PR description.

## Reporting bugs

Please include:

- OS + Python version (`python --version`).
- Whether you're on a Jetson, Mac, or other Linux/Windows.
- The full traceback (paste, don't screenshot).
- The command you ran and what you expected vs. what happened.

Thanks!
