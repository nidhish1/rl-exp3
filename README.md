# Exercise 3: Panda box pushing with MPPI

Complete Python implementation for the three MuJoCo Panda pushing tasks. The controller uses parallel MuJoCo rollouts, seven joint-position controls, and four linearly interpolated control knots.

## MP4 videos

| Task | Successful example | Failure example |
| --- | --- | --- |
| A: goal position | [Play/download MP4](videos/taskA_seed00.mp4) | — |
| B: goal position and yaw | [Play/download MP4](videos/taskB_seed00.mp4) | [Yaw outside tolerance](videos/taskB_seed01.mp4) |
| C: goal with an obstacle | [Play/download MP4](videos/taskC_seed02.mp4) | [Stalled route](videos/taskC_seed00.mp4) |

The five videos are actual simulation recordings at 1280 × 720. Playback follows simulation time, with an additional one-second final hold. Use GitHub's download button if inline playback is unavailable.

## Setup

The supplied scene and robot meshes are required to run the code. Copy the complete `assets/` directory from the assignment's original `HW4_exercise 3(1).zip` into this repository, beside `mppi_push.py`. Preserve its directory structure and included license. The expected scene path is `assets/push_scene.xml`. Model assets are not included in this upload.

Tested on macOS ARM64 with Python 3.14 and the versions pinned in `requirements.txt`.

```bash
git clone https://github.com/nidhish1/rl-exp3.git
cd rl-exp3
# Copy the supplied assets/ folder here before running the simulation.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Evaluate all three tasks, ten episodes per task.
python evaluate.py --task all --runs 10

# Or evaluate one task.
python evaluate.py --task A --runs 10
```

The runner saves trajectories and metrics in `results/` and resumes completed episodes for the same configuration. For a changed controller or parameters, choose a fresh directory with `--output results_new`.

For an interactive viewer on macOS:

```bash
.venv/bin/mjpython mppi_push.py --task A --view
```

This runs the task benchmark first, then replays episode 0. On other supported desktop platforms use `python` instead of `mjpython`. Rendering requires graphics access.

## Included code

| File | Purpose |
| --- | --- |
| `mppi_push.py` | Complete MPPI controller and simulation framework |
| `mppi_push_template.py` | Original supplied Python template |
| `evaluate.py` | Reproducible evaluation, trajectory checkpoints and summaries |
| `build_media.py` | Plot, screenshot and MP4 generation; obstacle-contact audit |
| `build_report.py` | Generate LaTeX result tables, task pages and controller excerpts |
| `requirements.txt` | Dependency versions |

After evaluating all 30 episodes into the default `results/` directory:

```bash
python build_media.py
python build_report.py
```

These commands generate figures, screenshots, videos, a contact audit and LaTeX fragments locally. `build_report.py` generates fragments, not a standalone PDF. Existing videos may be replaced when media is regenerated. Saved result files and report PDFs are not included in this upload.
