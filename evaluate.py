"""Reproducible evaluation and trajectory checkpoints for the supplied tasks."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mppi_push as m

ROOT = Path(__file__).resolve().parent


def evaluate(tasks, start, runs, folder, duration=10.0):
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(parameters=dict(m.PARAMS, T_episode=duration),
                  controller_sha256=hashlib.sha256((ROOT / 'mppi_push.py').read_bytes()).hexdigest(),
                  mujoco_version=m.mujoco.__version__, numpy_version=np.__version__)
    config_path = folder / 'configuration.json'
    if config_path.exists() and json.loads(config_path.read_text()) != json.loads(json.dumps(config)):
        raise RuntimeError('Configuration changed: choose a new output folder')
    config_path.write_text(json.dumps(config, indent=2))
    for task in tasks:
        for seed in range(start, start + runs):
            stem = folder / f'task{task}_seed{seed:02d}'
            if stem.with_suffix('.json').exists():
                print(f'Skipping completed {stem.name}', flush=True)
                continue
            params, goal = m.task_setup(task, seed, config['parameters'])
            print(f'\nTASK {task}, SEED {seed}, goal={goal}', flush=True)
            log = m.run_episode(params, goal, seed)
            arrays = {k: log[k] for k in ('t', 'qpos', 'Y', 'ctrl')}
            arrays.update({f'replan_{k}': v for k, v in log['replan'].items()})
            np.savez_compressed(stem.with_suffix('.npz'), **arrays)
            meta = {k: log[k] for k in (
                'success', 'held_success', 'e_pos', 'T_sim', 'wall_total', 'rtf',
                'wall_per_replan', 'arm_table_steps', 'max_tilt_deg',
                'final_tilt_deg', 'box_off_table')}
            meta.update(task=task, seed=seed, goal_pos=goal['pos'].tolist(),
                        goal_yaw=goal['yaw'],
                        e_yaw=None if goal['yaw'] is None else log['e_yaw'])
            # Audit actual box/cylinder separation using the rotated square footprint.
            if params['obstacle'] is not None:
                xy = log['Y'][:, m.SENS['box_pos']][:, :2]
                yaw = m.quat_yaw(log['Y'][:, m.SENS['box_quat']])
                delta = np.asarray(params['obstacle']) - xy
                local = np.column_stack((np.cos(yaw)*delta[:, 0] + np.sin(yaw)*delta[:, 1],
                                         -np.sin(yaw)*delta[:, 0] + np.cos(yaw)*delta[:, 1]))
                clearance = np.linalg.norm(np.maximum(np.abs(local)-params['box_half'], 0), axis=1)
                meta['minimum_box_obstacle_clearance_m'] = float(clearance.min()-params['obstacle_r'])
            stem.with_suffix('.json').write_text(json.dumps(meta, indent=2, allow_nan=False))
            print('CHECKPOINT ' + json.dumps(meta), flush=True)
    summarize(folder)


def summarize(folder):
    records = [json.loads(f.read_text()) for f in sorted(folder.glob('task*_seed*.json'))]
    summary = {}
    for task in 'ABC':
        rows = [r for r in records if r['task'] == task]
        if not rows:
            continue
        summary[task] = dict(episodes=len(rows), successes=sum(r['success'] for r in rows),
            held_successes=sum(r['held_success'] for r in rows),
            success_rate=float(np.mean([r['success'] for r in rows])),
            mean_position_error_cm=float(100*np.mean([r['e_pos'] for r in rows])),
            mean_yaw_error_deg=None if task != 'B' else float(np.degrees(np.mean([r['e_yaw'] for r in rows]))),
            mean_simulation_seconds=float(np.mean([r['T_sim'] for r in rows])),
            mean_wall_seconds=float(np.mean([r['wall_total'] for r in rows])),
            mean_planning_ms=float(1000*np.mean([r['wall_per_replan'] for r in rows])),
            arm_table_steps=sum(r['arm_table_steps'] for r in rows),
            off_table=sum(r['box_off_table'] for r in rows))
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('\nSUMMARY\n' + json.dumps(summary, indent=2), flush=True)
    return records, summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task', choices=['A', 'B', 'C', 'all'], default='all')
    ap.add_argument('--runs', type=int, default=10)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--duration', type=float, default=10.0)
    ap.add_argument('--output', type=Path, default=ROOT / 'results')
    args = ap.parse_args()
    evaluate('ABC' if args.task == 'all' else args.task,
             args.start, args.runs, args.output, args.duration)
