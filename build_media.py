"""Render actual logged trajectories and plot all benchmark episodes."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import imageio.v2 as imageio
import mppi_push as m
from evaluate import summarize

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})


def load(task, seed):
    stem = ROOT / 'results' / f'task{task}_seed{seed:02d}'
    return json.loads(stem.with_suffix('.json').read_text()), np.load(stem.with_suffix('.npz'))


def render(task, seed=0):
    meta, log = load(task, seed)
    params, goal = m.task_setup(task, seed, m.PARAMS)
    model, _ = m.load_models(params)
    data = m.mujoco.MjData(model)
    m.set_goal_marker(model, data, goal)
    camera = m.mujoco.MjvCamera()
    camera.lookat[:] = [.55, .20, .46]
    camera.distance, camera.azimuth, camera.elevation = 1.0, 135, -55
    renderer = m.mujoco.Renderer(model, 720, 1280)
    screenshots, videos = ROOT / 'screenshots', ROOT / 'videos'
    screenshots.mkdir(exist_ok=True); videos.mkdir(exist_ok=True)
    t = log['t']
    try:
        for label, fraction in [('start', 0), ('middle', .5), ('final', 1)]:
            k = int(round(fraction*(len(t)-1)))
            data.qpos[:] = log['qpos'][k]
            m.mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera)
            imageio.imwrite(screenshots / f'task{task}_seed{seed:02d}_{label}.png', renderer.render())
        path = videos / f'task{task}_seed{seed:02d}.mp4'
        with imageio.get_writer(path, fps=30, quality=7, macro_block_size=8) as writer:
            indices = np.searchsorted(t, np.arange(0, t[-1], 1/30)).clip(0, len(t)-1)
            indices = np.r_[indices, np.repeat(len(t)-1, 30)]
            for k in indices:
                data.qpos[:] = log['qpos'][k]
                m.mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera)
                writer.append_data(renderer.render())
        print('Rendered', path.name, flush=True)
    finally:
        renderer.close()


def square(center, yaw, ax, color, label):
    corners = .04*np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    ax.add_patch(Polygon(corners @ rot.T + center, closed=True, fill=False,
                         edgecolor=color, lw=1.3, label=label))
    direction = .04*np.array([np.cos(yaw), np.sin(yaw)])
    ax.annotate('', xy=center+direction, xytext=center,
                arrowprops={'arrowstyle': '->', 'color': color})


def representative_seed(task):
    for seed in range(10):
        meta, _ = load(task, seed)
        if meta['success']:
            return seed
    return 0


def plots(task):
    chosen = representative_seed(task)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.25))
    for seed in range(10):
        meta, log = load(task, seed)
        box = log['Y'][:, m.SENS['box_pos']]
        goal = dict(pos=np.array(meta['goal_pos']), yaw=meta['goal_yaw'])
        ep, ey, tilt = m.episode_errors(log['Y'], goal)
        color = '#25834d' if meta['success'] else '#c34c42'
        axes[1].plot(log['t'], 100*ep, color=color, alpha=.7, lw=1)
        axes[2].plot(log['t'], np.degrees(ey) if task == 'B' else tilt,
                     color=color, alpha=.7, lw=1)
        if seed == chosen:
            axes[0].plot(box[:, 0], box[:, 1], color='#23609a', lw=2, label=f'Box path (seed {chosen})')
            axes[0].scatter(*box[0, :2], marker='o', color='#23609a', s=25, label='Start')
            axes[0].scatter(*goal['pos'][:2], marker='x', color='#25834d', s=40, label='Goal')
            yaw = m.quat_yaw(log['Y'][-1, m.SENS['box_quat']])
            square(box[-1, :2], yaw, axes[0], '#c46b1a', 'Final box')
            square(goal['pos'][:2], meta['goal_yaw'] or 0, axes[0], '#25834d', 'Target box')
    if task == 'C':
        axes[0].add_patch(Circle(m.TASK_C['obstacle'], m.PARAMS['obstacle_r'], color='#555555', label='Obstacle'))
        axes[0].plot([.42, .42, .55], [.14, .34, .33], ':', color='#777777', lw=1, label='Guide waypoints')
    axes[0].set(xlabel='World x (m)', ylabel='World y (m)', title='Representative trajectory')
    axes[0].set_aspect('equal', adjustable='datalim')
    axes[0].legend(fontsize=6.6, loc='best', framealpha=.85)
    axes[1].axhline(2, ls='--', color='black', lw=1)
    axes[1].set(xlabel='Simulation time (s)', ylabel='Position error (cm)', title='All 10 episodes')
    axes[2].axhline(10, ls='--', color='black', lw=1)
    axes[2].set(xlabel='Simulation time (s)', ylabel='Yaw error (deg)' if task == 'B' else 'Box tilt (deg)',
                title='Green: success; red: failure')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    (ROOT / 'figures').mkdir(exist_ok=True)
    fig.savefig(ROOT / 'figures' / f'task{task}_results.pdf', bbox_inches='tight')
    fig.savefig(ROOT / 'figures' / f'task{task}_results.png', dpi=170, bbox_inches='tight')
    plt.close(fig)


def audit_obstacle():
    audits = []
    params, _ = m.task_setup('C', 0, m.PARAMS)
    model, _ = m.load_models(params)
    data = m.mujoco.MjData(model)
    box_id, obstacle_id = model.geom('box').id, model.geom('obstacle').id
    for seed in range(10):
        _, log = load('C', seed)
        contacts = 0
        for qpos in log['qpos']:
            data.qpos[:] = qpos
            m.mujoco.mj_forward(model, data)
            contacts += any({c.geom1, c.geom2} == {box_id, obstacle_id} and c.dist <= 0
                            for c in data.contact)
        audits.append(dict(seed=seed, box_obstacle_contact_steps=int(contacts)))
    (ROOT / 'results' / 'obstacle_audit.json').write_text(json.dumps(audits, indent=2))
    print('Obstacle contact audit:', audits, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tasks', default='ABC')
    ap.add_argument('--skip-render', action='store_true')
    args = ap.parse_args()
    for task in args.tasks:
        plots(task)
        if not args.skip_render:
            chosen = representative_seed(task)
            render(task, chosen)
            failures = [s for s in range(10) if not load(task, s)[0]['success']]
            if failures and failures[0] != chosen:
                render(task, failures[0])
    records, summary = summarize(ROOT / 'results')
    with (ROOT / 'results' / 'episodes.csv').open('w', newline='') as f:
        keys = list(dict.fromkeys(k for row in records for k in row))
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(records)
    if 'C' in args.tasks:
        audit_obstacle()
