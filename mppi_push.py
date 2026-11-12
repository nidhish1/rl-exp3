"""
mppi_push.py  —  ROB 6323  MPPI: the Franka Panda pushes a box
================================================================
Model predictive path integral control (MPPI) for non-prehensile pushing.
MuJoCo is both the "real" robot and the planner's model; rollouts are batched
with mujoco.rollout.

IMPLEMENTATION
--------------
mppi_step implements batched MPPI with sensor-based costs. Task C adds
left-side guide waypoints and a clearance penalty. The provided models,
control spline, random-goal generation, and success test are retained.
Use evaluate.py to save all 30 benchmark trajectories and build_media.py
to produce plots, screenshots, videos, and an obstacle-contact audit.


CONVENTIONS
-----------
n = planner stage (0 … N), 
N = horizon in planner steps, 
M = number of samples,
i = sample index (k is the iteration index in this course).
β = MPPI temperature - given, 
Σ = sampling covariance.
ℓ_n, ℓ_N = running and terminal cost, S_i = total cost of rollout i,
w_i = weight of rollout i.
x_n = full MuJoCo physics state (mj_getState with mjSTATE_FULLPHYSICS).
u_n = ctrl of the 7 arm actuators = joint position targets [rad].  The 8th
      actuator (gripper) is held closed at PARAMS['gripper_ctrl'].

Control parameterization.  The plan is K = PARAMS['n_knots'] knots
U ∈ R^{K×7} (joint targets), equally spaced over the horizon T_h = N·dt_plan,
knot 0 at the current time.  Per-step controls are the linear interpolation
u_n = Σ_k B[n, k] U[k]  (B from interp_matrix).  Sample i uses U + ε_i with
ε_i ∈ R^{K×7}, ε_i ~ N(0, Σ), Σ = diag(σ²).

MPPI update:
    w_i = exp(−(S_i − min_j S_j)/β) / Σ_j exp(−(S_j − min_j S_j)/β)
    U  ←  U + Σ_i w_i ε_i

Two models from the same scene (assets/push_scene.xml):
    real    — timestep dt_real, every collision: the "robot";
    planner — timestep dt_plan (coarser), upper-arm collisions off: MPPI's model.
The simulation waits while MPPI plans (it is not real time; the logs report
the real-time factor).

Frames: world z up, robot base at the origin, table top at z = 0.42 m.
The "fingertip" point is hand_pos + PARAMS['tip_offset'] · hand_zaxis.

USAGE
-----
  python mppi_push.py --task A     # push the box 20 cm               (10 episodes)
  python mppi_push.py --task B     # position + yaw, random goals     (10 episodes)
  python mppi_push.py --task C     # push around an obstacle          (10 episodes)
  options: --video  (videos/taskX.mp4 of episode 0)
           --view   (replay episode 0 in the MuJoCo viewer: use mjpython
                     instead of python on macOS)
"""

import argparse
import os
import time
import numpy as np
import mujoco
from mujoco import rollout

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE_PATH = os.path.join(HERE, 'assets', 'push_scene.xml')
FIG_DIR = os.path.join(HERE, 'figures')
VID_DIR = os.path.join(HERE, 'videos')
FULLPHYSICS = mujoco.mjtState.mjSTATE_FULLPHYSICS

N_ARM = 7          # arm actuators; actuator 7 (the 8th) is the gripper


# ── GIVEN: parameters ─────────────────────────────────────────────────────────

PARAMS = dict(
    # models
    dt_real=0.002,              # "real" simulation timestep [s] (menagerie default)
    dt_plan=0.005,              # planner timestep [s]
    planner_reduced_collisions=True,   # planner: no collisions for link0–link5 and the floor
    gripper_ctrl=0.0,           # gripper actuator, held closed (0 = closed, 255 = open)
    tip_offset=0.105,           # fingertip point below the hand frame [m]
    # task geometry
    table_z=0.42,               # table top [m]
    box_half=0.04,              # box half size [m]
    box_start=(0.55, 0.10, 0.0),        # box start (x, y, yaw) [m, m, rad]
    goal_A=(0.55, 0.30, None),          # task A goal (x, y, yaw); yaw None = not scored
    region_B=dict(x=(0.45, 0.65), y=(0.20, 0.35), yaw_deg=(45., 90.)),  # task B goals
    obstacle=None,              # task C: (x, y) of the cylinder on the table, None = parked out of the way
    obstacle_r=0.03,            # cylinder radius [m] (assets/push_scene.xml)
    T_episode=10.0,             # simulated time per episode [s]
    tol_pos=0.02,               # success: box xy error [m]
    tol_yaw_deg=10.0,           # success: box yaw error [deg] (task B)
    tol_tilt_deg=10.0,          # success: box upright, tilt of its z-axis [deg]
    t_hold=0.5,                 # success when within tolerance for this long [s]
    # MPPI
    M=256,                      # number of samples
    N=120,                      # horizon in planner steps (N · dt_plan = 0.6 s)
    n_knots=4,                  # knots of the control spline
    sigma=0.08,                 # std of the knot offsets [rad] (Σ = σ² I)
    beta=0.3,                   # temperature (same units as the cost S)
    dt_replan=0.04,             # replan period [s]
    z_noise_scale=0.3,          # damp_vertical_noise: keep this fraction of the noise that moves the
                                # fingertip vertically or tilts the hand (1 = plain joint-space sampling)
    n_threads=None,             # rollout threads (None = all cores)
)

# task C: the obstacle sits between the box and the goal
TASK_C = dict(goal=(0.55, 0.33, None), obstacle=(0.55, 0.225), w_obs=2000.0)


# ── GIVEN: models, state, sensors ─────────────────────────────────────────────

def load_models(params):
    """Return (m_real, m_plan), two compiled copies of the scene."""
    m_real = mujoco.MjModel.from_xml_path(SCENE_PATH)
    m_plan = mujoco.MjModel.from_xml_path(SCENE_PATH)
    m_real.opt.timestep = params['dt_real']
    m_plan.opt.timestep = params['dt_plan']
    if params['planner_reduced_collisions']:
        off = {'link0', 'link1', 'link2', 'link3', 'link4', 'link5'}
        for g in range(m_plan.ngeom):
            body = mujoco.mj_id2name(m_plan, mujoco.mjtObj.mjOBJ_BODY, m_plan.geom_bodyid[g])
            if body in off or mujoco.mj_id2name(m_plan, mujoco.mjtObj.mjOBJ_GEOM, g) == 'floor':
                m_plan.geom_contype[g] = 0
                m_plan.geom_conaffinity[g] = 0
    if params['obstacle'] is not None:
        for m in (m_real, m_plan):
            m.body_pos[m.body('obstacle').id] = [*params['obstacle'], params['table_z'] + 0.05]
    return m_real, m_plan


def sensor_slices(m):
    """{sensor name: slice into sensordata}."""
    out = {}
    for s in range(m.nsensor):
        a, n = m.sensor_adr[s], m.sensor_dim[s]
        out[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, s)] = slice(a, a + n)
    return out


_m0 = mujoco.MjModel.from_xml_path(SCENE_PATH)
SENS = sensor_slices(_m0)          # same layout for the real and planner models
NSENS = _m0.nsensordata
del _m0
# SENS keys: box_pos, box_quat, box_zaxis, hand_pos, hand_zaxis (frame sensors) and the contact
# sensors arm_table (arm links – table), hand_box (hand and fingers – box), palm_box (hand body – box);
# a contact sensor reads > 0 when there is contact.


def quat_yaw(q):
    """Yaw [rad] of quaternions (w, x, y, z), any leading shape."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def tip_pos(Y, params=PARAMS):
    """Fingertip point from sensordata Y (..., NSENS) → (..., 3)."""
    return Y[..., SENS['hand_pos']] + params['tip_offset'] * Y[..., SENS['hand_zaxis']]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def reset_start(m, d, params):
    """Arm at the 'home' keyframe with the gripper closed, box at params['box_start']."""
    mujoco.mj_resetData(m, d)
    d.qpos[:N_ARM] = m.key_qpos[m.key('home').id, :N_ARM]
    d.qpos[N_ARM:9] = 0.0
    x, y, yaw = params['box_start']
    d.qpos[9:16] = [x, y, params['table_z'] + params['box_half'],
                    np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    d.ctrl[:N_ARM] = d.qpos[:N_ARM]
    d.ctrl[N_ARM] = params['gripper_ctrl']
    mujoco.mj_forward(m, d)


def make_goal(x, y, yaw, params=PARAMS):
    """Goal dict: 'pos' (3,) box centre, 'yaw' [rad] or None (not scored)."""
    return dict(pos=np.array([x, y, params['table_z'] + params['box_half']]), yaw=yaw)


def set_goal_marker(m, d, goal):
    d.mocap_pos[0] = goal['pos']
    yaw = 0.0 if goal['yaw'] is None else goal['yaw']
    d.mocap_quat[0] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]


def get_state(m, d):
    x = np.empty(mujoco.mj_stateSize(m, FULLPHYSICS))
    mujoco.mj_getState(m, d, x, FULLPHYSICS)
    return x


# ── GIVEN: the control spline ─────────────────────────────────────────────────

def knot_times(params):
    T_h = params['N'] * params['dt_plan']
    return np.linspace(0.0, T_h, params['n_knots'])


def spline_eval(U, t, params):
    """Joint targets (7,) of the plan U (K, 7) at time t ≥ 0 from now (held after the last knot)."""
    tk = knot_times(params)
    t = np.clip(t, tk[0], tk[-1])
    return np.array([np.interp(t, tk, U[:, j]) for j in range(U.shape[1])])


def interp_matrix(params):
    """B (N, K): u_n = Σ_k B[n, k] U[k] at the planner times t_n = n · dt_plan (linear hat functions)."""
    tk = knot_times(params)
    t = np.arange(params['N']) * params['dt_plan']
    B = np.zeros((params['N'], params['n_knots']))
    for k in range(params['n_knots']):
        e = np.zeros(params['n_knots']); e[k] = 1.0
        B[:, k] = np.interp(t, tk, e)
    return B


def shift_knots(U, dt_shift, params):
    """Warm start: the plan seen dt_shift later, U'[k] = spline(U, t_k + dt_shift) (held after the end)."""
    tk = knot_times(params)
    return np.stack([spline_eval(U, t + dt_shift, params) for t in tk])


class Planner:
    """What MPPI needs on the planner side.

    m       planner model;  params  the PARAMS dict;  datas  one MjData per rollout thread;
    pool    a persistent mujoco.rollout.Rollout thread pool (call close() at the end);
    B       interpolation matrix (N, K);  ctrl_range  (7, 2) arm ctrlrange.
    """

    def __init__(self, m_plan, params):
        self.m = m_plan
        self.params = params
        self.nthread = params['n_threads'] or os.cpu_count()
        self.datas = [mujoco.MjData(m_plan) for _ in range(self.nthread)]
        self.pool = rollout.Rollout(nthread=self.nthread)
        self.B = interp_matrix(params)
        self.ctrl_range = m_plan.actuator_ctrlrange[:N_ARM].copy()

    def close(self):
        self.pool.close()


def damp_vertical_noise(eps, scale, m, d, params):
    """Optional: scale the part of joint-space noise eps (M, K, 7) that moves the fingertip
    vertically or tilts the hand (roll, pitch) by `scale`, using the Jacobian at the
    current configuration:  ε ← ε − (1 − scale) J_v⁺ J_v ε,   J_v = [J_tip,z; J_rot,x; J_rot,y].
    Without it, joint-space noise often lowers the palm onto the box top, which tips the box."""
    hid = m.body('hand').id
    tip = d.xpos[hid] + params['tip_offset'] * d.xmat[hid].reshape(3, 3)[:, 2]
    jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jp, jr, tip, hid)
    Jv = np.vstack([jp[2:3, :N_ARM], jr[:2, :N_ARM]])            # (3, 7)
    P = np.linalg.pinv(Jv) @ Jv                                   # projector onto the row space
    return eps - (1.0 - scale) * eps @ P.T


# ── IMPLEMENTED: one MPPI step ───────────────────────────────────────────────────────

def mppi_step(planner, U, d_real, goal, rng, u_prev):
    """One MPPI iteration from the current real state.  Returns (U_new, info).

    planner  Planner (planner.m, planner.params, planner.datas, planner.pool, planner.B, planner.ctrl_range)
    U        (K, 7) current plan (knots of joint targets, see "Control parameterization" at the top)
    d_real   MjData of the real robot; d_real.sensordata holds the current sensor readings
    goal     dict(pos=(3,), yaw=float or None)
    rng      np.random.Generator: draw all random numbers from it (episodes are then reproducible)
    u_prev   (7,) control applied just before
    U_new    (K, 7) new plan
    info     dict of floats, printed and plotted when present (optional):
             'S_min', 'S_mean', 'ess' (effective sample size 1/Σ w_i²)

    The implementation follows the six template steps below (p = planner.params).
    Sampling, costs, and the weighted update are vectorized across rollouts.

    1. Sample ε (M, K, 7), ε_i ~ N(0, σ² I) with σ = p['sigma'], and set ε_0 = 0 so that the
       current plan is always a candidate.  Then call
           ε = damp_vertical_noise(ε, p['z_noise_scale'], planner.m, d_real, p)
       (otherwise the noise often lowers the palm onto the box top).
    2. Controls u (M, N, 7):  u[i, n] = Σ_k B[n, k] (U + ε_i)[k]  with B = planner.B,
       clipped to planner.ctrl_range (np.einsum).
    3. Rollouts: sensordata Y (M, N, NSENS) of the M rollouts on the planner model, from the
       current real state, in one call:
           state, Y = planner.pool.rollout(planner.m, planner.datas, x0, ctrl)
       x0 = get_state(planner.m, d_real)[None] (the real and planner states have the same
       layout); ctrl (M, N, 8) = u with the gripper column p['gripper_ctrl'] appended.
       Y[:, n] is the sensor reading of x_n.  (MuJoCo Python docs, "rollout".)
    4. Costs S (M,), with the sensor readings in Y (use SENS, tip_pos, quat_yaw):
        you need to define the cost function to get to the goal this will likely
        include more than just the end goal to "guide" the optimizer
        for example you might want to 
          penalize both square distance to the box + absolute value to the box
          penalize yaw (orientation) error using (1-cos(yaw_box - yaw_goal)) if for task B
          penalize the velocity of the tip
          penalize collision with the table (arm-table contact) and with the palm (palm-box contact)
          etc
    5. Weights w_i = exp(−(S_i − min S)/β) / Σ_j exp(−(S_j − min S)/β), β = p['beta'].
    6. Update U_new = U + Σ_i w_i ε_i, clipped to planner.ctrl_range.
    """
    p = planner.params
    eps = rng.normal(0.0, p['sigma'], (p['M'], p['n_knots'], N_ARM))
    eps[0] = 0.0
    eps = damp_vertical_noise(eps, p['z_noise_scale'], planner.m, d_real, p)
    controls = np.einsum('nk,mkj->mnj', planner.B, U[None] + eps)
    controls = np.clip(controls, planner.ctrl_range[:, 0], planner.ctrl_range[:, 1])
    ctrl = np.full((p['M'], p['N'], planner.m.nu), p['gripper_ctrl'])
    ctrl[:, :, :N_ARM] = controls
    x0 = get_state(planner.m, d_real)[None]
    _, Y = planner.pool.rollout(planner.m, planner.datas, x0, ctrl)

    # Sensor-based running costs, averaged over the planning horizon.
    box = Y[..., SENS['box_pos']]
    tip = tip_pos(Y, p)
    target = goal['pos'].copy()
    if p['obstacle'] is not None:
        # Left-side route around the cylinder, with a conservative clearance.
        current = d_real.sensordata[SENS['box_pos']]
        phase = getattr(planner, 'route_phase', 0)
        waypoints = np.array([[0.42, 0.14], [0.42, 0.34], goal['pos'][:2]])
        if phase < 2 and np.linalg.norm(current[:2] - waypoints[phase]) < 0.025:
            phase += 1
        planner.route_phase = phase
        target[:2] = waypoints[phase]
    error = box[..., :2] - target[:2]
    distance = np.linalg.norm(error, axis=-1)
    direction = -error / np.maximum(distance[..., None], 0.02)
    contact = box.copy()
    contact[..., :2] -= 0.055 * direction
    contact[..., 2] -= 0.005
    tip_error = tip - contact
    stage = 100.0 * distance**2 + 10.0 * distance
    stage += 40.0 * np.sum(tip_error[..., :2]**2, axis=-1)
    stage += 2.0 * np.linalg.norm(tip_error[..., :2], axis=-1)
    stage += 400.0 * tip_error[..., 2]**2
    stage += 40.0 * (1.0 - Y[..., SENS['box_zaxis']][..., 2])
    stage += 4.0 * (1.0 + Y[..., SENS['hand_zaxis']][..., 2])
    stage += 25.0 * (Y[..., SENS['arm_table']][..., 0] > 0)
    stage += 15.0 * (Y[..., SENS['palm_box']][..., 0] > 0)
    tip_velocity = np.diff(tip, axis=1) / p['dt_plan']
    stage[:, 1:] += 0.10 * np.sum(tip_velocity**2, axis=-1)
    S = stage.mean(axis=1) + 200.0 * distance[:, -1]**2 + 15.0 * distance[:, -1]
    S += 0.2 * np.mean((controls - u_prev)**2, axis=(1, 2))
    if goal['yaw'] is not None:
        yaw_error = wrap(quat_yaw(Y[..., SENS['box_quat']]) - goal['yaw'])
        orientation = 1.0 - np.cos(yaw_error)
        S += 3.0 * orientation.mean(axis=1) + 6.0 * orientation[:, -1]
    if p['obstacle'] is not None:
        clearance = p['obstacle_r'] + np.sqrt(2.0) * p['box_half'] + 0.01
        obs_dist = np.linalg.norm(box[..., :2] - p['obstacle'], axis=-1)
        intrusion = np.maximum(clearance - obs_dist, 0.0)
        S += p['w_obs'] * np.mean(intrusion**2, axis=1)
        S += 500.0 * np.max(intrusion, axis=1)
    S *= 5.0  # cost scale relative to the fixed template temperature
    S = np.nan_to_num(S, nan=1e12, posinf=1e12, neginf=1e12)
    weights = np.exp(-(S - S.min()) / p['beta'])
    weights /= weights.sum()
    U_new = np.clip(U + np.einsum('m,mkj->kj', weights, eps),
                    planner.ctrl_range[:, 0], planner.ctrl_range[:, 1])
    info = {'S_min': float(S.min()), 'S_mean': float(S.mean()),
            'ess': float(1.0 / np.sum(weights**2))}
    return U_new, info


# ── GIVEN: the closed-loop episode ────────────────────────────────────────────

def box_tilt_deg(Y):
    """Angle between the box z-axis and the vertical [deg], from sensordata (..., NSENS)."""
    return np.degrees(np.arccos(np.clip(Y[..., SENS['box_zaxis']][..., 2], -1.0, 1.0)))


def episode_errors(Y_real, goal):
    """Box xy error [m], |yaw error| [rad] (nan if yaw is not scored) and tilt [deg] from real sensordata."""
    e_pos = np.linalg.norm(Y_real[..., SENS['box_pos']][..., :2] - goal['pos'][:2], axis=-1)
    if goal['yaw'] is None:
        e_yaw = np.full(e_pos.shape, np.nan)
    else:
        e_yaw = np.abs(wrap(quat_yaw(Y_real[..., SENS['box_quat']]) - goal['yaw']))
    return e_pos, e_yaw, box_tilt_deg(Y_real)


def within_tol(e_pos, e_yaw, tilt, params):
    """Success test: position (and yaw, if scored) within tolerance, box upright."""
    ok = (e_pos < params['tol_pos']) & (tilt < params['tol_tilt_deg'])
    return ok & ((np.isnan(e_yaw)) | (e_yaw < np.radians(params['tol_yaw_deg'])))


def run_episode(params, goal, seed=0, verbose=True):
    """MPPI in closed loop on the real model.  Returns a log dict (see the end)."""
    m, m_plan = load_models(params)
    d = mujoco.MjData(m)
    reset_start(m, d, params)
    set_goal_marker(m, d, goal)
    mujoco.mj_forward(m, d)
    planner = Planner(m_plan, params)
    rng = np.random.default_rng(seed)
    U = np.tile(d.ctrl[:N_ARM], (params['n_knots'], 1))       # initial plan: hold the current targets
    u_prev = d.ctrl[:N_ARM].copy()
    n_sub = int(round(params['dt_replan'] / params['dt_real']))
    n_replan = int(round(params['T_episode'] / params['dt_replan']))
    n_hold = int(round(params['t_hold'] / params['dt_real']))

    L = dict(t=[], qpos=[], Y=[], ctrl=[])
    R = dict(t=[], wall=[], S_min=[], S_mean=[], ess=[])
    hold = 0
    success = False
    t_start = time.perf_counter()
    try:
        for r in range(n_replan):
            t0 = time.perf_counter()
            U, info = mppi_step(planner, U, d, goal, rng, u_prev)
            wall = time.perf_counter() - t0
            for j in range(n_sub):                 # apply the plan for dt_replan on the real robot
                d.ctrl[:N_ARM] = spline_eval(U, j * params['dt_real'], params)
                d.ctrl[N_ARM] = params['gripper_ctrl']
                mujoco.mj_step(m, d)
                L['t'].append(d.time); L['qpos'].append(d.qpos.copy())
                L['Y'].append(d.sensordata.copy()); L['ctrl'].append(d.ctrl[:N_ARM].copy())
            u_prev = d.ctrl[:N_ARM].copy()
            U = shift_knots(U, params['dt_replan'], params)
            mujoco.mj_forward(m, d)            # sensordata of the current state, for the next replan
            e_pos, e_yaw, tilt = episode_errors(d.sensordata, goal)
            R['t'].append(d.time - params['dt_replan']); R['wall'].append(wall)
            for k in ('S_min', 'S_mean', 'ess'):
                R[k].append(info.get(k, np.nan))
            hold = hold + n_sub if within_tol(e_pos, e_yaw, tilt, params) else 0
            if verbose and r % 25 == 0:
                print(f'  t={d.time:5.2f}s  e_pos={100*e_pos:5.1f}cm'
                      + ('' if np.isnan(e_yaw) else f'  e_yaw={np.degrees(e_yaw):5.1f}°')
                      + f'  S_min={info.get("S_min", np.nan):9.3f}  ESS={info.get("ess", np.nan):6.1f}'
                      f'  wall={1e3*wall:5.1f}ms')
            if hold >= n_hold:
                success = True
                break
            if d.sensordata[SENS['box_pos']][2] < params['table_z']:   # box fell off the table
                break
    finally:
        planner.close()
    wall_total = time.perf_counter() - t_start
    log = {k: np.array(v) for k, v in L.items()}
    log['replan'] = {k: np.array(v) for k, v in R.items()}
    e_pos, e_yaw, tilt_end = episode_errors(log['Y'][-1], goal)
    tilt = box_tilt_deg(log['Y'])
    log.update(goal=goal, held_success=bool(success), success=bool(success or within_tol(e_pos, e_yaw, tilt_end, params)),
               e_pos=float(e_pos), e_yaw=float(e_yaw), T_sim=float(log['t'][-1]),
               wall_total=wall_total, rtf=float(log['t'][-1]) / wall_total,
               wall_per_replan=float(np.mean(R['wall'])),
               arm_table_steps=int(np.sum(log['Y'][:, SENS['arm_table']] > 0)),
               max_tilt_deg=float(tilt.max()), final_tilt_deg=float(tilt_end),
               box_off_table=bool(log['Y'][-1, SENS['box_pos']][2] < params['table_z']),
               model=m)
    if verbose:
        print(f'  -> success={log["success"]}  e_pos={100*log["e_pos"]:.1f}cm'
              + ('' if goal['yaw'] is None else f'  e_yaw={np.degrees(log["e_yaw"]):.1f}°')
              + f'  T_sim={log["T_sim"]:.2f}s  wall={wall_total:.1f}s  RTF={log["rtf"]:.2f}'
              f'  plan={1e3*log["wall_per_replan"]:.1f}ms/replan  arm-table steps={log["arm_table_steps"]}'
              f'  max tilt={log["max_tilt_deg"]:.1f}° (end {log["final_tilt_deg"]:.1f}°)')
    return log


def task_setup(task, run, params):
    """(params, goal) of episode `run` of task 'A', 'B' or 'C'."""
    if task == 'A':
        return params, make_goal(*params['goal_A'], params)
    if task == 'B':                                  # random goal, the same for a given run
        rng = np.random.default_rng(1000 + run)
        reg = params['region_B']
        x, y = rng.uniform(*reg['x']), rng.uniform(*reg['y'])
        yaw = np.radians(rng.uniform(*reg['yaw_deg'])) * rng.choice([-1.0, 1.0])
        return params, make_goal(x, y, yaw, params)
    params = dict(params, obstacle=TASK_C['obstacle'], w_obs=TASK_C['w_obs'])
    return params, make_goal(*TASK_C['goal'], params)


def run_task(task, n_runs=10, video=False, view=False):
    """n_runs episodes of a task (run s uses MPPI seed s; task B: goal s), plots and video of run 0."""
    logs = []
    for s in range(n_runs):
        params, goal = task_setup(task, s, PARAMS)
        print(f'task {task}, run {s}: goal {goal["pos"][:2].round(3)}'
              + ('' if goal['yaw'] is None else f', yaw {np.degrees(goal["yaw"]):.0f}°'))
        logs.append(run_episode(params, goal, s))
        if s == 0:
            if video:
                record_video(logs[0], f'task{task}')
    print(f'task {task}: success {sum(l["success"] for l in logs)}/{n_runs}'
          f'  mean e_pos {100*np.mean([l["e_pos"] for l in logs]):.1f}cm'
          f'  mean T {np.mean([l["T_sim"] for l in logs]):.2f}s'
          f'  mean wall {np.mean([l["wall_total"] for l in logs]):.1f}s'
          f'  arm-table steps {sum(l["arm_table_steps"] for l in logs)}'
          f'  off table {sum(l["box_off_table"] for l in logs)}')
    if view:
        view_episode(logs[0])
    return logs


# ── GIVEN: video ─────────────────────────────────────────────────

def record_video(log, name, fps=30, width=1280, height=720):
    """Render the logged episode offscreen (works with plain python on macOS)."""
    import imageio.v2 as imageio
    m = log['model']
    d = mujoco.MjData(m)
    set_goal_marker(m, d, log['goal'])
    r = mujoco.Renderer(m, height, width)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.55, 0.18, 0.45]; cam.distance = 1.1; cam.azimuth = 135; cam.elevation = -40
    os.makedirs(VID_DIR, exist_ok=True)
    path = os.path.join(VID_DIR, f'{name}.mp4')
    step = max(1, int(round(1.0 / (fps * m.opt.timestep))))
    with imageio.get_writer(path, fps=fps, quality=7, macro_block_size=8) as w:
        for k in range(0, len(log['t']), step):
            d.qpos[:] = log['qpos'][k]
            mujoco.mj_forward(m, d)
            r.update_scene(d, cam)
            frame = r.render()
            w.append_data(frame)
        for _ in range(fps):                     # hold the last frame for 1 s
            w.append_data(frame)
    r.close()
    print(f'  wrote {os.path.relpath(path, HERE)}')


def view_episode(log):
    """Replay in the passive viewer (needs mjpython on macOS)."""
    import mujoco.viewer
    m = log['model']
    d = mujoco.MjData(m)
    set_goal_marker(m, d, log['goal'])
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            for k in range(0, len(log['t']), 10):
                if not v.is_running():
                    break
                d.qpos[:] = log['qpos'][k]
                mujoco.mj_forward(m, d)
                v.sync()
                time.sleep(10 * m.opt.timestep)
            time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', default='A', choices=['A', 'B', 'C'])
    ap.add_argument('--video', action='store_true', help='write videos/taskX.mp4 of episode 0')
    ap.add_argument('--view', action='store_true', help='replay episode 0 in the viewer (mjpython on macOS)')
    args = ap.parse_args()
    run_task(args.task, video=args.video, view=args.view)


if __name__ == '__main__':
    main()
