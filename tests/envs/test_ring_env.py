"""
Test the ring environment with IDM controllers.
"""

from collections import deque
import copy
import gymnasium as gym
import numpy as np

from platoon_gym.ctrl.dmpc import DMPC
from platoon_gym.ctrl.idm import IntelligentDriverModel
from platoon_gym.dyn.linear_accel import LinearAccel
from platoon_gym.veh.vehicle import Vehicle

# IDM params
v0 = 30.0
s0 = 1.0
T = 2.0

# envt params
N = 30
dt = 0.1
circumference = 500.0

# dmpc params
N_dmpc_start = 15
th = 2.0
th_tran = 2.0
dh = 1.0
dh_tran = 1.0


def test_idm_ring():

    # set up dynamics
    tau = 0.5
    dt = 0.1
    x_lims = np.array([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]])
    u_lims = np.array([[-3.0, 1.5]])
    dyn = LinearAccel(dt, x_lims, u_lims, tau)

    # set up idm controller
    idm = IntelligentDriverModel(v0=v0, s0=s0, T=T)

    # set up environment params
    env_args = {
        "ring circumference": circumference,
        "dt": dt,
        "render_fps": 30,
        "record": False,
    }

    # set up vehicles
    vehicles = [Vehicle(dyn, -i * circumference / N, 0.0, 0.0) for i in range(N)]

    # set up ring env
    env = gym.make(
        "ring_env-v0", vehicles=vehicles, env_args=env_args, render_mode="human"
    )
    obs, info = env.reset()
    env.render()
    veh_states = info["vehicle states"]

    while True:
        try:
            actions = []
            for i, o in enumerate(obs):
                action = idm.control(o[0], veh_states[i][1], -o[1])
                # cbf to prevent collisions
                # want to make sure u satisfies the constraints
                # u <= (1 / tmin) * (vl - vf) + (k / tmin) * (s - tmin * vf)
                tmin = 0.25
                k1 = 0.1
                s = o[0]  # distance to predecessor
                vdiff = o[1]  # speed relative to predecessor, vl - vf
                vf = veh_states[i][1]
                af = veh_states[i][2]
                max_action = np.array(
                    [(1 / tmin) * vdiff + (k1 / tmin) * (s - tmin * vf)]
                )
                k2 = 1.0
                min_action = np.array(
                    [-k2 * dt / dyn.tau * (vf + dt * af) - (1 - dt / dyn.tau) * af]
                )
                action = np.minimum(max_action, np.maximum(min_action, action))
                action = np.clip(action, dyn.u_lims[:, 0], dyn.u_lims[:, 1])

                actions.append(action)
            obs, _, _, trunc, info = env.step(action=actions)
            veh_states = info["vehicle states"]
            if trunc:
                break
            env.render()
        except KeyboardInterrupt:
            env.close()
            break


def test_dmpc_ring():

    # set up environment params
    veh_colors = [(255, 0, 0)] * N_dmpc_start + [(0, 0, 255)] * (N - N_dmpc_start)
    env_args = {
        "ring circumference": circumference,
        "dt": dt,
        "render_fps": 30,
        "record": False,
        "vehicle colors": veh_colors,
    }

    # set up dynamics
    tau_lims = (0.5, 0.5)
    rng = np.random.default_rng()
    taus = rng.uniform(tau_lims[0], tau_lims[1], N)
    x_lims = np.array([[-np.inf, np.inf], [0.0, np.inf], [-3.0, 1.5]])
    u_lims = np.array([[-3.0, 1.5]])
    dyns = [LinearAccel(dt, x_lims, u_lims, tau, full_obs=False) for tau in taus]

    # set up controllers
    idm = IntelligentDriverModel(v0=v0, s0=s0, T=T)
    H = 20
    dhs = [None] * N_dmpc_start + [dh] * (N - N_dmpc_start)
    dhs[N_dmpc_start] = dh_tran
    # distance_headway = dh
    ths = [None] * N_dmpc_start + [th] * (N - N_dmpc_start)
    ths[N_dmpc_start] = th_tran
    # time_headway = th
    Qs = [None] * N_dmpc_start + [np.diag([10.0, 1.0])] * (N - N_dmpc_start)
    Q_neighbors = [None] * N_dmpc_start + [np.diag([10.0, 1.0])] * (N - N_dmpc_start)
    Qf_neighbors = [None] * N_dmpc_start + [np.diag([1000.0, 100.0])] * (
        N - N_dmpc_start
    )
    ctrls = [idm for _ in range(N_dmpc_start)] + [
        DMPC(
            H=H,
            Q=Qs[i],
            Q_neighbors=[Q_neighbors[i]],
            R=np.diag([1.0]),
            A=dyns[i].Ad,
            B=dyns[i].Bd,
            C=dyns[i].C,
            x_lims=dyns[i].x_lims,
            u_lims=dyns[i].u_lims,
            u_slew_rate=np.array([float("inf")]),
            distance_headways=[dhs[i]],
            time_headways=[ths[i]],
            terminal_constraint=False,
            Qf=0.0,
            Qf_neighbors=[Qf_neighbors[i]],
            output_norm="l1",
            input_norm="quadratic",
        )
        for i in range(N_dmpc_start, N)
    ]

    # set up vehicles
    vehicles = [Vehicle(dyns[i], -i * circumference / N, 0.0, 0.0) for i in range(N)]

    # set up ring env
    env = gym.make(
        "ring_env-v0", vehicles=vehicles, env_args=env_args, render_mode="human"
    )
    obs, info = env.reset()
    env.render()
    veh_states = info["vehicle states"]

    # initialize assumed states
    prev_assumed_states = []
    for i in range(N):
        if type(ctrls[i]) is IntelligentDriverModel:
            prev_assumed_states.append(None)
            continue
        uref = np.zeros((H, dyns[i].m))
        xa = ctrls[i].initialize_assumed_trajectory(veh_states[i], uref)
        prev_assumed_states.append(xa)

    # platoon leader velocity history
    T_hist = 60.0
    vel_hist = deque(maxlen=round(T_hist / dt) + 1)

    # simulate platoon
    while True:
        try:
            actions = []
            assumed_states = []
            for i in range(N):
                if type(ctrls[i]) is IntelligentDriverModel:
                    action = ctrls[i].control(obs[i][0], veh_states[i][1], -obs[i][1])
                    # cbf to prevent collisions
                    # want to make sure u satisfies the constraints
                    # u <= (1 / tmin) * (vl - vf) + (k / tmin) * (s - tmin * vf)
                    tmin = 0.25
                    k1 = 0.1
                    s = obs[i][0]  # distance to predecessor
                    vdiff = obs[i][1]  # speed relative to predecessor, vl - vf
                    vf = veh_states[i][1]
                    af = veh_states[i][2]
                    max_action = np.array(
                        [(1 / tmin) * vdiff + (k1 / tmin) * (s - tmin * vf)]
                    )
                    k2 = 1.0
                    min_action = np.array(
                        [
                            -k2 * dt / dyns[i].tau * (vf + dt * af)
                            - (1 - dt / dyns[i].tau) * af
                        ]
                    )
                    action = np.minimum(max_action, np.maximum(min_action, action))
                    action = np.clip(action, dyns[i].u_lims[:, 0], dyns[i].u_lims[:, 1])
                    actions.append(action)
                    assumed_states.append(None)
                    continue
                elif type(ctrls[i - 1]) is IntelligentDriverModel:
                    vf = veh_states[i][1]
                    vdiff = obs[i][1]
                    vl = vdiff + vf
                    x0 = np.array([0.0, vf, veh_states[i][2]])
                    yref = np.zeros((H + 1, dyns[i].p))
                    yref[:, 1] = vl
                    yref[1:, 0] = np.cumsum(dt * yref[1:, 1])
                    yref[:, 0] += obs[i][0]
                    yneighbors = [yref]
                    # calculate desired final velocity using average velocity
                    # over last some number of seconds
                    vel_hist.append(vl)
                    if len(vel_hist) == vel_hist.maxlen:
                        vdes = np.mean(vel_hist)
                    else:
                        vdes = vl
                    end_state = np.array([yref[-1, 0], vdes, 0.0])
                    end_state[0] -= dhs[i] + ths[i] * vdes
                    xa = copy.deepcopy(prev_assumed_states[i])
                    xa[:, 0] -= xa[0, 0]
                else:
                    x0 = veh_states[i].copy()
                    x0[0] = 0.0
                    yref = prev_assumed_states[i - 1][:, : dyns[i].p]
                    yref[:, 0] -= yref[0, 0]
                    yref[:, 0] += obs[i][0]
                    yneighbors = [yref]
                    xa = copy.deepcopy(prev_assumed_states[i])
                    xa[:, 0] -= xa[0, 0]
                    end_state = yref[-1, :].copy()
                    end_state[0] -= dhs[i] + ths[i] * end_state[1]

                action, ctrl_info = ctrls[i].control(
                    x0=x0,
                    y_neighbors=yneighbors,
                    xa=xa,
                    xf=end_state.copy(),
                )
                actions.append(action)
                xopt = ctrl_info["x"]
                xa = np.zeros_like(xopt)
                xa[:-1] = xopt[1:]
                xa[-1] = dyns[i].forward(xopt[-1], np.zeros(dyns[i].m))
                assumed_states.append(xa)

            # step environment
            obs, _, _, trunc, info = env.step(action=actions)
            veh_states = info["vehicle states"]
            prev_assumed_states = copy.deepcopy(assumed_states)
            if trunc:
                break
            env.render()
        except KeyboardInterrupt:
            env.close()
            break


if __name__ == "__main__":
    test_idm_ring()
    test_dmpc_ring()
