import argparse
import numpy as np
import torch

from platoon_gym.ctrl.ccmpc import CCMPC
from platoon_gym.ctrl.dmpc import DMPC
from platoon_gym.envs.platoon_env import PlatoonEnv
from platoon_gym.veh.virtual_leader import VirtualLeader

from i24_forecasting.utils.general_utils import get_project_dir

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

seed = 42


def main():
    parser = argparse.ArgumentParser(
        description="Simulate a platoon using forecasting and chance-constrained MPC."
    )
    parser.add_argument(
        "--use_truncated_gaussian", 
        action="store_true", 
        help="Use truncated Gaussian distribution"
    )
    args = parser.parse_args()

    # Load the forecasting model
    model_dir = get_project_dir() / "models" / "transformer"
    model_type = "truncated_gaussian" if args.use_truncated_gaussian else "gaussian"
    model_dir = model_dir / model_type
    model_name = "forecasting_model_best"

    # Set up virtual leader
    dt = 0.1
    vl_traj_type = "i24_trajectory"
    vl_traj_args = {'dt': dt, 'seed': seed}
    vl = VirtualLeader(vl_traj_type, vl_traj_args)

    # Set up platoon environment
    n_vehicles = 10
    t_h, d_h = 0.5, 2.0
    t_h_safe, d_h_safe = 0.1, 1.0
    render_fps = 100
    render_mode = "human"
    config = {
        "virtual_leader": vl,
        "render_fps": render_fps,
        "dt": dt,
        "distance_headway": [d_h for _ in range(n_vehicles)],
        "time_headway": [t_h for _ in range(n_vehicles)],
        "virtual_leader": vl,
    }
    env = PlatoonEnv(config, render_mode)

    # Set up CCMPC controller
    qp, qv, r = 1.0, 1.0, 10.0
    qms = 1.0
    confidence_level = 0.95
    u_min, u_max = -3., 3.
    v_min, v_max, a_min, a_max = 0., 100., u_min, u_max
    headway_args = (t_h, d_h, t_h_safe, d_h_safe)
    bound_args = (v_min, v_max, a_min, a_max, u_min, u_max)
    A = env.vehs[0].dyn.Ad
    B = env.vehs[0].dyn.Bd
    ccmpc_args = (A, B, qp, qv, r, dt, model_type, confidence_level)
    model_args = (model_dir, model_name, device)
    ccmpc = CCMPC(*model_args, *ccmpc_args, *headway_args, *bound_args, qms=qms)
    H = ccmpc.N  # horizon length

    # Set up DMPC controllers for the rest of the vehicles
    dmpcs : list[DMPC] = []
    for i in range(1, n_vehicles):
        dyn = env.vehs[i].dyn
        n, m, p = dyn.n, dyn.m, dyn.p
        Q = np.eye(p)
        Q_neighbors = [np.eye(p)]
        R = 10 * np.eye(m)
        A = dyn.Ad
        B = dyn.Bd
        C = dyn.C
        x_lims = np.array([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]])
        u_lims = np.array([[-np.inf, np.inf]])
        u_slew_rate = np.array([np.inf])
        distance_headways = [d_h]
        time_headways = [t_h]
        terminal_constraint = False
        Qf = Q
        Qf_neighbors = Q_neighbors
        output_norm = "quadratic"
        input_norm = "quadratic"
        dmpc_args = (H, Q, Q_neighbors, R, A, B, C)
        veh_args = (x_lims, u_lims, u_slew_rate, distance_headways, time_headways)
        dmpc_extra_args = (
            terminal_constraint, Qf, Qf_neighbors, output_norm, input_norm
        )
        dmpcs.append(DMPC(*dmpc_args, *veh_args, *dmpc_extra_args))
    
    ctrls = [ccmpc] + dmpcs
    
    # Start sim
    obs, env_info = env.reset()
    for n, ctrl in enumerate(ctrls):
        y_neighbors = [ctrls[n - 1].prev_assumed_state] if n > 0 else []
        ctrl.reset(env_info['vehicle_states'][n], obs[n], y_neighbors=y_neighbors)

    while True:
        try:
            env.render()

            actions = []
            for i in range(n_vehicles):
                action, _ = ctrls[i]()
                actions.append(action)
                
            # Step environment
            obs, _, term, trunc, env_info = env.step(actions)
            for n, ctrl in enumerate(ctrls):
                y_neighbors = [ctrls[n - 1].prev_assumed_state] if n > 0 else []
                ctrl.step(env_info['vehicle_states'][n], obs[n], y_neighbors=y_neighbors)
            if term or trunc:
                obs, env_info = env.reset()
                for n, ctrl in enumerate(ctrls):
                    y_neighbors = [ctrls[n - 1].prev_assumed_state] if n > 0 else []
                    ctrl.reset(env_info['vehicle_states'][n], obs[n], y_neighbors=y_neighbors)
            
        except KeyboardInterrupt:
            break
    env.close()


if __name__ == '__main__':
    main()