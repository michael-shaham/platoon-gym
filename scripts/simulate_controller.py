import argparse
import copy
import numpy as np
import torch

from i24_forecasting.utils.general_utils import get_project_dir

from platoon_gym.veh.virtual_leader import VirtualLeader
from platoon_gym.sim.platoon_simulation import PlatoonSimulation
from platoon_gym.ctrl.ccmpc import CCMPC
from platoon_gym.ctrl.dmpc import DMPC
from platoon_gym.ctrl.linear_feedback import LinearFeedback

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Simulate a platoon with a specified controller.")

    # Required arguments
    parser.add_argument("--controller", type=str, choices=["CCMPC", "DMPC", "LFBK"], required=True, help="Controller type.")
    
    # Environment arguments
    parser.add_argument("--dt", type=float, default=0.1, help="Simulation timestep.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--num_vehicles", type=int, default=50, help="Number of vehicles in the platoon.")
    parser.add_argument("--distance_headway", type=float, default=4.0, help="Desired distance headway between vehicles.")
    parser.add_argument("--time_headway", type=float, default=1.0, help="Desired time headway between vehicles.")
    parser.add_argument("--render", action="store_true", help="Render if true.")
    parser.add_argument("--max_trajectories", type=int, default=100, help="Maximum simulation steps.")
    parser.add_argument("--log_filename", type=str, default=None, help="HDF5 log filename. Defaults to <controller_name>_simulation_logs.hdf5.")
    
    # Controller agnostic arguments
    parser.add_argument("--u_min", type=float, default=-3.0, help="Minimum control input.")
    parser.add_argument("--u_max", type=float, default=3.0, help="Maximum control input.")
    parser.add_argument("--v_min", type=float, default=-2.0, help="Minimum velocity.")
    parser.add_argument("--v_max", type=float, default=100.0, help="Maximum velocity.")
    parser.add_argument("--a_min", type=float, default=-3.0, help="Minimum acceleration.")
    parser.add_argument("--a_max", type=float, default=3.0, help="Maximum acceleration.")
    
    # CCMPC-specific arguments
    parser.add_argument("--model_dir", type=str, default="models", help="Directory containing forecasting model.")
    parser.add_argument("--model_name", type=str, default="forecasting_model_best", help="Forecasting model filename.")
    parser.add_argument("--use_truncated_gaussian", action="store_true", help="Use truncated Gaussian distribution.")
    parser.add_argument("--use_quantile", action="store_true", help="Use quantile regression.")
    parser.add_argument("--model_architecture", type=str, default="transformer", help="Forecasting model embedding architecture.")
    parser.add_argument("--r_ccmpc", type=float, default=100.0, help="Control effort weight for CCMPC.")
    parser.add_argument("--qp_ccmpc", type=float, default=0.1, help="Position error weight for CCMPC.")
    parser.add_argument("--qv_ccmpc", type=float, default=2.0, help="Velocity error weight for CCMPC.")
    parser.add_argument("--qms_ccmpc", type=float, default=0.0, help="Move suppression weight for CCMPC.")
    parser.add_argument("--confidence_levels", nargs="*", type=float, default=[0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5], help="Confidence levels for CCMPC.")
    parser.add_argument("--t_h_safe", type=float, default=0.1, help="Safe time headway.")
    parser.add_argument("--d_h_safe", type=float, default=1.0, help="Safe distance headway.")
    parser.add_argument("--ctrl_alpha", type=float, default=1.0, help="Exponential filtering hyperparameter for control inputs.")

    # DMPC-specific arguments
    parser.add_argument("--use_terminal_constraint", action="store_true", help="Use terminal state constraint.")
    parser.add_argument("--qp_dmpc", type=float, default=1.0, help="Position error weight for DMPC.")
    parser.add_argument("--qv_dmpc", type=float, default=2.0, help="Velocity error weight for DMPC.")
    parser.add_argument("--r_dmpc", type=float, default=1.0, help="Control effort weight for DMPC.")
    parser.add_argument("--qf_dmpc", type=float, default=0.0, help="Terminal state error weight for DMPC.")
    parser.add_argument("--qnf_dmpc", type=float, default=1000.0, help="Terminal neighboring vehicle state error weight for DMPC.")
    parser.add_argument("--output_norm", type=str, default="quadratic", help="Output norm type for DMPC.")
    parser.add_argument("--input_norm", type=str, default="quadratic", help="Input norm type for DMPC.")

    # LFBK-specific arguments
    parser.add_argument("--not_saturate", action="store_true", help="Saturate control input.")
    parser.add_argument("--k", nargs="+", type=float, default=[0.5, 1.0], help="Linear feedback gain.")
    
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set up virtual leader
    vl_traj_args = {"dt": args.dt, "seed": args.seed}
    vl = VirtualLeader("i24_trajectory", vl_traj_args)
    
    # Set up environment
    env_config = {
        "dt": args.dt,
        "seed": args.seed,
        "num_vehicles": args.num_vehicles,
        "distance_headway": args.distance_headway,
        "time_headway": args.time_headway,
        "virtual_leader": vl,
    }
    if args.render:
        env_config.update({'render_mode': 'human', 'render_fps': 100})

    # Initialize controllers
    controllers = []
    controller_args_list = []

    # CCMPC controller
    model_dir = get_project_dir() / args.model_dir / args.model_architecture
    model_type = "truncated_gaussian" if args.use_truncated_gaussian else "gaussian"
    model_type = "quantile" if args.use_quantile else model_type  # overrides
    model_dir = model_dir / model_type if not args.use_quantile else model_dir / "gaussian"
    ccmpc_args = {
        "model_dir": model_dir,
        "model_name": args.model_name,
        "device": device,
        "qp": args.qp_ccmpc,
        "qv": args.qv_ccmpc,
        "r": args.r_ccmpc,
        "dt": args.dt,
        "model_type": model_type,
        "t_h": args.time_headway,
        "d_h": args.distance_headway,
        "t_h_safe": args.t_h_safe,
        "d_h_safe": args.d_h_safe,
        "v_min": args.v_min,
        "v_max": args.v_max,
        "a_min": args.a_min,
        "a_max": args.a_max,
        "u_min": args.u_min,
        "u_max": args.u_max,
        "qms": args.qms_ccmpc,
    }

    # DMPC controller
    dmpc_args = {
        "Q": np.diag(np.array([args.qp_dmpc, args.qv_dmpc])),
        "Q_neighbors": [np.diag(np.array([args.qp_dmpc, args.qv_dmpc]))],
        "R": args.r_dmpc,
        "u_slew_rate": np.array([np.inf]),
        "distance_headways": [args.distance_headway],
        "time_headways": [args.time_headway],
        "terminal_constraint": args.use_terminal_constraint,
        "Qf": args.qf_dmpc,
        "Qf_neighbors": [args.qnf_dmpc],
        "output_norm": args.output_norm,
        "input_norm": args.input_norm,
    }

    # LFBK controller
    saturate = not args.not_saturate
    lfbk_args = {
        "k": np.array(args.k),
        "distance_headway": args.distance_headway,
        "time_headway": args.time_headway,
        "saturate": saturate,
        "control_min": args.u_min,
        "control_max": args.u_max,
    }

    if 'mpc' in args.controller.lower():
        controllers.append(CCMPC)  # first vehicle always uses CCMPC
        ctrl_args = copy.deepcopy(ccmpc_args)
        controller_args_list.append(ctrl_args)

    if args.controller.lower() == "ccmpc":
        for _ in range(1, args.num_vehicles):
            controllers.append(CCMPC)
            ctrl_args = copy.deepcopy(ccmpc_args)
            controller_args_list.append(ctrl_args)
    elif args.controller.lower() == "dmpc":
        for _ in range(1, args.num_vehicles):
            x_lims = np.array([
                [-np.inf, np.inf], [args.v_min, args.v_max], [-np.inf, np.inf]
            ])
            u_lims = np.array([[args.u_min, args.u_max]])
            ctrl_args = copy.deepcopy(dmpc_args)
            ctrl_args.update({"x_lims": x_lims, "u_lims": u_lims})
            controllers.append(DMPC)
            controller_args_list.append(ctrl_args)
    elif args.controller.lower() == "lfbk":
        for _ in range(args.num_vehicles):
            ctrl_args = copy.deepcopy(lfbk_args)
            controllers.append(LinearFeedback)
            controller_args_list.append(ctrl_args)

    # Backup controllers
    backup_controllers = []
    backup_controller_args_list = []
    for _ in range(args.num_vehicles):
        backup_controllers.append(LinearFeedback)
        backup_controller_args_list.append(copy.deepcopy(lfbk_args))
    
    # Run simulation
    log_filename = args.controller.lower() 
    log_filename += f"_{model_type}_{args.max_trajectories}_trajectories"
    log_filename += f"_{args.num_vehicles}_vehicles"
    log_filename += f"_{args.time_headway:.2f}s_{args.distance_headway:.2f}m_headways"
    log_filename += f"_{args.qms_ccmpc:.2f}_qms.hdf5"
    simulation = PlatoonSimulation(
        env_config, 
        controllers, 
        controller_args_list, 
        backup_controllers,
        backup_controller_args_list,
        args.confidence_levels, 
        log_filename=log_filename,
        ctrl_alpha=args.ctrl_alpha,
    )
    simulation.simulate(args.max_trajectories)
    simulation.close()
    
    print(f"Simulation completed. Results saved in {simulation.log_filename}")


if __name__ == "__main__":
    main()
