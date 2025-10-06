import copy
import h5py
import numpy as np
import torch
from tqdm import tqdm
import traceback

from platoon_gym.ctrl.ccmpc import CCMPC
from platoon_gym.ctrl.dmpc import DMPC
from platoon_gym.ctrl.linear_feedback import LinearFeedback
from platoon_gym.envs.platoon_env import PlatoonEnv
from platoon_gym.utils.general_utils import get_project_dir
from platoon_gym.veh.virtual_leader import VirtualLeader


class PlatoonSimulation:
    """
    Simulates the platoon with the given controller classes and arguments 
    provided to each class. Used to log the results of the simulation, to then 
    compare the performance of platoons using different controllers.
    """

    def __init__(
        self, 
        env_config: dict, 
        controller_classes: list[type], 
        controller_args_list: list[dict], 
        backup_controller_classes: list[type],
        backup_controller_args_list: list[dict],
        confidence_levels: list[float],
        log_filename: str | None = None,
        ctrl_alpha: float = 1.0,
        attack_frequency: float = 0.0,
        attack_duration: float = 0.0,
        attacked_vehicles: list[int] = [],
        attack_times: list[float] = [],
        noise_std: float = 0.0
    ):
        """
        Initializes the platoon simulation with a given list of controllers.

        Args:
            env_config: Dictionary of PlatoonGym environment arguments.
            controller_classes: List of controller classes to instantiate.
            controller_args_list: List of init arguments for each controller.
            backup_controller_classes: List of backup controller classes.
            backup_controller_args_list: List of init arguments for each backup 
                controller.
            confidence_levels: List of confidence levels for CCMPC controllers.
            log_filename: Name of the HDF5 file to save logs. 
                Default: test_simulation_logs.hdf5
            ctrl_alpha: Hyperparameter in exponential filtering for control 
                inputs: u_t = alpha * u_t + (1 - alpha) * u_t-1.
                Default: 1.0 (no smoothing)
            attack_frequency: Frequency of DoS attacks.
            attack_duration: Duration of DoS attacks.
            attacked_vehicles: List of vehicle indices to be attacked.
            attack_times: List of attack times.
            noise_std: Standard deviation of noise to add to states and observations.
        """
        self.env_config = env_config
        self.confidence_levels = confidence_levels
        self.ctrl_alpha = ctrl_alpha
        self.seed = env_config.get("seed", 42)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        self.num_vehicles = env_config["num_vehicles"]
        self.dt = env_config["dt"]
        self.noise_std = noise_std
        log_filename = log_filename or "test_simulation_logs.hdf5"
        log_dir = get_project_dir() / "data"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_filename = log_dir / log_filename

        # set up attack scenario
        self.attack_times = attack_times
        if attack_frequency > 0.0 and attack_duration > 0.0:
            self.attack_frequency = attack_frequency
            self.attack_duration = attack_duration
            self.vehicle_attacked = [False for _ in range(self.num_vehicles)]
        elif attack_duration > 0.0 and len(attacked_vehicles) > 0 and len(attack_times) > 0:
            self.attack_duration = attack_duration
            self.attacked_vehicle_inds = attacked_vehicles
            self.vehicle_attacked = [False for _ in range(self.num_vehicles)]
        
        # Set up virtual leader using i24_trajectories
        vl_traj_args = {"dt": self.dt, "seed": self.seed}
        self.virtual_leader = VirtualLeader("i24_trajectory", vl_traj_args)

        d_h = env_config["distance_headway"]
        t_h = env_config["time_headway"]
        env_config["distance_headway"] = [d_h for _ in range(self.num_vehicles)]
        env_config["time_headway"] = [t_h for _ in range(self.num_vehicles)]
        
        # Set up platoon environment
        render_mode = env_config.get("render_mode", None)
        env_config["virtual_leader"] = self.virtual_leader
        self.env = PlatoonEnv(env_config, render_mode=render_mode)
        self.num_vehicles = len(self.env.vehs)
        
        # Initialize controllers
        self.og_controllers: list[DMPC | CCMPC | LinearFeedback] = []
        for i in range(self.num_vehicles):
            if 'MPC' in controller_classes[i].__name__:
                A = self.env.vehs[i].dyn.Ad
                B = self.env.vehs[i].dyn.Bd
                controller_args_list[i].update({"A": A, "B": B})
            if controller_classes[i].__name__ == 'DMPC':
                C = self.env.vehs[i].dyn.C
                controller_args_list[i].update({"H": self.og_controllers[0].N, "C": C})
            controller = controller_classes[i](**controller_args_list[i])
            self.og_controllers.append(controller)
        
        # Initialize backup controllers
        self.og_backup_controllers: list[LinearFeedback | CCMPC] = []
        for i in range(self.num_vehicles):
            if 'MPC' in controller_classes[i].__name__:
                A = self.env.vehs[i].dyn.Ad
                B = self.env.vehs[i].dyn.Bd
                backup_controller_args_list[i].update({"A": A, "B": B})
            controller = backup_controller_classes[i](**backup_controller_args_list[i])
            self.og_backup_controllers.append(controller)
        
        # Prepare HDF5 logging
        self.log_file = h5py.File(self.log_filename, "a")
        self.log_file.attrs["num_vehicles"] = self.num_vehicles
        self.log_file.attrs["dt"] = self.dt
        self.log_file.attrs["distance_headway"] = d_h
        self.log_file.attrs["time_headway"] = t_h
        self.traj_count = self.log_file.attrs.get("num_traj", 0)
    
    def simulate(self, num_trajectories: int):
        """Simulates the environment for a given number of trajectories."""
        print(f"Already completed {self.traj_count} trajectories.")
        for epoch in tqdm(range(num_trajectories)):
            self.run_one_epoch(epoch)
    
    def run_one_epoch(self, epoch: int):
        """Runs the simulation and logs results to HDF5."""
        self.controllers = copy.deepcopy(self.og_controllers)
        self.backup_controllers = copy.deepcopy(self.og_backup_controllers)
        _, env_info = self.reset()

        if epoch < self.traj_count:
            return

        # Initialize lists to store results
        vl_states = []
        veh_states = [[] for _ in range(self.num_vehicles)]
        veh_controls = [[] for _ in range(self.num_vehicles)]
        veh_attacked = [[] for _ in range(self.num_vehicles)]
        collided = False

        prev_ctrls = [0.0 for _ in range(self.num_vehicles)]

        while True:
            self.env.render()

            vl_states.append(self.env.vl.state)
            actions = []

            # figure out attack scenario
            for at in self.attack_times:
                if at <= self.env.time < at + self.attack_duration:
                    for attacked_veh_ind in self.attacked_vehicle_inds:
                        self.vehicle_attacked[attacked_veh_ind] = True
                else:
                    for attacked_veh_ind in self.attacked_vehicle_inds:
                        self.vehicle_attacked[attacked_veh_ind] = False
            
            for i, ctrl in enumerate(self.controllers):
                veh_states[i].append(env_info["vehicle_states"][i])
                if self.vehicle_attacked[i]:
                    veh_attacked[i].append(True)
                else:
                    veh_attacked[i].append(False)
                
                for j, c in enumerate(self.confidence_levels):
                    try:
                        if self.vehicle_attacked[i] or (i > 0 and self.vehicle_attacked[i - 1]):
                            # print(f'Vehicle {i} is under attack! Using backup controller {self.backup_controllers[i].__class__.__name__}')
                            action, _ = self.backup_controllers[i](confidence_level=c)
                        else:
                            action, _ = ctrl(confidence_level=c)
                        break
                    except Exception as e:
                        print(f"Error in controller {i} with confidence level {c}: {e}")
                        if j == len(self.confidence_levels) - 1:
                            print("All confidence levels failed. Using default action.")
                            action, _ = self.backup_controllers[i]()
                        print(traceback.format_exc())

                # Exponential smoothing
                if isinstance(self.controllers[i], CCMPC):
                    action = self.ctrl_alpha * action + (1 - self.ctrl_alpha) * prev_ctrls[i]
                    prev_ctrls[i] = action
                actions.append(action)
                veh_controls[i].append(action)

            _, _, term, trunc, env_info = self.step(actions)

            if term:
                collided = True

            if trunc:
                for i in range(self.num_vehicles):
                    veh_states[i].append(env_info["vehicle_states"][i])
                vl_states.append(self.env.vl.state)
                break
        
        # Store results in HDF5
        traj_key = f"traj_{self.traj_count}"
        traj_group = self.log_file.create_group(traj_key)
        traj_group.create_dataset("vl_states", data=np.array(vl_states))
        traj_group.create_dataset("veh_states", data=np.array(veh_states))
        traj_group.create_dataset("veh_controls", data=np.array(veh_controls))
        traj_group.create_dataset("veh_attacked", data=np.array(veh_attacked))
        traj_group.attrs["collided"] = collided
        
        self.traj_count += 1
    
    def reset(self):
        """Resets the simulation environment."""
        obs, env_info = self.env.reset()
        self.vehicle_attacked = [False for _ in range(self.num_vehicles)]
        for n in range(self.num_vehicles):
            # add noise to states and obs
            state_dim = env_info["vehicle_states"][n].shape[0]
            state = env_info["vehicle_states"][n] + np.random.normal(0, self.noise_std, state_dim)
            observation = obs[n] + np.random.normal(0, self.noise_std, obs[n].shape[0])

            # reset default controllers
            use_neighbors = isinstance(self.controllers[n], DMPC)
            y_neighbors = [self.controllers[n - 1].prev_assumed_state] if use_neighbors else None
            self.controllers[n].reset(state, observation, y_neighbors=y_neighbors)

            # reset backup controllers (e.g., if comms failure)
            self.backup_controllers[n].reset(state, observation)
        return obs, env_info
    
    def step(self, actions: list) -> tuple:
        """Steps the simulation environment."""
        obs, reward, term, trunc, env_info = self.env.step(actions)

        # update assumed states for controller/backup controller
        for n in range(self.num_vehicles):
            # update assumed states for controller/backup controller
            if self.vehicle_attacked[n] or (n > 0 and self.vehicle_attacked[n - 1]):
                if isinstance(self.controllers[n], DMPC) and isinstance(self.backup_controllers[n], CCMPC):
                    self.controllers[n].assumed_state = copy.deepcopy(self.backup_controllers[n].assumed_state)
            else:
                if isinstance(self.controllers[n], DMPC) and isinstance(self.backup_controllers[n], CCMPC):
                    self.backup_controllers[n].assumed_state = copy.deepcopy(self.controllers[n].assumed_state)

        # step controllers
        for n in range(self.num_vehicles):
            # add noise to states and obs
            state_dim = env_info["vehicle_states"][n].shape[0]
            state = env_info["vehicle_states"][n] + np.random.normal(0, self.noise_std, state_dim)
            observation = obs[n] + np.random.normal(0, self.noise_std, obs[n].shape[0])

            # step default controller
            use_neighbors = isinstance(self.controllers[n], DMPC)
            y_neighbors = [self.controllers[n - 1].prev_assumed_state] if use_neighbors else None
            self.controllers[n].step(state, observation, y_neighbors=y_neighbors)

            # step backup controller (e.g., if comms failure)
            self.backup_controllers[n].step(state, observation)

        return obs, reward, term, trunc, env_info

    def render(self):
        """Renders the simulation environment."""
        self.env.render()

    def close(self):
        """Closes the HDF5 file."""
        self.log_file.attrs["num_traj"] = self.traj_count
        self.log_file.close()