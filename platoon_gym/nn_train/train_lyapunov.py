import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import torch
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from typing import List, Optional, Tuple

from platoon_gym.dyn.linear_vel import LinearVel
from platoon_gym.dyn.platoon_sys import PlatoonDoubleIntegrator
from platoon_gym.envs.platoon_env import PlatoonEnv
from platoon_gym.nn_verify.lyap_verify import (
    verify_lyapunov_positivity,
    verify_lyapunov_decreasing,
)
from platoon_gym.nn_train.lyapunov_utils import (
    create_control_model,
    create_lyapunov_model,
    generate_new_lyapunov_model,
    initialize_dynamics,
    initialize_vehicles,
)
from platoon_gym.veh.virtual_leader import VirtualLeader

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


class DoubleIntLyapunovControllerTrainer:
    """
    Implements the work described in the paper "Learning a Stable, Safe,
    Distributed Feedback Controller for a Heterogeneous Platoon of Vehicles" by
    Michael Shaham and Taskin Padir. This class is not general --- it is
    specifically designed for the system described in the paper (a platoon of
    vehicles with linear velocity dynamics that we can then model as double
    integrators.)
    """

    def __init__(
        self,
        save_dir: str,
        guide_control: bool = True,
        max_episodes: int = 1000,
        control_hidden_dimensions: List[int] = [8, 8],
        lyapunov_hidden_dimensions: List[int] = [8, 8],
        control_learning_rate: float = 1e-2,
        lyapunov_learning_rate: float = 1e-2,
        lyapunov_loss_weight: float = 10.0,
        lyapunov_lambdas: List[int] = [1.0, 1.0],
        control_lambdas: List[int] = [1e-1, 1e-2, 1e-2, 1e-3, 0, 1e-3, 1e-2],
        control_limit: float = 3.0,
        desired_distance: float = 5.0,
        distance_safety_margin: float = 2.0,
        far_distance_margin: float = 3.0,
        num_vehicles_start: int = 1,
        num_vehicles_end: int = 3,
        error_bounds_lower: float = 0.1,
        error_bounds_upper: float = 2.0,
        leader_speed_error_lower: float = 0.1,
        leader_speed_error_upper: float = 5.0,
        reset_time: float = 30.0,
        train_dataset: bool = True,
        lyapunov_condition_factors: List[int] = [0.01, 0.01],
        error_decreasing_weight: float = 0.1,
        dataset_size: int = 100000,
        batch_size: int = 1024,
        num_epochs: int = 300,
        num_vehicles_increment: int = 1,
        check_frequency: int = 1,
        optimization_time_limit: int = 120,
        activation_function: torch.nn.Module = torch.nn.LeakyReLU,
        activation_function_slope: Optional[float] = 0.1,
        tau_lims: List[float] = [0.2, 0.8],
        velocity_initial_limits: List[float] = [15.0, 30.0],
        control_loss_forward_prop: int = 50,
        max_desired_action: float = 1.2,
        control_slew_max: float = 0.1,
        render_mode: Optional[str] = None,
        device: str = device,
        discrete_timestep: float = 0.1,
    ):
        self.save_dir = save_dir
        self.gc = guide_control
        self.max_episodes = max_episodes
        self.lhd = lyapunov_hidden_dimensions
        self.clr = control_learning_rate
        self.llr = lyapunov_learning_rate
        self.llw = torch.tensor(lyapunov_loss_weight).float().to(device)
        self.lcf = torch.tensor(lyapunov_condition_factors).float().to(device)
        self.llams = torch.tensor(lyapunov_lambdas).float().to(device)
        self.clams = torch.tensor(control_lambdas).float().to(device)
        self.cl = control_limit
        self.d_des = desired_distance
        self.dsm = distance_safety_margin
        self.fdm = far_distance_margin
        self.edw = error_decreasing_weight
        self.nvs = num_vehicles_start
        self.nve = num_vehicles_end
        self.nvi = num_vehicles_increment
        self.ebl = error_bounds_lower
        self.ebu = error_bounds_upper
        self.lsel = leader_speed_error_lower
        self.lseu = leader_speed_error_upper
        self.td = train_dataset
        self.ds = dataset_size
        self.bs = batch_size
        self.ne = num_epochs
        self.cf = check_frequency
        self.otl = optimization_time_limit
        if isinstance(activation_function, torch.nn.LeakyReLU):
            self.af = torch.nn.LeakyReLU(activation_function_slope)
        else:
            self.af = torch.nn.ReLU()
        self.tau_lims = tau_lims
        self.v_init_lims = velocity_initial_limits
        self.clfp = control_loss_forward_prop
        self.mda = max_desired_action
        self.csm = control_slew_max
        self.rm = render_mode
        self.dt = discrete_timestep

        # initialize neural networks
        self.ctrl, self.ctrl_file = create_control_model(
            control_hidden_dimensions, self.af, self.gc, self.cl, device, save_dir
        )
        self.lyap, self.lyap_file = None, None

        # algorithm args
        self.rng = np.random.default_rng()
        self.error_bounds = None
        self.ploss = None
        self.dloss = None
        self.perr = None
        self.derr = None

        # environment args
        self.vl_traj_type = "constant velocity"
        self.vl_traj_args = {"horizon": 1, "dt": self.dt}
        plot_size = (6, 4)
        dpi = 100 if sys.platform.startswith("linux") else 50
        if ("linux" not in sys.platform) and ("darwin" not in sys.platform):
            exit("Unsupported OS found: {}".format(sys.platform))
        self.env_args = {
            "headway": "CDH",
            "topology": "PF",
            "dt": self.dt,
            "plot size": plot_size,
            "render dpi": dpi,
            "reset time": reset_time,
        }

        # save algorithm data
        self.eval_eps, self.pos_losses, self.dec_losses = None, None, None

    def train_lyapunov_controller(self):
        env = None
        ctrl_opt = torch.optim.SGD(self.ctrl.parameters(), lr=self.clr)
        try:
            for n_vehs in range(self.nvs, self.nve + 1, self.nvi):
                self.eval_eps, self.pos_losses, self.dec_losses = [], [], []
                if self.lyap is None:
                    self.lyap, self.lyap_file = create_lyapunov_model(
                        self.nvs, self.lhd, self.af, self.gc, device, self.save_dir
                    )
                else:
                    self.lyap, self.lyap_file = generate_new_lyapunov_model(
                        n_vehs - self.nvi,
                        n_vehs,
                        self.lhd,
                        self.af,
                        self.gc,
                        device,
                        self.save_dir,
                        self.lyap,
                    )
                lyap_opt = torch.optim.SGD(self.lyap.parameters(), lr=self.llr)
                print(f"\nTraining for {n_vehs} vehicles...\n")
                # initialize platooning environment
                d_des_list = [0] + [self.d_des] * n_vehs
                self.env_args["desired distance"] = d_des_list
                self.error_bounds, self.max_error_bounds = self.update_error_bounds(
                    n_vehs, d_des_list
                )
                self.platoon_control_bounds = [np.array([-self.cl, self.cl])] * n_vehs
                self.platoon_sys = PlatoonDoubleIntegrator(
                    n_vehs, self.dt, use_torch=True, device=device
                )
                # evaluate the lyapunov conditions initially if we are checking
                if self.cf >= 1:
                    self.ploss, self.dloss, self.perr, self.derr = self.eval_lyap()
                    self.eval_eps.append(0)
                    self.pos_losses.append(self.ploss)
                    self.dec_losses.append(self.dloss)

                # track best model
                min_ploss = self.ploss if self.ploss is not None else float("inf")
                min_dloss = self.dloss if self.dloss is not None else float("inf")
                best_ctrl = copy.deepcopy(self.ctrl)
                best_lyap = copy.deepcopy(self.lyap)

                dataset = []
                for ep_num in range(self.max_episodes):
                    # initialize env
                    v_init = self.rng.uniform(*self.v_init_lims)
                    dyns = initialize_dynamics(
                        n_vehs, self.dt, *self.tau_lims, self.rng
                    )
                    taus = [dyn.tau for dyn in dyns]
                    errs = self.initialize_errors()
                    vehs = initialize_vehicles(n_vehs, dyns, d_des_list, v_init, errs)
                    vl = VirtualLeader(
                        self.vl_traj_type, self.vl_traj_args, velocity=v_init
                    )
                    if env is None:
                        env = PlatoonEnv(vehs, vl, self.env_args, render_mode=self.rm)
                        obs, env_info = env.reset()
                    else:
                        options = {
                            "vehicles": vehs,
                            "virtual leader": vl,
                            "desired distance": self.d_des,
                        }
                        obs, env_info = env.reset(options=options)
                    veh_states = env_info["vehicle states"]

                    print(f"\nEpisode {ep_num + 1} of {self.max_episodes}...\n")
                    print(f"error bounds:\n{self.error_bounds}")
                    print(f"virtual leader velocity: {env.vl.state[1]:.2f}")
                    print(f"starting states:")
                    for i, v in enumerate(vehs):
                        print(f"{i}: {v.state[0]:.2f}, {v.state[1]:.2f}")
                    print()

                    while True:
                        env.render()
                        errs = []
                        for i in range(len(obs)):
                            err = [*(obs[i] - np.array([d_des_list[i], 0.0]))]
                            errs.append(err)
                        errs = torch.from_numpy(np.array(errs)).float().to(device)
                        actions = self.ctrl(errs).detach().cpu()
                        actions = actions.squeeze() if n_vehs > 1 else actions
                        actions_env = [
                            np.array([actions[i].item() * taus[i] + veh_states[i][1]])
                            for i in range(n_vehs)
                        ]

                        # step the lyapunov network and controller
                        control_loss = (
                            self.control_loss(errs.reshape(1, -1)) if self.gc else 0.0
                        )
                        lyapunov_loss = self.lyapunov_loss(errs.reshape(1, -1))
                        loss = self.llw * lyapunov_loss + control_loss
                        if self.td and lyapunov_loss > 0:
                            dataset.append(errs.reshape(1, -1).cpu().numpy())

                        ctrl_opt.zero_grad()
                        lyap_opt.zero_grad()
                        loss.backward()
                        ctrl_opt.step()
                        lyap_opt.step()

                        print(
                            f"ep: {ep_num+1}, "
                            + f"lyap loss: {lyapunov_loss.item():.3f}, "
                            + f"1-norm err: {torch.norm(errs[:, :2], 1):.2f}, "
                            + f"loss: {loss.item():.3f}, "
                            + f"maxact: {actions.abs().max().item():.3f}, "
                            + f"avgact: {actions.abs().mean().item():.3f}, "
                        )

                        # step environment
                        obs, _, term, trunc, env_info = env.step(actions_env)
                        if trunc:
                            veh_states = env_info["vehicle states"]
                            if self.td:
                                self.train_dataset(dataset)
                            break
                        if term:
                            print("collided")
                        veh_states = copy.deepcopy(env_info["vehicle states"])

                    if (ep_num + 1) % self.cf == 0:
                        self.ploss, self.dloss, self.perr, self.derr = self.eval_lyap()
                        self.eval_eps.append(ep_num + 1)
                        self.pos_losses.append(self.ploss)
                        self.dec_losses.append(self.dloss)
                        if self.ploss < min_ploss and self.dloss < min_dloss:
                            best_ctrl = copy.deepcopy(self.ctrl)
                            best_lyap = copy.deepcopy(self.lyap)
                        min_ploss = min(min_ploss, self.ploss)
                        min_dloss = min(min_dloss, self.dloss)
                        if self.ploss < 1e-5 and self.dloss < 3e-5:
                            if np.allclose(self.error_bounds, self.max_error_bounds):
                                torch.save(best_ctrl.state_dict(), self.ctrl_file)
                                torch.save(best_lyap.state_dict(), self.lyap_file)
                                self.save_loss_plot(n_vehs)
                                break
                            print("Increasing error bounds to:\n", self.error_bounds)
                            self.error_bounds, self.max_error_bounds = (
                                self.update_error_bounds(
                                    n_vehs,
                                    d_des_list,
                                    prev_error_bounds=self.error_bounds,
                                )
                            )

                    else:
                        self.ploss = None
                        self.dloss = None
                        best_ctrl = copy.deepcopy(self.ctrl)
                        best_lyap = copy.deepcopy(self.lyap)

        except KeyboardInterrupt:
            env.close()
            torch.save(best_ctrl.state_dict(), self.ctrl_file)
            torch.save(best_lyap.state_dict(), self.lyap_file)
            self.save_loss_plot(n_vehs)
            exit()

    def lyapunov_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Lyapunov loss for the given error state x. The
        positivity loss is given by
            L_pos = ReLU(a_1 * ||x||_1 - V(x))
        and the decreasing loss is given by
            L_dec = ReLU(V(f(x, pi(x))) - V(x) + a_2 * V(x))
        where a is the Lyapunov condition weights. The returned loss is then
            L = lam_1 * L_pos + lam_2 * L_dec.

        Args:
            x: torch.Tensor, shape (n_samples, 2 * n_vehs), error state

        Returns:
            torch.Tensor, the Lyapunov loss averaged over the samples
        """
        assert x.ndim == 2
        n_samples = x.shape[0]
        n_vehs = x.shape[1] // 2
        lyap_curr = self.lyap(x).view(n_samples)
        action = self.ctrl(x.view(n_samples, n_vehs, 2)).view(n_samples, -1)
        x_next = self.platoon_sys.forward_error(x.t(), action.t()).t()
        lyap_next = self.lyap(x_next).view(n_samples)
        pos_loss = F.relu(self.lcf[0] * torch.norm(x.t(), p=1, dim=0) - lyap_curr)
        dec_loss = F.relu(lyap_next - (1 - self.lcf[1]) * lyap_curr)
        loss = self.llams[0] * pos_loss + self.llams[1] * dec_loss

        # check to make sure all is well
        # assert lyap_curr.shape == (n_samples,)
        # assert lyap_next.shape == (n_samples,)
        # assert pos_loss.shape == (n_samples,)
        # assert dec_loss.shape == (n_samples,)
        # for i in range(n_samples):
        #     xi = x[i]
        #     ai = self.ctrl(xi.view(n_vehs, -1))
        #     ai = ai.squeeze() if n_vehs > 1 else ai.view(1)
        #     x_nexti = self.platoon_sys.forward_error(xi, ai)
        #     lyap_curri = self.lyap(xi.view(1, -1)).squeeze().item()
        #     lyap_nexti = self.lyap(x_nexti.view(1, -1)).squeeze().item()
        #     plossi = F.relu(self.lcf[0] * torch.norm(xi, 1) - lyap_curri).item()
        #     dlossi = F.relu(lyap_nexti - (1 - self.lcf[1]) * lyap_curri).item()
        #     assert np.isclose(lyap_curri, lyap_curr[i].item(), rtol=0.0, atol=1e-4)
        #     assert np.isclose(lyap_nexti, lyap_next[i].item(), rtol=0.0, atol=1e-4)
        #     assert np.isclose(plossi, pos_loss[i].item(), rtol=0.0, atol=1e-4)
        #     assert np.isclose(dlossi, dec_loss[i].item(), rtol=0.0, atol=1e-4)

        return loss.mean()

    def control_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the control loss for the given error state x. The control
        loss is a user-defined function that penalizes the controller for
        doing bad things (as specified by a user).

        Args:
            x: torch.Tensor, shape (n_samples, 2 * n_vehs), the error state

        Returns:
            torch.Tensor, the control loss
        """
        if torch.count_nonzero(self.clams) == 0:
            return torch.tensor(0.0).to(device)
        assert x.ndim == 2
        n_samples = x.shape[0]
        n_vehs = x.shape[1] // 2
        x = x.view(n_samples, n_vehs, -1)
        safety_loss = self.safety_loss(x)
        distance_loss = self.distance_loss(x)
        err_dec_loss = torch.zeros((n_samples, n_vehs)).to(device)
        pos_err_loss = self.position_error_loss(x)
        vel_err_loss = self.velocity_error_loss(x)
        action_loss = torch.zeros((n_samples, n_vehs)).to(device)
        slew_loss = torch.zeros((n_samples, n_vehs)).to(device)
        prev_actions = None
        for _ in range(self.clfp):
            actions = self.ctrl(x).squeeze(-1)
            x_next = (
                self.platoon_sys.forward_error(
                    x.reshape(n_samples, -1).t(), actions.t()
                )
                .t()
                .view(n_samples, n_vehs, -1)
            )

            # check to make sure all is well
            # for i in range(n_samples):
            #     xi = x[i].view(-1)
            #     ai = self.ctrl(xi.view(n_vehs, -1))
            #     ai = ai.squeeze() if n_vehs > 1 else ai.view(1)
            #     xi_next = self.platoon_sys.forward_error(xi, ai)
            #     assert torch.allclose(xi_next, x_next[i].view(-1), rtol=0.0, atol=1e-4)

            safety_loss += self.safety_loss(x_next)
            distance_loss += self.distance_loss(x_next)
            err_dec_loss += self.error_decreasing_loss(x, x_next)
            pos_err_loss += self.position_error_loss(x_next)
            vel_err_loss += self.velocity_error_loss(x_next)
            action_loss += self.action_loss(actions)
            if prev_actions is not None:
                slew_loss += self.slew_loss(prev_actions, actions)

            x = x_next.clone()
            prev_actions = actions.clone()

        # print(f'{safety_loss.mean(-1)=}')
        # print(f'{distance_loss.mean(-1)=}')
        # print(f'{err_dec_loss.mean(-1)=}')
        # print(f'{pos_err_loss.mean(-1)=}')
        # print(f'{vel_err_loss.mean(-1)=}')
        # print(f'{action_loss.mean(-1)=}')
        # print(f'{slew_loss.mean(-1)=}')
        # exit()

        return (
            self.clams[0] * safety_loss.mean(-1)
            + self.clams[1] * distance_loss.mean(-1)
            + self.clams[2] * F.relu(err_dec_loss).mean(-1)
            + self.clams[3] * pos_err_loss.mean(-1)
            + self.clams[4] * vel_err_loss.mean(-1)
            + self.clams[5] * action_loss.mean(-1)
            + self.clams[6] * slew_loss.mean(-1)
        ).mean(0)

    def safety_loss(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(-x[:, :, 0] - self.d_des + self.dsm)

    def distance_loss(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x[:, :, 0] - self.fdm)

    def error_decreasing_loss(
        self, x: torch.Tensor, x_next: torch.Tensor
    ) -> torch.Tensor:
        return (x_next - x + self.edw * torch.abs(x)).sum(-1)

    def position_error_loss(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, 0] ** 2

    def velocity_error_loss(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, 1] ** 2

    def action_loss(self, actions: torch.Tensor) -> torch.Tensor:
        return F.relu(actions.abs() - self.mda)

    def slew_loss(
        self, actions: torch.Tensor, next_actions: torch.Tensor
    ) -> torch.Tensor:
        return F.relu((next_actions - actions).abs() - self.csm)

    def eval_lyap(self) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Evaluates the Lyapunov optimization problems to determine the maximum
        violation of the Lyapunov conditions given the current Lyapunov
        network and controller.

        Returns:
            float: the maximum positivity violation
            float: the maximum decreasing violation
            np.ndarray: the error state that maximizes the positivity violation
            np.ndarray: the error state that maximizes the decreasing violation
        """
        print("\nVerifying Lyapunov function positivity...\n")
        pos_loss, eopt = verify_lyapunov_positivity(
            self.lyap,
            self.error_bounds,
            self.lcf[0].item(),
            time_limit=self.otl,
        )
        err_opt_pos = eopt.reshape((-1, 2))
        err_opt = torch.from_numpy(eopt).float().to(device).view(1, -1)
        pos_loss_calc = (
            self.lcf[0] * torch.norm(err_opt.view(-1), 1) - self.lyap(err_opt)
        ).item()
        print(f"err_opt=\n{err_opt.view(-1, 2)}")
        print(f"calculated optimal value: {pos_loss_calc}")
        print(f"cp optimal value: {pos_loss}")

        print("\nVerifying Lyapunov function decreasing...\n")
        dec_loss, eopt, eopt_next, aopt = verify_lyapunov_decreasing(
            self.lyap,
            self.ctrl,
            self.error_bounds,
            self.lcf[1].item(),
            self.dt,
            state_size=2,
            ctrl_lims=self.platoon_control_bounds,
            time_limit=self.otl,
        )
        err_opt_dec = eopt.reshape((-1, 2))
        err_opt = torch.from_numpy(eopt).float().to(device).view(1, -1)
        err_next_opt = torch.from_numpy(eopt_next).float().to(device).view(1, -1)
        dec_loss_calc = (
            self.lyap(err_next_opt) - (1 - self.lcf[1]) * self.lyap(err_opt)
        ).item()

        print(f"err_opt=\n{err_opt.view(-1, 2)}")
        print(f"a_opt=\n{aopt}")
        print(f"calculated optimal value: {dec_loss_calc}")
        print(f"cp optimal value: {dec_loss}\n")

        time.sleep(1.0)

        return pos_loss_calc, dec_loss_calc, err_opt_pos, err_opt_dec

    def initialize_errors(self) -> np.ndarray:
        """
        Initialize the starting platoon errors (distances from desired
        positions and velocity differences). If checked the Lyapunov conditions
        previously during the episode, use one of the error states given by the
        max positivity or decreasing condition violation. If not, randomly
        select an error state on the boundary of the error bounds.

        Returns:
            np.ndarray, shape (n_vehs, 2), the initial error state
        """
        if self.perr is not None and self.derr is not None:
            rb = self.rng.integers(0, 2)
            if np.isclose(self.ploss, 0.0, rtol=0.0, atol=1e-5):
                rb = 1
            elif np.isclose(self.dloss, 0.0, rtol=0.0, atol=1e-5):
                rb = 0
            return self.perr if rb == 0 else self.derr
        else:
            nbounds = self.error_bounds.shape[0]
            nvehs = nbounds // 2
            ris = self.rng.integers(0, 2, size=nbounds)
            return self.error_bounds[np.arange(nbounds), ris].reshape(nvehs, -1)

    def train_dataset(self, dataset: List[np.ndarray]):
        """
        Performs stochastic gradient descent on the dataset augmented with
        random points such that the size of the datset is equal to self.ds.

        Args:
            dataset: list of np.ndarray of shape (1, 2 * n_vehs)
        """
        if self.dloss is not None and self.ploss is not None:
            if np.isclose(self.dloss, 0.0, rtol=0.0, atol=1e-5) and np.isclose(
                self.ploss, 0.0, rtol=0.0, atol=1e-5
            ):
                return
        print(f"dataset size: {len(dataset)}")
        if len(dataset) < self.ds:
            rand_errs = (
                torch.from_numpy(
                    self.rng.uniform(
                        low=1.1 * self.max_error_bounds[:, 0],
                        high=1.1 * self.max_error_bounds[:, 1],
                        size=(self.ds - len(dataset), self.error_bounds.shape[0]),
                    )
                )
                .float()
                .to(device)
            )
            torch_data = torch.vstack(
                (
                    torch.from_numpy(np.concatenate(dataset, axis=0))
                    .float()
                    .to(device),
                    rand_errs,
                )
            )
        else:
            torch_data = (
                torch.from_numpy(np.concatenate(dataset, axis=0)).float().to(device)
            )
        lopt = torch.optim.AdamW(self.lyap.parameters(), lr=self.llr)
        copt = torch.optim.AdamW(self.ctrl.parameters(), lr=self.clr)
        warmup_epochs = self.ne // 2

        def lr_warmup(epoch: int):
            return epoch / warmup_epochs if epoch < warmup_epochs else 1.0

        lopt_warmup = LambdaLR(lopt, lr_warmup)
        copt_warmup = LambdaLR(copt, lr_warmup)
        for i in range(self.ne):
            # train on a batch
            self.train_batch(torch_data, lopt, copt)
            # evaluate on the dataset
            lyapunov_loss = self.lyapunov_loss(torch_data)
            print(f"epoch {i+1}, lyapunov loss: {lyapunov_loss.item():.8f}")
            lopt_warmup.step()
            copt_warmup.step()
            if lyapunov_loss < 5e-9:
                return

    def train_batch(
        self, x: torch.Tensor, lopt: torch.optim.Optimizer, copt: torch.optim.Optimizer
    ) -> None:
        """
        Trains the Lyapunov and control networks on the given dataset for
        one batch.

        Args:
            x: torch.Tensor, shape (n_samples, 2 * n_vehs), the dataset
            lopt: torch.optim.Optimizer, the Lyapunov optimizer
            copt: torch.optim.Optimizer, the control optimizer
        """
        N = x.shape[0]  # num_samples
        inds = torch.multinomial(
            torch.ones(N), num_samples=min(self.bs, N), replacement=False
        )
        xb = x[inds]
        lyapunov_loss = self.lyapunov_loss(xb)
        control_loss = self.control_loss(xb) if self.gc else 0.0
        loss = self.llw * lyapunov_loss + control_loss
        lopt.zero_grad()
        copt.zero_grad()
        loss.backward()
        lopt.step()
        copt.step()

    def update_error_bounds(
        self,
        num_vehicles: int,
        d_des_list: List[float],
        error_bounds_increase_factor: float = 1.5,
        prev_error_bounds: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates the error bounds we will verify over. If prev_error_bounds is None,
        then we create the error bounds from scratch using the minimum values.
        Otherwise, we update the error bounds based on the previous error bounds.
        Also returns the max error bounds based on the max values provided.
        """
        assert num_vehicles >= 1
        assert self.ebl >= 0.0
        assert self.ebu >= self.ebl
        assert self.lsel >= self.ebl
        assert self.lseu >= self.lsel
        assert self.lseu >= self.ebu

        max_error_bounds = np.array(
            [[-self.ebu, self.ebu] for _ in range(2 * num_vehicles)]
        )
        max_error_bounds[1, :] = np.array([-self.lseu, self.lseu])

        if prev_error_bounds is None:
            error_bounds = np.array(
                [[-self.ebl, self.ebl] for _ in range(2 * num_vehicles)]
            )
            error_bounds[1, :] = np.array([-self.lsel, self.lsel])
        else:
            error_bounds = error_bounds_increase_factor * prev_error_bounds
            if (np.abs(error_bounds[0, :]) > self.ebu).any():
                error_bounds[0, 0] = -self.ebu
                error_bounds[0, 1] = self.ebu
                if num_vehicles > 1:
                    error_bounds[2:, 0] = -self.ebu
                    error_bounds[2:, 1] = self.ebu
            if (np.abs(error_bounds[1, :]) > self.lseu).any():
                error_bounds[1, 0] = -self.lseu
                error_bounds[1, 1] = self.lseu

        for k, i in enumerate(range(2, 2 * num_vehicles, 2)):
            error_bounds[i, 0] = max(error_bounds[i, 0], -d_des_list[k + 1] / 2.0)
            max_error_bounds[i, 0] = max(
                max_error_bounds[i, 0], -d_des_list[k + 1] / 2.0
            )

        return error_bounds, max_error_bounds

    def save_loss_plot(self, n_vehs: int):
        """
        Creates a plot of the maximum Lyapunov condition violation for each
        episode in which it was evaluated.
        """
        fig_file = os.path.join(
            self.save_dir, "results", f"verification_losses_{n_vehs}.pdf"
        )
        data_file = os.path.join(
            self.save_dir, "data", f"verification_losses_{n_vehs}.csv"
        )
        if not os.path.exists(os.path.dirname(fig_file)):
            os.makedirs(os.path.dirname(fig_file))
        if not os.path.exists(os.path.dirname(data_file)):
            os.makedirs(os.path.dirname(data_file))
        plt.figure()
        plt.title("Maximum Lyapunov condition violation")
        plt.plot(self.eval_eps, self.pos_losses, label="positivity violation")
        plt.plot(self.eval_eps, self.dec_losses, label="decreasing violation")
        plt.xlabel("episode")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(fig_file, bbox_inches="tight")
        np_data = np.vstack((self.eval_eps, self.pos_losses, self.dec_losses)).T
        np.savetxt(data_file, np_data)
        # plt.show()
