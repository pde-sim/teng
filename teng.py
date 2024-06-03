import json
import logging
import os
import time
from argparse import ArgumentParser
from datetime import datetime
from decimal import Decimal
from functools import partial

import jax
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp

import numpy as np

from src.model import SimplePDENet3
from src.sampler import PeriodicQuadratureSampler
from src.var_state import SimpleVarStateReal
from src.operator import HeatOperatorNoLog, AllenCahnOperator, BurgersOperator
from src.utils import RandomNaturalPolicyGradTDVP

now = datetime.now()


def get_config():
    parser = ArgumentParser()

    ### general configs ###
    parser.add_argument("--D", type=float, nargs='+', default=[1 / 10])
    parser.add_argument("--equation", type=str, default='heat', choices=['heat', 'allen_cahn', 'burgers'])
    parser.add_argument("--nb_steps", type=int, default=800)
    parser.add_argument("--nb_iters_per_step", type=int, default=5)
    parser.add_argument("--dt", type=str, default='0.005')
    parser.add_argument("--integrator", type=str, default='heun', choices=['euler', 'heun'])
    parser.add_argument("--save_dir", type=str, nargs='?', default=None)  # can be emtpy

    ### model configs ###
    parser.add_argument("--load_model_state_from", type=str, nargs='?', default='init_model_state.pickle')
    parser.add_argument("--model_seed", type=int, default=1234)

    ### sampler configs ###
    parser.add_argument("--nb_samples", type=int, default=65536) # we used 262144 in the paper
    parser.add_argument("--sampler_seed", type=int, default=4321)

    ### policy grad configs ###
    parser.add_argument("--policy_grad_nb_params", type=int, default=1536)
    parser.add_argument("--policy_grad_seed", type=int, default=8844)
    parser.add_argument("--policy_grad2_nb_params", type=int, default=1024)
    parser.add_argument("--policy_grad2_seed", type=int, default=8848)

    args = parser.parse_args()
    if args.save_dir is None or args.save_dir.lower() == 'none':
        args.save_dir = f'./results/run_{now.strftime("%m-%d-%Y-%H-%M-%S")}/'
    else:
        args.save_dir = "./results/" + args.save_dir

    os.makedirs(args.save_dir, exist_ok=True)
    with open(args.save_dir + '/config.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    return args


def write_to_file(file, *items, flush=False):
    if type(items[0]) == list or type(items[0]) == tuple:
        items = items[0]
    for item in items:
        file.write('%s ' % item)
    file.write('\n')
    if flush:
        file.flush()


def square_loss_func(u, v):
    reward = -(u - v)
    loss = reward ** 2 / 2
    return reward, loss


@partial(jax.pmap, in_axes=(None, None, 0, 0, 0), static_broadcasted_argnums=0)
def loss_func_pure(var_state_pure, state, samples, sqrt_weights, u_target):
    u = var_state_pure.evaluate(state, samples)
    reward, losses = square_loss_func(u, u_target)
    loss = (losses * sqrt_weights ** 2).sum()
    return reward, loss


def loss_func(var_state, samples, sqrt_weights, u_target):
    return loss_func_pure(var_state.pure_funcs, var_state.get_state(), samples, sqrt_weights, u_target)


class CompareWithExact:
    def __init__(self, points_per_dim=512, config=None):

        self.points_per_dim = points_per_dim
        grid = jnp.linspace(0, 2 * jnp.pi, points_per_dim, endpoint=False)
        grid2d = jnp.stack(jnp.meshgrid(grid, grid, indexing='ij'), axis=-1).reshape(1, -1, 2)
        self.xs = grid2d
        if config.equation == 'heat':
            self.exact_solution_dir = 'heat_equation_2d_spectral_fourier'
        elif config.equation == 'allen_cahn':
            self.exact_solution_dir = 'allen_cahn_equation_2d_spectral_fourier'
        elif config.equation == 'burgers':
            self.exact_solution_dir = 'burgers_equation_2d_spectral_fourier'
        else:
            raise NotImplementedError
        if not os.path.exists(self.exact_solution_dir):
            raise FileNotFoundError(f'{self.exact_solution_dir} does not exist')

    def __call__(self, var_state, T: Decimal):
        try:
            exact_u_hat = np.load(os.path.join(self.exact_solution_dir, f'T_{T.normalize()}.npy'))
        except FileNotFoundError as e:
            logging.warning(
                f'Failed to load exact solution at {T=}, if you are using the provided exact solution, only selected time steps are provided due to file size limitations, {e=}')
            return np.nan, np.nan
        exact_u = self.ifft(exact_u_hat, max_N=self.points_per_dim).ravel()
        var_state_u = var_state.evaluate(self.xs).squeeze(0)
        abs_err = jnp.linalg.norm(exact_u - var_state_u)
        rel_err = abs_err / jnp.linalg.norm(exact_u)
        return abs_err.item() / self.points_per_dim * (
                2 * jnp.pi) ** 2, rel_err.item()  # points_per_dim is the same as sqrt(N)

    def ifft(self, x_hat, max_N):
        """Compute the inverse fourier transform of the given fourier coefficients"""
        x_hat = jnp.fft.ifftshift(x_hat)
        if max_N is not None:
            max_k = x_hat.shape[0] // 2
            new_x_hat = jnp.zeros((max_N, max_N), dtype=jnp.complex128)
            new_x_hat = new_x_hat.at[:max_k + 1, :max_k + 1].set(x_hat[:max_k + 1, :max_k + 1])
            new_x_hat = new_x_hat.at[:max_k + 1, -max_k:].set(x_hat[:max_k + 1, -max_k:])
            new_x_hat = new_x_hat.at[-max_k:, :max_k + 1].set(x_hat[-max_k:, :max_k + 1])
            new_x_hat = new_x_hat.at[-max_k:, -max_k:].set(x_hat[-max_k:, -max_k:])
            x_hat = new_x_hat
        x = jnp.fft.ifft2(x_hat, norm='forward')
        return x


def euler_step(config, fiters, T, step, dt, var_state_new, var_state_old, var_state_temps, pde_operator, policy_grad, policy_grad2):
    var_state_old.set_state(var_state_new.get_state())
    samples, _, sqrt_weights = var_state_old.sampler.sample(start=0)
    final_losses = []

    # first train var_state_temp0
    stage = 0
    var_state_new.set_state(var_state_old.get_state())  # var_state_old is a better initial guess
    u_old = var_state_old.evaluate(samples)
    u_old_dot = pde_operator(var_state_old, samples, u_old, compile=True)
    u_target = u_old + u_old_dot * float(dt)
    for iter in range(config.nb_iters_per_step + 2):
        reward, loss = loss_func(var_state_new, samples, sqrt_weights, u_target)
        update, info = (policy_grad if iter == 0 else policy_grad2)(samples=samples, sqrt_weights=sqrt_weights,
                                                                    rewards=reward, var_state=var_state_new,
                                                                    resample_params=True)
        info = tuple(each_info.item() if isinstance(each_info, jnp.ndarray) else each_info for each_info in info)
        var_state_new.update_parameters(update)
        loss = loss.squeeze().item()
        logging.info(f'{step=}, {T=}, {stage=}, {iter=}, {loss=}, {info=}')
        write_to_file(fiters, T, step, stage, iter, loss, *info, flush=False)
    loss = loss_func(var_state_new, samples, sqrt_weights, u_target)[1].squeeze().item()
    logging.info(f'{step=}, {T=}, {stage=}, {loss=}')
    final_losses.append(loss)

    return final_losses


def heun_step(config, fiters, T, step, dt, var_state_new, var_state_old, var_state_temps, pde_operator, policy_grad, policy_grad2):
    var_state_temp0 = var_state_temps[0]
    var_state_old.set_state(var_state_new.get_state())
    samples, _, sqrt_weights = var_state_old.sampler.sample(start=0)
    final_losses = []

    # first train var_state_temp0
    stage = 0
    var_state_temp0.set_state(var_state_old.get_state())  # var_state_old is a better initial guess
    u_old = var_state_old.evaluate(samples)
    u_old_dot = pde_operator(var_state_old, samples, u_old, compile=True)
    u_target = u_old + u_old_dot * float(dt)
    for iter in range(config.nb_iters_per_step + 2):
        reward, loss = loss_func(var_state_temp0, samples, sqrt_weights, u_target)
        update, info = (policy_grad if iter == 0 else policy_grad2)(samples=samples, sqrt_weights=sqrt_weights,
                                                                    rewards=reward, var_state=var_state_temp0,
                                                                    resample_params=True)
        info = tuple(each_info.item() if isinstance(each_info, jnp.ndarray) else each_info for each_info in info)
        var_state_temp0.update_parameters(update)
        loss = loss.squeeze().item()
        logging.info(f'{step=}, {T=}, {stage=}, {iter=}, {loss=}, {info=}')
        write_to_file(fiters, T, step, stage, iter, loss, *info, flush=False)
    loss = loss_func(var_state_temp0, samples, sqrt_weights, u_target)[1].squeeze().item()
    logging.info(f'{step=}, {T=}, {stage=}, {loss=}')
    final_losses.append(loss)

    # then train var_state_new
    stage = 1
    var_state_new.set_state(var_state_temp0.get_state())  # var_state_temp0 is a better initial guess
    u_temp0 = var_state_temp0.evaluate(samples)
    u_temp0_dot = pde_operator(var_state_temp0, samples, u_temp0, compile=True)
    u_target = u_old + (u_old_dot + u_temp0_dot) * float(dt / 2)
    for iter in range(config.nb_iters_per_step):
        reward, loss = loss_func(var_state_new, samples, sqrt_weights, u_target)
        update, info = policy_grad2(samples=samples, sqrt_weights=sqrt_weights, rewards=reward, var_state=var_state_new,
                                    resample_params=True)
        info = tuple(each_info.item() if isinstance(each_info, jnp.ndarray) else each_info for each_info in info)
        var_state_new.update_parameters(update)
        loss = loss.squeeze().item()
        logging.info(f'{step=}, {T=}, {stage=}, {iter=}, {loss=}, {info=}')
        write_to_file(fiters, T, step, stage, iter, loss, *info, flush=False)
    loss = loss_func(var_state_new, samples, sqrt_weights, u_target)[1].squeeze().item()
    logging.info(f'{step=}, {T=}, {stage=}, {loss=}')
    final_losses.append(loss)

    return final_losses


def save_states(config, var_state_new, var_state_old, var_state_temps, step):
    var_state_new.save_state(os.path.join(config.save_dir, f'var_state_new_{step}.pickle'))
    var_state_old.save_state(os.path.join(config.save_dir, f'var_state_old_{step}.pickle'))
    for i, var_state_temp in enumerate(var_state_temps):
        var_state_temp.save_state(os.path.join(config.save_dir, f'var_state_temp_{i}_{step}.pickle'))


def train(config, var_state_new, var_state_old, var_state_temps, pde_operator, policy_grad, policy_grad2):
    try:
        error_from_exact = CompareWithExact(config=config)
    except Exception as e:
        logging.warning(f'Exact solution directory not found, {e=}')
        error_from_exact = lambda *args: (np.nan, np.nan)
    if config.integrator == 'euler':
        stepper = euler_step
    elif config.integrator == 'heun':
        stepper = heun_step
    else:
        raise ValueError(f'Unknown integrator {config.integrator}')
    training_time = 0
    with open(os.path.join(config.save_dir, f'iters.txt'), 'w') as fiters, open(
            os.path.join(config.save_dir, f'steps.txt'), 'w') as fsteps:
        T = Decimal('0')
        dt = Decimal(config.dt)
        err = error_from_exact(var_state_new, T)
        logging.info(f'step={-1}, {T=}, {err=}, loss={(0, 0)}, {training_time=}')
        write_to_file(fsteps, -1, T, *err, 0, 0, flush=True)
        for step in range(config.nb_steps):
            T += dt
            start_time = time.perf_counter()
            loss = stepper(config, fiters, T, step, dt, var_state_new, var_state_old, var_state_temps, pde_operator, policy_grad, policy_grad2)
            training_time += time.perf_counter() - start_time
            fiters.flush()
            err = error_from_exact(var_state_new, T)
            logging.info(f'{step=}, {T=}, {err=}, {loss=}, {training_time=}')
            write_to_file(fsteps, step, T, *err, *loss, flush=True)
            save_states(config, var_state_new, var_state_old, var_state_temps, step)


def main():
    # get config
    config = get_config()

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    file_handler = logging.FileHandler(os.path.join(config.save_dir, "log.txt"), mode="w")
    logger.addHandler(file_handler)
    start_time = time.perf_counter()

    net = SimplePDENet3(width=40, depth=7, period=jnp.pi * 2)

    # the var_states can share the same sampler in this case
    sampler = PeriodicQuadratureSampler(nb_sites=2, nb_samples=config.nb_samples, minvals=0.,
                                        maxvals=jnp.pi * 2, quad_rule='trapezoid', rand_seed=config.sampler_seed)

    # define the var_state
    # we need to define multiple copies of the var_state for the intermediate results of heun's method
    # the net can be shared because it is just a pure function, which will not cause any issue

    var_state_new = SimpleVarStateReal(net=net, system_shape=(2,), sampler=sampler,
                                       init_seed=config.model_seed)
    var_state_old = SimpleVarStateReal(net=net, system_shape=(2,), sampler=sampler,
                                       init_seed=config.model_seed)
    # temporary var_states for storing the intermediate results of heun's method
    var_state_temps = []
    if config.integrator == 'heun':
        for _ in range(1):
            var_state_temps.append(SimpleVarStateReal(net=net, system_shape=(2,), sampler=sampler,
                                                      init_seed=config.model_seed))

    # load model state if needed
    if config.load_model_state_from is not None and config.load_model_state_from.lower() != 'none':
        # we will only load the state to the new var_state, and the old var_state will be updated by the new var_state
        var_state_new.load_state(config.load_model_state_from)

    # define the operator of the pde
    # first parse the input
    if len(config.D) == 1:
        diffusion_coefs = jnp.diag(jnp.ones(2) * config.D[0])
    elif len(config.D) == config.nb_dims:
        diffusion_coefs = jnp.diag(jnp.array(config.D))
    elif len(config.D) == config.nb_dims ** 2:
        diffusion_coefs = jnp.array(config.D).reshape(2, 2)
    else:
        raise ValueError(
            f'D can take either 1 argument or {config.nb_dims=} arguments or {config.nb_dims**2=} arguments, but got {len(config.D)=}')

    # then define the operator
    if config.equation == 'heat':
        drift_coefs = jnp.zeros(2)
        pde_operator = HeatOperatorNoLog(2, drift_coefs, diffusion_coefs, check_validity=True)
    elif config.equation == 'allen_cahn':
        pde_operator = AllenCahnOperator(2, diffusion_coefs, check_validity=True)
    elif config.equation == 'burgers':
        pde_operator = BurgersOperator(2, diffusion_coefs, check_validity=True)
    else:
        raise ValueError(f'Unknown equation: {config.equation}')

    # define policy grad function
    policy_grad = RandomNaturalPolicyGradTDVP(var_state=var_state_new, ls_solver=None,
                                              nb_params_to_take=config.policy_grad_nb_params,
                                              rand_seed=config.policy_grad_seed)
    policy_grad2 = RandomNaturalPolicyGradTDVP(var_state=var_state_new, ls_solver=None,
                                               nb_params_to_take=config.policy_grad2_nb_params,
                                               rand_seed=config.policy_grad2_seed)

    train(config, var_state_new, var_state_old, var_state_temps, pde_operator, policy_grad, policy_grad2)

    end_time = time.perf_counter()
    logger.info(f"Total time: {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    main()
