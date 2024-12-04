from typing import Literal
from erg_map import ErgodicMap
from optimizer import Optimizer
import shapely
import numpy as np
import time
import argparse

import shapely.plotting
import matplotlib.pyplot as plt
import matplotlib

from problem_util import ProblemUtil


def square_nonuniform(mp: ErgodicMap):
    g1 = np.exp(-100*np.sum((mp.mesh_points - [.15, .4])**2, axis=1))
    g2 = np.exp(-np.sum([200, 40] * (mp.mesh_points - [.825, .75])**2, axis=1))
    return g1 + 2*g2


def maze_nonuniform(mp: ErgodicMap):
    points = mp.mesh_points
    g1 = np.exp(-100*(points[:, 0]-.5)**2 - 400*(points[:, 1]-.55)**2)
    g2 = np.exp(-50*(points[:, 0]-.25)**2 - 50*(points[:, 1]-.25)**2)
    return .5*g1 + g2


def rooms_nonuniform(mp: ErgodicMap):
    points = mp.mesh_points
    g1 = np.exp(-100*(points[:, 0]-.1)**2 - 100*(points[:, 1]+.5)**2)
    g2 = np.exp(-300*(points[:, 0]-.05)**2 - 300*(points[:, 1]-.5)**2)
    return g1 + g2


distributions = {
    'uniform': lambda mp: np.ones(len(mp.mesh_points)),
    'square_nonuniform': square_nonuniform,
    'maze_nonuniform': maze_nonuniform,
    'rooms_nonuniform': rooms_nonuniform,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='square', help='Map type (square/maze/rooms)')
    parser.add_argument('--uniform', type=bool, default=True,
                        help='Uniform or non-uniform distribution', action=argparse.BooleanOptionalAction)
    parser.add_argument('--log_dir', type=str, default='logs/', help='Logging directory')
    parser.add_argument('--agents', type=int, default=1, help='Number of agents')
    parser.add_argument('--vmax', type=float, default=1, help='Max agent velocity')
    parser.add_argument('--steps', type=int, default=1000, help='Num optimization steps')
    parser.add_argument('--dt', type=float, default=1e-2, help='Timestep size')
    parser.add_argument('--alpha', type=float, default=1, help='Optimizer step size')
    args = parser.parse_args()
    print(args)

    # which_map: Literal['square', 'maze', 'rooms'] = 'rooms'
    which_map = args.map
    if which_map not in ['square', 'maze', 'rooms']:
        print(f'Invalid map: {which_map}')
        exit(0)

    # which_distr: Literal['uniform', 'nonuniform'] = 'uniform'
    which_distr = 'uniform' if args.uniform else 'nonuniform'
    max_v = {
        'square': 1,
        'maze': 1,
        'rooms': 1,
    }[which_map]

    mp = ErgodicMap(f'{which_map}/{which_map}.npz')
    if which_distr == 'uniform':
        distr = distributions['uniform'](mp)
    else:
        distr = distributions[f'{which_map}_nonuniform'](mp)

    mp.set_info_distr(distr)
    opt = Optimizer(mp, dt=args.dt, alpha=args.alpha, v_max=1)
    prob = ProblemUtil(f'{args.log_dir}', mp, opt)

    i = 0
    num_steps = args.steps
    if i == 0:
        np.random.seed(0)
        x0 = np.array([mp.sample_in_region() for _ in range(args.agents)])

        prob.make_weights(0, tot_dur=3)
        ts, xs = prob.gen_initial_traj(x0)
        print(xs.shape)

    else:
        ts, xs = prob.load_weights(i)

    plt.ion()
    fig = plt.figure(figsize=(8, 8))
    shapely.plotting.plot_polygon(mp.map_geom, add_points=False)
    plt.tricontourf(mp.mesh_points[:, 0], mp.mesh_points[:, 1],
                    distr, levels=100, triangles=mp.mesh_triangles)
    prob.draw_traj(xs)
    fig.canvas.draw()
    fig.canvas.flush_events()

    print('init erg val', mp.ergodicity(xs))
    t = time.time()

    for ii in range(i+1, i + num_steps + 1):
        step = opt.get_socp_step(ts, xs)

        # perform update
        xs = xs + step[:xs.size].reshape(xs.shape)
        xs = opt.map.proj_region(xs)
        opt.from_paramvec(opt.paramvec() + step[xs.size:])

        prob.update_traj(xs)
        print(f'Iter {ii}; Time: {time.time() - t}')
        print('erg metric', mp.ergodicity(xs))
        x2 = opt.one_timestep(np.tile(ts[:-1], len(xs)),
                              xs[:, :-1].reshape(-1, 2))
        # print('constraint', np.abs(x2.flatten() - xs[:, 1:].flatten()).max())
        mdx = np.linalg.norm(xs[:,1:] - xs[:,:-1], axis=-1).max()
        # print('MAX DX', mdx, 'SHOULD BE', opt.max_dx, f'({mdx / opt._dt:.02f})')
        prob.save_weights(ii, ts, xs)

    print('Done!')
    plt.ioff()
    plt.show()
