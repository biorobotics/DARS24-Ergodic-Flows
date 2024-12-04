from erg_map import ErgodicMap
from optimizer import Optimizer
import matplotlib.pyplot as plt

import numpy as np
import os


class ProblemUtil:
    def __init__(self, name: str, map: ErgodicMap, opt: Optimizer):
        self.map = map
        self.opt = opt

        if not os.path.exists(name):
            os.mkdir(name)
        self.save_path = name

    def make_weights(self, seed: int, legdur=1., tot_dur=10.):
        opt = self.opt
        self.tot_dur = tot_dur

        np.random.seed(seed=seed)
        num_legs = int(np.ceil((tot_dur / legdur) + 3))
        wtraj0 = np.random.random((num_legs, self.map.num_vfs)) - .5
        # wtraj0 = np.zeros((num_legs, self.map.num_vfs))

        def wvec2(t):
            i = int(t / legdur)
            j = i + 1
            perc = np.mod(t / legdur, 1)
            out = (1 - perc) * wtraj0[i] + perc * wtraj0[j]
            return out

        retimed = np.arange(0, tot_dur + 2*opt._dur, opt._dur)
        wtraj = np.array([wvec2(t) for t in retimed])
        opt.set_weights(wtraj)

    def gen_initial_traj(self, x0):
        tot_dur = self.tot_dur
        opt = self.opt
        ts, path, erg = opt.forward(0, x0, dur=tot_dur, progress_bar=True)

        ts = ts[:, 0]
        xs = np.moveaxis(path, -2, 0)
        fp = os.path.join(self.save_path, 'path0.npz')
        np.savez_compressed(fp, ts=ts, xs=xs, weights=opt.weights)
        return ts, xs

    def save_weights(self, i, ts, xs):
        fp = os.path.join(self.save_path, f'path{i}.npz')
        np.savez_compressed(fp, ts=ts, xs=xs, weights=self.opt.weights)

    def load_weights(self, i):
        fp = os.path.join(self.save_path, f'path{i}.npz')
        data = np.load(fp)
        ts = data['ts']
        xs = data['xs']
        self.opt.set_weights(data['weights'])
        return ts, xs

    def draw_traj(self, xs):
        plt.ion()
        if len(xs.shape) == 2:
            self.lineplot, = plt.plot(*xs.T, '.-', c='r')
        else:
            self.lineplot = []
            self.scatterplots = []
            for i in range(xs.shape[0]):
                l, = plt.plot(*xs[i].T, '-', c='r')
                self.lineplot.append(l)
                ss = plt.scatter(*xs[i].T, c='r', s=10, zorder=4)
                self.scatterplots.append(ss)
            self.line_ends = plt.scatter(*xs[:, -1].T, c='k', s=50, marker='x', zorder=5)
            self.line_starts = plt.scatter(*xs[:, 0].T, c='w', s=50, marker='*', zorder=5)

    def update_traj(self, xs):
        if type(self.lineplot) is list:
            self.line_ends.set_offsets(xs[:, -1])
            for i in range(len(self.lineplot)):
                self.lineplot[i].set_data(*xs[i].T)
                self.scatterplots[i].set_offsets(xs[i])

                near_viols = self.map.dist2boundary(xs[i]) <= 2e-3
                col = np.array(['b' if z else 'r'for z in near_viols])
                self.scatterplots[i].set_color(col)
            plt.gcf().canvas.draw()
            plt.gcf().canvas.flush_events()
        else:
            self.lineplot.set_data(*xs.T)  # type: ignore
            plt.gcf().canvas.draw()
            plt.gcf().canvas.flush_events()
