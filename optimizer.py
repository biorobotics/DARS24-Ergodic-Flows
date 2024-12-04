from erg_map import ErgodicMap
import numpy as np
import scipy.sparse as sp
import sparse as sp2
from tqdm import tqdm
import cvxpy as cp

from opt_einsum import contract


class Optimizer:
    weights: np.ndarray
    max_t: float
    eps = 1e-8
    socp_iters = 2000

    def __init__(self, map: ErgodicMap, leg_dur=.1, dt=1e-3, v_max=1., alpha=1.):
        self.map = map
        self.num_vfs = map.num_vfs

        self._dur = leg_dur
        self._dt = dt
        self.v_max = v_max
        self.max_dx = v_max * dt
        self.alpha = alpha

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_weights(self, weights: np.ndarray):
        if weights.shape[1] != self.num_vfs:
            raise RuntimeError(
                f'Given weights have wrong shape. Expected {self.num_vfs}; got {weights.shape[1]}')
        self.weights = weights
        self.max_t = weights.shape[0] * self._dur

    def paramvec(self):
        return self.weights.flatten()

    def from_paramvec(self, paramvec: np.ndarray):
        self.set_weights(paramvec.reshape(-1, self.num_vfs))

    def _wvec_pattern(self, ts: np.ndarray):
        i = np.floor(ts / self._dur).astype(int)
        _i = np.eye(len(self.weights))[i]
        _j = np.eye(len(self.weights))[i + 1]

        perc = np.mod(ts / self._dur, 1)[..., np.newaxis]
        return sp.csr_array((1 - perc) * _i + perc * _j)

    def _wvec(self, ts: np.ndarray):
        return self._wvec_pattern(ts) @ self.weights

    def forward(self, ts, xs, dur: float | None = None, progress_bar=False, return_path=True, return_erg=True, return_coeffs=False):
        ''' Rolls ODE forwards for `dur / dt` timesteps.
        args:
            xs (ndarray):        position(s) at (ts)
            ts (float):          start time(s), same shape as `xs`. If scalar, then aligned starts.
            dur (float):         sim duration
            progress_bar:        enable tqdm
            returns:      {path, erg, coeffs}
        @Returns
            ts:      times
            path:    trajectory
            erg:     ergodic metric
            coeffs:  trajectory fourier coeffs
        '''

        ts = np.atleast_1d(ts).astype(float)
        xs = np.atleast_2d(xs).astype(float)
        if xs.shape[0] != ts.shape[0]:
            if ts.shape == (1,):
                ts = np.tile(ts, xs.shape[0])
            else:
                raise RuntimeError(
                    f'x ({xs.shape}) and t ({ts.shape}) shape must match in 1st dim!')

        if dur is None:
            dur = self.max_t

        if np.any(ts + dur > self.max_t):
            raise RuntimeError(
                f'Given sim duration exceeds maximum time ({self.max_t})')

        dt = self._dt
        it = np.arange(0, dur, dt)
        if progress_bar:
            it = tqdm(it)

        def f(t, x) -> np.ndarray:
            x = self.map.proj_region(x)
            return contract('nv,vna->na', self._wvec(t),
                            self.map.vector_field(x))  # type: ignore

        x = xs + 0
        t = ts + 0
        path = [x]
        ts = [t]
        for _ in it:
            k1 = f(t, x)
            k2 = f(t + dt/2, x + dt*k1/2)
            k3 = f(t + dt/2, x + dt*k2/2)
            k4 = f(t + dt, x + dt*k3)

            if np.any(np.isnan([k1, k2, k3, k4])):
                raise RuntimeError(
                    f'Function evaluation failed at time {t}: nans encountered.')

            x = x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
            x = self.map.proj_region(x)
            t = t + dt

            path.append(x)
            ts.append(t)

        path = np.array(path)
        ret = []
        if return_path:
            ret.append(np.array(ts))
            ret.append(path)
        if return_erg:
            ret.append(self.map.ergodicity(path))
        if return_coeffs:
            ret.append(self.map.coeffs(path))
        return ret

    def dpath(self, ts: np.ndarray, xs: np.ndarray):
        ''' Rolls ODE forwards for one timestep, for (t, x) points.
        Tracks derivative of f(t, x) 

        args:
            ts (float):          start time(s), same shape as `xs`. If scalar, then aligned starts.
            xs (ndarray):        position(s) at (ts)
        @Returns
            dx_{t+1}/dx_{t} and dx_{t+1}/dw
        '''
        dt = self._dt

        def f(t, x) -> np.ndarray:
            x = self.map.proj_region(x)
            return contract('nv,vna->na', self._wvec(t),
                            self.map.vector_field(x))  # type: ignore

        def rk4_update(t, x):
            k1 = f(t, x)
            k2 = f(t + dt/2, x + dt*k1/2)
            k3 = f(t + dt/2, x + dt*k2/2)
            k4 = f(t + dt, x + dt*k3)
            return (k1 + 2*k2 + 2*k3 + k4) / 6

        v = rk4_update(ts, xs)
        Jx = (rk4_update(ts, xs + [self.eps, 0]) - v) / self.eps
        Jy = (rk4_update(ts, xs + [0, self.eps]) - v) / self.eps

        Jv = np.stack([Jx, Jy], axis=-1)
        dxdx = np.eye(2) + dt * Jv  # dense array

        _t, _w = self._wvec_pattern(ts).shape  # (T, W)
        _v, _t, _a = self.map.vector_field(xs).shape  # (V, T, 2)

        def dfdw(t, x) -> sp2.COO:
            x = self.map.proj_region(x)
            wpat = self._wvec_pattern(t)  # (T, W)
            vf = self.map.vector_field(x)  # (V, T, 2)

            w_up = sp.kron(wpat, np.ones((1, _a*_v)))  # (T, [W, 2, V])
            v_up = np.repeat(vf.transpose(1, 2, 0)
                             [:, np.newaxis], _w, axis=1)  # (N, W, 2, V)

            dvdw = w_up.multiply(v_up.reshape(_t, _a*_v*_w))
            return sp2.COO.from_scipy_sparse(dvdw).reshape([_t, _w, _a, _v]).transpose(
                [0, 2, 1, 3]).reshape([_t, _a, _w*_v])

        def dfdx(t, x):
            x = self.map.proj_region(x)
            f0 = f(t, x)
            fx = f(t, x + [self.eps, 0])
            fy = f(t, x + [0, self.eps])
            return np.stack([
                (fx - f0) / self.eps,
                (fy - f0) / self.eps
            ], axis=-1)

        k1 = f(ts, xs)
        k2 = f(ts + dt/2, xs + dt*k1/2)
        k3 = f(ts + dt/2, xs + dt*k2/2)
        k4 = f(ts + dt, xs + dt*k3)
        dk1dw = dfdw(ts, xs)

        dk2dw_partial = dfdw(ts + dt/2, xs + dt*k1/2)
        dk2dx_partial = dfdx(ts + dt/2, xs + dt*k1/2)

        dk2dw = dk2dw_partial + dt/2 * \
            contract('tab,tbw->taw', dk2dx_partial, dk1dw)  # type: ignore

        dk3dw_partial = dfdw(ts + dt/2, xs + dt*k2/2)
        dk3dx_partial = dfdx(ts + dt/2, xs + dt*k2/2)
        dk3dw = dk3dw_partial + dt/2 * \
            contract('tab,tbw->taw', dk3dx_partial, dk2dw)  # type: ignore

        dk4dw_partial = dfdw(ts + dt, xs + dt*k3)
        dk4dx_partial = dfdx(ts + dt, xs + dt*k3)
        dk4dw = dk4dw_partial + dt * \
            contract('tab,tbw->taw', dk4dx_partial, dk3dw)  # type: ignore

        dvdw = (dk1dw + 2*dk2dw + 2*dk3dw + dk4dw) / 6
        return dxdx, dt * dvdw

    def one_timestep(self, ts: np.ndarray, xs: np.ndarray):
        dt = self._dt

        def f(t, x) -> np.ndarray:
            x = self.map.proj_region(x)
            return contract('nv,vna->na',
                            self._wvec(t),
                            self.map.vector_field(x)
                            )  # type: ignore

        def rk4_update(t, x):
            k1 = f(t, x)
            k2 = f(t + dt/2, x + dt*k1/2)
            k3 = f(t + dt/2, x + dt*k2/2)
            k4 = f(t + dt, x + dt*k3)

            return (k1 + 2*k2 + 2*k3 + k4) / 6

        v = rk4_update(ts, xs)
        return xs + v*dt

    def get_socp_step(self, ts, xs) -> np.ndarray:
        alpha = self.alpha
        derg_dx = self.map.dergodicity(xs)
        obj_vec = np.r_[derg_dx.flatten(), np.zeros(self.weights.size)]

        num_trajs = int(np.prod(xs.shape[:-2]))
        traj_size = int(np.prod(xs.shape[-2:]))

        # Constraint matrix
        _ts, _xs = ts[:-1], xs[..., :-1, :]
        _ts_flat = np.tile(_ts, num_trajs)
        _xs_flat = _xs.reshape(-1, 2)
        z = self.one_timestep(_ts_flat, _xs_flat).reshape(num_trajs, -1, 2)

        dzdx, dzdw = self.dpath(_ts_flat, _xs_flat)

        # un-flatten dxdx and dxdw
        dzdx = dzdx.reshape((num_trajs, -1, 2, 2))
        dzdw = dzdw.reshape((num_trajs, -1, 2, self.weights.size))

        sp_dzdx = contract('ntij,mn,st->ntimsj', dzdx, sp2.eye(num_trajs),
                           sp2.eye(len(_ts)+1, len(_ts))
                           ).reshape((_xs.size, -1)).tocsr()  # type: ignore
        sp_dzdw = dzdw.reshape((_xs.size, self.weights.size)).tocsr()
        sp_cons = sp.bmat([[sp_dzdx, sp_dzdw]])

        end_lhs = sp.kron(sp.eye(num_trajs), sp.eye(traj_size-2, traj_size, 2))
        end_rhs = sp.csr_matrix((_xs.size, self.weights.size))
        sp_cons = sp_cons - sp.bmat([[end_lhs, end_rhs]])

        _xs1 = xs[..., 1:, :]
        b = (_xs1 - z).flatten()

        delta_vec = cp.Variable(xs.size + self.weights.size)
        dx_size = np.prod(xs.shape[-2:])
        dxs = [delta_vec[i*dx_size:(i+1)*dx_size].reshape(xs.shape[-2:], order='C')
               for i in range(num_trajs)]

        # CONSTRAINT 1: collocation
        cons = [sp_cons @ delta_vec == b]

        # CONSTRAINT 2: velocity limit
        for i in range(num_trajs):
            # resulting distance between points <= d
            xpost = xs[i] + dxs[i]
            cons.append(cp.SOC(self.max_dx * np.ones(len(_ts)),
                        xpost[:-1] - xpost[1:], axis=1))

        # CONSTRAINT 3: Geometry constraints
        near_viols = self.map.dist2boundary(xs, full=True) <= 2e-3
        for traj_i, time_j in zip(*np.where(np.any(near_viols, axis=-1))):
            xpost = xs[traj_i, time_j] + dxs[traj_i][time_j]
            A, b = self.map.as_lin_constr(near_viols[traj_i, time_j])
            cons.append(A@xpost + b >= 2e-3)

        # CONSTRAINT 4: FIXED START?
        for i in range(num_trajs):
            cons.append(dxs[i][0] == 0)

        # Quadratic objective: |c - x| <= t
        scales = np.r_[.05 * np.ones(xs.size), np.ones(self.weights.size)]
        M_inv = sp.diags(scales)
        M = sp.diags(1 / scales)
        t = cp.Variable()

        cons.append(cp.SOC(t, M @ (-alpha*obj_vec - delta_vec)))

        prob = cp.Problem(cp.Minimize(t), cons)
        prob.solve(solver=cp.SCS, max_iters=self.socp_iters, verbose=False)
        return delta_vec.value  # type: ignore
