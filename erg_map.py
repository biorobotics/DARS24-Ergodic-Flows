import numpy as np
from abc import ABC
import scipy.interpolate
import os

import shapely
from shapely.geometry.polygon import orient
import pygmsh
import spharapy.trimesh as tm
import spharapy.spharatransform as st

from opt_einsum import contract


class ErgodicMap(ABC):
    '''Uses spharapy basis functions.
    '''
    distr: np.ndarray
    map_coeffs: np.ndarray
    flow_scale = 1/400

    def __init__(self, cache_path: str, vectors_path: str | None = None, num_bases=100) -> None:
        super().__init__()
        with np.load(cache_path) as data:
            self.mesh_points = data['mesh_points']
            self.mesh_triangles = data['mesh_triangles']
            self.mesh_freqs = data['mesh_freqs'][:num_bases]
            self.mesh_basis = data['mesh_basis'][:, :num_bases]
            self.sphara_bary_A = data['A']
            self.sphara_bary_b = data['b']
            self.exterior_pts = data['exterior_pts']
            self.holes_pts = data['holes_pts']
            self.massmatrix = data['massmatrix']

        self.sobolev_norm = (1 + np.sqrt(self.mesh_freqs)) ** (-2)
        self.map_geom = shapely.Polygon(self.exterior_pts, self.holes_pts)

        geom_points = []
        for lines in [self.exterior_pts, *self.holes_pts]:
            for frm, to in zip(lines[:-1], lines[1:]):
                geom_points.append((frm, to))
        self.geom_points = np.array(geom_points)

        if vectors_path is None:
            vectors_path = os.path.join(os.path.dirname(
                os.path.abspath(cache_path)), 'vector-summary.npz')
        with np.load(vectors_path) as data:
            vec_xy = data['xy']
            vec_vals = data['vals'].reshape(len(vec_xy), -1)
            self.vf_interpolator = scipy.interpolate.CloughTocher2DInterpolator(
                vec_xy, vec_vals)
            self.num_vfs = vec_vals.shape[1] // 2

    @staticmethod
    def triangulate_geometry(cache_path: str, map_geom: shapely.Polygon,
                             mesh_size=.04, scale=1):
        # Extract map geometry & create trimesh
        map_geom = orient(map_geom)
        exterior_pts = np.array(map_geom.exterior.xy).T
        holes = [np.array(hole.xy).T for hole in map_geom.interiors]

        with pygmsh.occ.Geometry() as geom:
            total = geom.add_polygon(exterior_pts[:-1], mesh_size=mesh_size)
            for inner_pts in holes:
                inner = geom.add_polygon(inner_pts[:-1], mesh_size=mesh_size)
                total = geom.boolean_difference(total, inner)

            geom.synchronize()
            mesh = geom.generate_mesh()
        mesh.points = scale * mesh.points
        mesh_points = mesh.points[:, :2]
        mesh_triangles = mesh.cells[1].data.astype(int)

        # Calculate A & b matrices for finding nearest cell
        A = np.zeros((*mesh_triangles.shape, 2))
        b = np.zeros(mesh_triangles.shape)
        for i, tri in enumerate(mesh_triangles):
            _pts = mesh.points[tri].T + 0
            _pts[2] = 1
            mat = np.linalg.pinv(_pts)
            A[i] = mat[:, :2]
            b[i] = mat[:, 2]

        # Calculate Laplace-Beltrami eigenfunctions.
        sphara_mesh = tm.TriMesh(mesh_triangles, mesh.points)
        sphara_basis = st.SpharaBasis(sphara_mesh, 'fem')
        eigvec, eigval = sphara_basis.basis()
        eigval[np.isclose(eigval, 0)] = 0  # force near-zero to 0
        M = sphara_basis.massmatrix()

        np.savez_compressed(cache_path,
                            mesh_points=mesh_points,
                            mesh_triangles=mesh_triangles,
                            mesh_freqs=eigval,
                            mesh_basis=eigvec,
                            massmatrix=M,
                            A=A,
                            b=b,
                            exterior_pts=exterior_pts,
                            holes_pts=holes)

    def set_info_distr(self, distr: np.ndarray):
        if not np.isclose(distr @ self.massmatrix @ np.ones(len(distr)), 1):
            integral = distr @ self.massmatrix @ np.ones(len(distr))
            print(f'WARNING: info distr not normalized ({integral}). Normalizing.')
            distr = distr / integral

        self.distr = distr
        self.map_coeffs = distr @ self.massmatrix @ self.mesh_basis

    def vector_field(self, xy: np.ndarray) -> np.ndarray:
        out = self.vf_interpolator(xy).reshape(*xy.shape[:-1], -1, 2)
        return self.flow_scale * np.moveaxis(out, -2, 0)

    # Geometry constraint checking & region projection.
    def dist2boundary(self, xy: np.ndarray, full=False) -> np.ndarray:
        shape = xy.shape
        xy = xy.reshape(-1, 1, 2)

        frm = self.geom_points[:, 0]
        vec = self.geom_points[:, 1] - self.geom_points[:, 0]
        vnorm = np.linalg.norm(vec, axis=1)
        vhat = vec / vnorm[:, np.newaxis]

        dot = np.sum(vhat * (xy - frm), axis=-1)
        is_between = (0 <= dot) & (dot < vnorm)

        line_dist = np.sqrt(np.linalg.norm((xy - frm), axis=-1)**2 - dot**2)
        point_dist = np.amin(np.linalg.norm(
            self.geom_points - xy.reshape(-1, 1, 1, 2), axis=-1), axis=-1)

        dists = point_dist
        dists[is_between] = line_dist[is_between]

        out = dists.reshape(*shape[:-1], len(self.geom_points))
        if not full:
            return np.amin(out, axis=-1)
        return out

    def in_region(self, xy: np.ndarray, buf=0.) -> np.ndarray:
        shape = xy.shape
        xy = xy.reshape(-1, 2)

        a = self.geom_points[:, 0]
        b = self.geom_points[:, 1]
        v = b - a

        va = (a - xy[:, np.newaxis])
        vb = (b - xy[:, np.newaxis])

        v1v2 = np.divide(v[:, 0], v[:, 1], out=np.zeros(
            len(v)), where=v[:, 1] != 0)
        s = va[:, :, 0] - v1v2 * va[:, :, 1]
        positive_ray = s > 0
        intersects_line = (va @ [0, 1]) * (vb @ [0, 1]) < 0
        in_reg = np.sum((positive_ray & intersects_line)
                        .astype(int), axis=-1) % 2 == 1
        if buf == 0:
            return in_reg.reshape(shape[:-1])
        return in_reg.reshape(shape[:-1]) & (self.dist2boundary(xy) > buf)
    
    def sample_in_region(self):
        minv = np.amin(self.geom_points[:, 0], axis=0)
        maxv = np.amax(self.geom_points[:, 0], axis=0)

        prob_in_map = self.map_geom.area / np.prod(maxv - minv)
        prob_in_map = max(prob_in_map, 1e-4)
        for _ in range(5):
            pts = (maxv - minv) * np.random.random((int(1 / prob_in_map), 2)) + minv
            valid = self.in_region(pts)
            if np.any(valid):
                return pts[self.in_region(pts)][0]
        print('ERROR: Failed to sample any points in region!')
        exit(0)

    def proj_region(self, xy: np.ndarray) -> np.ndarray:
        sel = ~self.in_region(xy)
        if not np.any(sel):
            return xy

        out = xy + 0
        xy2 = xy[sel]
        dists = self.dist2boundary(xy2, full=True)
        proj = []

        for xi, line_ix in zip(xy2, np.argmin(dists, axis=-1)):
            a, b = self.geom_points[line_ix]
            vhat = (b - a) / np.linalg.norm(b-a)
            dot = (xi - a) @ vhat

            if (0 <= dot) & (dot < np.linalg.norm(b-a)):
                proj.append(a + dot * vhat)
            elif np.linalg.norm(xi-a) < np.linalg.norm(xi-b):
                proj.append(a)
            else:
                proj.append(b)

        out[sel] = np.array(proj)
        return out

    def as_lin_constr(self, which_lines: np.ndarray):
        '''Express near-boundary constraints as Ax + b >= 0
        '''
        which_lines = np.array(which_lines).reshape(-1)

        boundary_points = self.geom_points[which_lines]
        frm = boundary_points[:, 0]
        to = boundary_points[:, 1]

        A = (to - frm) / np.linalg.norm(to-frm, axis=-1)[:, np.newaxis]
        A = A @ [[0, 1], [-1, 0]]
        b = -np.sum(A * frm, axis=-1)
        return A, b

    # Calculating ergodicity
    def exp(self, xy: np.ndarray) -> np.ndarray:
        shape = xy.shape
        xy = xy.reshape(-1, 2)
        bary_coords: np.ndarray = contract(
            'ijk,ak->aij', self.sphara_bary_A, xy) + self.sphara_bary_b  # type: ignore
        ixs = np.argmax(np.all(bary_coords >= 0, axis=2), axis=1)
        tri_ixs = self.mesh_triangles[ixs]

        mx = contract('abj,aj,abc->ac',
                      self.sphara_bary_A[ixs], xy, self.mesh_basis[tri_ixs])
        b = contract(
            'ab,abc->ac', self.sphara_bary_b[ixs], self.mesh_basis[tri_ixs])
        tmp: np.ndarray = mx + b  # type: ignore
        return tmp.reshape(*shape[:-1], self.mesh_basis.shape[-1])

    def dexp(self, xy: np.ndarray) -> np.ndarray:
        shape = xy.shape
        xy = xy.reshape(-1, 2)
        bary_coords: np.ndarray = contract(
            'ijk,ak->aij', self.sphara_bary_A, xy) + self.sphara_bary_b  # type: ignore
        ixs = np.argmax(np.all(bary_coords >= 0, axis=2), axis=1)
        tri_ixs = self.mesh_triangles[ixs]
        tmp: np.ndarray = contract('abj,abc->acj', self.sphara_bary_A[ixs],
                                   self.mesh_basis[tri_ixs])  # type: ignore
        return tmp.reshape(*shape[:-1], self.mesh_basis.shape[1], 2)

    def coeffs(self, xy: np.ndarray) -> np.ndarray:
        return np.mean(self.exp(xy), axis=tuple(range(len(xy.shape) - 1)))

    def dcoeffs(self, xy: np.ndarray) -> np.ndarray:
        numel = xy.size / 2  # assume xy last dim shape 2
        return np.moveaxis(self.dexp(xy), -2, 0) / numel

    def ergodicity(self, xy: np.ndarray) -> float:
        return np.sum(self.sobolev_norm * (self.coeffs(xy) - self.map_coeffs) ** 2)

    def dergodicity(self, xy: np.ndarray) -> np.ndarray:
        front = 2 * self.sobolev_norm * (self.coeffs(xy) - self.map_coeffs)
        return np.sum(front.reshape(-1, 1, 1, 1) * self.dcoeffs(xy), axis=0)
