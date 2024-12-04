from erg_map import ErgodicMap
import numpy as np


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


def get_distribution(map: str, uniform: bool):
    if uniform:
        return lambda mp: np.ones(len(mp.mesh_points))
    if map == 'square':
        return square_nonuniform
    elif map == 'maze':
        return maze_nonuniform
    elif map == 'rooms':
        return rooms_nonuniform
