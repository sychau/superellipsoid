""" 
Adopted from "Superquadrics Revisited: Learning 3D Shape Parsing beyond Cuboids" by Paschalidou et al.
https://github.com/paschalidoud/superquadric_parsing/blob/master/scripts/visualization_utils.py
"""

import numpy as np
import trimesh
from pyquaternion import Quaternion

def fexp(x, p):
    return np.sign(x)*(np.abs(x)**p)


def sq_surface(a1, a2, a3, e1, e2, eta, omega):
    x = a1 * fexp(np.cos(eta), e1) * fexp(np.cos(omega), e2)
    y = a2 * fexp(np.cos(eta), e1) * fexp(np.sin(omega), e2)
    z = a3 * fexp(np.sin(eta), e1)
    return x, y, z


def points_on_sq_surface(a1, a2, a3, e1, e2, R, t, Kx, Ky, n_samples=100):
    """Computes a SQ given a set of parameters and saves it into a np array
    """
    assert R.shape == (3, 3)
    assert t.shape == (3, 1)

    eta = np.linspace(-np.pi/2, np.pi/2, n_samples, endpoint=True)
    omega = np.linspace(-np.pi, np.pi, n_samples, endpoint=True)
    eta, omega = np.meshgrid(eta, omega)
    x, y, z = sq_surface(a1, a2, a3, e1, e2, eta, omega)

    # Apply the deformations
    fx = Kx * z / a3
    fx += 1
    fy = Ky * z / a3
    fy += 1
    fz = 1

    x = x * fx
    y = y * fy
    z = z * fz

    # Get an array of size 3x10000 that contains the points of the SQ
    points = np.stack([x, y, z]).reshape(3, -1)
    points_transformed = R.T.dot(points) + t
    # print "R:", R
    # print "t:", t
    # print "e:", [e1, e2]
    # print "K:", [Kx, Ky]

    x_tr = points_transformed[0].reshape(n_samples, n_samples)
    y_tr = points_transformed[1].reshape(n_samples, n_samples)
    z_tr = points_transformed[2].reshape(n_samples, n_samples)

    return x_tr, y_tr, z_tr, points_transformed

def _from_primitive_parms_to_mesh(translation, rotations, sizes, shapes, deformation,color):
    # Extract the parameters of the primitives
    a1, a2, a3 = sizes
    e1, e2 = shapes
    Kx, Ky = deformation
    t = np.array(translation).reshape(3, 1)
    R = Quaternion(rotations).rotation_matrix.reshape(3, 3)

    # Sample points on the surface of its mesh
    _, _, _, V = points_on_sq_surface(a1, a2, a3, e1, e2, R, t, Kx, Ky)
    assert V.shape[0] == 3

    # Build a mesh object using the vertices loaded before and get its convex hull
    m = trimesh.Trimesh(vertices=V.T).convex_hull

    # Apply color
    for i in range(len(m.faces)):
        m.visual.face_colors[i] = color

    return m

def save_prediction_as_ply(prediction, output_path, prob_threshold):
    (prob, translation, rotations, sizes, shapes, deformation) = prediction
    M = prob.shape[1]
    mesh = None
    random_colors = [(np.random.randint(0, 256), 
                  np.random.randint(0, 256), 
                  np.random.randint(0, 256), 230) for _ in range(M)]

    for m in range(M):
        if (prob[0][m] < prob_threshold):
            print("skip ", m)
            continue
        color = random_colors[m]
        _mesh = _from_primitive_parms_to_mesh(
            translation.cpu().detach().reshape(M, 3).numpy()[m],
            rotations.cpu().detach().reshape(M, 4).numpy()[m],
            sizes.cpu().detach().reshape(M, 3).numpy()[m],
            shapes.cpu().detach().reshape(M, 2).numpy()[m],
            deformation.cpu().detach().reshape(M, 2).numpy()[m],
            color
        )
        mesh = trimesh.util.concatenate(_mesh, mesh)

    mesh.export(output_path, file_type="ply")

def visualize_mesh(path):
    mesh = trimesh.load(path)
    mesh.show()

def visualize_point_cloud(path):
    point_cloud = trimesh.load(path)
    point_cloud.colors = [0, 255, 255]
    point_cloud.show()

def visualize_voxels(path):
    voxels = np.load(path)["voxels"],
    voxels = np.squeeze(np.asarray(voxels), axis=0)
    voxelsGrid = trimesh.voxel.VoxelGrid(voxels)
    voxelsGrid.show()