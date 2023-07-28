import numpy as np # numpy backend
import pygmtools as pygm
import matplotlib.pyplot as plt # for plotting
from matplotlib.patches import ConnectionPatch # for plotting matching result
import networkx as nx # for plotting graphs
import functools
import torch

from c4il.models.robots import DifferentiableInvertedPendulum, DifferentiableDoubleInvertedPendulum, DifferentiableCartPole

from torch_robotics.torch_kinematics_tree.geometrics.skeleton import get_skeleton_from_model
from correspondence_metric.utils.visualization import draw_correspondence

pygm.BACKEND = 'numpy' # set default backend for pygmtools
np.random.seed(1) # fix random seed


if __name__ == '__main__':
    random = False
    method = 'analytic'
    draw_argmax = True
    scale_embodiment = True
    alpha = 1.
    beta = 0.01
    
    invert_pendulum = DifferentiableInvertedPendulum()
    double_invert_pendulum = DifferentiableDoubleInvertedPendulum()
    link_names1 = invert_pendulum.get_link_names()
    link_names2 = double_invert_pendulum.get_link_names()
    print(link_names1, link_names2)

    # link names to index
    link_names_map1 = {link_names1[i]: i for i in range(len(link_names1))}
    link_names_map2 = {link_names2[i]: i for i in range(len(link_names2))}
    print(link_names_map1, link_names_map2)

    q1 = torch.rand(1, invert_pendulum._n_dofs)
    G1 = get_skeleton_from_model(invert_pendulum, q1)

    q2 = torch.rand(1, double_invert_pendulum._n_dofs)
    G2 = get_skeleton_from_model(double_invert_pendulum, q2)
    pos1 = {n: v[[0, 2]] for n, v in G1.node_pos.items()}  # x-z positions from 3D pos
    pos2 = {n: v[[0, 2]] for n, v in G2.node_pos.items()}  # x-z positions from 3D pos

    print("Invert pendulum pos: ", pos1)
    print("Invert double pendulum pos: ", pos2)

    # A1 = G1.compute_adjacency_matrix()
    # A2 = G2.compute_adjacency_matrix()
    # conn1, edge1 = pygm.utils.dense_to_sparse(A1)
    # conn2, edge2 = pygm.utils.dense_to_sparse(A2)

    n1 = np.array([G1.num_node])
    n2 = np.array([G2.num_node])
    conn1, edge1 = G1.get_edges(weighted=True)
    conn2, edge2 = G2.get_edges(weighted=True)
    node1 = G1.get_all_neighbors()[:, np.newaxis]
    node2 = G2.get_all_neighbors()[:, np.newaxis]
    conn1 = np.array([[link_names_map1[i], link_names_map1[j]] for i, j in conn1])
    conn2 = np.array([[link_names_map2[i], link_names_map2[j]] for i, j in conn2])

    # print("Node 1: ", node1)
    # print("Node 2: ", node2)
    # print("Edge 1: ", edge1)
    # print("Edge 2: ", edge2)
    # print("Conn 1: ", conn1)
    # print("Conn 2: ", conn2)

    # build affinity matrix
    gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=.001) # set affinity function
    K = pygm.utils.build_aff_mat(node1, edge1, conn1, node2, edge2, conn2, n1, None, n2, None, node_aff_fn=gaussian_aff, edge_aff_fn=gaussian_aff)

    # solve using SM
    X = pygm.sm(K, n1, n2)

    # get discrete matching
    Xd = pygm.hungarian(X)

    plt.figure(figsize=(8, 4))
    plt.suptitle(f'SM Matching Result')
    ax1 = plt.subplot(1, 2, 1)
    plt.title('Inverted Pendulum Skeleton')
    plt.gca().margins(0.4)
    G1.three_d = False
    G1.draw_skeleton(pos=pos1, ax=ax1)
    ax2 = plt.subplot(1, 2, 2)
    plt.title('Double Inverted Pendulum Skeleton')
    G2.three_d = False
    G2.draw_skeleton(pos=pos2, ax=ax2)
    for i in range(G1.num_node):
        j = np.argmax(Xd[i]).item()
        con = ConnectionPatch(xyA=pos1[link_names1[i]], xyB=pos2[link_names2[j]], coordsA="data", coordsB="data",
                            axesA=ax1, axesB=ax2, color="green")
        # compute the distances between matched joints
        plt.gca().add_artist(con)
    # draw_correspondence(G1, G2, X, pos1=pos1, pos2=pos2, shift=np.array([5., 0.]))
    plt.tight_layout()
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    plt.show()
