import numpy as np 
import pygmtools as pygm
import functools
import torch

from c4il.models.robots import DifferentiableInvertedPendulum, DifferentiableDoubleInvertedPendulum
from torch_robotics.torch_kinematics_tree.geometrics.skeleton import get_skeleton_from_model

pygm.BACKEND = 'numpy' 
np.random.seed(1) 

def compute_correspondence(kinematic_tree1, kinematic_tree2, device='cpu'):

    link_names1 = kinematic_tree1.get_link_names()
    link_names2 = kinematic_tree2.get_link_names()

    link_names_map1 = {link_names1[i]: i for i in range(len(link_names1))}
    link_names_map2 = {link_names2[i]: i for i in range(len(link_names2))}

    q1 = torch.rand(1, kinematic_tree1._n_dofs, device=device)
    G1 = get_skeleton_from_model(kinematic_tree1, q1)

    q2 = torch.rand(1, kinematic_tree2._n_dofs, device=device)
    G2 = get_skeleton_from_model(kinematic_tree2, q2)

    n1 = np.array([G1.num_node])
    n2 = np.array([G2.num_node])
    conn1, edge1 = G1.get_edges(weighted=True)
    conn2, edge2 = G2.get_edges(weighted=True)
    node1 = G1.get_all_neighbors()[:, np.newaxis]
    node2 = G2.get_all_neighbors()[:, np.newaxis]
    conn1 = np.array([[link_names_map1[i], link_names_map1[j]] for i, j in conn1])
    conn2 = np.array([[link_names_map2[i], link_names_map2[j]] for i, j in conn2])
    gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=.001) 
    K = pygm.utils.build_aff_mat(node1, edge1, conn1, node2, edge2, conn2, n1, None, n2, None, node_aff_fn=gaussian_aff, edge_aff_fn=gaussian_aff)
    X = pygm.sm(K, n1, n2)
    Xd = pygm.hungarian(X)

    return Xd

def compute_distances(Xd, q1, q2, kinematic_tree1, kinematic_tree2):

    link_names1 = kinematic_tree1.get_link_names()
    link_names2 = kinematic_tree2.get_link_names()

    link_names_map1 = {link_names1[i]: i for i in range(len(link_names1))}
    link_names_map2 = {link_names2[i]: i for i in range(len(link_names2))}

    pose_list1 = kinematic_tree1.compute_forward_kinematics_all_links(q1)
    pose_list2 = kinematic_tree2.compute_forward_kinematics_all_links(q2)

    distances = []
    for i in range(len(link_names1)):
        j = np.argmax(Xd[i]).item()
        pose_node1 = pose_list1[:, link_names_map1[link_names1[i]], :3]  # get the pose of the node in G1
        pose_node2 = pose_list2[:, link_names_map2[link_names2[j]], :3]  # get the pose of the node in G2
        distance = torch.norm(pose_node1 - pose_node2)  # calculate Euclidean distance
        distances.append(distance)

    return distances

def compute_total_distance(Xd, q1, q2, kinematic_tree1, kinematic_tree2):

    distances = compute_distances(Xd, q1, q2, kinematic_tree1, kinematic_tree2)
    total_distance = torch.sum(torch.stack(distances))
    
    return total_distance

if __name__ == '__main__':

    invert_pendulum = DifferentiableInvertedPendulum()
    double_invert_pendulum = DifferentiableDoubleInvertedPendulum()

    Xd = compute_correspondence(invert_pendulum, double_invert_pendulum)

    q1 = torch.rand(1, invert_pendulum._n_dofs)
    q2 = torch.rand(1, double_invert_pendulum._n_dofs)
    distances = compute_distances(Xd, q1, q2, invert_pendulum, double_invert_pendulum)
    print(distances) 
    reward = lambda _q1, _q2: compute_distances(Xd, _q1, _q2, invert_pendulum, double_invert_pendulum)
    #reward = lambda _q1, _q2: math.exp(-compute_total_distance(Xd, _q1, _q2, invert_pendulum, double_invert_pendulum))
    #print(reward(q1, q2))    
    for i in range(100):
        q1 = torch.rand(1, invert_pendulum._n_dofs)
        q2 = torch.rand(1, double_invert_pendulum._n_dofs)
        print(reward(q1, q2))