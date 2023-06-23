import torch
import time
from c4il.models.robots import DifferentiableInvertedPendulum, DifferentiableDoubleInvertedPendulum, DifferentiableCartPole
from torch_robotics.torch_kinematics_tree.geometrics.skeleton import get_skeleton_from_model

class CorrespondenceTrainer():
    pass 

#if __name__ == "__main__":
def get_skeletons(batch_size=10, device="cpu"):

    # batch_size = 10
    # device = "cpu"
    print("===========================Inverted Pendulum Model===============================")
    ip_kin = DifferentiableInvertedPendulum(device=device)
    ip_kin.print_link_names()
    print(ip_kin.get_joint_limits())
    print(ip_kin._n_dofs)
    time_start = time.time()
    q_ip = torch.rand(batch_size, ip_kin._n_dofs).to(device)
    q_ip.requires_grad_(True)
    data_ip = ip_kin.compute_forward_kinematics_all_links(q_ip)
    print(data_ip.shape)
    # lin_jacs, ang_jacs = ip_kin.compute_fk_and_jacobian(q, 'link_ee')
    time_end = time.time()
    # print(lin_jacs.shape)
    # print(ang_jacs.shape)
    skeleton_ip = get_skeleton_from_model(ip_kin, q_ip)
    print("Computational Time {}".format(time_end - time_start))
 
    print("===========================Double Inverted Pendulum Model===============================")
    dip_kin = DifferentiableDoubleInvertedPendulum(device=device)
    dip_kin.print_link_names()
    print(dip_kin.get_joint_limits())
    print(dip_kin._n_dofs)
    time_start = time.time()
    q_dip = torch.rand(batch_size, dip_kin._n_dofs).to(device)
    q_dip.requires_grad_(True)
    data_dip = dip_kin.compute_forward_kinematics_all_links(q_dip)
    print(data_dip.shape)
    # lin_jacs, ang_jacs = dip_kin.compute_fk_and_jacobian(q, 'link_ee')
    time_end = time.time()
    # print(lin_jacs.shape)
    # print(ang_jacs.shape)
    skeleton_dip = get_skeleton_from_model(dip_kin, q_dip)
    print("Computational Time {}".format(time_end - time_start))
    
    print("===========================Cart Pole Model===============================")
    cp_kin = DifferentiableCartPole(device=device)
    cp_kin.print_link_names()
    print(cp_kin.get_joint_limits())
    print(cp_kin._n_dofs)
    time_start = time.time()
    q_cp = torch.rand(batch_size, cp_kin._n_dofs).to(device)
    q_cp.requires_grad_(True)
    data_cp = cp_kin.compute_forward_kinematics_all_links(q_cp)
    print(data_cp.shape)
    # lin_jacs, ang_jacs = cp_kin.compute_fk_and_jacobian(q, 'link_ee')
    time_end = time.time()
    # print(lin_jacs.shape)
    # print(ang_jacs.shape)
    skeleton_cp = get_skeleton_from_model(cp_kin, q_cp)
    print("Computational Time {}".format(time_end - time_start))

    return skeleton_ip, skeleton_dip, skeleton_cp




