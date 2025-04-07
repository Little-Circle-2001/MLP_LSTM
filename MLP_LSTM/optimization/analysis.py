import os
import sys
import torch
import argparse

import transformer.manage as DT_manager

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
data_dir = root_folder + '/dataset'

parser = argparse.ArgumentParser(description='transformer-rpod')
parser.add_argument('--data_dir', type=str, default='dataset',
                    help='defines directory from where to load files')
args = parser.parse_args()
args.data_dir = root_folder + '/' + args.data_dir
#print(sys.path)

import numpy as np
import numpy.linalg as la
import numpy.matlib as matl
import scipy.io as io
import matplotlib.pyplot as plt

from dynamics.orbit_dynamics import *
from rpod_scenario import *
from ocp import *
# import transformer.manage as DT_manager

import MLP_LSTM.inference as inference
from MLP_LSTM.inference import *




# Initializations
warmstart = 'both' # 'cvx'/'transformer'/'both'
scenario_test_dataset = True
state_representation = 'rtn' # 'roe'/'rtn'
dataset_to_use = 'both' # 'scp'/'cvx'/'both'
transformer_ws = 'dyn' # 'dyn'/'ol'
network_model_name = 'checkpoint_MLP_LSTM_v6'
transformer_model_name = 'checkpoint_rtn_art'
select_idx = True # set to True to manually select a test trajectory via its index (idx)
# idx = 89 # index of the test trajectory (e.g., idx = 18111)56

# Get the datasets and loaders from the torch data
# datasets, dataloaders = DT_manager.get_train_val_test_data(state_representation, dataset_to_use,transformer_model_name)
# train_loader, eval_loader, test_loader = dataloaders

test_dataset_MLP_LSTM, test_loader_MLP_LSTM = inference.get_test_data()

feasible_number_MLP_LSTM = 0
list_runtime_ws_MLP_LSTM = []
list_runtime_scp_MLP_LSTM = []
list_total_runtime_MLP_LSTM = []
list_cost_ws_MLP_LSTM = []
list_cost_scp_MLP_LSTM = []

feasible_number_cvx = 0
list_runtime_ws_cvx = []
list_runtime_scp_cvx = []
list_total_runtime_cvx = []
list_cost_ws_cvx = []
list_cost_scp_cvx = []

# data_stats = np.load(data_dir + '/dataset-rpod-stats.npz')

print(f'test dataset number: {len(test_dataset_MLP_LSTM)}')
# Sample from test dataset
for idx in range(len(test_dataset_MLP_LSTM)):
    exclude_scp_cvx = False
    exclude_scp_DT = False
    test_sample_MLP_LSTM = test_dataset_MLP_LSTM[idx]

# if select_idx:
#     test_sample = test_loader.dataset.getix(idx)
# else:
#     test_sample = next(iter(test_loader))

    states_i, actions_i, obstacles_i, mask_i, _, oe, time_discr, time_sec, horizons, ix = test_sample_MLP_LSTM
# _, _, rtgs_i, ctgs_i, timesteps_i, attention_mask_i, _, dt, _, _, _ = test_sample



    print('Sampled trajectory ' + str(ix) + ' from test_dataset.')


    hrz = horizons.item()

    if state_representation == 'roe':
        state_roe_0 = np.array((states_i[0, :] * data_stats['states_std'][0]) + data_stats['states_mean'][0])
    elif state_representation == 'rtn':
        state_rtn_0 = np.array((states_i[0, :] * data_stats['states_std'][0]) + data_stats['states_mean'][0])
        state_roe_0 = map_rtn_to_roe(state_rtn_0, oe_0_ref)

    # Dynamics Matrices Precomputations
    # 动力学转移矩阵预计算
    stm_hrz, cim_hrz, psi_hrz, oe_hrz, time_hrz, dt_hrz = dynamics_roe_optimization(oe_0_ref, t_0, hrz, n_time_rpod)

    # Build the oe vector including the target instant
    oe_hrz_trg = np.append(oe_hrz,np.array([oe_0_ref.item(0), oe_0_ref.item(1), oe_0_ref.item(2), oe_0_ref.item(3), oe_0_ref.item(4), oe_0_ref.item(5) + n_ref*(time_hrz[-1]+dt_hrz-t_0)]).reshape((6,1)),1)
    time_hrz_trg = np.append(time_hrz, time_hrz[-1]+dt_hrz)

    # Warmstarting and optimization
    if warmstart == 'cvx' or warmstart == 'both':
        # Solve Convex Problem
        runtime_cvx0 = time.time()
        states_roe_cvx, actions_cvx, feas_cvx = ocp_cvx(stm_hrz, cim_hrz, psi_hrz, state_roe_0, n_time_rpod)
        states_rtn_cvx = roe_to_rtn_horizon(states_roe_cvx, oe_hrz_trg, n_time_rpod)
        states_rtn = np.empty(shape=(1, n_time_rpod, 6))
        states_rtn[0,:,:] = np.transpose(states_rtn_cvx)
        runtime_cvx = time.time() - runtime_cvx0

        states_roe_cvx_trg = np.append(states_roe_cvx, (states_roe_cvx[:,-1]+cim_hrz[:,:,-1].dot(actions_cvx[:,-1])).reshape((6,1)), 1)
        states_roe_ws_cvx = states_roe_cvx # set warm start
        states_rtn_ws_cvx = roe_to_rtn_horizon(states_roe_cvx_trg, oe_hrz_trg, n_time_rpod+1)
        # Evaluate Constraint Violation
        constr_cvx, constr_viol_cvx = check_koz_constraint(states_rtn_ws_cvx, n_time_rpod+1)
        # Solve SCP
        states_roe_scp_cvx, actions_scp_cvx, feas_scp_cvx, iter_scp_cvx , J_vect_scp_cvx, runtime_scp_cvx = solve_scp(stm_hrz, cim_hrz, psi_hrz, state_roe_0, states_roe_ws_cvx, n_time_rpod)
        if states_roe_scp_cvx is None:
            exclude_scp_cvx = True
        else:

            feasible_number_cvx += 1
            # print('SCP cost:', la.norm(actions_scp_cvx, axis=0).sum())
            # print('J vect', J_vect_scp_cvx)
            # print('SCP runtime:', runtime_scp_cvx)
            # print('CVX+SCP runtime:', runtime_cvx+runtime_scp_cvx)
            states_roe_scp_cvx_trg = np.append(states_roe_scp_cvx, (states_roe_scp_cvx[:,-1]+cim_hrz[:,:,-1].dot(actions_scp_cvx[:,-1])).reshape((6,1)), 1)
            states_rtn_scp_cvx = roe_to_rtn_horizon(states_roe_scp_cvx_trg, oe_hrz_trg, n_time_rpod+1)
            constr_scp_cvx, constr_viol_scp_cvx = check_koz_constraint(states_rtn_scp_cvx, n_time_rpod+1)

    if warmstart == 'MLP_LSTM' or warmstart == 'both':

        # Import the MLP_LSTM

        model = inference.get_MLP_LSTM_model(network_model_name)
        # print(model.config)
        inference_func = getattr(inference, 'model_inference_dyn')
        # print('Using MLP_LSTM model \'', network_model_name, '\' with inference function inference.'+inference_func.__name__+'()')
        action0 = np.array([0, 0, 0])
        MLP_LSTM_trajectory, runtime_MLP_LSTM = inference_func(model, state_roe_0, state_rtn_target, obstacle, action0, stm_hrz, cim_hrz, psi_hrz, state_representation, n_time_rpod)
        states_roe_ws_MLP_LSTM = MLP_LSTM_trajectory['roe_' + transformer_ws] # set warm start
        # states_rtn_ws_MLP_LSTM = MLP_LSTM_trajectory['rtn_' + transformer_ws]
        actions_rtn_ws_MLP_LSTM = MLP_LSTM_trajectory['control_' + transformer_ws]
        states_roe_MLP_LSTM_trg = np.append(states_roe_ws_MLP_LSTM, (states_roe_ws_MLP_LSTM[:,-1]+cim_hrz[:,:,-1].dot(actions_rtn_ws_MLP_LSTM[:,-1])).reshape((6,1)), 1)
        states_rtn_ws_MLP_LSTM = roe_to_rtn_horizon(states_roe_MLP_LSTM_trg, oe_hrz_trg, n_time_rpod+1)
        constr_MLP_LSTM, constr_viol_MLP_LSTM = check_koz_constraint(states_rtn_ws_MLP_LSTM, n_time_rpod+1)

        # Solve SCP
        states_roe_scp_MLP_LSTM, actions_scp_MLP_LSTM, feas_scp_MLP_LSTM, iter_scp_MLP_LSTM, J_vect_scp_MLP_LSTM, runtime_scp_MLP_LSTM = solve_scp(stm_hrz, cim_hrz, psi_hrz, state_roe_0, states_roe_ws_MLP_LSTM, n_time_rpod)
        if states_roe_scp_MLP_LSTM is None:
            exclude_scp_DT = True
            # print('No scp-MLP_LSTM solution!')
        else:
            feasible_number_MLP_LSTM += 1
            states_roe_scp_MLP_LSTM_trg = np.append(states_roe_scp_MLP_LSTM, (states_roe_scp_MLP_LSTM[:,-1]+cim_hrz[:,:,-1].dot(actions_scp_MLP_LSTM[:,-1])).reshape((6,1)), 1)
            states_rtn_scp_MLP_LSTM = roe_to_rtn_horizon(states_roe_scp_MLP_LSTM_trg, oe_hrz_trg, n_time_rpod+1)
            constr_scp_MLP_LSTM, constr_viol_scp_MLP_LSTM = check_koz_constraint(states_rtn_scp_MLP_LSTM, n_time_rpod+1)

    if (not exclude_scp_cvx) & (not exclude_scp_DT):

        list_cost_ws_cvx.append(la.norm(actions_cvx, axis=0).sum())
        list_cost_scp_cvx.append(la.norm(actions_scp_cvx, axis=0).sum())
        list_runtime_ws_cvx.append(runtime_cvx)
        list_runtime_scp_cvx.append(runtime_scp_cvx)
        list_total_runtime_cvx.append(runtime_cvx + runtime_scp_cvx)

        list_cost_ws_MLP_LSTM.append(la.norm(actions_rtn_ws_MLP_LSTM, axis=0).sum())
        list_cost_scp_MLP_LSTM.append(la.norm(actions_scp_MLP_LSTM, axis=0).sum())
        list_runtime_ws_MLP_LSTM.append(runtime_MLP_LSTM)
        list_runtime_scp_MLP_LSTM.append(runtime_scp_MLP_LSTM)
        list_total_runtime_MLP_LSTM.append(runtime_MLP_LSTM + runtime_scp_MLP_LSTM)

average_ws_cost_cvx = np.mean(list_cost_ws_cvx)
average_scp_cost_cvx = np.mean(list_cost_scp_cvx)
average_runtime_ws_cvx = np.mean(list_runtime_ws_cvx)
average_runtime_scp_cvx = np.mean(list_runtime_scp_cvx)
average_total_runtime_cvx = np.mean(list_total_runtime_cvx)

average_ws_cost_MLP_LSTM = np.mean(list_cost_ws_MLP_LSTM)
average_scp_cost_MLP_LSTM  = np.mean(list_cost_scp_MLP_LSTM)
average_runtime_ws_MLP_LSTM  = np.mean(list_runtime_ws_MLP_LSTM)
average_runtime_scp_MLP_LSTM  = np.mean(list_runtime_scp_MLP_LSTM)
average_total_runtime_MLP_LSTM  = np.mean(list_total_runtime_MLP_LSTM)

print(f'cvx feasible ratio: {(100 * feasible_number_cvx/len(test_dataset_MLP_LSTM)):.2f}%')
print(f'average_ws_cost_cvx: {average_ws_cost_cvx:.2f}')
print(f'average_scp_cost_cvx: {average_scp_cost_cvx:.2f}')
print(f'average_runtime_ws_cvx: {average_runtime_ws_cvx:.2f}')
print(f'average_runtime_scp_cvx: {average_runtime_scp_cvx:.2f}')
print(f'average_total_runtime_cvx: {average_total_runtime_cvx:.2f}')

print(f'MLP_LSTM feasible ratio: {(100*feasible_number_MLP_LSTM/len(test_dataset_MLP_LSTM)):.2f}%')
print(f'average_ws_cost_MLP_LSTM: {average_ws_cost_MLP_LSTM:.2f}')
print(f'average_scp_cost_MLP_LSTM: {average_scp_cost_MLP_LSTM:.2f}')
print(f'average_runtime_ws_MLP_LSTM: {average_runtime_ws_MLP_LSTM:.2f}')
print(f'average_runtime_scp_MLP_LSTM: {average_runtime_scp_MLP_LSTM:.2f}')
print(f'average_total_runtime_MLP_LSTM: {average_total_runtime_MLP_LSTM:.2f}')

    # if warmstart == 'transformer' or warmstart == 'both' and scenario_test_dataset:
    #
    #     # Import the Transformer
    #     model = DT_manager.get_DT_model(transformer_model_name, train_loader, eval_loader)
    #     inference_func = getattr(DT_manager, 'torch_model_inference_'+transformer_ws)
    #     print('Using ART model \'', transformer_model_name, '\' with inference function DT_manage.'+inference_func.__name__+'()')
    #     rtg = la.norm(actions_cvx, axis=0).sum()
    #     DT_trajectory, runtime_DT = inference_func(model, test_loader, test_sample, stm_hrz, cim_hrz, psi_hrz, state_representation, rtg_perc=1., ctg_perc=0., rtg=rtg)
    #     states_roe_ws_DT = DT_trajectory['roe_' + transformer_ws] # set warm start
    #     # states_rtn_ws_DT = DT_trajectory['rtn_' + transformer_ws]
    #     actions_rtn_ws_DT = DT_trajectory['dv_' + transformer_ws]
    #     states_roe_DT_trg = np.append(states_roe_ws_DT, (states_roe_ws_DT[:,-1]+cim_hrz[:,:,-1].dot(actions_rtn_ws_DT[:,-1])).reshape((6,1)), 1)
    #     states_rtn_ws_DT = roe_to_rtn_horizon(states_roe_DT_trg, oe_hrz_trg, n_time_rpod+1)
    #     print('ART cost:', la.norm(actions_rtn_ws_DT, axis=0).sum())
    #     print('ART runtime:', runtime_DT)
    #     constr_DT, constr_viol_DT = check_koz_constraint(states_rtn_ws_DT, n_time_rpod+1)
    #
    #     # Solve SCP
    #     states_roe_scp_DT, actions_scp_DT, feas_scp_DT, iter_scp_DT, J_vect_scp_DT, runtime_scp_DT = solve_scp(stm_hrz, cim_hrz, psi_hrz, state_roe_0, states_roe_ws_DT, n_time_rpod)
    #     if states_roe_scp_DT is None:
    #         exclude_scp_DT = True
    #         print('No scp-DT solution!')
    #     else:
    #         print('SCP cost:', la.norm(actions_scp_DT, axis=0).sum())
    #         print('J vect', J_vect_scp_DT)
    #         print('SCP runtime:', runtime_scp_DT)
    #         print('ART+SCP runtime:', runtime_DT + runtime_scp_DT)
    #         states_roe_scp_DT_trg = np.append(states_roe_scp_DT, (states_roe_scp_DT[:,-1]+cim_hrz[:,:,-1].dot(actions_scp_DT[:,-1])).reshape((6,1)), 1)
    #         states_rtn_scp_DT = roe_to_rtn_horizon(states_roe_scp_DT_trg, oe_hrz_trg, n_time_rpod+1)
    #         constr_scp_DT, constr_viol_scp_DT = check_koz_constraint(states_rtn_scp_DT, n_time_rpod+1)
    #
    # else:
    #     # Import the Transformer
    #     model = DT_manager.get_DT_model(transformer_model_name, train_loader, eval_loader)
    #     inference_func = getattr(DT_manager, 'torch_model_inference_'+transformer_ws+'_no_use_dataset')
    #     print('Using ART model \'', transformer_model_name, '\' with inference function DT_manage.'+inference_func.__name__+'()')
    #
    #     rtg = la.norm(actions_cvx, axis=0).sum()
    #     ctg = compute_constraint_to_go(states_rtn, 1, n_time_rpod)
    #     timesteps = [[i for i in range(n_time_rpod)]]
    #     attention_mask = np.ones([1, n_time_rpod])
    #
    #     DT_trajectory, runtime_DT = inference_func(model, test_loader, hrz, states_rtn, actions_cvx, rtg, ctg, timesteps, attention_mask, oe_hrz, time_hrz, dt_hrz, stm_hrz, cim_hrz, psi_hrz, state_representation, ctg_perc=0.)
    #     states_roe_ws_DT = DT_trajectory['roe_' + transformer_ws] # set warm start
    #     # states_rtn_ws_DT = DT_trajectory['rtn_' + transformer_ws]
    #     actions_rtn_ws_DT = DT_trajectory['dv_' + transformer_ws]
    #     states_roe_DT_trg = np.append(states_roe_ws_DT, (states_roe_ws_DT[:,-1]+cim_hrz[:,:,-1].dot(actions_rtn_ws_DT[:,-1])).reshape((6,1)), 1)
    #     states_rtn_ws_DT = roe_to_rtn_horizon(states_roe_DT_trg, oe_hrz_trg, n_time_rpod+1)
    #     print('ART cost:', la.norm(actions_rtn_ws_DT, axis=0).sum())
    #     print('ART runtime:', runtime_DT)
    #     constr_DT, constr_viol_DT = check_koz_constraint(states_rtn_ws_DT, n_time_rpod+1)
    #
    #     # Solve SCP
    #     states_roe_scp_DT, actions_scp_DT, feas_scp_DT, iter_scp_DT, J_vect_scp_DT, runtime_scp_DT = solve_scp(stm_hrz, cim_hrz, psi_hrz, state_roe_0, states_roe_ws_DT, n_time_rpod)
    #     if states_roe_scp_DT is None:
    #         exclude_scp_DT = True
    #         print('No scp-DT solution!')
    #     else:
    #         print('SCP cost:', la.norm(actions_scp_DT, axis=0).sum())
    #         print('J vect', J_vect_scp_DT)
    #         print('SCP runtime:', runtime_scp_DT)
    #         print('ART+SCP runtime:', runtime_DT + runtime_scp_DT)
    #         states_roe_scp_DT_trg = np.append(states_roe_scp_DT, (states_roe_scp_DT[:,-1]+cim_hrz[:,:,-1].dot(actions_scp_DT[:,-1])).reshape((6,1)), 1)
    #         states_rtn_scp_DT = roe_to_rtn_horizon(states_roe_scp_DT_trg, oe_hrz_trg, n_time_rpod+1)
    #         constr_scp_DT, constr_viol_scp_DT = check_koz_constraint(states_rtn_scp_DT, n_time_rpod+1)
