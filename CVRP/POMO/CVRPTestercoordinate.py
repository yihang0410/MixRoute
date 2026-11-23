
import torch

import os
from logging import getLogger

from CVRPEnv import CVRPEnv as Env
from CVRPModel import CVRPModel as Model

from utils.utils import *


class CVRPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = self.tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        # ====== helper: convert an LKH-style npz to the model's native coordinate npz ======
        def _convert_lkh_npz_to_model_npz(src_npz_path: str, dst_npz_path: str):
            """
            Read an LKH-style npz holding object arrays:
            - coords[i] : (N+1, 2) int pixel coordinates, row 0 is the depot
            - demands[i]: (N,) int demands (1..9)
            - capacities[i]: int capacity (= demand_scaler)
            and convert it to the native coordinate format:
            - depot_xy: (B, 1, 2) in [0,1]
            - node_xy : (B, N, 2) in [0,1]
            - node_demand: (B, N)  = demands / capacity
            - demand_scaler: (B,)   = capacities
            Coordinates are scaled back to [0,1] by 10000 by default; if the max coordinate is <= 2000, scale by the max instead.
            """
            import numpy as _np

            data = _np.load(src_npz_path, allow_pickle=True)
            coords_obj = data['coords']      # object array, each (N+1,2)
            demands_obj = data['demands']    # object array, each (N,)
            caps = data['capacities']        # (B,)
            B = len(coords_obj)

            # infer N
            N = coords_obj[0].shape[0] - 1

            depot_xy = _np.zeros((B, 1, 2), dtype=_np.float32)
            node_xy  = _np.zeros((B, N, 2), dtype=_np.float32)
            node_demand = _np.zeros((B, N), dtype=_np.float32)
            demand_scaler = _np.array(caps, dtype=_np.int32)

            for i in range(B):
                cxy = _np.asarray(coords_obj[i])   # (N+1,2) int
                dem = _np.asarray(demands_obj[i])  # (N,)    int
                cap = int(caps[i])

                # scale back to [0,1]: 10000 by default, otherwise adapt to the data
                mx = int(cxy.max())
                scale = 10000.0 if mx > 2000 else float(max(mx, 1))
                cxy01 = cxy.astype(_np.float32) / scale

                depot_xy[i, 0, :] = cxy01[0, :]
                node_xy[i, :, :]  = cxy01[1:, :]
                node_demand[i, :] = dem.astype(_np.float32) 

            _np.savez(dst_npz_path,
                    depot_xy=depot_xy,
                    node_xy=node_xy,
                    node_demand=node_demand,
                    capacity=demand_scaler)

            return B, N

        td = self.tester_params.get('test_data_load', {'enable': False})
        if td.get('enable', False):
            fname = td['filename']
            if str(fname).lower().endswith('.npz'):
                import numpy as np
                data = np.load(fname, allow_pickle=True)
                keys = set(list(data.keys()))
                data.close()

                # 1) LKH-style npz: has coords / demands / capacities
                if {'coords', 'demands', 'capacities'}.issubset(keys):
                    # write a one-off intermediate npz under result_folder
                    tmp_npz = os.path.join(self.result_folder, "_converted_from_lkh.npz")
                    B, problem = _convert_lkh_npz_to_model_npz(fname, tmp_npz)

                    # set problem size, episode count and batch limit from the dataset
                    self.env.problem_size = problem

                    # use B as the total number of test episodes
                    self.tester_params['test_episodes'] = int(B)
                    # keep the batch size bounded
                    self.tester_params['test_batch_size'] = min(self.tester_params['test_batch_size'], 1024)

                    # load the converted coordinate npz
                    self.env.use_saved_problems_npz(tmp_npz, self.device)

                # 2) native coordinate npz
                elif {'node_xy'}.issubset(keys):
                    # read directly and set problem_size
                    data2 = np.load(fname, allow_pickle=True)
                    problem = int(data2['node_xy'].shape[1])  # (B, problem, 2)
                    B = int(data2['node_xy'].shape[0])
                    data2.close()

                    self.env.problem_size = problem
                    self.tester_params['test_episodes'] = int(B)
                    self.tester_params['test_batch_size'] = min(self.tester_params['test_batch_size'], 1024)

                    self.env.use_saved_problems_npz(fname, self.device)

                else:
                    raise ValueError(f"Unknown npz schema in {fname}: keys={keys}")

            else:
                # legacy .pt loading
                self.env.use_saved_problems(fname, self.device)

        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score, aug_score = self._test_one_batch(batch_size)

            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size

            # Logs
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))
                return score_AM.avg, aug_score_AM.avg

    def _test_one_batch(self, batch_size):

        # Augmentation
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

        # Return
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

        return no_aug_score.item(), aug_score.item()
