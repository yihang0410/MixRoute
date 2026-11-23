from dataclasses import dataclass
import torch

import sys
sys.path.insert(0, "..")
from CVRPTWProblemDef import get_random_problems, augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)
    node_tw: torch.Tensor = None
    # shape: (batch, problem, 2)  [earliest, latest]
    node_service: torch.Tensor = None
    # shape: (batch, problem)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    current_time: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)


class CVRPTWEnv:
    def __init__(self, **env_params):

        # Const @INIT
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']

        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_node_tw = None
        self.saved_node_service = None
        self.saved_index = None

        # Const @Load_Problem
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)
        self.depot_node_tw = None
        # shape: (batch, problem+1, 2)
        self.depot_node_service = None
        # shape: (batch, problem+1)

        # Dynamic
        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None
        self.at_the_depot = None
        self.load = None
        self.current_time = None
        self.visited_ninf_flag = None
        self.ninf_mask = None
        self.finished = None

        # states to return
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        if not self.FLAG__use_saved_problems:
            depot_xy, node_xy, node_demand, node_tw, node_service, depot_tw_end = \
                get_random_problems(batch_size, self.problem_size)
        else:
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index + batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index + batch_size]
            node_demand = self.saved_node_demand[self.saved_index:self.saved_index + batch_size]
            node_tw = self.saved_node_tw[self.saved_index:self.saved_index + batch_size]
            node_service = self.saved_node_service[self.saved_index:self.saved_index + batch_size]
            depot_tw_end = torch.ones(batch_size, 1)
            self.saved_index += batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
                node_tw = node_tw.repeat(8, 1, 1)
                node_service = node_service.repeat(8, 1)
                depot_tw_end = depot_tw_end.repeat(8, 1)
            else:
                raise NotImplementedError

        # Assemble depot+node tensors
        # Coordinates
        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)

        # Demand: depot has 0 demand
        depot_demand = torch.zeros(self.batch_size, 1, device=node_demand.device)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)

        # Time windows: depot has [0, depot_tw_end]
        depot_tw = torch.cat([
            torch.zeros(self.batch_size, 1, 1, device=node_tw.device),
            depot_tw_end[:, :, None]
        ], dim=2)
        # shape: (batch, 1, 2)
        self.depot_node_tw = torch.cat((depot_tw, node_tw), dim=1)
        # shape: (batch, problem+1, 2)

        # Service time: depot has 0 service time
        depot_service = torch.zeros(self.batch_size, 1, device=node_service.device)
        self.depot_node_service = torch.cat((depot_service, node_service), dim=1)
        # shape: (batch, problem+1)

        # Precompute distance matrix for travel times
        self.dist_mat = (self.depot_node_xy[:, :, None, :] - self.depot_node_xy[:, None, :, :]).norm(p=2, dim=-1)
        # shape: (batch, problem+1, problem+1)

        # Index caches
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        # Reset state
        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        self.reset_state.node_tw = node_tw
        self.reset_state.node_service = node_service

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)

        self.at_the_depot = torch.ones(self.batch_size, self.pomo_size, dtype=torch.bool)
        self.load = torch.ones(self.batch_size, self.pomo_size)
        self.current_time = torch.zeros(self.batch_size, self.pomo_size)

        self.visited_ninf_flag = torch.zeros(self.batch_size, self.pomo_size, self.problem_size + 1)
        self.ninf_mask = torch.zeros(self.batch_size, self.pomo_size, self.problem_size + 1)
        self.finished = torch.zeros(self.batch_size, self.pomo_size, dtype=torch.bool)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.current_time = self.current_time
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        # Dynamic-1: update selected list
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat(
            (self.selected_node_list, self.current_node[:, :, None]), dim=2
        )

        # Dynamic-2: update load, time, masks
        self.at_the_depot = (selected == 0)

        # --- Load update ---
        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        gathering_index = selected[:, :, None]
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1  # refill at depot

        # --- Time update ---
        # Travel time from previous node to selected node
        if self.selected_count == 1:
            # First move: from depot (node 0)
            prev_node = torch.zeros_like(selected)
        else:
            # prev_node was set in last step
            prev_node = self.selected_node_list[:, :, -2]

        # Gather travel time
        travel_time = self.dist_mat[
            self.BATCH_IDX,
            prev_node,
            selected
        ]
        # shape: (batch, pomo)

        # Arrival time = current_time + travel_time
        arrival_time = self.current_time + travel_time

        # Gather time window [earliest, latest] and service time for selected node
        tw_list = self.depot_node_tw[:, None, :, :].expand(self.batch_size, self.pomo_size, -1, -1)
        selected_tw = tw_list.gather(
            dim=2, index=selected[:, :, None, None].expand(-1, -1, -1, 2)
        ).squeeze(dim=2)
        # shape: (batch, pomo, 2)
        selected_earliest = selected_tw[:, :, 0]
        selected_latest = selected_tw[:, :, 1]

        service_list = self.depot_node_service[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        selected_service = service_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)

        # Wait if arrive early, start service at max(arrival, earliest)
        service_start = torch.max(arrival_time, selected_earliest)
        departure_time = service_start + selected_service
        # shape: (batch, pomo)

        self.current_time = departure_time
        # At depot: reset time to 0
        self.current_time[self.at_the_depot] = 0

        # --- Masking ---
        # Mark visited
        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot always visitable

        self.ninf_mask = self.visited_ninf_flag.clone()

        # Capacity constraint: mask nodes with demand > remaining load
        round_error_epsilon = 1e-5
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        self.ninf_mask[demand_too_large] = float('-inf')

        # Time window constraint: mask nodes we can't reach before their latest time
        # For each candidate node j, check:
        #   current_time + dist(current_node, j) <= latest_j
        # dist_mat shape: (batch, problem+1, problem+1)
        # current_node shape: (batch, pomo)
        # We need dist_mat[b, current_node[b,p], :] for all b, p
        cur_idx = self.current_node[:, :, None, None].expand(
            self.batch_size, self.pomo_size, 1, self.problem_size + 1
        )
        # shape: (batch, pomo, 1, problem+1)
        travel_to_all = self.dist_mat[:, None, :, :].expand(
            self.batch_size, self.pomo_size, self.problem_size + 1, self.problem_size + 1
        ).gather(dim=2, index=cur_idx).squeeze(dim=2)
        # shape: (batch, pomo, problem+1)

        arrival_at_all = self.current_time[:, :, None] + travel_to_all
        # shape: (batch, pomo, problem+1)

        latest_all = self.depot_node_tw[:, None, :, 1].expand(
            self.batch_size, self.pomo_size, self.problem_size + 1
        )
        # shape: (batch, pomo, problem+1)

        too_late = arrival_at_all > latest_all + round_error_epsilon
        self.ninf_mask[too_late] = float('-inf')

        # Check finished
        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        self.finished = self.finished + newly_finished

        # Don't mask depot for finished episodes
        self.ninf_mask[:, :, 0][self.finished] = 0

        # Update step state
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.current_time = self.current_time
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # Return
        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance()
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()

        travel_distances = segment_lengths.sum(2)
        return travel_distances
