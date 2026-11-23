import torch
import numpy as np
import random


# Vehicle capacity (following RouteFinder / RL4CO convention)

def get_vehicle_capacity(num_loc):
    """RouteFinder convention: capacity = 30 + num_loc/5 for num_loc > 20."""
    if num_loc > 1000:
        extra_cap = 1000 // 5 + (num_loc - 1000) / 33.3
    elif num_loc > 20:
        extra_cap = num_loc // 5
    else:
        extra_cap = 0
    return int(30 + extra_cap)


# Main generator (following RouteFinder / RL4CO generate_mtvrp_data)

MAX_TIME = 4.6    # RouteFinder default time horizon
SPEED = 1.0       # RouteFinder default speed


def get_random_problems(batch_size, problem_size):
    """
    Generate random CVRPTW instances following the RouteFinder/RL4CO convention.

    Time window generation (from RouteFinder, generate_mtvrp_data):
        service_time ~ U[0.12, 0.15]
        tw_length    ~ U[0.15, 0.30]
        d_0i         = Euclidean distance from depot to node i
        h_max        = (max_time - service_time - tw_length) / d_0i * speed - 1
        tw_start     = (1 + (h_max - 1) * U[0,1]) * d_0i / speed
        tw_end       = tw_start + tw_length

    Depot time window: [0, max_time]

    Returns:
        depot_xy:       (batch, 1, 2)          in [0, 1]
        node_xy:        (batch, problem, 2)    in [0, 1]
        node_demand:    (batch, problem)        normalized by capacity (sum ≈ 1)
        node_tw:        (batch, problem, 2)    [tw_start, tw_end]
        node_service:   (batch, problem)        service time per node
        depot_tw_end:   (batch, 1)             = max_time
    """
    capacity = get_vehicle_capacity(problem_size)

    # --- Coordinates in [0, 1] ---
    depot_xy = torch.rand(size=(batch_size, 1, 2))
    node_xy = torch.rand(size=(batch_size, problem_size, 2))

    # --- Demand: integer [1, 9] normalized by capacity ---
    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)).float() / float(capacity)
    # shape: (batch, problem)

    # --- Time windows (RouteFinder convention) ---
    a, b, c = 0.12, 0.15, 0.3

    # Service time ~ U[a, b] = U[0.12, 0.15]
    node_service = torch.rand(batch_size, problem_size) * (b - a) + a
    # shape: (batch, problem)

    # Time window length ~ U[b, c] = U[0.15, 0.30]
    tw_length = torch.rand(batch_size, problem_size) * (c - b) + b
    # shape: (batch, problem)

    # Distance from depot to each node
    d_0i = ((node_xy - depot_xy) ** 2).sum(dim=-1).sqrt().clamp(min=1e-6)
    # shape: (batch, problem)

    # h_max = (max_time - service_time - tw_length) / d_0i * speed - 1
    h_max = (MAX_TIME - node_service - tw_length) / d_0i * SPEED - 1
    # Clamp h_max to be at least 1 to ensure feasibility
    h_max = h_max.clamp(min=1.0)

    # tw_start = (1 + (h_max - 1) * U[0,1]) * d_0i / speed
    tw_start = (1.0 + (h_max - 1.0) * torch.rand(batch_size, problem_size)) * d_0i / SPEED
    # shape: (batch, problem)

    # tw_end = tw_start + tw_length
    tw_end = tw_start + tw_length
    # shape: (batch, problem)

    node_tw = torch.stack([tw_start, tw_end], dim=-1)
    # shape: (batch, problem, 2)

    # --- Depot time window: [0, max_time] ---
    depot_tw_end = torch.full((batch_size, 1), MAX_TIME)

    return depot_xy, node_xy, node_demand, node_tw, node_service, depot_tw_end


def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)
    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    return aug_xy_data
