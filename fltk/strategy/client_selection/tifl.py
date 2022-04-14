import math
from typing import Tuple, List
import numpy as np
from dataclasses import dataclass


@dataclass
class tiflTier:
    id: any
    accuracy: float
    prob: float
    credits: int
    client_ids: List[str]


TiflData = List[tiflTier]


# explicit function to normalize array
def norm_1d(arr):
    summed = np.sum(arr.astype('float64'))
    return arr / summed


def tier_selection(tiers: TiflData, round_id, I):
    print(f'{round_id=}')
    print(f'{I=}')
    print(f'{tiers=}')
    if np.mod(round_id, I) == 0 and round_id > 1:
        tiers = change_probs(tiers)
    selected_id = select_tier(tiers)
    selected = [x for x in tiers if x.id == selected_id][0]
    # selected = [x for x in tiers if x[0] == selected_id][0]
    tiers = decrement_tier(tiers, selected)
    return tiers, selected


def decrement_tier(tiers: TiflData, selected: tiflTier):
    for tier in tiers:
        if tier.id == selected.id and tier.credits > 0:
            tier.credits -= 1
        # if tier[0] == selected[0] and tier[2] > 0:
        #     tier[2] -= 1
    return tiers


def select_tier(tiers: TiflData):
    # Only use tiers with credits
    available_tiers = [x for x in tiers if x.credits]
    probs = [x.prob for x in available_tiers]
    print(f'[1] {probs=}')

    probs = norm_1d(np.asarray(probs))

    print(f'[2] {probs=}')
    print(sum(probs))
    # return np.random.choice([x[0] for x in available_tiers], 1, p=probs)
    return np.random.choice([x.id for x in available_tiers], 1, p=probs)


def tifl_init(n_tiers: int, n_rounds: int):
    credits = np.ceil(float(n_rounds) / float(n_tiers))
    tier_data = []
    prob = 1.0 / n_tiers
    for i in range(n_tiers):
        # tier_data.append([i, prob, credits, []])
        tier_data.append(tiflTier(i, 0, prob, credits, []))
    return tier_data


def change_probs(tiers: TiflData) -> TiflData:
    '''
    Implementation of the paper: "TiFL: A Tier-based Federated Learning System"
    Structure of the input data:
    List of tuples:
    Each tuple is (tier_id, mean_tier_accuracy, credits_tier)

    mean_tier_accuracy is notated in the paper a A^r_T
    :param tiers:
    :return:
    '''
    print('Changing probs!')
    n = float(len([x for x in tiers if x.credits > 0]))
    D = n * (n + 1) / 2
    print(f'{D=}')
    # n_no_credits = len([x for x in tiers if x[2]])

    tiers.sort(key=lambda x: x.accuracy)
    new_probs = []
    idx_decr = 0
    for idx, tier in enumerate(tiers):
        if tier.credits > 0:
            # Reduce precision for calculation
            updated_prob = np.round(float(n - (idx - idx_decr)) / D, 5)
        else:
            idx_decr += 1
            updated_prob = 0
        tier.prob = updated_prob
        # new_probs.append([, updated_prob, tier_credits, client_ids])
    # probs = np.asarray([x[1] for x in new_probs])
    # for item, prob in zip(new_probs, probs):
    #     item[1] = prob
    return tiers


def create_tiers(profiling_data: List[Tuple[str, float]], num_tiers: int, num_rounds: int) -> TiflData:
    tier_data = tifl_init(num_tiers, num_rounds)
    sorted_clients = sorted(profiling_data, key=lambda item: item[1])
    print(f'{sorted_clients=}')
    chunk_size = math.ceil(len(sorted_clients) / num_tiers)
    chunks = [sorted_clients[i:i + chunk_size] for i in range(0, len(sorted_clients), chunk_size)]
    print(f'{chunks=}')
    for idx, c in enumerate(chunks):
        tier_data[idx].client_ids = [x[0] for x in c]
        # tier_data[idx][3] = [x[0] for x in c]
    return tier_data

