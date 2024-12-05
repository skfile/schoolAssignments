import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm  # Import tqdm for progress bars

def eefx_allocation_composite(df_blocks, block_names, school_names, dist, P_k, max_distance,
                              max_possible_imbalance, lambda_distance, lambda_racial, lambda_capacity,
                              capacity_per_school, capacity_slack):
    # Initialize allocation and unallocated items
    allocation = {school: [] for school in school_names}
    unallocated_blocks = set(block_names)

    # Initialize capacities
    total_population = df_blocks['Population'].sum()
    remaining_capacity = {school: capacity_slack * (total_population / len(school_names)) for school in school_names}

    # While there are unallocated blocks
    iteration = 0
    with tqdm(total=len(unallocated_blocks), desc='Allocating Blocks') as pbar:
        while unallocated_blocks:
            iteration += 1
            # Compute composite valuations
            valuations = compute_composite_valuations(
                allocation, remaining_capacity, unallocated_blocks, df_blocks, school_names, dist, P_k,
                max_distance, max_possible_imbalance, lambda_distance, lambda_racial, lambda_capacity
            )

            # Each school selects its most valued unallocated block
            proposals = {}
            proposed_blocks = set()
            for school in school_names:
                if remaining_capacity[school] > 0:
                    # Only consider blocks that have valuations for this school
                    blocks_with_valuation = [block for block in unallocated_blocks if block in valuations[school]]
                    if not blocks_with_valuation:
                        continue  # No blocks to propose for this school

                    # Get the unallocated blocks sorted by valuation
                    preferred_blocks = sorted(
                        blocks_with_valuation,
                        key=lambda block: -valuations[school][block]
                    )

                    # Find a block that fits in the remaining capacity and hasn't been proposed yet
                    for block in preferred_blocks:
                        if block in proposed_blocks:
                            continue  # Skip if block is already proposed
                        block_population = df_blocks.loc[df_blocks['UniqueTract'] == block, 'Population'].values[0]
                        if remaining_capacity[school] - block_population >= 0:
                            proposals[school] = block
                            proposed_blocks.add(block)
                            break

            # If no proposals, break
            if not proposals:
                break

            # Build envy graph
            envy_graph = defaultdict(list)
            for school_i in proposals:
                for school_j in proposals:
                    if school_i != school_j:
                        # If school_i values school_j's proposed block more than its own
                        if valuations[school_i][proposals[school_j]] > valuations[school_i][proposals[school_i]]:
                            envy_graph[school_i].append(school_j)

            # Detect cycles in envy graph
            cycle = find_envy_cycle(envy_graph, proposals)

            allocated_blocks = set()
            if cycle:
                # Allocate blocks along the cycle
                for school in cycle:
                    block = proposals[school]
                    allocation[school].append(block)
                    allocated_blocks.add(block)
                    block_population = df_blocks.loc[df_blocks['UniqueTract'] == block, 'Population'].values[0]
                    remaining_capacity[school] -= block_population
            else:
                # Allocate proposed blocks
                for school, block in proposals.items():
                    allocation[school].append(block)
                    allocated_blocks.add(block)
                    block_population = df_blocks.loc[df_blocks['UniqueTract'] == block, 'Population'].values[0]
                    remaining_capacity[school] -= block_population

            # Update unallocated blocks and progress bar
            unallocated_blocks -= allocated_blocks
            pbar.update(len(allocated_blocks))

    return allocation

def compute_composite_valuations(
    allocation, remaining_capacity, unallocated_blocks, df_blocks, school_names, dist, P_k,
    max_distance, max_possible_imbalance, lambda_distance, lambda_racial, lambda_capacity
):
    valuations = {}

    for school in school_names:
        valuations[school] = {}
        # Get current school allocation
        school_blocks = allocation[school]
        school_data = df_blocks[df_blocks['UniqueTract'].isin(school_blocks)]
        n_s = school_data['Population'].sum()
        n_s_k = {race: school_data[race].sum() for race in ['White', 'Black', 'Asian']}
        for block in unallocated_blocks:
            # Distance valuation
            try:
                distance = dist[block][school]
            except KeyError:
                # If distance data is missing, skip this block-school pair
                continue
            V_distance = (max_distance - distance) / max_distance

            # Racial balance valuation
            block_row = df_blocks.loc[df_blocks['UniqueTract'] == block]
            n_b = block_row['Population'].values[0]
            n_b_k = {race: block_row[race].values[0] for race in ['White', 'Black', 'Asian']}
            n_total = n_s + n_b
            n_s_k_new = {race: n_s_k[race] + n_b_k[race] for race in ['White', 'Black', 'Asian']}
            racial_imbalance_after = sum([abs((n_s_k_new[race]/n_total if n_total > 0 else 0) - P_k[race]) for race in ['White', 'Black', 'Asian']])
            V_racial = - (racial_imbalance_after) / (3 * max_possible_imbalance)  # Divided by 3 since there are 3 races

            # Capacity impact valuation
            remaining_cap = remaining_capacity[school]
            V_capacity = 1 if remaining_cap >= n_b else 0  # 1 if within capacity, 0 otherwise

            # Composite valuation
            valuations[school][block] = (lambda_distance * V_distance +
                                         lambda_racial * V_racial +
                                         lambda_capacity * V_capacity)
    return valuations

def find_envy_cycle(envy_graph, proposals):
    # Detect cycles in envy graph using DFS
    visited = set()
    stack = []

    def dfs(school):
        if school in stack:
            return stack[stack.index(school):]
        if school in visited:
            return None
        visited.add(school)
        stack.append(school)
        for neighbor in envy_graph[school]:
            cycle = dfs(neighbor)
            if cycle:
                return cycle
        stack.pop()
        return None

    for school in proposals:
        cycle = dfs(school)
        if cycle:
            return cycle
    return None

def compute_metrics(df_results, df_blocks, dist, P_k, school_names, max_distance):
    # Calculate total family cost
    family_cost = 0
    for idx, row in df_results.iterrows():
        block = row['UniqueTract']
        school = row['UniqueSchoolID']
        try:
            distance = dist[block][school]
        except KeyError:
            # Skip if distance data is missing
            continue
        population = row['Population']
        family_cost += population * distance

    # Calculate racial imbalance
    racial_imbalance = 0
    for school in school_names:
        school_data = df_results[df_results['UniqueSchoolID'] == school]
        n_j = school_data['Population'].sum()
        racial_imbalance_school = 0
        for race in ['White', 'Black', 'Asian']:
            n_jk = school_data[race].sum()
            expected = P_k[race] * n_j
            racial_imbalance_school += abs(n_jk - expected)
        racial_imbalance += racial_imbalance_school

    # Calculate compactness penalty (max distance to assigned blocks)
    compactness_penalty = 0
    for school in school_names:
        school_data = df_results[df_results['UniqueSchoolID'] == school]
        if school_data.empty:
            continue
        max_distance_school = school_data.apply(lambda row: dist[row['UniqueTract']][school], axis=1).max()
        compactness_penalty += max_distance_school

    # Calculate entropy
    school_entropies = {}
    for school in school_names:
        school_data = df_results[df_results['UniqueSchoolID'] == school]
        total_population = school_data['Population'].sum()
        white_population = school_data['White'].sum()
        black_population = school_data['Black'].sum()
        asian_population = school_data['Asian'].sum()
        total_race_population = white_population + black_population + asian_population
        if total_race_population == 0:
            continue
        proportions = {
            'White': white_population / total_race_population,
            'Black': black_population / total_race_population,
            'Asian': asian_population / total_race_population
        }
        entropy = -sum([p * np.log(p) if p > 0 else 0 for p in proportions.values()])
        school_entropies[school] = entropy

    average_entropy = np.mean(list(school_entropies.values()))

    # Compute envy count
    envy_count = 0
    for idx, row in df_results.iterrows():
        block = row['UniqueTract']
        assigned_school = row['UniqueSchoolID']
        try:
            assigned_distance = dist[block][assigned_school]
        except KeyError:
            continue
        for school in school_names:
            if school != assigned_school:
                try:
                    other_distance = dist[block][school]
                except KeyError:
                    continue
                if other_distance + 1e-6 < assigned_distance:
                    envy_count += 1
                    break  # Count each block only once

    return {
        'family_cost': family_cost,
        'racial_imbalance': racial_imbalance,
        'compactness_penalty': compactness_penalty,
        'average_entropy': average_entropy,
        'envy_count': envy_count
    }