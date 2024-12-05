from ortools.linear_solver import pywraplp
import pandas as pd
import numpy as np

def solve_optimization(
    df_blocks, df_schools, dist, P_k, BigM, capacity_slack,
    lambda_cost, lambda_imbalance, lambda_compactness, solver_time_limit=60000
):
    # Create optimization model
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        print('Solver not available.')
        return None

    # Set solver parameters for time limit
    solver.SetTimeLimit(solver_time_limit)  # Time limit in milliseconds

    # Decision variables
    x = {}
    block_names = list(df_blocks['UniqueTract'])
    school_names = list(df_schools['UniqueSchoolID'])

    # Ensure dist contains all block-school pairs
    missing_pairs = []
    for block in block_names:
        for school in school_names:
            if block not in dist or school not in dist[block]:
                missing_pairs.append((block, school))

    if missing_pairs:
        print(f"Missing distance data for {len(missing_pairs)} block-school pairs.")
        # Optionally recompute distances for missing pairs or skip
        # For now, we will skip these pairs in the model

    for block in block_names:
        for school in school_names:
            # Only create variables for block-school pairs present in dist
            if block in dist and school in dist[block]:
                x[(block, school)] = solver.IntVar(0, 1, f'x_{block}_{school}')

    # Auxiliary variables for racial imbalance
    d = {}
    for school in school_names:
        for race in ['White', 'Black', 'Asian']:
            d[(school, race)] = solver.NumVar(0, solver.infinity(), f'd_{school}_{race}')

    # Variables for total assigned populations
    n_j = {}
    n_jk = {}
    for school in school_names:
        n_j[school] = solver.NumVar(0, solver.infinity(), f'n_{school}')
        for race in ['White', 'Black', 'Asian']:
            n_jk[(school, race)] = solver.NumVar(0, solver.infinity(), f'n_{school}_{race}')

    # Variables for compactness
    M = {}
    for school in school_names:
        M[school] = solver.NumVar(0, solver.infinity(), f'M_{school}')

    # Constraints

    # Assignment Constraint: Each block must be assigned to exactly one school
    for block in block_names:
        # Only sum over schools that have variables for this block
        x_vars = [x[(block, school)] for school in school_names if (block, school) in x]
        solver.Add(solver.Sum(x_vars) == 1)

    # Capacity Constraint: Total assigned population should not exceed capacity (with slack)
    total_population = df_blocks['Population'].sum()
    num_schools = len(school_names)
    capacity_per_school = capacity_slack * (total_population / num_schools)

    for school in school_names:
        x_vars = [x[(block, school)] for block in block_names if (block, school) in x]
        solver.Add(
            solver.Sum([
                df_blocks.loc[df_blocks['UniqueTract'] == block, 'Population'].values[0] * x[(block, school)]
                for block in block_names if (block, school) in x
            ]) <= capacity_per_school
        )

    # Define n_j and n_jk (Total assigned populations)
    for school in school_names:
        # Total population assigned to school
        solver.Add(n_j[school] == solver.Sum([
            df_blocks.loc[df_blocks['UniqueTract'] == block, 'Population'].values[0] * x[(block, school)]
            for block in block_names if (block, school) in x
        ]))
        for race in ['White', 'Black', 'Asian']:
            # Total population of race assigned to school
            solver.Add(n_jk[(school, race)] == solver.Sum([
                df_blocks.loc[df_blocks['UniqueTract'] == block, race].values[0] * x[(block, school)]
                for block in block_names if (block, school) in x
            ]))
            # Racial Imbalance Constraints: d_{jk} >= |n_{jk} - P_k * n_j|
            solver.Add(d[(school, race)] >= n_jk[(school, race)] - P_k[race] * n_j[school])
            solver.Add(d[(school, race)] >= - (n_jk[(school, race)] - P_k[race] * n_j[school]))
            solver.Add(d[(school, race)] >= 0)

    # Compactness Constraints: M_j >= d_{ij} - (1 - x_{ij}) * BigM
    for school in school_names:
        for block in block_names:
            if (block, school) in x:
                distance = dist[block][school]
                solver.Add(M[school] >= distance - (1 - x[(block, school)]) * BigM)

    # Compute Family Cost (Total travel distance for students)
    family_cost_terms = []
    for block in block_names:
        block_population = df_blocks.loc[df_blocks['UniqueTract'] == block, 'Population'].values[0]
        for school in school_names:
            if (block, school) in x:
                distance_to_school = dist[block][school]
                cost = block_population * distance_to_school * x[(block, school)]
                family_cost_terms.append(cost)

    family_cost = solver.Sum(family_cost_terms)

    # Racial Imbalance Term
    racial_imbalance = solver.Sum([d[(school, race)] for school in school_names for race in ['White', 'Black', 'Asian']])

    # Compactness Penalty Term
    compactness_penalty = solver.Sum([M[school] for school in school_names])

    # Objective Function: Minimize weighted sum of family cost, racial imbalance, and compactness penalty
    solver.Minimize(lambda_cost * family_cost + lambda_imbalance * racial_imbalance + lambda_compactness * compactness_penalty)

    # Solve the model
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print('Optimal solution found.')
    elif status == pywraplp.Solver.FEASIBLE:
        print('Feasible solution found (may not be optimal).')
    else:
        print('No solution found.')
        return None

    # Collect the assignments
    assignments = []
    for block in block_names:
        for school in school_names:
            if (block, school) in x and x[(block, school)].solution_value() > 0.5:
                assignments.append({'UniqueTract': block, 'UniqueSchoolID': school})

    df_assignments = pd.DataFrame(assignments)

    # Merge assignments with block data
    df_results = pd.merge(df_assignments, df_blocks, on='UniqueTract', how='left')

    # Calculate total cost components
    total_cost = solver.Objective().Value()
    family_cost_value = family_cost.solution_value()
    racial_imbalance_value = racial_imbalance.solution_value()
    compactness_penalty_value = compactness_penalty.solution_value()

    return {
        'assignments': df_results,
        'total_cost': total_cost,
        'family_cost': family_cost_value,
        'racial_imbalance': racial_imbalance_value,
        'compactness_penalty': compactness_penalty_value
    }