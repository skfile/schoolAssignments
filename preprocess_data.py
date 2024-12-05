#!/usr/bin/env python3

import os
import glob
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import argparse
import pickle
import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute the great-circle distance between two points on the Earth surface.
    Input parameters: latitudes and longitudes of two points in decimal degrees.
    Output: distance between the two points in meters.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Earth's radius in meters
    R = 6371000

    # Compute differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))

    distance = R * c
    return distance

def process_prepped_file(filename, district_code):
    # Read the CSV file, ensuring that 'block_id' and 'NCESSCH' are read as strings
    df = pd.read_csv(filename, dtype={'block_id': str, 'NCESSCH': str})

    # Process Blocks Data
    blocks_cols = [
        'block_id', 'block_centroid_lat', 'block_centroid_long',
        'num_white_to_school', 'num_black_to_school', 'num_asian_to_school',
        'num_native_to_school', 'num_hispanic_to_school', 'num_total_to_school',
        'block_idx'
    ]
    df_blocks = df[blocks_cols].copy()
    df_blocks = df_blocks.drop_duplicates(subset=['block_id'])
    df_blocks.rename(columns={
        'block_id': 'Tract',
        'block_centroid_lat': 'Latitude',
        'block_centroid_long': 'Longitude',
        'num_white_to_school': 'White',
        'num_black_to_school': 'Black',
        'num_asian_to_school': 'Asian',
        'num_total_to_school': 'Population'
    }, inplace=True)
    df_blocks['district_code'] = district_code  # Add district_code column
    df_blocks['geometry'] = df_blocks.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
    df_blocks['UniqueTract'] = df_blocks['district_code'] + '_' + df_blocks['Tract']
    gdf_blocks = gpd.GeoDataFrame(df_blocks, geometry='geometry')

    # Process Schools Data
    schools_cols = [
        'NCESSCH', 'SCH_NAME', 'LEA_NAME',
        'lat', 'long', 'total_white', 'total_black', 'total_asian',
        'total_native', 'total_hispanic', 'total_enrollment', 'school_idx'
    ]
    df_schools = df[schools_cols].copy()
    df_schools = df_schools.drop_duplicates(subset=['NCESSCH'])
    df_schools.rename(columns={
        'lat': 'Latitude',
        'long': 'Longitude',
        'total_white': 'White',
        'total_black': 'Black',
        'total_asian': 'Asian',
        'total_enrollment': 'Capacity',
        'SCH_NAME': 'Name',
        'NCESSCH': 'SchoolID'
    }, inplace=True)
    df_schools['district_code'] = district_code  # Add district_code column
    df_schools['geometry'] = df_schools.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
    df_schools['UniqueSchoolID'] = df_schools['district_code'] + '_' + df_schools['SchoolID']
    gdf_schools = gpd.GeoDataFrame(df_schools, geometry='geometry')

    return gdf_blocks, gdf_schools

def compute_distances(gdf_blocks, gdf_schools):
    dist = {}
    for _, block in gdf_blocks.iterrows():
        block_id = block['UniqueTract']
        dist[block_id] = {}
        lat1 = block['Latitude']
        lon1 = block['Longitude']
        for _, school in gdf_schools.iterrows():
            school_id = school['UniqueSchoolID']
            lat2 = school['Latitude']
            lon2 = school['Longitude']
            try:
                distance = haversine_distance(lat1, lon1, lat2, lon2)
            except Exception as e:
                print(f"Error computing distance between Block {block_id} and School {school_id}: {e}")
                continue
            dist[block_id][school_id] = distance
    return dist

def main():
    parser = argparse.ArgumentParser(description='Process prepped files and compute distances.')
    parser.add_argument('--data_directory', type=str, default='MD', help='Directory containing the data files')
    parser.add_argument('--output_directory', type=str, default='.', help='Directory to save the output files')
    parser.add_argument('--num_districts', type=int, default=None, help='Number of districts to process (default: all)')
    args = parser.parse_args()

    data_directory = args.data_directory
    output_directory = args.output_directory
    num_districts = args.num_districts

    # Correct the glob pattern to find prepped files
    prepped_files = sorted(glob.glob(os.path.join(data_directory, 'prepped_file_for_solver_*.csv')))

    if not prepped_files:
        print(f"No prepped files found in {data_directory}. Please check the directory path.")
        return

    # Limit the number of districts if specified
    if num_districts is not None:
        prepped_files = prepped_files[:num_districts]

    all_blocks = []
    all_schools = []
    dist_all = {}
    skipped_districts = []

    for prepped_file in prepped_files:
        district_code = prepped_file.split('_')[-1].split('.')[0]
        print(f"\nProcessing district {district_code}")
        print(f"Prepped file: {prepped_file}")

        gdf_blocks, gdf_schools = process_prepped_file(prepped_file, district_code)
        if gdf_blocks.empty or gdf_schools.empty:
            print(f"Empty DataFrame for district {district_code}. Skipping this district.")
            skipped_districts.append(district_code)
            continue

        dist = compute_distances(gdf_blocks, gdf_schools)

        all_blocks.append(gdf_blocks)
        all_schools.append(gdf_schools)
        dist_all.update(dist)

    if not all_blocks or not all_schools:
        print("No data processed. Exiting.")
        return

    # Combine all blocks and schools
    gdf_blocks = pd.concat(all_blocks, ignore_index=True)
    gdf_schools = pd.concat(all_schools, ignore_index=True)
    dist = dist_all

    # Save the data
    blocks_file = os.path.join(output_directory, 'gdf_blocks.pkl')
    schools_file = os.path.join(output_directory, 'gdf_schools.pkl')
    dist_file = os.path.join(output_directory, 'dist.pkl')

    gdf_blocks.to_pickle(blocks_file)
    gdf_schools.to_pickle(schools_file)
    with open(dist_file, 'wb') as f:
        pickle.dump(dist, f)

    print(f"\nSaved gdf_blocks to {blocks_file}")
    print(f"Saved gdf_schools to {schools_file}")
    print(f"Saved dist to {dist_file}")

    if skipped_districts:
        print(f"\nSkipped districts due to issues: {skipped_districts}")

if __name__ == '__main__':
    main()

# #!/usr/bin/env python3

# import os
# import glob
# import pandas as pd
# import geopandas as gpd
# from shapely.geometry import Point
# import argparse
# import pickle
# import numpy as np

# def haversine_distance(lat1, lon1, lat2, lon2):
#     """
#     Compute the great-circle distance between two points on the Earth surface.
#     Input parameters: latitudes and longitudes of two points in decimal degrees.
#     Output: distance between the two points in meters.
#     """
#     # Convert decimal degrees to radians
#     lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

#     # Earth's radius in meters
#     R = 6371000

#     # Compute differences
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1

#     # Haversine formula
#     a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
#     c = 2 * np.arcsin(np.sqrt(a))

#     distance = R * c
#     return distance

# def process_prepped_file(filename, district_code):
#     # Read the CSV file, ensuring that 'block_id' and 'NCESSCH' are read as strings
#     df = pd.read_csv(filename, dtype={'block_id': str, 'NCESSCH': str})

#     # Process Blocks Data
#     blocks_cols = [
#         'block_id', 'block_centroid_lat', 'block_centroid_long',
#         'num_white_to_school', 'num_black_to_school', 'num_asian_to_school',
#         'num_native_to_school', 'num_hispanic_to_school', 'num_total_to_school',
#         'block_idx'
#     ]
#     df_blocks = df[blocks_cols].copy()
#     df_blocks = df_blocks.drop_duplicates(subset=['block_id'])
#     df_blocks.rename(columns={
#         'block_id': 'Tract',
#         'block_centroid_lat': 'Latitude',
#         'block_centroid_long': 'Longitude',
#         'num_white_to_school': 'White',
#         'num_black_to_school': 'Black',
#         'num_asian_to_school': 'Asian',
#         'num_total_to_school': 'Population'
#     }, inplace=True)
#     df_blocks['district_code'] = district_code  # Add district_code column
#     df_blocks['geometry'] = df_blocks.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
#     df_blocks['UniqueTract'] = df_blocks['district_code'] + '_' + df_blocks['Tract']
#     gdf_blocks = gpd.GeoDataFrame(df_blocks, geometry='geometry')

#     # Process Schools Data
#     schools_cols = [
#         'NCESSCH', 'SCH_NAME', 'LEA_NAME',
#         'lat', 'long', 'total_white', 'total_black', 'total_asian',
#         'total_native', 'total_hispanic', 'total_enrollment', 'school_idx'
#     ]
#     df_schools = df[schools_cols].copy()
#     df_schools = df_schools.drop_duplicates(subset=['NCESSCH'])
#     df_schools.rename(columns={
#         'lat': 'Latitude',
#         'long': 'Longitude',
#         'total_white': 'White',
#         'total_black': 'Black',
#         'total_asian': 'Asian',
#         'total_enrollment': 'Capacity',
#         'SCH_NAME': 'Name',
#         'NCESSCH': 'SchoolID'
#     }, inplace=True)
#     df_schools['district_code'] = district_code  # Add district_code column
#     df_schools['geometry'] = df_schools.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
#     df_schools['UniqueSchoolID'] = df_schools['district_code'] + '_' + df_schools['SchoolID']
#     gdf_schools = gpd.GeoDataFrame(df_schools, geometry='geometry')

#     return gdf_blocks, gdf_schools

# def compute_distances(gdf_blocks, gdf_schools):
#     dist = {}
#     for _, block in gdf_blocks.iterrows():
#         block_id = block['UniqueTract']
#         dist[block_id] = {}
#         lat1 = block['Latitude']
#         lon1 = block['Longitude']
#         for _, school in gdf_schools.iterrows():
#             school_id = school['UniqueSchoolID']
#             lat2 = school['Latitude']
#             lon2 = school['Longitude']
#             try:
#                 distance = haversine_distance(lat1, lon1, lat2, lon2)
#             except Exception as e:
#                 print(f"Error computing distance between Block {block_id} and School {school_id}: {e}")
#                 continue
#             dist[block_id][school_id] = distance
#     return dist

# def main():
#     parser = argparse.ArgumentParser(description='Process prepped files and compute distances.')
#     parser.add_argument('--data_directory', type=str, default='MD', help='Directory containing the data files')
#     parser.add_argument('--output_directory', type=str, default='.', help='Directory to save the output files')
#     parser.add_argument('--num_districts', type=int, default=None, help='Number of districts to process (default: all)')
#     args = parser.parse_args()

#     data_directory = args.data_directory
#     output_directory = args.output_directory
#     num_districts = args.num_districts

#     # Correct the glob pattern to find prepped files
#     prepped_files = sorted(glob.glob(os.path.join(data_directory, 'prepped_file_for_solver_*.csv')))

#     if not prepped_files:
#         print(f"No prepped files found in {data_directory}. Please check the directory path.")
#         return

#     # Limit the number of districts if specified
#     if num_districts is not None:
#         prepped_files = prepped_files[:num_districts]

#     all_blocks = []
#     all_schools = []
#     dist_all = {}
#     skipped_districts = []

#     for prepped_file in prepped_files:
#         district_code = prepped_file.split('_')[-1].split('.')[0]
#         print(f"\nProcessing district {district_code}")
#         print(f"Prepped file: {prepped_file}")

#         gdf_blocks, gdf_schools = process_prepped_file(prepped_file, district_code)
#         if gdf_blocks.empty or gdf_schools.empty:
#             print(f"Empty DataFrame for district {district_code}. Skipping this district.")
#             skipped_districts.append(district_code)
#             continue

#         dist = compute_distances(gdf_blocks, gdf_schools)

#         all_blocks.append(gdf_blocks)
#         all_schools.append(gdf_schools)
#         dist_all.update(dist)

#     if not all_blocks or not all_schools:
#         print("No data processed. Exiting.")
#         return

#     # Combine all blocks and schools
#     gdf_blocks = pd.concat(all_blocks, ignore_index=True)
#     gdf_schools = pd.concat(all_schools, ignore_index=True)
#     dist = dist_all

#     # Save the data
#     blocks_file = os.path.join(output_directory, 'gdf_blocks.pkl')
#     schools_file = os.path.join(output_directory, 'gdf_schools.pkl')
#     dist_file = os.path.join(output_directory, 'dist.pkl')

#     gdf_blocks.to_pickle(blocks_file)
#     gdf_schools.to_pickle(schools_file)
#     with open(dist_file, 'wb') as f:
#         pickle.dump(dist, f)

#     print(f"\nSaved gdf_blocks to {blocks_file}")
#     print(f"Saved gdf_schools to {schools_file}")
#     print(f"Saved dist to {dist_file}")

#     if skipped_districts:
#         print(f"\nSkipped districts due to issues: {skipped_districts}")

# if __name__ == '__main__':
#     main()