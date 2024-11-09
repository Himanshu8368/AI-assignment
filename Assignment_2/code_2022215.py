# Boilerplate for AI Assignment — Knowledge Representation, Reasoning and Planning
# CSE 643

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from pyDatalog import pyDatalog
from collections import defaultdict, deque
# -------------------------------------we add this 
import pickle
import os
#-------------------------------------------end

## ****IMPORTANT****
## Don't import or use any other libraries other than defined above
## Otherwise your code file will be rejected in the automated testing

# ------------------ Global Variables ------------------
route_to_stops = defaultdict(list)  # Mapping of route IDs to lists of stops
trip_to_route = {}                   # Mapping of trip IDs to route IDs
stop_trip_count = defaultdict(int)    # Count of trips for each stop
fare_rules = {}                      # Mapping of route IDs to fare information
merged_fare_df = None                # To be initialized in create_kb()

# Load static data from GTFS (General Transit Feed Specification) files
df_stops = pd.read_csv('GTFS/stops.txt')
df_routes = pd.read_csv('GTFS/routes.txt')
df_stop_times = pd.read_csv('GTFS/stop_times.txt')
df_fare_attributes = pd.read_csv('GTFS/fare_attributes.txt')
df_trips = pd.read_csv('GTFS/trips.txt')
df_fare_rules = pd.read_csv('GTFS/fare_rules.txt')

# ------------------ Function Definitions ------------------

# ------------------ Function Implementations ------------------

def create_kb():
    """
    Create the knowledge base by populating global variables with information from loaded datasets.
    This function establishes the relationships between routes, trips, stops, and fare rules.

    Returns:
        None
    """
    global route_to_stops, trip_to_route, stop_trip_count, fare_rules, merged_fare_df

    # Step 1: Create trip_id to route_id mapping using df_trips.
    for _, row in df_trips.iterrows():
        trip_to_route[row['trip_id']] = row['route_id']

    # Step 2: Map route_id to a list of stops in the correct sequence using df_stop_times.
    for _, row in df_stop_times.iterrows():
        route_id = trip_to_route.get(row['trip_id'])
        if route_id:
            # Ensure we store stops with stop_sequence information
            if route_id not in route_to_stops:
                route_to_stops[route_id] = []
                
            # Add a tuple with (stop_id, stop_sequence)
            route_to_stops[route_id].append((row['stop_id'], row['stop_sequence']))
            # print(route_to_stops)

    # Step 3: Ensure each route only has unique stops by removing duplicates and sorting by stop_sequence.
    for route_id, stops in route_to_stops.items():
        # Filter out any entries that aren't tuples, to avoid errors
        stops = [stop for stop in stops if isinstance(stop, tuple) and len(stop) == 2]
        # print(stops)
        
        # Now, remove duplicates and sort by stop_sequence
        unique_stops = sorted(set(stops), key=lambda x: x[1])  # Sort by stop sequence
        # print(unique_stops)
        route_to_stops[route_id] = [stop_id for stop_id, _ in unique_stops]

    # Step 4: Count trips for each stop using df_stop_times.
    for _, row in df_stop_times.iterrows():
        stop_trip_count[row['stop_id']] += 1
        
    # print(stop_trip_count)
    # Step 5: Create fare rules for routes using df_fare_rules and df_fare_attributes.
    fare_rules = df_fare_rules.set_index('route_id').to_dict()['fare_id']
    fare_attributes = df_fare_attributes.set_index('fare_id').to_dict()
    # print(fare_rules)
    # print(fare_attributes)
    
    # Step 6: Merge fare_rules and fare_attributes into a single DataFrame for easy access.
    merged_fare_df = df_fare_rules.merge(df_fare_attributes, on='fare_id')
    # print(merged_fare_df)

# def create_kb():
#     """
#     Create or load the knowledge base by populating global variables with information from loaded datasets.
#     This function establishes the relationships between routes, trips, stops, and fare rules.
#     If a knowledge base file already exists, it loads the data from the file instead of reprocessing.

#     Returns:
#         None
#     """
#     global route_to_stops, trip_to_route, stop_trip_count, fare_rules, merged_fare_df

#     # Define the file path for the .pkl file
#     kb_file_path = 'knowledge_base.pkl'

#     # Check if the knowledge base file already exists
#     if os.path.exists(kb_file_path):
#         # Load data from the .pkl file
#         with open(kb_file_path, 'rb') as f:
#             route_to_stops, trip_to_route, stop_trip_count, fare_rules, merged_fare_df = pickle.load(f)
#         print("Knowledge base loaded from file.")
#         return  # Exit function after loading

#     # Step 1: Create trip_id to route_id mapping using df_trips.
#     for _, row in df_trips.iterrows():
#         trip_to_route[row['trip_id']] = row['route_id']

#     # Step 2: Map route_id to a list of stops in the correct sequence using df_stop_times.
#     for _, row in df_stop_times.iterrows():
#         route_id = trip_to_route.get(row['trip_id'])
#         if route_id:
#             # Ensure we store stops with stop_sequence information
#             if route_id not in route_to_stops:
#                 route_to_stops[route_id] = []
#             # Add a tuple with (stop_id, stop_sequence)
#             route_to_stops[route_id].append((row['stop_id'], row['stop_sequence']))

#     # Step 3: Ensure each route only has unique stops by removing duplicates and sorting by stop_sequence.
#     for route_id, stops in route_to_stops.items():
#         stops = [stop for stop in stops if isinstance(stop, tuple) and len(stop) == 2]
#         unique_stops = sorted(set(stops), key=lambda x: x[1])  # Sort by stop sequence
#         route_to_stops[route_id] = [stop_id for stop_id, _ in unique_stops]

#     # Step 4: Count trips for each stop using df_stop_times.
#     for _, row in df_stop_times.iterrows():
#         stop_trip_count[row['stop_id']] += 1

#     # Step 5: Create fare rules for routes using df_fare_rules and df_fare_attributes.
#     fare_rules = df_fare_rules.set_index('route_id').to_dict()['fare_id']
#     fare_attributes = df_fare_attributes.set_index('fare_id').to_dict()

#     # Step 6: Merge fare_rules and fare_attributes into a single DataFrame for easy access.
#     merged_fare_df = df_fare_rules.merge(df_fare_attributes, on='fare_id')

#     # Save the populated knowledge base to a .pkl file for future use
#     with open(kb_file_path, 'wb') as f:
#         pickle.dump((route_to_stops, trip_to_route, stop_trip_count, fare_rules, merged_fare_df), f)
#     print("Knowledge base created and saved to file.")

def get_busiest_routes():
    """
    Identify the top 5 busiest routes based on the number of trips.

    Returns:
        list: A list of tuples, where each tuple contains:
              - route_id (int): The ID of the route.
              - trip_count (int): The number of trips for that route.
    """
    # Count the number of trips for each route using trip_to_route.
    route_trip_count = defaultdict(int)
    for trip_id, route_id in trip_to_route.items():
        route_trip_count[route_id] += 1

    # Sort routes by trip count in descending order and get the top 5.
    top_routes = sorted(route_trip_count.items(), key=lambda x: x[1], reverse=True)[:5]
    # print(top_routes)
    return top_routes


def get_most_frequent_stops():
    """
    Identify the top 5 stops with the highest number of trips.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - trip_count (int): The number of trips for that stop.
    """
    # Sort stops by trip count in descending order and get the top 5.
    top_stops = sorted(stop_trip_count.items(), key=lambda x: x[1], reverse=True)[:5]
    # print(top_stops)
    return top_stops


def get_top_5_busiest_stops():
    """
    Identify the top 5 stops with the highest number of different routes passing through them.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - route_count (int): The number of routes passing through that stop.
    """
    # Create a mapping from stop_id to unique route_ids passing through it.
    stop_route_count = defaultdict(set)
    for route_id, stops in route_to_stops.items():
        for stop_id in stops:
            stop_route_count[stop_id].add(route_id)
    # print(stop_route_count)
    # Count unique routes per stop and get the top 5 busiest stops.
    stop_route_counts = {stop_id: len(routes) for stop_id, routes in stop_route_count.items()}
    top_stops = sorted(stop_route_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    # print(top_stops)
    return top_stops


def get_stops_with_one_direct_route():
    """
    Identify the top 5 pairs of consecutive stops (start and end) connected by exactly one direct route.
    The pairs are sorted by the combined frequency of trips passing through both stops.

    Returns:
        list: A list of tuples, where each tuple contains:
              - pair (tuple): A tuple with two stop IDs (stop_1, stop_2).
              - route_id (int): The ID of the route connecting the two stops.
    """
    # Map pairs of consecutive stops to routes.
    stop_pairs = defaultdict(list)
    for route_id, stops in route_to_stops.items():
        for i in range(len(stops) - 1):
            pair = (stops[i], stops[i + 1])
            stop_pairs[pair].append(route_id)
    # print(stop_pairs)
    # Filter pairs that only have one direct route and sort by trip frequency.
    one_route_pairs = [(pair, routes[0]) for pair, routes in stop_pairs.items() if len(routes) == 1]
    one_route_pairs_sorted = sorted(one_route_pairs, 
                                    key=lambda x: stop_trip_count[x[0][0]] + stop_trip_count[x[0][1]], 
                                    reverse=True)[:5]
    # print(one_route_pairs)
    return one_route_pairs_sorted


# Function to get merged fare DataFrame
# No need to change this function
def get_merged_fare_df():
    """
    Retrieve the merged fare DataFrame.

    Returns:
        DataFrame: The merged fare DataFrame containing fare rules and attributes.
    """
    global merged_fare_df
    return merged_fare_df


# Visualize the stop-route graph interactively
def visualize_stop_route_graph_interactive(route_to_stops):
    """
    Visualize the stop-route graph using Plotly for interactive exploration.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    # Create a NetworkX graph
    G = nx.Graph()
    for route_id, stops in route_to_stops.items():
        for i in range(len(stops) - 1):
            G.add_edge(stops[i], stops[i + 1], route=route_id)

    # Plot with Plotly for interactivity
    pos = nx.spring_layout(G)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(size=10, color='blue'))
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(showlegend=False)
    fig.show()

# import plotly.graph_objs as go
# import random

# # Function to visualize the stop-route graph interactively
# def visualize_stop_route_graph_interactive(route_to_stops):
#     """
#     Visualize the stop-route graph using Plotly for interactive exploration.

#     Args:
#         route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

#     Returns:
#         None
#     """
#     # Generate random 3D coordinates for each stop
#     stop_coordinates = {}
#     for route_id, stops in route_to_stops.items():
#         for stop in stops:
#             if stop not in stop_coordinates:
#                 stop_coordinates[stop] = (random.uniform(0, 100), random.uniform(0, 100), random.uniform(0, 100))

#     # Create lists for edges (lines) between stops on the same route
#     edge_x = []
#     edge_y = []
#     edge_z = []
#     node_x = []
#     node_y = []
#     node_z = []
#     node_text = []
    
#     for route_id, stops in route_to_stops.items():
#         for i in range(len(stops) - 1):
#             x0, y0, z0 = stop_coordinates[stops[i]]
#             x1, y1, z1 = stop_coordinates[stops[i + 1]]
#             edge_x.extend([x0, x1, None])
#             edge_y.extend([y0, y1, None])
#             edge_z.extend([z0, z1, None])
        
#         # Add each stop as a node
#         for stop in stops:
#             x, y, z = stop_coordinates[stop]
#             node_x.append(x)
#             node_y.append(y)
#             node_z.append(z)
#             node_text.append(f"Stop: {stop}, Route: {route_id}")

#     # Create 3D scatter plot for nodes (stops)
#     node_trace = go.Scatter3d(
#         x=node_x, y=node_y, z=node_z,
#         mode='markers',
#         marker=dict(size=6, color='blue', opacity=0.8),
#         text=node_text,
#         hoverinfo='text'
#     )

#     # Create 3D line plot for edges (connections between stops)
#     edge_trace = go.Scatter3d(
#         x=edge_x, y=edge_y, z=edge_z,
#         mode='lines',
#         line=dict(color='black', width=2),
#         hoverinfo='none'
#     )

#     # Create figure and add traces
#     fig = go.Figure(data=[edge_trace, node_trace])

#     # Update layout for better 3D visualization
#     fig.update_layout(
#         title="Stop-Route Graph Visualization",
#         scene=dict(
#             xaxis=dict(title="X"),
#             yaxis=dict(title="Y"),
#             zaxis=dict(title="Z")
#         ),
#         margin=dict(l=0, r=0, b=0, t=40)
#     )

#     # Show the interactive 3D graph
#     fig.show()

# Brute-Force Approach for finding direct routes
def direct_route_brute_force(start_stop, end_stop):
    """
    Find all valid routes between two stops using a brute-force method.

    Args:
        start_stop (int): The ID of the starting stop.
        end_stop (int): The ID of the ending stop.

    Returns:
        list: A list of route IDs (int) that connect the two stops directly.
    """
    direct_routes = []

    # Iterate through each route and its stops
    for route_id, stops in route_to_stops.items():
        if start_stop in stops and end_stop in stops:
            
            # Check if end_stop appears after start_stop in the route
            start_index = stops.index(start_stop)
            end_index = stops.index(end_stop)
            if start_index < end_index:
                # append direct_route
                direct_routes.append(route_id)
    # print(direct_routes)
    return direct_routes

# Initialize Datalog predicates for reasoning
pyDatalog.create_terms('RouteHasStop, DirectRoute, OptimalRoute, X, Y, Z, R, R1, R2')  


# def initialize_datalog():
#     """
#     Initialize Datalog terms and predicates for reasoning about routes and stops.

#     Returns:
#         None
#     """
#     pyDatalog.clear()  # Clear previous terms
#     print("Terms initialized: DirectRoute, RouteHasStop, OptimalRoute")  # Confirmation print

#     # Define base predicates for routes and stops
#     # RouteHasStop(R, X) means route R includes stop X
#     # DirectRoute(X, Y, R) means there is a direct route R between stops X and Y
#     DirectRoute(X, Y, R) <= RouteHasStop(R, X) & RouteHasStop(R, Y)
    
#     # Populate the knowledge base with data
#     create_kb()  # This will populate the global route_to_stops data structure
#     add_route_data(route_to_stops)  # Add route data to Datalog

def initialize_datalog():
    """
    Initialize Datalog terms and predicates for reasoning about routes and stops.

    Returns:
        None
    """
    pyDatalog.clear()  # Clear previous terms
    # print("Terms initialized: DirectRoute, RouteHasStop, OptimalRoute")  # Confirmation print

    # Define base predicates for routes and stops
    DirectRoute(X, Y, R) <= RouteHasStop(R, X) & RouteHasStop(R, Y)

    # Populate the knowledge base with data
    create_kb()  # This will populate the global route_to_stops data structure
    add_route_data(route_to_stops)  # Add route data to Datalog



def add_route_data(route_to_stops):
    """
    Add the route data to Datalog for reasoning.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    for route_id, stops in route_to_stops.items():
        for stop in stops:
            # Add facts to the knowledge base
            +RouteHasStop(route_id, stop)

def query_direct_routes(start, end):
    """
    Query for direct routes between two stops.

    Args:
        start (int): The ID of the starting stop.
        end (int): The ID of the ending stop.

    Returns:
        list: A sorted list of route IDs (int) connecting the two stops directly.
    """
    # Query for routes that have both start and end stops
    query_result = DirectRoute(start, end, R)
    
    # route_ids = (set(route_id for route_id, in query_result))
    route_ids = sorted(set(route_id for route_id, in query_result))
    return route_ids

def forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform forward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria.
    """
    paths = []

    #  Find all routes that start from the starting stop
    direct_routes = query_direct_routes(start_stop_id, stop_id_to_include)
    # print(direct_routes)
    
    # For each direct route, find paths to the ending stop
    for route_id in direct_routes:
        # Get all stops on this route
        stops_on_route = route_to_stops[route_id]
        # print(stops_on_route)
        
        # Check if via stop is included and find routes from there to the end stop
        if stop_id_to_include in stops_on_route:
            index_of_via = stops_on_route.index(stop_id_to_include)
            # print(index_of_via)
            
            # Check for paths from the via stop to the end stop within the same route
            for stop_id in stops_on_route[index_of_via + 1:]:
                if stop_id == end_stop_id:
                    paths.append([(route_id, stop_id)])
                # print(paths)
            # If transfers are allowed, find other routes from the via stop to the end stop
            if max_transfers > 0:
                transfer_routes = query_direct_routes(stop_id_to_include, end_stop_id)
                # print(transfer_routes)
                for transfer_route_id in transfer_routes:
                    # paths.append([(route_id, stop_id_to_include), (transfer_route_id, end_stop_id)])
                    paths.append((route_id, stop_id_to_include,transfer_route_id))
    # print(paths)
    return paths
    # return direct_routes



def backward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform backward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria.
    """
    paths = []

    # Step 1: Find all routes that end at the ending stop
    direct_routes = query_direct_routes(stop_id_to_include, end_stop_id)
    # print(direct_routes)
    
    # Step 2: For each direct route, find paths back to the starting stop
    for route_id in direct_routes:
        # Get all stops on this route
        stops_on_route = route_to_stops[route_id]
        # print(stops_on_route)
        
        # Check if via stop is included and find routes from the start stop
        if stop_id_to_include in stops_on_route:
            index_of_via = stops_on_route.index(stop_id_to_include)
            # print(index_of_via)
            
            # Check for paths from the via stop to the start stop
            for stop_id in reversed(stops_on_route[:index_of_via]):
                if stop_id == start_stop_id:
                    paths.append([(route_id, stop_id)])
            # print(paths)
            
            # If transfers are allowed, find routes from the start stop to the via stop
            if max_transfers > 0:
                transfer_routes = query_direct_routes(start_stop_id, stop_id_to_include)
                # print(transfer_routes)
                
                for transfer_route_id in transfer_routes:
                    # paths.append([(transfer_route_id, start_stop_id), (route_id, stop_id_to_include)])
                    paths.append((route_id, stop_id_to_include,transfer_route_id))
    print(paths)
    return paths
    # return direct_routes




# def pddl_planning(start_stop_id, end_stop_id, stop_id_to_include=None, max_transfers=3):
#     """
#     Optimized PDDL-style planning to find routes with optional transfers.

#     Args:
#         start_stop_id (int): The starting stop ID.
#         end_stop_id (int): The ending stop ID.
#         stop_id_to_include (int): The stop ID for a transfer.
#         max_transfers (int): The maximum number of transfers allowed.

#     Returns:
#         list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
#               - route_id (int): The ID of the route.
#               - stop_id (int): The ID of the stop.
#     """
#     # Queue for BFS, with each element: (current stop, path taken, current transfers)
#     queue = deque([(start_stop_id, [], 0)])  # (current stop, path, transfers)
#     visited = set()
#     valid_paths = []  # Store all paths meeting the criteria

#     # Precompute a reverse map of stops to routes for efficient lookup
#     stop_to_routes = defaultdict(list)
#     for route_id, stops in route_to_stops.items():
#         for stop in stops:
#             stop_to_routes[stop].append(route_id)
    
#     while queue and len(valid_paths) < 5:  # Limit to 5 paths
#         current_stop, path, transfers = queue.popleft()

#         # Skip if we've already visited this stop with the same or fewer transfers
#         if (current_stop, transfers) in visited:
#             continue
#         visited.add((current_stop, transfers))

#         # Check if destination is reached within transfer limits
#         if current_stop == end_stop_id and transfers <= max_transfers:
#             if stop_id_to_include is None or any(stop == stop_id_to_include for _, stop in path):
#                 valid_paths.append(path + [(None, current_stop)])
#                 continue

#         # Expand routes from the current stop using precomputed map
#         for route_id in stop_to_routes[current_stop]:
#             stops_in_route = route_to_stops[route_id]
#             start_index = stops_in_route.index(current_stop)

#             # Traverse only forward from the current stop to avoid revisits
#             for next_stop in stops_in_route[start_index + 1:]:
#                 if (next_stop, transfers + 1) not in visited:
#                     # Add new stop to the queue with updated path and transfers
#                     queue.append((next_stop, path + [(route_id, current_stop)], transfers + 1))

#     print(f"Found {len(valid_paths)} valid paths in total.")
#     return valid_paths[:5]  # Return up to 5 valid paths

def pddl_planning(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Implement PDDL-style planning to find routes with optional transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID for a transfer.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id (int): The ID of the route.
              - stop_id (int): The ID of the stop.
    """
    
    # List to store the optimal routes
    optimal_routes = []

    # Step 1: Query all direct routes from start_stop_id to end_stop_id (both inclusive)
    direct_routes = query_direct_routes(start_stop_id, end_stop_id)
    # print(direct_routes)
    
    # If direct routes exist, add them to optimal routes
    if direct_routes:
        for route in direct_routes:
            optimal_routes.append([(route, start_stop_id), (route, end_stop_id)])
    # print(optimal_routes)
    
    # Step 2: If no direct route or multiple transfers are allowed, we explore possible transfers
    if not direct_routes or max_transfers > 0:
        # Perform forward chaining to explore routes with potential transfer at stop_id_to_include
        forward_routes = forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers)
        # print(forward_routes)
        
        # Add valid forward chaining routes
        for route_path in forward_routes:
            optimal_routes.append(route_path)
        # print(optimal_routes)
        # # Perform backward chaining to explore routes with potential transfer at stop_id_to_include
        # backward_routes = backward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers)
        
        # # Add valid backward chaining routes
        # for route_path in backward_routes:
        #     optimal_routes.append(route_path)
    
    # Step 3: Return the list of optimal routes found
    # print(optimal_routes)
    return optimal_routes

# Function to filter fare data based on an initial fare limit
def prune_data(merged_fare_df, initial_fare):
    """
    Filter fare data based on an initial fare limit.

    Args:
        merged_fare_df (DataFrame): The merged fare DataFrame.
        initial_fare (float): The maximum fare allowed.

    Returns:
        DataFrame: A filtered DataFrame containing only routes within the fare limit.
    """
    # Filter the DataFrame based on the initial fare constraint using 'price' instead of 'fare'
    pruned_df = merged_fare_df[merged_fare_df['price'] <= initial_fare]
    # print(f"Pruned data to include fares within {initial_fare}. Remaining rows: {len(pruned_df)}")
    return pruned_df


# Pre-computation of Route Summary
def compute_route_summary(pruned_df):
    """
    Generate a summary of routes based on fare information.

    Args:
        pruned_df (DataFrame): The filtered DataFrame containing fare information.

    Returns:
        dict: A summary of routes with the following structure:
              {
                  route_id (int): {
                      'min_price': float,          # The minimum fare for the route
                      'stops': set                # A set of stop IDs for that route
                  }
              }
    """
    route_summary = {}

    # Group by route_id to calculate min_price and collect stops for each route
    for route_id, group in pruned_df.groupby('route_id'):
        min_price = group['price'].min()  # Use 'price' instead of 'fare'
        stops = set(group['origin_id']).union(group['destination_id'])  # Collect stops from origin and destination

        route_summary[route_id] = {
            'min_price': min_price,
            'stops': stops
        }
    # print(min_price)
    # print(stops)
    
    # print(f"Computed route summary for {len(route_summary)} routes.")
    return route_summary

# BFS for optimized route planning with fare constraints


def bfs_route_planner_optimized(start_stop_id, end_stop_id, initial_fare, route_summary, max_transfers=3):
    """
    Use Breadth-First Search (BFS) to find the optimal route while considering fare constraints.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        initial_fare (float): The available fare for the trip.
        route_summary (dict): A summary of routes with fare and stop information.
        max_transfers (int): The maximum number of transfers allowed (default is 3).

    Returns:
        list: A list representing the optimal route with stops and routes taken, structured as:
              [
                  (route_id (int), stop_id (int)),  # Tuple for each stop taken in the route
                  ...
              ]
    """
    # Initialize queue for BFS: each entry is (current stop, route taken, path, remaining fare, transfers)
    queue = deque([(start_stop_id, None, [], initial_fare, 0)])
    visited = set()  # Keep track of (stop, route, transfers) to avoid revisiting the same route configuration

    while queue:
        current_stop, current_route, path, remaining_fare, transfers = queue.popleft()
        
        # If we reach the destination with remaining fare and within transfer limit, return the path
        if current_stop == end_stop_id and transfers <= max_transfers:
            # print(f"Optimal path found with remaining fare {remaining_fare}: {path + [(current_route, current_stop)]}")
            # print(len(path))
            if(len(path)==1):
                # print([(path[0][0],current_stop)])
                return [(path[0][0],current_stop)]
            # return path + [(current_route, current_stop)]
            # print([(path[0][0],path [1][1])] + [(current_route, current_stop)])
            return [(path[0][0],path [1][1])] + [(current_route, current_stop)]

        # Mark the current configuration as visited
        visited.add((current_stop, current_route, transfers))
        # print(visited)
        
        # Explore all routes available from the current stop
        for route_id, route_data in route_summary.items():
            if remaining_fare < route_data['min_price']:  # Skip routes exceeding remaining fare
                continue
            if current_stop not in route_data['stops']:  # Skip routes not containing the current stop
                continue

            # Calculate fare after taking the route and count transfers if switching routes
            fare_after_route = remaining_fare - route_data['min_price']
            new_transfers = transfers + (1 if route_id != current_route else 0)
            # print(fare_after_route)
            # print(new_transfers)
            
            if new_transfers > max_transfers:
                continue

            # Enqueue all subsequent stops in this route
            for stop_id in route_data['stops']:
                if (stop_id, route_id, new_transfers) not in visited:
                    queue.append((stop_id, route_id, path + [(route_id, current_stop)], fare_after_route, new_transfers))
            # print(queue)
            
    # print("No valid path found within fare and transfer constraints.")
    return []



# create_kb()  # Ensure the data is loaded before testing
# merged_fare_df = get_merged_fare_df()  # Use the function to retrieve the DataFrame
# initialize_datalog()
# visualize_stop_route_graph_interactive(route_to_stops)

