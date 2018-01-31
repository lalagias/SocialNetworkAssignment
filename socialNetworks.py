import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import datetime

# Variable N (N cuts of timestamp)
N = 3

# Read data and convert it to dataframe
data = pd.read_csv("E:\MOCCA\Downloads\SocialNetwork\SocialMini.csv", sep=' ')
data
graph_data = pd.DataFrame(data)
timestamps = graph_data['timestamp']
timestamps = pd.DataFrame(timestamps)
# timestamps
# timestamps['timestamp']

# timestampList = timestamps['timestamp'].values.tolist()
# timestampList = [datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in timestampList]
# timestamps = pd.DataFrame(timestampList)
# timestamps.columns

# 1st question
# Find tmin
tmin = timestamps.min()
tmin

# Find tmax
tmax = timestamps.max()
tmax

# Find intervals
intervals = np.array_split(data, N)
intervals
intervalsList = []
for i in intervals:
    intervalsList.append('(' + str(i['timestamp'].iloc[0]) + ',' + str(i['timestamp'].iloc[-1]) + ')')

intervalsList
intervals[0]
# TODO FOR all dicts for intervals
intervalsDict = intervals[0].to_dict()
# intervalsDict['target_id']
intervals[0].values

DG = nx.DiGraph()
edges = [ (x[0], x[1], {'timestamp': x[2] }) for x in intervals[0].values ]
DG.add_edges_from(edges)
[e for e in DG.edges]

graphs = []
for i in range(0, N):
    directed_graph = nx.DiGraph()
    edges = [ (edge[0], edge[1], {'timestamp': edge[2] }) for edge in intervals[i].values ]
    directed_graph.add_edges_from(edges)
    graphs.append(directed_graph)
graphs

# 4th question
# 1) DEGREE CENTRALITY
degree_centrality = {}
for i in range(0, N):
    degree_centrality[i] = nx.degree_centrality(graphs[i])

degree_centrality[2]

# 2) IN-DEGREE CENTRALITY
in_degree_centrality = {}
for i in range(0, N):
    in_degree_centrality[i] = nx.in_degree_centrality(graphs[i])

in_degree_centrality[2]

# 3) OUT-DEGREE CENTRALITY
out_degree_centrality = {}
for i in range(0, N):
    out_degree_centrality[i] = nx.out_degree_centrality(graphs[i])

# 4) CLOSENESS CENTRALITY
closeness_centrality = {}
# for i in range(0, N):
#     closeness_centrality[i] = nx.closeness_centrality(graphs[i], u='35233')
closeness_centralityKappa = nx.closeness_centrality(graphs[1], reverse=True)
closeness_centralityKappa
