import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import datetime
import sys
import operator

file_path = sys.argv[1]

# Variable N (N divsions of time interval)
N = int(input("\nPlease give the number of divisions"))

# Read data and convert it to dataframe
data = pd.read_csv(file_path, sep=' ')
graph_data = pd.DataFrame(data)
timestamps = graph_data['timestamp']
timestamps = pd.DataFrame(timestamps)

# 1st Question

# Find tmin
tmin = timestamps.min()
print("\ntmin = " + datetime.datetime.fromtimestamp(tmin).strftime('%Y-%m-%d %H:%M:%S'))

# Find tmax
tmax = timestamps.max()
print("\ntmax = " + datetime.datetime.fromtimestamp(tmax).strftime('%Y-%m-%d %H:%M:%S'))

# 2nd Question

# Find intervals
intervals = np.array_split(data, N)
intervalsList = []
for i in intervals:
    intervalsList.append('(' + str(i['timestamp'].iloc[0]) + ',' + str(i['timestamp'].iloc[-1]) + ')')
for i, val in enumerate(intervalsList):
    print("\nTime Interval (" + str(i) + ") : From " +
         datetime.datetime.fromtimestamp(int(val.split(",")[0][1:])).strftime('%Y-%m-%d %H:%M:%S') + " To " +
         datetime.datetime.fromtimestamp(int(val.split(",")[1][:-1])).strftime('%Y-%m-%d %H:%M:%S')
         )

# 3rd Question

#Create graphs for each time interval
graphs = []
for i in range(0, N):
    directed_graph = nx.DiGraph()
    edges = [ (edge[0], edge[1], {'timestamp': edge[2] }) for edge in intervals[i].values ]
    directed_graph.add_edges_from(edges)
    graphs.append(directed_graph)

# 4th Question

# Centralities
centralities = input("\nType 0 to compute graph centralities or 1 to skip")

if centralities == "0":
    # 1) Degree Centrality
    degree_centrality = []
    for i in range(0, N):
        degree_centrality.append(nx.degree_centrality(graphs[i]))
        plt.bar(range(len(degree_centrality[i])), list(degree_centrality[i].values()), align='center')
        plt.xticks(range(len(degree_centrality[i])), list(degree_centrality[i].keys()))
        plt.show()

    # 2) In-Degree Centrality
    in_degree_centrality = []
    for i in range(0, N):
        in_degree_centrality.append(nx.in_degree_centrality(graphs[i]))
        plt.bar(range(len(in_degree_centrality[i])), list(in_degree_centrality[i].values()), align='center')
        plt.xticks(range(len(in_degree_centrality[i])), list(in_degree_centrality[i].keys()))
        plt.show()

    # 3) Out-Degree Centrality
    out_degree_centrality = []
    for i in range(0, N):
        out_degree_centrality.append(nx.out_degree_centrality(graphs[i]))
        plt.bar(range(len(out_degree_centrality[i])), list(out_degree_centrality[i].values()), align='center')
        plt.xticks(range(len(out_degree_centrality[i])), list(out_degree_centrality[i].keys()))
        plt.show()

    # 4) Clossness Centrality
    closeness_centrality = []
    for i in range(0, N):
        closeness_centrality.append(nx.closeness_centrality(graphs[i]))
        plt.bar(range(len(closeness_centrality[i])), list(closeness_centrality[i].values()), align='center')
        plt.xticks(range(len(closeness_centrality[i])), list(closeness_centrality[i].keys()))
        plt.show()

    # 5) Betweenness Centrality
    betweenness_degree_centrality = []
    for i in range(0, N):
        betweenness_degree_centrality.append(nx.betweenness_centrality(graphs[i]))
        plt.bar(range(len(betweenness_degree_centrality[i])), list(betweenness_degree_centrality[i].values()), align='center')
        plt.xticks(range(len(betweenness_degree_centrality[i])), list(betweenness_degree_centrality[i].keys()))
        plt.show()

    # 6) Eigenvector Centrality
    eigenvector_degree_centrality = []
    for i in range(0, N):
        eigenvector_degree_centrality.append(nx.eigenvector_centrality_numpy(graphs[i]))
        plt.bar(range(len(eigenvector_degree_centrality[i])), list(eigenvector_degree_centrality[i].values()), align='center')
        plt.xticks(range(len(eigenvector_degree_centrality[i])), list(eigenvector_degree_centrality[i].keys()))
        plt.show()

    # 7) Katz Centrality
    katz_degree_centrality = {}
    for i in range(0, N):
        katz_degree_centrality[i] = nx.katz_centrality_numpy(graphs[i])
        plt.bar(range(len(katz_degree_centrality[i])), list(katz_degree_centrality[i].values()), align='center')
        plt.xticks(range(len(katz_degree_centrality[i])), list(katz_degree_centrality[i].keys()))
        plt.show()


# 5th Question

# Similarity Measures
similarities = input("\nType 0 to compute prediction success rates or 1 to skip")

if similarities == "0":
    # graph vars contains the neighbors
    # common_nodes contains the intersection of graphs
    # graph_pairs is a list of dictionaries of the common nodes containing the neighbors of each subgraph
    # .neighbors returns only succesors nodes not predeccesors
    # We checked the source nodes, if they got neighbors and are succesors.

    # In general, we found the neighbors for the first subgraph and the neighbors for the second subgraph only for source nodes.
    graph_pairs = []
    for i in range(0, N - 1):
        graph1 = [x for x in graphs[i].nodes if any(graphs[i].neighbors(x))]
        graph2 = [x for x in graphs[i+1].nodes if any(graphs[i+1].neighbors(x))]
        common_nodes = set(graph1).intersection(graph2)
        graph_pairs.append( dict( (source_id, [list(graphs[i].edges(source_id)), list(graphs[i+1].edges(source_id)) ]) for source_id in common_nodes) )

    # Graph creation
    # sub_graphs is a list that contains the sub graphs
    sub_graphs = []
    # we iterate from 0 to N-1 and we create a Directed Graph every iteration
    for i in range(0, N-1):
        sub_directed_graph = nx.DiGraph()
        # we iterate all the keys in tha graph_pairs dictionary from the last question
        for j in graph_pairs[i]:
            # var edges contains the first list of edges(edges that existed in the previous time period)
            edges = graph_pairs[i][j][0]
            # we add the edges in the sub_directed_graph
            sub_directed_graph.add_edges_from(edges)
        # we append the final sub_directed_graph in the sub_graphs list
        sub_graphs.append(sub_directed_graph)
    list(sub_graphs[0].nodes)

    # 6th Question

    # Graph Distance
    length_shortest_path = []
    for i in range(0, N-1):
        shortest_path_iterator = dict(nx.all_pairs_shortest_path_length(sub_graphs[i]))
        length_shortest_path.append(shortest_path_iterator)

    # Common Neighbors
    sub_common_neighbors_list = []
    for i in range(0, N-1):
        sub_common_neighbors = {u:  {v: len(set(sub_graphs[i].successors(u)).intersection(set(sub_graphs[i].successors(v)))) for v in list(sub_graphs[i].nodes) } for u in list(sub_graphs[i].nodes) }
        sub_common_neighbors_list.append(sub_common_neighbors)

    # Convert Digraph to undirected
    sub_graphs_undirected = []
    for i in range(0, N-1):
        sub_graphs_undirected_graph = nx.Graph()
        sub_graphs_undirected_graph = sub_graphs[i].to_undirected()
        sub_graphs_undirected.append(sub_graphs_undirected_graph)
    sub_graphs_undirected

    # Jaccard's Coefficient
    jaccard_list = []
    for i in range(0, N-1):
        jaccard_iterator = nx.jaccard_coefficient(sub_graphs_undirected[i])
        dummylist = []
        for j in jaccard_iterator:
            x, y, z = j
            dummylist.append((x, y ,z))
        jaccard_dictionary = { x[0]: { y[1]: y[2] for y in dummylist if y[0] == x[0] } for x in dummylist }
        jaccard_list.append(jaccard_dictionary)

    # Adamic/Adar
    adamic_list = []
    for i in range(0, N-1):
        adamic_iterator = nx.adamic_adar_index(sub_graphs_undirected[i])
        dummylist = []
        for j in adamic_iterator:
            x, y, z = j
            dummylist.append((x, y ,z))
        adamic_dictionary = { x[0]: { y[1]: y[2] for y in dummylist if y[0] == x[0] } for x in dummylist }
        adamic_list.append(adamic_dictionary)

    # Preferential Attachment
    preferential_attachment_list = []
    for i in range(0, N-1):
        preferential_attachment_iterator = nx.preferential_attachment(sub_graphs_undirected[i])
        dummylist = []
        for j in preferential_attachment_iterator:
            x, y, z = j
            dummylist.append((x, y ,z))
        preferential_attachment_dictionary = { x[0]: { y[1]: y[2] for y in dummylist if y[0] == x[0] } for x in dummylist }
        preferential_attachment_list.append(preferential_attachment_dictionary)

    # 7th Question

    #Get percentages of top similarity measures from user
    pGD = float(input("\nType percentage of top for Graph Distance"))
    pCN = float(input("\nType percentage of top for Common Neighbors"))
    pJC = float(input("\nType percentage of top for Jaccard's Coefficient"))
    pA = float(input("\nType percentage of top for Adamic Adar"))
    pPA = float(input("\nType percentage of top for Preferential Attachment"))

    # List of dictionaries with similarity measures for every subgraph
    similarity_measures_dicts = { 'pGD': length_shortest_path, 'pCN':
        sub_common_neighbors_list , 'pJC': jaccard_list, 'pA': adamic_list ,'pPA': preferential_attachment_list }
    # Percentage of top similarity measures
    similarity_measures_top = { 'pGD': pGD, 'pCN': pCN , 'pJC': pJC, 'pA': pA ,'pPA': pPA }
    # Dictionary with final results for each measure
    similarity_measures_final_results = { }
    # Each iteration add final result of similarity measure combining results
    # of pairs of graphs.
    for measure, top in similarity_measures_top.items():
        top_int = int(sub_graphs[i].size()*top)
        pair_graph_results = []
        # Calculate success rate for each pair of graphs
        for i in range(0, N-1):
            max_dict = {}
            # Find max similarity measure for each key
            for key, value in similarity_measures_dicts[measure][i].items():
                max_key = max(value.items(), key=operator.itemgetter(1))[0]
                max_dict[(key,max_key)] = value[max_key]
            # Get top% edges based on similarity measures
            max_dict = dict(sorted(max_dict.items(), key=operator.itemgetter(1), reverse=True)[:top_int])
            successes = 0
            for edge in max_dict.keys():
                if edge in [ edges[1][i] for edges in graph_pairs[i].values() for i, val in enumerate(edges[1]) ]:
                    successes += 1
            pair_graph_results.append(successes / top_int)
        similarity_measures_final_results[measure] = pair_graph_results
    print(similarity_measures_final_results)

# NA ALLAKSOUME KANA ONOMA SAN TO DUMMY LIST
# ANTE KANA SXOLIO
# FOR PRINT SIMILARITIES SAN SXOLIO
# WORD ME EKSIGISEIS NA DOSOUME TA RESTA
# NA TELIOSOUME GIA NA PAME VOLTA