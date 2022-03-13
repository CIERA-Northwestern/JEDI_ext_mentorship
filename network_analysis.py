import networkx.algorithms.community as nx_comm

def get_pods(this_network):
    """ partition into "pods," constructed in 2 steps:
        step 1: separate into 'communities' maximizing 'modularity'
        step 2: reintroduce edges into community for nodes that are part of other
        communities. NOTE: a node may appear in multiple axes this way! """

    pods = []
    missing_edgess = []

    ## used to figure out which edges are missing in a community
    all_edges = set(this_network.edges.keys())

    ## split into communities, maximizing modularity (apparently)
    comms = nx_comm.greedy_modularity_communities(this_network,resolution=1)

    for comm in comms:
        ## make a copy so we can edit the graph
        this_comm = this_network.subgraph(comm).copy()

        ## find nodes which have edges that either start or end
        ##  in this community but are not part of the community
        missing_edges = find_missing_edges(
            all_edges,
            list(this_comm.nodes),
            list(this_comm.edges.keys()))
        print(this_comm,'but missing',len(missing_edges),'edges')

        ## reintroduce edges to community to form a 'pod'
        #this_comm.add_edges_from(new_missing_edges)
        pods += [this_comm]
        missing_edgess += [missing_edges]

    return pods,missing_edgess

def find_missing_edges(all_edges,sub_nodes,sub_edges):
    """ Find edges that exist in the graph but not 
        in the sub-graph defined by nodes and edges 
        (i.e. edges that start or end in the sub-graph
        but /not/ both)."""
    expanded_edges = set([
        edge for edge in all_edges if 
        (edge[0] in sub_nodes or edge[1] in sub_nodes)])
    sub_edges = set(sub_edges)

    ## take set difference
    return list(expanded_edges-sub_edges)
