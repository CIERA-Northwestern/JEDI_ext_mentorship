import numpy as np
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


def find_planar_crossing_edges(graph,pos_dict):
    ## could be useful: 
    #https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.boundary.edge_boundary.html#networkx.algorithms.boundary.edge_boundary

    crosses = []
    edges = list(graph.edges.keys())
    for i,edge1 in enumerate(edges):
        for edge2 in edges[i+1:]:
            if edge1 == edge2: continue
            if determine_if_cross(edge1,edge2,pos_dict):
                crosses += [edge1,edge2]

    return crosses

def determine_if_cross(edge1,edge2,pos_dict):
    # y-y0 = m(x-x0) + b
    node1a = edge1[0]
    node1b = edge1[1]
    x1a,y1a = pos_dict[node1a]
    x1b,y1b = pos_dict[node1b]

    node2a = edge2[0]
    node2b = edge2[1]
    x2a,y2a = pos_dict[node2a]
    x2b,y2b = pos_dict[node2b]


    xs = [x1a,x1b,x2a,x2b]
    ys = [y1a,y1b,y2a,y2b]
    xmin,xmax = np.min(xs),np.max(xs)
    ymin,ymax = np.min(ys),np.max(ys)

    if np.isclose(x1a,x1b) and np.isclose(x2a,x2b): 
        ## two vertical lines that overlap
        if np.isclose(x1a,x2b): return True
        ## two vertical lines that do not overlap
        else: return False
    
    m1 = (y1b-y1a)/(x1b-x1a)
    m2 = (y2b-y2a)/(x2b-x2a)
    if np.isinf(m1): m1 = 0
    if np.isinf(m2): m2 = 0

    ## parallel lines (but not vertical)
    if np.isclose(m1,m2): return False

    # y - y1a = m1(x-x1a)
    # y - y2a = m2(x-x2a)

    # y - m1x = -m1x1a + y1a
    # y - m2x = -m2x2a + y2a

    #[-m1 , 1][x] = [-m1x1a + y1a] -> [x] = ([-m1 , 1])^-1 [-m1x1a + y1a]
    #[-m2 , 1][y] = [-m2x2a + y2a] -> [y] = ([-m2 , 1])    [-m2x2a + y2a]
    for x1,y1 in zip([x1a,x1b],[y1a,y1b]):
        for x2,y2 in zip([x2a,x2b],[y2a,y2b]):
            if np.isclose(x1,x2) and np.isclose(y1,y2): return False 
            vec = np.array([-m1*x1+y1,-m2*x2+y2]).reshape(2,1)

    mat = np.array([[-m1,1],[-m2,1]])
    try:
        xcross,ycross = np.matmul(np.linalg.inv(mat),vec)[:,0]
    except:
        print(mat,vec)
        import pdb; pdb.set_trace()

    cross = True
    for x1,y1 in zip([x1a,x2a],[y1a,y2a]):
        for x2,y2 in zip([x1b,x2b],[y1b,y2b]):
            xbad = (x1<xcross<x2 or x1>xcross>x2)
            ybad = (y1<ycross<y2 or y1>ycross>y2)
            #if np.isclose(x1,xcross) or np.isclose(x2,xcross): return False
            #if np.isclose(y1,ycross) or np.isclose(y2,ycross): return False
            cross = (cross and xbad)# and ybad)
    return cross