import numpy as np
import networkx.algorithms.community as nx_comm

class Point(object):
    def __repr__(self):
        return f"({self.x:.2f},{self.y:.2f})"
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def contained_in_domain(self,p_other1,p_other2):
        ## swap so other1 is left-most
        if p_other2.x < p_other1.x: p_other1,p_other2 = p_other2,p_other1

        return p_other1.x-1e-3 < self.x < p_other2.x+1e-3
    
    def contained_in_range(self,p_other1,p_other2):
        if p_other2.y < p_other1.y: p_other1,p_other2 = p_other2,p_other1

        return p_other1.y-1e-3 < self.y < p_other2.y+1e-3
    
    def share_x(self,other): return np.isclose(self.x,other.x)
    def share_y(self,other): return np.isclose(self.y,other.y)

    def on_line(self,p_other1,p_other2):
        ## swap so other1 is left-most
        if p_other2.x < p_other1.x: p_other1,p_other2 = p_other2,p_other1
        m = (p_other2.y - p_other1.y)/(p_other2.x-p_other1.x)

        return np.isclose((self.x-p_other1.x)*m+p_other1.y,self.y)
    
    def distance(self,other):
        return np.sqrt((self.x-other.x)**2+(self.y-other.y)**2)

def get_pods(this_network, resolution = 1, loud = False):
    """ partition into "pods," constructed in 2 steps:
        step 1: separate into 'communities' maximizing 'modularity'
        step 2: reintroduce edges into community for nodes that are part of other
        communities. NOTE: a node may appear in multiple axes this way! """

    pods = []
    missing_edgess = []

    ## used to figure out which edges are missing in a community
    all_edges = set(this_network.edges.keys())

    ## split into communities, maximizing modularity (apparently)
    comms = nx_comm.greedy_modularity_communities(this_network,resolution=resolution)

    for comm in comms:
        ## make a copy so we can edit the graph
        this_comm = this_network.subgraph(comm).copy()

        ## find nodes which have edges that either start or end
        ##  in this community but are not part of the community
        missing_edges = find_missing_edges(
            all_edges,
            list(this_comm.nodes),
            list(this_comm.edges.keys()))
        if (loud):
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

def detangle_edges(graph,pos_dict):

    keep_going = True
    crossing_edges = find_planar_crossing_edges(graph,pos_dict)
    all_nodes = list(graph.nodes)

    for i,node1 in enumerate(all_nodes):
        for node2 in all_nodes[i+1:]:
            new_pos_dict = {**pos_dict}
            new_pos_dict[node1],new_pos_dict[node2] = pos_dict[node2],pos_dict[node1]
            new_crossing_edges = find_planar_crossing_edges(graph,new_pos_dict)
            if len(new_crossing_edges) < len(crossing_edges): return detangle_edges(graph,new_pos_dict)

    return pos_dict

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

def find_overlapping_nodes(graph,pos_dict,thresh=0.5):
    items = list(pos_dict.items())
    bad_nodes = []
    for i,(node1,pos1) in enumerate(items):
        for node2,pos2 in items[i:]:
            p1 = Point(*pos1)
            p2 = Point(*pos2)
            if p1.distance(p2) < thresh: bad_nodes += [node1,node2]
    return set(bad_nodes)


def determine_if_cross(edge1,edge2,pos_dict):
    # y-y0 = m(x-x0) + b
    node1a = edge1[0]
    node1b = edge1[1]

    node2a = edge2[0]
    node2b = edge2[1]

    p1a = Point(*pos_dict[node1a])
    p1b = Point(*pos_dict[node1b])
    if p1a.x > p1b.x: p1a,p1b = p1b,p1a

    p2a = Point(*pos_dict[node2a])
    p2b = Point(*pos_dict[node2b])
    if p2a.x > p2b.x: p2a,p2b = p2b,p2a

    m1 = (p1b.y-p1a.y)/(p1b.x-p1a.x)
    m2 = (p2b.y-p2a.y)/(p2b.x-p2a.x)

    # y - y1a = m1(x-x1a)
    # y - y2a = m2(x-x2a)

    # y - m1x = -m1x1a + y1a
    # y - m2x = -m2x2a + y2a

    #[-m1 , 1][x] = [-m1x1a + y1a] -> [x] = ([-m1 , 1])^-1 [-m1x1a + y1a]
    #[-m2 , 1][y] = [-m2x2a + y2a] -> [y] = ([-m2 , 1])    [-m2x2a + y2a]

    vert_1 = p1a.share_x(p1b)
    vert_2 = p2a.share_x(p2b)

    ## parallel lines don't cross unless they are co-linear
    if m1 == m2 and not (vert_1 and vert_2): return(
        p1a.on_line(p2a,p2b) or
        p1b.on_line(p2a,p2b) or
        p2a.on_line(p1a,p1b) or
        p2b.on_line(p1a,p1b))
    
    node_set = set([node1a,node1b,node2a,node2b])
    if len(node_set) < 4 and not (vert_1 and vert_2): return False

    if vert_1 and vert_2 and p1a.share_x(p2a):
        return (
        p1a.contained_in_range(p2a,p2b) or 
        p1b.contained_in_range(p2a,p2b) or 
        p2a.contained_in_range(p1a,p1b) or 
        p2b.contained_in_range(p1a,p1b))
    elif vert_1 and p1a.contained_in_domain(p2a,p2b):
        pcross = Point(p1a.x,m2*(p1a.x-p2a.x)+p2a.y)
        return pcross.contained_in_range(p1a,p1b)
    elif vert_2 and p2a.contained_in_domain(p1a,p1b):
        pcross = Point(p2a.x,m1*(p2a.x-p1a.x)+p1a.y)
        return pcross.contained_in_range(p2a,p2b)

    ## choose this combination of points to make a vector of point slope form
    vec = np.array([-m1*p1a.x+p1a.y,-m2*p2a.x+p2a.y]).reshape(2,1)
    mat = np.array([[-m1,1],[-m2,1]])
    xcross,ycross = np.matmul(np.linalg.pinv(mat),vec)[:,0]
    return ( Point(xcross,ycross).contained_in_domain(p1a,p1b) and
        Point(xcross,ycross).contained_in_domain(p2a,p2b) and
        Point(xcross,ycross).contained_in_range(p1a,p1b) and
        Point(xcross,ycross).contained_in_range(p2a,p2b))