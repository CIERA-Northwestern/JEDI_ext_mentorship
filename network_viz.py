import matplotlib.pyplot as plt

import numpy as np
import networkx as nx

from mentor_matching import Person,GLOBAL_max_mentees
from network_analysis import get_pods,find_planar_crossing_edges

hexcols = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', 
           '#CC6677', '#882255', '#AA4499', '#661100', '#6699CC', '#AA4466',
           '#4477AA']

colors = [12, 3, 5, 6]
colors = [hexcols[color] for color in colors]

color_map = {
    'Undergrads':colors[0],
    'GradStudents':colors[1],
    'Postdocs':colors[2],
    'Faculty':colors[3] }

def iterate_force_directed(graph,pos_dict):

    i = 0 
    fixed = find_good_nodes(graph,pos_dict)
    while len(fixed) != len(graph.nodes) and i < 50:
        new_pos_dict = add_force_directed(graph,pos_dict,50+i,fixed=fixed if len(fixed) else None)
        new_fixed = find_good_nodes(graph,pos_dict)
        i+=1
        if len(new_fixed) < len(fixed): continue
        fixed = new_fixed
        pos_dict = new_pos_dict
    return pos_dict

def find_good_nodes(graph,pos_dict):
    exclude_nodes = []
    crossing_edges = find_planar_crossing_edges(graph,pos_dict)
    bad_nodes = list(graph.edge_subgraph(crossing_edges).nodes)

    count = dict(zip(bad_nodes,np.zeros(len(bad_nodes))))

    for j,edge1 in enumerate(crossing_edges[::2]):
        edge2 = crossing_edges[2*j+1]
        these_nodes = list(edge1[:2])+list(edge2[:2])
        for node in these_nodes:
            count[node]+=1

    for j,edge1 in enumerate(crossing_edges[::2]):
        edge2 = crossing_edges[2*j+1]
        these_nodes = np.array(list(edge1[:2])+list(edge2[:2]))
        this_count = np.zeros(4)
        for i,node in enumerate(these_nodes): this_count[i] = count[node]

        these_nodes = these_nodes[np.argsort(this_count)[::-1]]
        for node in these_nodes: 
            if node not in exclude_nodes: 
                exclude_nodes.append(node)
                break

    fixed = [node for node in graph.nodes if node not in exclude_nodes]
    return fixed

def add_force_directed(graph,pos_dict:dict,seed:int=100,fixed=None):

    pos_dict = nx.spring_layout(
        graph,
        k=.01, ## target distance between nodes
        weight=100, ## attraction between nodes
        pos=pos_dict, ## initial position
        iterations=1000,
        seed=seed,
        fixed=fixed)

    return pos_dict

def get_edge_colors(edges):
    colors = []
    ## loop over every edge and assign it a color
    ##  based on how "preferred" the relationship is,
    ##  gold is the best!
    for (mentor,mentee,_) in edges:
        mentor:Person = mentor ## for typehinting
        mentee:Person = mentee ## for typehinting

        if (mentor.name in mentee.mentors_prefr and 
            mentee.name in mentor.mentees_prefr): color = 'gold'
        elif mentor.name in mentee.mentors_prefr: color = 'lightgreen'
        elif  mentee.name in mentor.mentees_prefr: color = 'darkgreen'
        else: color = 'k'

        ## add this color to the list
        colors += [color]

    return colors

def draw_remaining_spots(ax,nodes,pos_dict,dr=0.2):
    for node in nodes:
        node:Person = node ## for typehinting
        x,y = pos_dict[node]

        remaining_spots = ''
        flags = []
        mentors_remaining = node.n_role_mentors-node.has_n_role_mentors 
        mentees_remaining = node.n_role_mentees-node.has_n_role_mentees 
        ## didn't offer to take more than the global maximum
        if np.sum(mentors_remaining) == node.mentors_remaining:
            for char,n in zip('ugpf',mentors_remaining):
                if n <= 0: continue
                remaining_spots+=char*int(n)
                flags = np.append(flags,np.repeat(0,n))
        else: pass 
        for char,n in zip('ugpf',mentees_remaining):
            if n <= 0: continue
            remaining_spots+=char*int(n)
            flags = np.append(flags,np.repeat(1,n))

        thetas = np.linspace(0,2*np.pi,len(remaining_spots),endpoint=False)+np.pi/2

        dxs = np.cos(thetas)*dr
        dys = np.sin(thetas)*dr

        for dx,dy,this_char,flag in zip(dxs,dys,remaining_spots,flags):
            ax.text(x+dx,y+dy,
                this_char,
                verticalalignment='center',
                horizontalalignment='center',
                c='magenta' if not flag else 'red',
                fontsize=12)



def draw_network(
    this_network,
    simple_pos:bool=False,
    scale_fact:float=1,
    seed:int=300):

    ## partition into "pods," constructed in 2 steps:
    ##  step 1: separate into 'communities' maximizing 'modularity'
    ##  step 2: reintroduce edges into community for nodes that are part of other
    ##      communities. NOTE: a node may appear in multiple axes this way!
    pods,missing_edgess = get_pods(this_network)


    ## initialize matplotlib axes
    fig,axs = plt.subplots(nrows=len(pods)//4+(len(pods)%4>0),ncols=4)
    axs = np.array(axs)

    ## draw each 'pod' in its own separate axis
    for ax,this_pod,missing_edges in zip(axs.flatten(),pods,missing_edgess):
 
        nodes = list(this_pod.nodes)
        edges = list(this_pod.edges.keys())
        print(nodes)
        this_pod.add_edges_from(missing_edges)
        anti_nodes = [node for node in this_pod.nodes if node not in nodes]
    
        if simple_pos:
            ## manually position each node according to their rank
            pos_dict = {}
            ## start offset so that you don't end up with a faculty
            ##  member mentoring an undergrad and you can't tell
            ##  whether the arrow starts at a postdoc or grad, for example.
            np.random.seed(seed)
            role_counts = np.zeros(4) + [0,np.random.random(),np.random.random(),0]
            for node in list(this_pod.nodes):
                role_counts[node.rank]+=1
                pos_dict[node] = [node.rank,role_counts[node.rank]]

            max_counts = np.max(role_counts)
            for node in list(this_pod.nodes): pos_dict[node][0]*=max_counts/role_counts.shape[0]
        else: pos_dict = nx.kamada_kawai_layout(this_pod)
            ## position the nodes in 2d space. kamada_kawai minimizes edge length and 
            ##  force_directed (spring) indirectly minimizes edge crossings 
            ##  (it's an NP hard problem apparently this is the best one can do).

        try: pos_dict = nx.planar_layout(this_pod,scale=0.1)
        except: pos_dict = iterate_force_directed(this_pod,pos_dict)

        ## draw each component of the graph separately
        ##  nodes
        for shape,llist in zip(['o','*'],[nodes,anti_nodes]):
            nx.draw_networkx_nodes(
                this_pod,
                pos_dict,
                ax=ax,
                node_shape=shape,
                node_color=[color_map[node.role] for node in llist],
                nodelist=llist)

        ##  labels
        nx.draw_networkx_labels(
            this_pod,
            pos_dict,
            labels=dict([(node,f"{node.initials}") for node in this_pod.nodes]),
            ax=ax)

        ##  edges, arrows point from mentor -> mentee
        for style,llist in zip(['-','--'],[edges,missing_edges]):
            nx.draw_networkx_edges(
                this_pod,
                pos_dict,
                ax=ax,
                edge_color=get_edge_colors(llist),
                style=style,
                width=2,
                edgelist=llist)

        crossing_edges = find_planar_crossing_edges(this_pod,pos_dict)
        if len(crossing_edges):
            nx.draw_networkx_edges(
                this_pod,
                pos_dict,
                ax=ax,
                edge_color='red',
                style='-',
                width=2,
                edgelist=crossing_edges)

        ## annotate any remaining mentor (o) or mentee (x) spots
        dx = np.diff(ax.get_xlim())[0]
        dy = np.diff(ax.get_ylim())[0]
        dr = np.sqrt(dx**2+dy**2)
        draw_remaining_spots(ax,nodes,pos_dict,dr=dr/30)

        ax.axis('off')
        #ax.set_aspect(1)
        #ax.set_xlim(left=-0.5)
    
    for ax in axs.flatten(): 
        ax.axis('off')
        if ax.is_last_col() and ax.is_last_row():
            ax.text(0.5,1,'willing to mentor',c='red',transform=ax.transAxes,ha='center',va='center',fontsize=12)
            ax.text(0.5,.95,'requesting mentor',c='magenta',transform=ax.transAxes,ha='center',va='center',fontsize=12)
                    
    ## format the figure to minimize whitespace
    fig.subplots_adjust(wspace=0,hspace=0,left=0,right=1,bottom=0,top=1)
        
    fig.set_facecolor('white')
    fig.set_size_inches(2*len(pods)*scale_fact,6*scale_fact)
    fig.set_dpi(120)
    return fig