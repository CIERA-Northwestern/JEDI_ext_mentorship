import matplotlib.pyplot as plt

import numpy as np
import networkx as nx

from mentor_matching import Person,GLOBAL_max_mentees
from network_analysis import get_pods

color_map = {
    'Undergrads':'blue',
    'GradStudents':'green',
    'Postdocs':'purple',
    'Faculty':'orange' }

def add_force_directed(graph,pos_dict:dict,seed:int=100):

    pos_dict = nx.spring_layout(
        graph,
        k=10, ## target distance between nodes
        weight=1, ## attraction between nodes
        pos=pos_dict, ## initial position
        iterations=1000,
        seed=seed)

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
        remaining_spots += 'o'*int(node.mentees_remaining)
        remaining_spots += 'x'*int(node.mentors_remaining)

        thetas = np.linspace(0,2*np.pi,len(remaining_spots),endpoint=False)+np.pi/2

        dxs = np.cos(thetas)*dr
        dys = np.sin(thetas)*dr

        for dx,dy,this_char in zip(dxs,dys,remaining_spots):
            ax.text(x+dx,y+dy,
                this_char,
                verticalalignment='center',
                horizontalalignment='center',
                c='r')

def draw_network(
    this_network,
    simple_pos:bool=False,
    scale_fact:float=1,
    seed:int=300):

    ## partition into "pods," constructed in 2 steps:
    ##  step 1: separate into 'communities' maximizing 'modularity'
    ##  step 2: reintroduce edges into community for nodes that are part of other
    ##      communities. NOTE: a node may appear in multiple axes this way!
    pods = get_pods(this_network)


    ## initialize matplotlib axes
    fig,axs = plt.subplots(nrows=2,ncols=len(pods)//2+len(pods)%2)
    axs = np.array(axs)

    ## draw each 'pod' in its own separate axis
    for ax,this_pod in zip(axs.flatten(),pods):
 
        nodes = list(this_pod.nodes)
        edges = list(this_pod.edges.keys())
        print(nodes)
    
        if simple_pos:
            ## manually position each node according to their rank
            pos_dict = {}
            ## start offset so that you don't end up with a faculty
            ##  member mentoring an undergrad and you can't tell
            ##  whether the arrow starts at a postdoc or grad, for example.
            role_counts = np.zeros(4) + [0,1,1,0]
            for node in list(this_pod.nodes):
                role_counts[node.rank]+=1
                pos_dict[node] = [node.rank,role_counts[node.rank]]

            max_counts = np.max(role_counts)
            for node in list(this_pod.nodes): pos_dict[node][0]*=max_counts/role_counts.shape[0]
        else:
            ## position the nodes in 2d space. kamada_kawai minimizes edge length and 
            ##  force_directed (spring) indirectly minimizes edge crossings 
            ##  (it's an NP hard problem apparently this is the best one can do).
            pos_dict = add_force_directed(
                this_pod,
                nx.kamada_kawai_layout(this_pod),
                seed=seed)

        ## draw each component of the graph separately
        ##  nodes
        nx.draw_networkx_nodes(
            this_pod,
            pos_dict,
            ax=ax,
            node_color=[color_map[node.role] for node in nodes])
        ##  labels
        nx.draw_networkx_labels(
            this_pod,
            pos_dict,
            ax=ax)

        ##  edges, arrows point from mentor -> mentee
        nx.draw_networkx_edges(
            this_pod,
            pos_dict,
            ax=ax,
            edge_color=get_edge_colors(edges),
            width=2)

        ## annotate any remaining mentor (o) or mentee (x) spots
        dx = np.diff(ax.get_xlim())[0]
        dy = np.diff(ax.get_ylim())[0]
        dr = np.sqrt(dx**2+dy**2)
        draw_remaining_spots(ax,nodes,pos_dict,dr=dr/30)

        ax.axis('off')
        ax.set_aspect(1)
    
    for ax in axs.flatten(): ax.axis('off')
                    
    ## format the figure to minimize whitespace
    fig.subplots_adjust(wspace=0,hspace=0,left=0,right=1,bottom=0,top=1)
        
    fig.set_facecolor('white')
    fig.set_size_inches(2*len(pods)*scale_fact,6*scale_fact)
    fig.set_dpi(120)
    return fig