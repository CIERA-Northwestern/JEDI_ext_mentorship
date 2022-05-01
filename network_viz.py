import matplotlib.pyplot as plt

import numpy as np
import networkx as nx

from mentor_matching import Person,GLOBAL_max_mentees
from network_analysis import detangle_edges, get_pods,find_planar_crossing_edges,find_overlapping_nodes

hexcols = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', 
           '#CC6677', '#882255', '#AA4499', '#661100', '#6699CC', '#AA4466',
           '#4477AA']

colors = [12, 3, 5, 6]
colors = [hexcols[color] for color in colors]

color_map = {
    'Undergraduate Student':colors[0],
    'Graduate Student':colors[1],
    'Postdoc':colors[2],
    'Faculty':colors[3] }

def iterate_random(
    graph,
    pos_dict,
    force_directed=False,
    iter_max=1000,
    only_change_one_at_a_time=True):

    i = 0 
    all_nodes = set(graph.nodes)

    ## determine how bad this setup is w.r.t. edge crossings
    crossing_edges = find_planar_crossing_edges(graph,pos_dict)
    ## nodes that participate in edge crossings
    bad_nodes = set(graph.edge_subgraph(crossing_edges).nodes)
    ## subset of bad nodes which have small number of edge crossings
    ##  or are nodes that are overlapping
    worse_nodes = pick_nodes_to_change(
        graph,
        pos_dict,
        crossing_edges,
        bad_nodes,
        only_change_one_at_a_time=only_change_one_at_a_time)

    ## nodes that will not be nudged
    fixed = all_nodes - worse_nodes

    ## try to improve the graph by nudging nodes that are not in fixed
    while len(crossing_edges) and i < iter_max:
        ## nudge using spring_layout
        if force_directed: new_pos_dict = add_force_directed(graph,pos_dict,100+i,fixed=fixed if len(fixed) else None)
        ## nudge using a gaussian for dx,dy 
        ##  centered on 0 and of width magnitude
        else: new_pos_dict = add_random(
            graph,
            pos_dict,
            i, ## seed
            fixed=fixed if len(fixed) else None,
            magnitude=2)

        ## determine how bad this setup is w.r.t. edge crossings
        new_crossing_edges = find_planar_crossing_edges(graph,pos_dict)
        new_bad_nodes = set(graph.edge_subgraph(new_crossing_edges).nodes)
        new_worse_nodes = pick_nodes_to_change(
            graph,
            new_pos_dict,
            new_crossing_edges,
            new_bad_nodes,
            only_change_one_at_a_time=only_change_one_at_a_time)
        new_fixed = all_nodes - new_worse_nodes
        i+=1

        ## we reduced the number of crossings,
        ##  accept the change
        if len(new_bad_nodes)<=len(bad_nodes):
            print(len(bad_nodes),len(new_bad_nodes))
            fixed = new_fixed
            bad_nodes = new_bad_nodes
            pos_dict = new_pos_dict
            crossing_edges = new_crossing_edges

    return pos_dict

def pick_nodes_to_change(
    graph,
    pos_dict,
    crossing_edges,
    bad_nodes,
    only_change_one_at_a_time=True):

    worse_nodes = []

    count = dict(zip(bad_nodes,np.zeros(len(bad_nodes))))

    ## count how many times each edge crosses another
    for j,edge1 in enumerate(crossing_edges[::2]):
        edge2 = crossing_edges[2*j+1]
        these_nodes = list(edge1[:2])+list(edge2[:2])
        for node in these_nodes: count[node]+=1
    
    nodes = np.array(list(count.keys()))
    counts = np.array(list(count.values()))
    nodes = nodes[counts>0]
    counts = counts[counts>0]
    nodes = nodes[np.argsort(counts)]

    if only_change_one_at_a_time and len(nodes)>0: 
        return set([nodes[0]]).union(find_overlapping_nodes(graph,pos_dict))
    else: 
        return set(nodes).union(find_overlapping_nodes(graph,pos_dict))
    

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

def add_random(graph,pos_dict,seed,fixed,magnitude=10):
    if fixed is None: fixed = []

    new_pos_dict = {**pos_dict}
    for node in list(graph.nodes): 
        if node in fixed: continue
        x,y = pos_dict[node]
        x = np.random.normal(x,magnitude)
        y = np.random.normal(y,magnitude)
        new_pos_dict[node] = [x,y]
    return new_pos_dict

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



def get_positions(
    this_network,
    simple_pos:bool=True,
    seed:int=300,
    add_missing_edges:bool=False):

    ## partition into "pods," constructed in 2 steps:
    ##  step 1: separate into 'communities' maximizing 'modularity'
    ##  step 2: reintroduce edges into community for nodes that are part of other
    ##      communities. NOTE: a node may appear in multiple axes this way!
    pods,missing_edgess = get_pods(this_network)

    pos_dicts = []

    ## draw each 'pod' in its own separate axis
    for i,(this_pod,missing_edges) in enumerate(zip(pods,missing_edgess)):

 
        nodes = list(this_pod.nodes)
        edges = list(this_pod.edges.keys())
        if add_missing_edges: this_pod.add_edges_from(missing_edges)
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
        else: pos_dict = nx.shell_layout(this_pod)
            ## position the nodes in 2d space. kamada_kawai minimizes edge length and 
            ##  force_directed (spring) indirectly minimizes edge crossings 
            ##  (it's an NP hard problem apparently this is the best one can do).

        #try: pos_dict = nx.planar_layout(this_pod,scale=0.1)
        #except: pos_dict = iterate_force_directed(this_pod,pos_dict)
        pos_dict = detangle_edges(this_pod,pos_dict)#iterate_random(this_pod,pos_dict,force_directed=False)
        pos_dicts += [pos_dict]

    return pods,pos_dicts

def draw_network(
    this_network,
    pods = None,
    pos_dicts = None,
    scale_fact:float=1,
    debug_crossing_edges:bool=False,
    single_axis=True,
    **kwargs):

    ## partition into "pods," constructed in 2 steps:
    ##  step 1: separate into 'communities' maximizing 'modularity'
    if pods is None or pos_dicts is None: pods,pos_dicts = get_positions(this_network,**kwargs)

    if not single_axis:
        ## initialize matplotlib axes
        fig,axs = plt.subplots(nrows=len(pods)//4+(len(pods)%4>0),ncols=4)
        axs = np.array(axs)
        offsets = np.zeros((len(pods),2))
    else:
        fig,axs = plt.subplots(1,1)
        axs = np.array([axs]*len(pods)).flatten()
        offsets = np.zeros((len(pods),2))
        thetas = np.linspace(0,2*np.pi,len(pods),endpoint=False)
        offsets[:,0] = np.cos(thetas)*scale_fact*3
        offsets[:,1] = np.sin(thetas)*scale_fact*3

    ## draw each 'pod' in its own separate axis
    for i,(ax,this_pod,pos_dict,offset) in enumerate(zip(axs.flatten(),pods,pos_dicts,offsets)):
        nodes = list(this_pod.nodes)
        edges = list(this_pod.edges.keys())
        anti_nodes = [node for node in this_pod.nodes if node not in nodes]
        copy_dict = {**pos_dict}
        for key,value in copy_dict.items(): copy_dict[key] = value+offset

        ## draw each component of the graph separately
        ##  nodes
        for shape,llist in zip(['o','*'],[nodes,anti_nodes]):
            nx.draw_networkx_nodes(
                this_pod,
                copy_dict,
                ax=ax,
                node_shape=shape,
                node_color=[color_map[node.role] for node in llist],
                nodelist=llist)

        ##  labels
        nx.draw_networkx_labels(
            this_pod,
            copy_dict,
            labels=dict([(node,f"{node.initials}") for node in this_pod.nodes]),
            ax=ax)

        missing_edges = []
        ##  edges, arrows point from mentor -> mentee
        for style,llist in zip(['-','--'],[edges,missing_edges]):
            try: nx.draw_networkx_edges(
                this_pod,
                copy_dict,
                ax=ax,
                edge_color=get_edge_colors(llist),
                style=style,
                width=2,
                edgelist=llist)
            except: pass

        if debug_crossing_edges:
            crossing_edges = find_planar_crossing_edges(this_pod,copy_dict)
            if len(crossing_edges):
                nx.draw_networkx_edges(
                    this_pod,
                    copy_dict,
                    ax=ax,
                    edge_color='red',
                    style='-',
                    width=2,
                    edgelist=crossing_edges)

        ## annotate any remaining mentor (o) or mentee (x) spots
        dx = np.diff(ax.get_xlim())[0]
        dy = np.diff(ax.get_ylim())[0]
        dr = np.sqrt(dx**2+dy**2)
        draw_remaining_spots(ax,nodes,copy_dict,dr=dr/30)

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
        
    if not single_axis:
        fig.set_facecolor('white')
        fig.set_size_inches(2*len(pods)*scale_fact,6*scale_fact)
        fig.set_dpi(120)
    else:
        fig.set_facecolor('white')
        fig.set_size_inches(2*len(pods)*scale_fact,2*len(pods)*scale_fact)
        fig.set_dpi(120)
    
    add_legend_to_ax(axs[-1],debug=show_remaining_spots,between=between)

    return pods,pos_dicts

def add_legend_to_ax(ax,debug=True,between=False):

    if debug:
        add_to_legend(ax,'willing to mentor',c='k',lw=0,labelcolor='red')
        add_to_legend(ax,'requesting mentor',c='k',lw=0,labelcolor='blue')
    for role,rank in role_ranks.items():
        add_to_legend(
            ax,
            role,
            c=color_map[role],
            line_kwargs={'marker':'.','markersize':25},
            lw=0) 

    if not between:
        add_to_legend(
            ax,
            'appears elsewhere',
            c='k',
            line_kwargs={'marker':'*','markersize':15},
            lw=0)

    for label,edge_color in zip(
        ['both prefer','mentee prefers','mentor prefers'],
        ['gold','lightgreen','darkgreen']):
        add_to_legend(
            ax,
            label,
            c=edge_color,
            labelspacing=1.2,
            frameon=False,
            labelcolor=debug*['red','blue']+['k']*8,
            loc='center' if not between else 'upper left')

def add_to_legend(
    ax,
    label='',
    shape='line',
    loc=0,
    ls='-',
    c=None,
    alpha=1,
    lw=None,
    legend_kwargs=None,
    make_new_legend=False,
    line_kwargs=None,
    **kwargs):

    from matplotlib.lines import Line2D


    legend = ax.get_legend()
    ## add the current legend to the tracked artists
    ##  and then pretend we weren't passed one
    prev_loc = None
    if make_new_legend and legend is not None:
        prev_loc = legend._loc
        ax.add_artist(legend)
        legend=None

    if legend is not None:
        lines = legend.get_lines()
        labels = [text.get_text() for text in legend.get_texts()]
    else:
        lines,labels=[],[]

    if legend_kwargs is None:
        legend_kwargs = {}

    ## make the new line
    if shape == 'line':
        ## this is wild, but have to handle when plotting
        ##  markers, apparently when you read the line from 
        ##  the legend it loses memory of the marker
        if 'ls' in kwargs and kwargs['ls'] == '':
            ax.markers = []
        if line_kwargs is None: line_kwargs = {}

        if lw is not None: line_kwargs['lw'] = lw
        else: line_kwargs['lw'] = 2

        line = Line2D(
        [0],[0],
        ls=ls,
        c=c,
        alpha=alpha,
        **line_kwargs
        )
    else:
        raise NotImplementedError

    if label not in labels:
        lines.append(line)
        labels.append(label)

    if loc in legend_kwargs:
        loc = legend_kwargs.pop('loc')

    for line in lines:
        if line.get_linestyle() == 'None':
            print(line.get_marker(),'marker')

    if prev_loc is not None and loc == prev_loc:
        loc+=1

    ## for backwards compatibility...
    legend_kwargs.update(kwargs)
    ax.legend(lines,labels,loc=loc,**legend_kwargs)

    return ax