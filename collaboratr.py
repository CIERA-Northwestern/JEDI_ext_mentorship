from __future__ import division

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from operator import truediv
#from nxpd import draw
from collections import Counter
import itertools
import random
from jinja2 import Template

def insert_node(G,name, email, learn, teach):
    G.add_node(name, email=email, learn=learn, teach=teach)
    
    for node in G.nodes(data=True):
        learn_other = node[1]['learn']
        teach_other = node[1]['teach']
        
        common = [l for l in learn if l in teach_other]
        w = len(common)
        G.add_edge(name, node[0], weight=w, common=common)
        
        common = [l for l in teach if l in learn_other]
        w = len(common)
        G.add_edge(node[0], name, weight=w, common=common)
        
def find_breakouts(users, teach, learn, BREAKOUT_REQ_RATIO, BREAKOUT_REQ_FRACTION):
    """
    Find subjects which deserve breakouts depending 
    on student to teach ratios.
    """
    big_list_teach = []
    for i in range(len(teach)):
        big_list_teach.append(teach[i].split(';'))

    big_list_learn = []
    for i in range(len(learn)):
        big_list_learn.append(learn[i].split(';'))
        
    teach_dict = dict(Counter(list((itertools.chain.from_iterable(big_list_teach)))))
    learn_dict = dict(Counter(list((itertools.chain.from_iterable(big_list_learn)))))
    
    common_names = [ n for n in set(teach_dict).intersection(set(learn_dict)) ]

    teach_dict = {k:teach_dict[k] for k in common_names if k in teach_dict} # Python3
    learn_dict = {k:learn_dict[k] for k in common_names if k in learn_dict} # Python3
    
    N_teach = list(teach_dict.values())
    N_learn = list(learn_dict.values())
    
    ratios =  list(map(truediv,N_learn,N_teach))

    N = len(users)
    subjects = list(teach_dict.keys())
    isBreakout = (np.array(ratios) > BREAKOUT_REQ_RATIO) & (np.array(N_learn) > BREAKOUT_REQ_FRACTION*N)
    breakouts = np.array(subjects)[list(np.where(isBreakout)[0])]
    
    return list(breakouts)

def assign_users(G,participants):
    """
    This uses LOTS of nested loops. Work to be done here.
    """
    assign = {}
    i=0
    for p in participants:
        assign_p = {}
        for l in p[1]['learn']:
            edges = G.edges(p[0], data=True)
            common = []
                
            for e in edges:
                if (l in e[2]['common']) and (l!=''):
                    common.append(e[1])
                else: 
                    pass

            if common != []:
                assign_p[l] = common
                
        assign[p[0]] = assign_p
        i+=1
        
    return assign

def drawIndividualGraphs(participants,connections):
    index=1
    for p in participants:
        con = [c for c in connections if c[0] == p]
        links = [c[1] for c in con]
        links.append(p)

        H = nx.DiGraph()
        H.add_nodes_from(links, style='filled', fillcolor='black', 
                             fontcolor='white', fontname='sans-serif', fontsize=24)

        for c in con:
            H.add_edge(c[0], c[1], label=c[2]['label'], style='dotted', fontcolor='red', fontsize=12,
                              fontname='sans-serif')

        draw(H, filename='Resources/'+str(index)+'.svg')
        index+=1
    return index

def createWebPage(obj):
    # Define jinja2 template
    TEMPLATE1 = Template("""<html><head><link rel="stylesheet" href="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">

    <!-- jQuery library -->
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.12.4/jquery.min.js"></script>

    <!-- Latest compiled JavaScript -->
    <script src="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script></head>
    <body>
    <nav class="navbar navbar-default">
      <div class="container-fluid">
        <div class="navbar-header">
          <a class="navbar-brand" href="#">Collaboratr</a>
        </div>
        <ul class="nav navbar-nav">
          <li class="active"><a href="#">Home</a></li>
          <li><a href="#0">Overall</a></li>
          {% for p in obj.participants%}
              <li><a href="#{{loop.index}}">{{p}}</a></li> 
          {% endfor %}
        </ul>
      </div>
    </nav>

    {% for i in range(0,obj.index)%}<a name="{{i}}"><br><br><br><br><br><br><center><img src="Resources/{{i}}.svg" align="middle"></center>{% endfor %}<script src="jquery/jquery-2.0.3.min.js"></script>
    </body>
    </html>""")
    return TEMPLATE1

def createBreakoutPage(participants,breakouts):
    objs = []
    for b in breakouts:
        learn = []
        teach = []

        for p in participants:
            if b in p[1]['learn']:
                learn.append(p[0].split(' ')[0])

            if b in p[1]['teach']:
                teach.append(p[0].split(' ')[0])


        learn = ', '.join(learn)
        teach = ', '.join(teach)
        o = {}
        o['learn'] = learn
        o['teach'] = teach
        o['name'] = b

        objs.append(o)
        
        
    # Define jinja2 template
    TEMPLATE2 = Template("""<html><head><head><link rel="stylesheet" href="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">

    <!-- jQuery library -->
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.12.4/jquery.min.js"></script>

    <!-- Latest compiled JavaScript -->
    <script src="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script></head>
    <div class="container">
      <h2>Breakouts Table</h2>
      <table class="table">
        <thead>
          <tr>
            <th>Subjects</th>
            <th>Learners</th>
            <th>Teachers</th>
          </tr>
        </thead>
        <tbody>
        {% for i in objs%}
          <tr>
            <td>{{i.name}}</td>
            <td>{{i.learn}}</td>
            <td>{{i.teach}}</td>
          </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>
    <script src="jquery/jquery-2.0.3.min.js"></script>
    </body>
    </html>""")
    return TEMPLATE2
