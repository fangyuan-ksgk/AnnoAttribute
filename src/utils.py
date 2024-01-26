# StarChars drawer function
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict

from plotly.subplots import make_subplots

def draw_two_star_charts(data1: dict, data2: dict, title1: str, title2: str, save_path: str = None):
    df1 = pd.DataFrame(data1.items(), columns=['Bias', 'Score'])
    df2 = pd.DataFrame(data2.items(), columns=['Bias', 'Score'])

    # Create a subplot with 1 row and 2 columns
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'polar'}, {'type': 'polar'}]],
                        subplot_titles=(title1, title2))

    # First Star Chart
    fig.add_trace(go.Scatterpolar(r=df1['Score'], theta=df1['Bias'], fill='toself', name=title1), row=1, col=1)

    # Second Star Chart
    fig.add_trace(go.Scatterpolar(r=df2['Score'], theta=df2['Bias'], fill='toself', name=title2), row=1, col=2)

    # Update layout for both plots
    fig.update_layout(
        title_text='POE (Pairwise Objective Evaluation)',
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        polar2=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=False
    )

    fig.show()

    if save_path is not None:
        fig.write_image(save_path)

def draw_star_chars(data: dict, title: str):
    df = pd.DataFrame(data.items(), columns=['Bias', 'Score'])
    fig = px.line_polar(df, r='Score', theta='Bias', line_close=True)
    fig.update_traces(fill='toself')
    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False
    )
    fig.show()

import networkx as nx
import random
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import json

# Hierarchical layout util function
def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike 
    
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    G: the graph (must be a tree)
    
    root: the root node of current branch 
    - if the tree is directed and this is not given, 
      the root will be found and used
    - if the tree is directed and this is given, then 
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, 
      then a random choice will be used.
    
    width: horizontal space allocated for this branch - avoids overlap with other branches
    
    vert_gap: gap between levels of hierarchy
    
    vert_loc: vertical location of root
    
    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''
    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def parse_node_name_color(node):
    if node.endswith('O'):
        return node[:-2], 'red'
    else:
        return node[:-2], 'lightblue'

def add_nodes_edges(tree, graph, color_map):
    parent_name, parent_color = parse_node_name_color(tree.value)

    for child in tree.children:
        child_name, child_color = parse_node_name_color(child.value)
        graph.add_node(child_name, color=child_color)
        graph.add_edge(parent_name, child_name)
        color_map.append(child_color)
        add_nodes_edges(child, graph, color_map)

def convert_tree_to_graph(tree):
    graph = nx.DiGraph()
    root_name, root_color = parse_node_name_color(tree.value)
    color_map = [root_color]
    graph.add_node(root_name, color=root_color)
    add_nodes_edges(tree, graph, color_map)
    return graph, color_map


class TreeNode:
    def __init__(self, value, color='lightblue', subjective_score=0.0):
        self.value = value
        self.children = []
        self.color = color
        self.subjective_score = subjective_score # addition of subjective scores for each attribute

    def add_child(self, child_node):
        self.children.append(child_node)

    def get_leaf_nodes(self): # dynamic programming trick! 
        if len(self.children) == 0:
            return [self]
        else:
            return [child for child in self.children for child in child.get_leaf_nodes()]

    # A method to return the current tree as a nested dictionary
    def to_dict(self):
        return {
            "value": self.value,
            "children": [child.to_dict() for child in self.children]
        }

    def save(self, save_path):
        with open(save_path, 'w') as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_dict(cls, dict_):
        """ Recursively (re)construct TreeNode-based tree from dictionary. """
        node = cls(dict_['value'])
        node.children = [cls.from_dict(child) for child in dict_['children']]
        return node

    @classmethod
    def load(cls, load_path):
        with open(load_path, 'r') as f:
            return cls.from_dict(json.load(f))
    
    def draw(self, save_path=None, figsize=(15,15)):
        # Assuming 'tree' is your TreeNode object
        graph, color_map = convert_tree_to_graph(self)
        # Use the hierarchy_pos function to get positions
        root_name = self.value[:-2]
        pos = hierarchy_pos(graph, root=root_name) 
        # Draw the graph
        # make the plot wider!
        plt.figure(figsize=figsize)
        nx.draw(graph, pos, node_color=color_map, with_labels=True)
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()














