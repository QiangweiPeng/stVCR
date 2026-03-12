import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import math
import anndata as ad
import plotly.graph_objects as go

from .utils import *

class CellNode:
    def __init__(self, cell_type):
        self.cell_type = cell_type
        self.children = {}
        self.count = 0

def build_transition_tree(all_cell_id, all_cell_type, all_time_point, time_points):
    positions = [all_time_point.index(cur_time_point) for cur_time_point in time_points]
    time_points_data = [(all_cell_type[index], all_cell_id[index]) for index in positions]
    
    if not time_points_data:
        return None
    
    # Initialize the root node with the first time point's unique cell type
    root_type = time_points_data[0][0][0]
    root = CellNode(root_type)
    root.count = len(time_points_data[0][1])
    
    # Dictionary to keep track of current nodes by their ID
    current_nodes = {i: root for i in time_points_data[0][1]}
    
    # Process each time point starting from the second one
    for types, ids in time_points_data[1:]:
        next_nodes = {}
        for cell_type, cell_id in zip(types, ids):
            if cell_id in current_nodes:
                parent_node = current_nodes[cell_id]
                if cell_type not in parent_node.children:
                    parent_node.children[cell_type] = CellNode(cell_type)
                child_node = parent_node.children[cell_type]
                child_node.count += 1
                next_nodes[cell_id] = child_node
                        
        # Update current_nodes for the next iteration
        current_nodes = next_nodes
    
    return root

def print_tree(node, level=0, threshold=20):
    if node.count > threshold:
        print("  " * level + f"{node.cell_type} ({node.count})")
        for child in node.children.values():
            print_tree(child, level + 1, threshold)

def tree_to_sankey(transition_tree, threshold, cell_type_color_map):
    labels = []
    sources = []
    targets = []
    values = []
    node_colors = []
    node_indices = {}
    current_index = 0

    def traverse_tree(node, parent_index=None):
        nonlocal current_index
        if node.count > threshold:
            node_label = f"{node.cell_type}<br>({node.count})"
            labels.append(node_label)
            node_colors.append(cell_type_color_map.get(node.cell_type, 'gray'))
            node_indices[node] = current_index
            current_node_index = current_index
            current_index += 1

            if parent_index is not None:
                sources.append(parent_index)
                targets.append(current_node_index)
                values.append(node.count)
            
            for child in node.children.values():
                traverse_tree(child, current_node_index)
    traverse_tree(transition_tree)
    
    return labels, sources, targets, values, node_colors

def plot_sankey(labels, sources, targets, values, node_colors, title=None, font_size=7, width = 725, height=400, scale=5, save_path=None, show_or_save='show'):
    sankey_data = go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )
    # create figure
    fig = go.Figure(sankey_data)
    
    # update
    fig.update_layout(title=title, font_size=font_size,  width = width, height=height)
    
    # show
    if show_or_save == 'show' or show_or_save == 'save_and_show':
        fig.show()

    if save_path and (show_or_save == 'save' or show_or_save == 'save_and_show'):
        fig.write_image(save_path, width = width, height=height, scale=scale)
    
    return fig


def generate_lineage(adata, 
                    model_path,
                    st_classifier_path,
                    label_to_cell_type_map,
                    init_cell_type,
                    init_time_index,
                    lineage_time_points,
                    delta_t = 0.1,
                    threshold = 25,
                    save_path = 'lineage.pdf', 
                    show_or_save='show',
                    cell_type_color_map = None,
                    title = None, 
                    font_size = 15, 
                    width = 725, 
                    height = 400, 
                    scale = 5,
                    ):
    '''Generate Sankey plot of cell lineage'''
    time_init_adjust = np.min(adata.obs['time'])

    init_time = adata.obs['time_input'].unique()[init_time_index]
    init_cell = torch.cat((torch.from_numpy(adata[adata.obs.time_input == init_time, :].obsm['X_spatial_aligned']), 
                           torch.from_numpy(adata[adata.obs.time_input == init_time, :].obsm['X_gene_input'])), 
                           dim=1).type(torch.float32)
    
    model = torch.load(model_path, map_location='cpu')
    model_anno = torch.load(st_classifier_path, map_location='cpu')
    
    # Pick out the cells that satisfy the initial cell type based on the classifier
    cur_cell_type = get_cell_type(init_cell, model_anno, torch.tensor(init_time), label_to_cell_type_map)
    index = [i for i,x in enumerate(cur_cell_type) if x==init_cell_type]
    init_cell = init_cell[index]

    spatial_dim = adata.obsm['X_spatial_aligned'].shape[1]

    lineage_time_points_adjust = [cur_time - time_init_adjust for cur_time in lineage_time_points]
    end_time = lineage_time_points_adjust[-1]

    _, _, cell_type_time_series_list, cell_id_time_series_list, time_points = evolution_forward(
        init_cell, model, model_anno, init_time, end_time, label_to_cell_type_map, 
        spatial_dim=spatial_dim, init_cell_type=init_cell_type, delta_t = delta_t, 
        other_time_points=lineage_time_points_adjust)
    
    if cell_type_color_map is None:
        cell_type_color_map = generate_annotation_colors_map(adata, 
                                                            annotation_key='Annotation', 
                                                            annotation_colors_key='Annotation_colors') 
    
    transition_tree = build_transition_tree(cell_id_time_series_list, cell_type_time_series_list, time_points, lineage_time_points_adjust)
    labels, sources, targets, values, node_colors = tree_to_sankey(transition_tree, threshold, cell_type_color_map)
    fig = plot_sankey(labels, sources, targets, values, node_colors, title=title, font_size=font_size, width = width, height=height, scale=scale, save_path=save_path, show_or_save=show_or_save)

    return fig, transition_tree


    

