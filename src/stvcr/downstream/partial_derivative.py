import torch
import os
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

from matplotlib.pyplot import rc_context
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as ticker

import networkx as nx
import gseapy as gp
import seaborn as sns

from .plot_3d_utils import *

def raw_gene_space_velocity(ot_model, ae_model, device=torch.device('cpu')):
    def velocity(cur_cell_gene, cur_cell_spatial, t):
        latent_embed = ae_model.encode_mlp1(cur_cell_gene)
        velocity_latent = ot_model.gene_velocity_net(torch.tensor(t, dtype=torch.float32), torch.cat((latent_embed, cur_cell_spatial), axis=1))
        velocity_raw = (ae_model.decode_mlp1(latent_embed + 0.1*velocity_latent) - ae_model.decode_mlp1(latent_embed - 0.1*velocity_latent))/0.2
        return velocity_raw, velocity_latent
    return velocity

def raw_spatital_velocity(ot_model, ae_model, device=torch.device('cpu')):
    def velocity(cur_cell_gene, cur_cell_spatial, t):
        latent_embed = ae_model.encode_mlp1(cur_cell_gene)
        spatial_velocity = ot_model.spatial_velocity_net(torch.tensor(t, dtype=torch.float32), torch.cat((latent_embed, cur_cell_spatial), axis=1))
        return spatial_velocity
    return velocity

def raw_gene_space_growth(ot_model, ae_model, device=torch.device('cpu')):
    def growth(cur_cell_gene, cur_cell_spatial, t):
        latent_embed = ae_model.encode_mlp1(cur_cell_gene)
        cell_growth = ot_model.growth_rate_net(torch.tensor(t, dtype=torch.float32), torch.cat((latent_embed, cur_cell_spatial), axis=1))
        return cell_growth
    return growth


def derivative_to_direction(adata,
                         model_path,
                         ae_model_path,
                         time_index,
                         function_target='gene', # 'gene' 'growth' or 'norm_spatial_velocity'
                         direction_vector=None,
                         ):
    model = torch.load(model_path, map_location='cpu')
    ae_model = torch.load(ae_model_path, map_location='cpu')
    
    cur_time = adata.obs['time_input'].unique()[time_index]
    cur_adata = adata[adata.obs.time_input == cur_time, :]
    cur_cell_gene = torch.tensor(cur_adata.X.toarray(), requires_grad=True, dtype=torch.float32)
    cur_cell_spatial = torch.tensor(cur_adata.obsm['X_spatial_aligned'], requires_grad=True, dtype=torch.float32)

    spatial_dim = adata.obsm['X_spatial_aligned'].shape[1]
    if direction_vector is None and spatial_dim == 2:
        direction_vector = np.array([1, 0], dtype=np.float32)
    elif direction_vector is None and spatial_dim == 3:
        direction_vector = np.array([1, 0, 0], dtype=np.float32)

    if function_target == 'gene':
        function = raw_gene_space_velocity(model, ae_model)
        cur_cell_velocity, _ = function(cur_cell_gene, cur_cell_spatial, cur_time)

        par_gene_velocity_par_direction = np.zeros((cur_adata.n_obs, cur_adata.n_vars))
        for i in range(cur_adata.n_vars):
            cur_cell_velocity_gene_i = cur_cell_velocity[:, i]

            # grad_output = torch.ones_like(cur_cell_velocity_gene_i)
            # cur_cell_velocity_gene_i.backward(grad_output, retain_graph=True)
            torch.sum(cur_cell_velocity_gene_i).backward(retain_graph=True)

            par_gene_velocity_par_direction[:, i] = np.sum(cur_cell_spatial.grad.numpy().copy() * direction_vector, axis=1)
            cur_cell_velocity_gene_i.grad = None
        cur_adata.layers['partial_gene_partial_direction'+str(direction_vector)] = par_gene_velocity_par_direction

    elif function_target == 'norm_spatial_velocity':
        function = raw_spatital_velocity(model, ae_model)
        cur_cell_spatial_velocity = function(cur_cell_gene, cur_cell_spatial, cur_time)
        cur_cell_spatial_velocity_norm = torch.sum(torch.square(cur_cell_spatial_velocity), dim=1)

        # grad_output = torch.ones_like(cur_cell_spatial_velocity_norm)
        # cur_cell_spatial_velocity_norm.backward(grad_output, retain_graph=True)
        torch.sum(cur_cell_spatial_velocity_norm).backward(retain_graph=True)

        derivative_n = np.sum(cur_cell_spatial.grad.numpy().copy() * direction_vector, axis=1)
        cur_adata.obsm['partial_norm_spatial_velocity_partial_direction'+str(direction_vector)] = derivative_n
        cur_cell_spatial_velocity_norm.grad = None

    elif function_target == 'growth':
        function = raw_gene_space_growth(model, ae_model)
        cur_cell_growth = function(cur_cell_gene, cur_cell_spatial, cur_time)

        # grad_output = torch.ones_like(cur_cell_growth)
        # cur_cell_growth.backward(grad_output, retain_graph=True)
        torch.sum(cur_cell_growth).backward(retain_graph=True)

        derivative_n = np.sum(cur_cell_spatial.grad.numpy().copy() * direction_vector, axis=1)
        cur_adata.obsm['partial_growth_partial_direction'+str(direction_vector)] = derivative_n
        cur_cell_growth.grad = None

    return cur_adata
    

def derivative_to_expression(adata,
                         model_path,
                         ae_model_path,
                         time_index,
                         function_target='gene', # 'gene' 'growth' or 'norm_spatial_velocity'
                         gene_list=None,
                         ):
    model = torch.load(model_path, map_location='cpu')
    ae_model = torch.load(ae_model_path, map_location='cpu')
    
    cur_time = adata.obs['time_input'].unique()[time_index]
    cur_adata = adata[adata.obs.time_input == cur_time, :].copy()
    cur_cell_gene = torch.tensor(cur_adata.X.toarray(), requires_grad=True, dtype=torch.float32)
    cur_cell_spatial = torch.tensor(cur_adata.obsm['X_spatial_aligned'], requires_grad=True, dtype=torch.float32)

    if function_target == 'gene':
        function = raw_gene_space_velocity(model, ae_model)
        cur_cell_velocity, _ = function(cur_cell_gene, cur_cell_spatial, cur_time)
        if gene_list is None:
            gene_list = cur_adata.var_names
        gene_index = [cur_adata.var_names.get_loc(gene) for gene in gene_list]

        par_gene_velocity_par_gene = np.zeros((cur_adata.n_obs, len(gene_index), len(gene_index)))
        for i in range(len(gene_index)):
            cur_gene_index = gene_index[i]
            cur_cell_velocity_gene_i = cur_cell_velocity[:, cur_gene_index]
            torch.sum(cur_cell_velocity_gene_i).backward(retain_graph=True)
            par_gene_velocity_par_gene[:, i, :] = cur_cell_gene.grad.numpy().copy()[:, gene_index]
            cur_cell_velocity_gene_i.grad = None
        cur_adata.obsm['partial_gene_velocity_partial_gene'] = par_gene_velocity_par_gene

    elif function_target == 'norm_spatial_velocity':
        function = raw_spatital_velocity(model, ae_model)
        cur_cell_spatial_velocity = function(cur_cell_gene, cur_cell_spatial, cur_time)
        cur_cell_spatial_velocity_norm = torch.sum(torch.square(cur_cell_spatial_velocity), dim=1)

        # grad_output = torch.ones_like(cur_cell_spatial_velocity_norm)
        # cur_cell_spatial_velocity_norm.backward(grad_output, retain_graph=True)
        torch.sum(cur_cell_spatial_velocity_norm).backward(retain_graph=True)

        cur_adata.layers['partial_norm_spatial_velocity_partial_gene'] = cur_cell_gene.grad.numpy().copy()
        cur_cell_spatial_velocity_norm.grad = None

    elif function_target == 'growth':
        function = raw_gene_space_growth(model, ae_model)
        cur_cell_growth = function(cur_cell_gene, cur_cell_spatial, cur_time)

        # grad_output = torch.ones_like(cur_cell_growth)
        # cur_cell_growth.backward(grad_output, retain_graph=True)
        torch.sum(cur_cell_growth).backward(retain_graph=True)

        cur_adata.layers['partial_growth_partial_gene'] = cur_cell_gene.grad.numpy().copy()
        cur_cell_growth.grad = None
    
    if function_target == 'gene':
        return cur_adata, gene_list
    else:
        return cur_adata
    

def derivative_scatter_plot(adata,
                    function_source='gene', # 'direction_vector'
                    function_target='gene', # 'gene' 'growth' or 'norm_spatial_velocity'
                    source_gene = None,
                    target_gene = None,
                    par_gene_velocity_par_gene_lsit = None,
                    direction_vector=None,
                    figsize = (6, (5*2/3)),
                    max_value = None,
                    cmap = 'coolwarm',
                    pointsize = 0.2,
                    save_path = None,
                    ):
    adata = adata.copy()
    
    def scatter_plot_with_scanpy(adata, title, color_value,
                                 figsize=(6, (5*2/3)), max_value = None, spatial_key='X_spatial_aligned', 
                                 cmap='coolwarm', pointsize=0.2, save_path=None):

        if max_value is None:
            max_value = np.maximum(np.abs(color_value.max()), np.abs(color_value.min()))
        norm = TwoSlopeNorm(vcenter=0, vmin=-max_value, vmax=max_value)
        mapper = ScalarMappable(norm=norm, cmap='coolwarm')
        
        # rc_context is used for the figure size
        with rc_context({"figure.figsize": figsize}):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.clear()
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(-1.0, 1.0)
            
            ax.scatter(adata.obsm[spatial_key][:, -2], adata.obsm[spatial_key][:, -1],
                        c=color_value, cmap=cmap, alpha=1, norm=norm, s=pointsize, marker='o',edgecolors='none')
            ax.axis('off')
            
            plt.xticks([])
            plt.yticks([])
            
            # plt.xlabel('X')
            # plt.ylabel('Y')
            
            cbar = plt.colorbar(mapper)
            cbar.formatter = ticker.ScalarFormatter()
            cbar.formatter.set_scientific(True)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.update_ticks()
            
            if title is not None:
                plt.title(title)
                # plt.title(rf'$\frac{{\partial g}}{{\partial \mathit{{{gene}}}}}$')
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
            plt.show()
    
    if function_source == 'gene':
        if function_target == 'gene':
            source_gene_index = par_gene_velocity_par_gene_lsit.index(source_gene)
            target_gene_index = par_gene_velocity_par_gene_lsit.index(target_gene)
            par_gene_velocity_par_gene = adata.obsm['partial_gene_velocity_partial_gene'][:, target_gene_index, source_gene_index]
            if adata.obsm['X_spatial_aligned'].shape[1] == 2:
                scatter_plot_with_scanpy(adata, title=rf'$\partial \mathit{{{target_gene}}} / \partial \mathit{{{source_gene}}}$', color_value=par_gene_velocity_par_gene,
                                        figsize=figsize, max_value = max_value, spatial_key='X_spatial_aligned', 
                                        cmap=cmap, pointsize=pointsize, save_path=save_path)
            else:
                colors_key = 'partial_gene_velocity_partial_gene_'+str(target_gene_index)+'_'+str(source_gene_index)
                adata.obs[colors_key] = par_gene_velocity_par_gene
                plot_from_adata_3d(adata, filename=save_path, colors_key=colors_key, spatial_key='X_spatial_aligned',
                    cell_type_color_map=cmap, cpo='iso', add_text=rf'$\partial \mathit{{{target_gene}}} / \partial \mathit{{{source_gene}}}$',
                    text_kwargs={"font_size": 7*5}, outline_kwargs={"font_size": 7*5}, legend_kwargs = {"label_font_size": 12,  "title":""},
                    show_legend=True, window_size=(512, 512))
        elif function_target == 'growth':
            source_gene_index = adata.var_names.get_loc(source_gene)
            partial_growth_partial_gene = adata.layers['partial_growth_partial_gene'][:, source_gene_index]
            if adata.obsm['X_spatial_aligned'].shape[1] == 2:
                scatter_plot_with_scanpy(adata, title=rf'$\partial g / \partial \mathit{{{source_gene}}}$', color_value=partial_growth_partial_gene,
                                     figsize=figsize, max_value = max_value, spatial_key='X_spatial_aligned', 
                                     cmap=cmap, pointsize=pointsize, save_path=save_path)
            else:
                colors_key = 'partial_growth_partial_gene_'+str(source_gene_index)
                adata.obs[colors_key] = partial_growth_partial_gene
                plot_from_adata_3d(adata, filename=save_path, colors_key=colors_key, spatial_key='X_spatial_aligned',
                    cell_type_color_map=cmap, cpo='iso', add_text=rf'$\partial g / \partial \mathit{{{source_gene}}}$',
                    text_kwargs={"font_size": 7*5*1}, outline_kwargs={"font_size": 7*5}, legend_kwargs = {"label_font_size": 12,  "title":""},
                    show_legend=True, window_size=(512, 512))
        elif function_target == 'norm_spatial_velocity':
            source_gene_index = adata.var_names.get_loc(source_gene)
            partial_norm_spatial_velocity_partial_gene = adata.layers['partial_norm_spatial_velocity_partial_gene'][:, source_gene_index]
            if adata.obsm['X_spatial_aligned'].shape[1] == 2:
                scatter_plot_with_scanpy(adata, title=rf'$\partial \|v_x\| / \partial \mathit{{{source_gene}}}$', color_value=partial_norm_spatial_velocity_partial_gene,
                                     figsize=figsize, max_value = max_value, spatial_key='X_spatial_aligned', 
                                     cmap=cmap, pointsize=pointsize, save_path=save_path)
            else:
                colors_key = 'partial_norm_spatial_velocity_partial_gene_'+str(source_gene_index)
                adata.obs[colors_key] = partial_norm_spatial_velocity_partial_gene
                plot_from_adata_3d(adata, filename=save_path, colors_key=colors_key, spatial_key='X_spatial_aligned',
                    cell_type_color_map=cmap, cpo='iso', add_text=rf'$\partial \|v_x\| / \partial \mathit{{{source_gene}}}$',
                    text_kwargs={"font_size": 7*5*1}, outline_kwargs={"font_size": 7*5}, legend_kwargs = {"label_font_size": 12,  "title":""},
                    show_legend=True, window_size=(512, 512))
    elif function_source == 'direction_vector':
        if function_target == 'gene':
            target_gene_index = adata.var_names.get_loc(target_gene)
            partial_gene_partial_direction = adata.layers['partial_gene_partial_direction'+str(direction_vector)][:, target_gene_index]
            if adata.obsm['X_spatial_aligned'].shape[1] == 2:
                scatter_plot_with_scanpy(adata, title=rf'$\partial \mathit{{{target_gene}}} / \partial \vec{{n}}$', color_value=partial_gene_partial_direction,
                                     figsize=figsize, max_value = max_value, spatial_key='X_spatial_aligned', 
                                     cmap=cmap, pointsize=pointsize, save_path=save_path)
            else:
                colors_key = 'partial_gene_partial_direction_'+str(target_gene_index)+'_'+str(direction_vector)
                adata.obs[colors_key] = partial_gene_partial_direction
                plot_from_adata_3d(adata, filename=save_path, colors_key=colors_key, spatial_key='X_spatial_aligned',
                    cell_type_color_map=cmap, cpo='iso', add_text=rf'$\partial \mathit{{{target_gene}}} / \partial \vec{{n}}$',
                    text_kwargs={"font_size": 7*5*1}, outline_kwargs={"font_size": 7*5}, legend_kwargs = {"label_font_size": 12,  "title":""},
                    show_legend=True, window_size=(512, 512))
        elif function_target == 'growth':
            partial_growth_partial_direction = adata.obsm['partial_growth_partial_direction'+str(direction_vector)]
            if adata.obsm['X_spatial_aligned'].shape[1] == 2:
                scatter_plot_with_scanpy(adata, title=rf'$\partial g / \partial \vec{{n}}$', color_value=partial_growth_partial_direction,
                            figsize=figsize, max_value = max_value, spatial_key='X_spatial_aligned', 
                            cmap=cmap, pointsize=pointsize, save_path=save_path)
            else:
                colors_key = 'partial_growth_partial_direction_'+str(direction_vector)
                # colors_key = 'temp'
                adata.obs[colors_key] = partial_growth_partial_direction
                plot_from_adata_3d(adata, filename=save_path, colors_key=colors_key, spatial_key='X_spatial_aligned',
                    cell_type_color_map=cmap, cpo='iso', add_text=rf'$\partial g / \partial \vec{{n}}$',
                    text_kwargs={"font_size": 7*5*1}, outline_kwargs={"font_size": 7*5}, legend_kwargs = {"label_font_size": 12,  "title":""},
                    show_legend=True, window_size=(512, 512))
        elif function_target == 'norm_spatial_velocity':
            partial_norm_spatial_velocity_partial_direction = adata.obsm['partial_norm_spatial_velocity_partial_direction'+str(direction_vector)]
            if adata.obsm['X_spatial_aligned'].shape[1] == 2:
                scatter_plot_with_scanpy(adata, title=rf'$\partial \|v_x\| / \partial \vec{{n}}$', color_value=partial_norm_spatial_velocity_partial_direction,
                            figsize=figsize, max_value = max_value, spatial_key='X_spatial_aligned', 
                            cmap=cmap, pointsize=pointsize, save_path=save_path)
            else:
                colors_key = 'partial_norm_spatial_velocity_partial_direction_'+str(direction_vector)
                adata.obs[colors_key] = partial_norm_spatial_velocity_partial_direction
                plot_from_adata_3d(adata, filename=save_path, colors_key=colors_key, spatial_key='X_spatial_aligned',
                    cell_type_color_map=cmap, cpo='iso', add_text=rf'$\partial \|v_x\| / \partial \vec{{n}}$',
                    text_kwargs={"font_size": 7*5*1}, outline_kwargs={"font_size": 7*5}, legend_kwargs = {"label_font_size": 12,  "title":""},
                    show_legend=True, window_size=(512, 512))
            

def plot_grn_heat_map(adata,
                    gene_list,
                    title=None,
                    figsize = (8, 8.5),
                    max_value = None,
                    cmap = 'coolwarm',
                    save_path = None,
                    ):
    regulation_matrix = np.mean(adata.obsm['partial_gene_velocity_partial_gene'], axis=0)
    if max_value:
        cur_max = max_value
    else:
        cur_max = np.abs(regulation_matrix).max()

    norm = TwoSlopeNorm(vcenter=0, vmin=-cur_max, vmax=cur_max)
    mapper = ScalarMappable(norm=norm, cmap=cmap)
    
    plt.figure(figsize=figsize)
    plt.imshow(regulation_matrix, cmap=cmap, norm=norm)
    plt.colorbar(mapper)
    
    plt.xticks(np.arange(len(gene_list)), gene_list)
    plt.yticks(np.arange(len(gene_list)), gene_list)
    plt.xticks(rotation=90)
    # plt.xlabel('target gene')
    # plt.ylabel('source gene')
    plt.xlabel('source gene')
    plt.ylabel('target gene')

    if title:
        plt.title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_grn_graph(adata,
                gene_list, 
                threshold=0.01, 
                node_colors='lightblue', 
                title=None, 
                figsize=(8,8.5), 
                font_size=12, 
                node_size=2000, 
                arrowsize=50, 
                min_width = 1.0, 
                max_width = 6.0, 
                save_path=None):
    
    regulation_matrix = np.mean(adata.obsm['partial_gene_velocity_partial_gene'], axis=0)
    # 创建一个有向图
    G = nx.DiGraph()

    num_genes = len(gene_list)
    
    # 添加节点（基因）
    for i in range(num_genes):
        G.add_node(i, label=gene_list[i])

    # 绘制网络图
    pos = nx.circular_layout(G)  # 使用圆形布局
    # pos = nx.spring_layout(G)
    
    # 添加边（调控关系），只添加调控强度大于某个阈值的边
    for i in range(num_genes):
        for j in range(num_genes):
            if i != j and abs(regulation_matrix[i, j]) > threshold:
                G.add_edge(j, i, weight=regulation_matrix[i, j])
            # G.add_edge(j, i, weight=regulation_matrix[i, j])
    
    # 根据权重调整边的宽度和颜色
    # 设置边宽度的范围
    min_width = min_width
    max_width = max_width
    edge_weights_list = [np.abs(G[u][v]['weight']) for u, v in G.edges()]
    if edge_weights_list:
        normalized_widths = np.clip(edge_weights_list, 0, max(edge_weights_list))  # 归一化
        edge_widths = min_width + (max_width - min_width) * (normalized_widths - min(edge_weights_list)) / (max(edge_weights_list) - min(edge_weights_list))
    else:
        edge_widths = 0.0
    
    # 边的颜色设置：负值为蓝色，正值为红色
    # edge_colors = ['blue' if G[u][v]['weight'] < 0 else 'red' for u, v in G.edges()]
    # edge_colors = ['darkblue' if G[u][v]['weight'] < 0 else 'darkred' for u, v in G.edges()]
    edge_colors = ['#003366' if G[u][v]['weight'] < 0 else '#CC0000' for u, v in G.edges()]
    
    
    
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=edge_widths, edge_color=edge_colors, arrowstyle='-|>', arrowsize=arrowsize, style='solid')
    nx.draw_networkx_labels(G, pos, labels={i: gene_list[i] for i in G.nodes()}, font_size=font_size)

    plt.axis('off')
    
    
    # 省略边标签的绘制
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def enrichment_analysis(adata,
                        growth_or_migration='growth',     # 'growth' or 'migration'
                        top_or_bottom='top',              # 'top' or 'bottom'
                        gene_number=100,
                        gene_sets = None,
                        organism='Human',                 # 'Human' or 'Mouse'
                        enrichment_type='GO',             # 'GO' or 'KEGG'
                        plot_type='bar',                  # 'bar' or 'bubble'
                        top_terms=10,
                        outdir='enrichment_results',      # 输出文件夹
                        cutoff=0.05,                  # p-value cutoff
                        ):
    # 选择基因列表
    if growth_or_migration == 'growth':
        values = np.mean(adata.layers['partial_growth_partial_gene'], axis=0)
    elif growth_or_migration == 'migration':
        values = np.mean(adata.layers['partial_norm_spatial_velocity_partial_gene'], axis=0)
    else:
        raise ValueError("growth_or_migration must be 'growth' or 'migration'")

    if top_or_bottom == 'top':
        gene_indices = np.argsort(values)[-gene_number:][::-1]
    elif top_or_bottom == 'bottom':
        gene_indices = np.argsort(values)[:gene_number]
    else:
        raise ValueError("top_or_bottom must be 'top' or 'bottom'")

    gene_list = list(adata.var_names[gene_indices])

    # 富集数据库
    if gene_sets is not None:
        if enrichment_type == 'GO':
            gene_sets = ['GO_Biological_Process_2021']
        elif enrichment_type == 'KEGG':
            gene_sets = ['KEGG_2021_Human'] if organism == 'Human' else ['KEGG_2021_Mouse']
        else:
            raise ValueError("enrichment_type must be 'GO' or 'KEGG'")

    # 创建输出文件夹
    os.makedirs(outdir, exist_ok=True)

    # 运行 enrichr 分析
    enr = gp.enrichr(gene_list=gene_list,
                     gene_sets=gene_sets,
                     organism=organism,
                     outdir=None,  # 不自动保存内部图表
                     cutoff=cutoff)

    # 筛选前若干term
    top_results = enr.results.sort_values('Adjusted P-value').head(top_terms)

    # 输出图路径
    barplot_path = os.path.join(outdir, f"{growth_or_migration}_{top_or_bottom}_{enrichment_type}_barplot.png")
    dotplot_path = os.path.join(outdir, f"{growth_or_migration}_{top_or_bottom}_{enrichment_type}_dotplot.png")

    # 画图
    if plot_type == 'bar':
        gp.barplot(top_results, title=f'{enrichment_type} Enrichment ({growth_or_migration}, {top_or_bottom})',
                   ofname=barplot_path, cutoff=cutoff)
    elif plot_type == 'bubble':
        gp.dotplot(top_results, title=f'{enrichment_type} Enrichment ({growth_or_migration}, {top_or_bottom})',
                   ofname=dotplot_path, cutoff=cutoff)
    else:
        raise ValueError("plot_type must be 'bar' or 'bubble'")

    return top_results