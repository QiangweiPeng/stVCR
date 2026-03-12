import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import math

from matplotlib.animation import FuncAnimation
from IPython.display import HTML,Image

from .utils import *
from .plot_3d_utils import *


def plot_2d_video(cell_type_time_series_list, spatial_time_series_list, time_points, save_path,
                              cell_type_color_map, figsize=(6,4), dpi=150, fps=10, time_unit='hour', time_init_adjust=0,
                              show_or_save='show'):
    
    # --- 第一步：预先遍历所有帧，找到全局的最大/最小坐标 ---
    all_coords = np.concatenate(spatial_time_series_list, axis=0)
    x_min, y_min = all_coords.min(axis=0)
    x_max, y_max = all_coords.max(axis=0)
    
    # 给边缘留一点 buffer（比如 5%），防止点贴在边框上
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    global_xlim = (x_min - x_pad, x_max + x_pad)
    global_ylim = (y_min - y_pad, y_max + y_pad)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    num_frames = len(time_points)

    def update(frame):

        ax.clear() 
        # --- 第二步：重新应用设置 (因为 clear 把它们删了) ---
        ax.set_xlim(global_xlim)
        ax.set_ylim(global_ylim) # 这里的顺序决定了 Y 轴是否翻转
        ax.set_aspect('equal', adjustable='box')
        # ax.invert_yaxis()

        adata_cur = ad.AnnData(np.zeros((len(cell_type_time_series_list[frame]),1)))
        adata_cur.obsm['X_spatial_aligned'] = spatial_time_series_list[frame]

        categories = cell_type_color_map.keys()
        adata_cur.obs['Annotation'] = pd.Series(pd.Categorical(cell_type_time_series_list[frame], categories=categories), 
                                                index=adata_cur.obs.index)
        adata_cur.uns['Annotation_colors'] = cell_type_color_map.values()
        

        # scatter = sc.pl.scatter(adata_cur, basis='spatial_aligned', color='Annotation',
        #               size = 10, legend_loc= 'right margin', show=False, ax=ax)
        scatter = sc.pl.scatter(adata_cur, basis='spatial_aligned', color='Annotation',
              size = 10, show=False, ax=ax, frameon=False)
        ax.set_title(f'{time_points[frame]+time_init_adjust} '+time_unit)
        # ax.axis('off')
        # ax.legend(bbox_to_anchor=(1.0, 0.05), fontsize=40, frameon=False, ncol=7)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=math.ceil(len(categories)/3), frameon=False,
                  markerscale=5, columnspacing=0.3, fontsize=7)
        # for handle in legend.legendHandles:
        #     handle.set_sizes([75])

        return scatter

    animation = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)
    if save_path and (show_or_save == 'save' or show_or_save == 'save_and_show'):
        animation.save(save_path, fps=fps, dpi=dpi)
        if show_or_save == 'save_and_show':
            return HTML(animation.to_jshtml())
    if show_or_save == 'show':
        return HTML(animation.to_jshtml())


def plot_2d_video_sim_rgb_data(exp_time_series_list, spatial_time_series_list, time_points, save_path,
                            figsize=(6,4), dpi=150, fps=10, time_unit='hour', time_init_adjust=0,
                            show_or_save='show', rgb_min_value=0, rgb_max_value=2, xlim=(-1.2, 1.2), ylim=(-1.2, 1.2),
                            invert_yaxis=True):

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    num_frames = len(time_points)

    exp_time_series_list_normlized = []
    for i in range(len(exp_time_series_list)):
        exp_time_series_list_normlized.append(
            np.minimum(np.maximum((exp_time_series_list[i] - rgb_min_value) / rgb_max_value, 0.0), 1.0))

    def update(frame):
        # ax.clear()
        # adata_cur = ad.AnnData(np.zeros((len(exp_time_series_list[frame]),1)))
        # adata_cur.obsm['X_spatial_aligned'] = spatial_time_series_list[frame]

        # categories = cell_type_color_map.keys()
        # adata_cur.obs['Annotation'] = pd.Series(pd.Categorical(cell_type_time_series_list[frame], categories=categories), 
        #                                         index=adata_cur.obs.index)
        # adata_cur.uns['Annotation_colors'] = cell_type_color_map.values()
        

        # # scatter = sc.pl.scatter(adata_cur, basis='spatial_aligned', color='Annotation',
        # #               size = 10, legend_loc= 'right margin', show=False, ax=ax)
        # scatter = sc.pl.scatter(adata_cur, basis='spatial_aligned', color='Annotation',
        #       size = 10, show=False, ax=ax, frameon=False)
        # ax.set_title(f'{time_points[frame]+time_init_adjust} '+time_unit)
        # # ax.axis('off')
        # # ax.legend(bbox_to_anchor=(1.0, 0.05), fontsize=40, frameon=False, ncol=7)
        # ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=math.ceil(len(categories)/3), frameon=False,
        #           markerscale=5, columnspacing=0.3, fontsize=7)
        # # for handle in legend.legendHandles:
        # #     handle.set_sizes([75])

        ax.clear()
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        if invert_yaxis:
            ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_title(fr't={time_points[frame]+time_init_adjust:.2f} '+time_unit)
        
        ax.scatter(spatial_time_series_list[frame][:, 0], spatial_time_series_list[frame][:, 1],
                                c=exp_time_series_list_normlized[frame])

    animation = FuncAnimation(fig, update, frames=num_frames, interval=int(1000/fps), blit=False)
    if save_path and (show_or_save == 'save' or show_or_save == 'save_and_show'):
        animation.save(save_path, fps=fps, dpi=dpi)
        if show_or_save == 'save_and_show':
            return HTML(animation.to_jshtml())
    if show_or_save == 'show':
        return HTML(animation.to_jshtml())


def generate_video(adata, 
                   model_path,
                   st_classifier_path,
                   label_to_cell_type_map,
                   init_time_index = 0,
                   end_time = None,
                   delta_t = 0.1,
                   init_cell_type = None,
                   fix_cell_type = False,
                   save_path = 'stvcr_video.gif', 
                   show_or_save='show',
                   cell_type_color_map = None,
                   cell_type_colors_key = 'Annotation_colors',
                   figsize=(6,4),
                   dpi = 150,
                   fps = 10,
                   time_unit = 'hour',
                   show_axes = True,
                   show_legend = True,
                   show_text = True,
                   text_kwargs_gif = {"font_size": 7*5*10},
                   outline_kwargs_gif = {"font_size": 7*5},
                   legend_kwargs_gif = {"label_font_size": 7*5,  "title_font_size": 7*5},
                   ):
    '''Generate video for spatiotemporal trajectory'''

    time_init_adjust = np.min(adata.obs['time'])

    init_time = np.unique(adata.obs['time_input'])[init_time_index]
    init_cell = torch.cat((torch.from_numpy(adata[adata.obs.time_input == init_time, :].obsm['X_spatial_aligned']), 
                           torch.from_numpy(adata[adata.obs.time_input == init_time, :].obsm['X_gene_input'])), 
                           dim=1).type(torch.float32)
    model = torch.load(model_path, map_location='cpu')
    model_anno = torch.load(st_classifier_path, map_location='cpu')
    spatial_dim = adata.obsm['X_spatial_aligned'].shape[1]
    if end_time is None:
        end_time = np.max(adata.obs['time_input'])
    else:
        end_time = end_time - time_init_adjust
    spatial_time_series_list, _, cell_type_time_series_list, _, time_points = evolution_forward(
        init_cell, model, model_anno, init_time, end_time, label_to_cell_type_map, 
        spatial_dim=spatial_dim, init_cell_type=init_cell_type, fix_cell_type=fix_cell_type, 
        delta_t = delta_t)


    if cell_type_color_map is None and cell_type_colors_key in adata.uns.keys() and cell_type_colors_key[0:-7] in adata.obs.keys():
        cell_type_color_map = generate_annotation_colors_map(adata, 
                                                             annotation_key=cell_type_colors_key[0:-7], 
                                                             annotation_colors_key=cell_type_colors_key)


    if adata.obsm['X_spatial_aligned'].shape[1] == 2:
        animation_html_show = plot_2d_video(cell_type_time_series_list, spatial_time_series_list, time_points, save_path,
                                cell_type_color_map, figsize=figsize, dpi=dpi, fps=fps, time_unit=time_unit, 
                                time_init_adjust=time_init_adjust, show_or_save=show_or_save)
        return animation_html_show
    elif adata.obsm['X_spatial_aligned'].shape[1] == 3:
        if show_or_save == 'save' or show_or_save== 'save_and_show':
            save_image = True
        p = plot_3d_video(cell_type_time_series_list, spatial_time_series_list, time_points, save_path,
                cell_type_color_map=cell_type_color_map, show_or_save='show', cpo=None, model_style='points', 
                                   jupyter="static", model_size = 15, fps=fps, save_image=save_image,
                                   text_kwargs=text_kwargs_gif, outline_kwargs=outline_kwargs_gif,
                                   legend_kwargs = legend_kwargs_gif, show_legend=show_legend, 
                                   show_axes=show_axes, show_text=show_text)
        return p
        


def generate_video_sim_rgb_data(adata, 
                   model_path,
                   init_time_index = 0,
                   end_time = None,
                   delta_t = 0.1,
                   save_path = 'stvcr_video.gif', 
                   show_or_save='show',
                   figsize=(6,4),
                   dpi = 150,
                   fps = 10,
                   time_unit = 'hour',
                   xlim = (-1.2, 1.2),
                   ylim = (-1.2, 1.2),
                   ):
    '''Generate video for spatiotemporal trajectory when using simulated RGB data'''

    time_init_adjust = np.min(adata.obs['time'])

    init_time = np.unique(adata.obs['time_input'])[init_time_index]
    init_cell = torch.cat((torch.from_numpy(adata[adata.obs.time_input == init_time, :].obsm['X_spatial_aligned']), 
                           torch.from_numpy(adata[adata.obs.time_input == init_time, :].obsm['X_gene_input'])), 
                           dim=1).type(torch.float32)
    model = torch.load(model_path, map_location='cpu')

    spatial_dim = adata.obsm['X_spatial_aligned'].shape[1]
    if end_time is None:
        end_time = np.max(adata.obs['time_input'])
    else:
        end_time = end_time - time_init_adjust
    spatial_time_series_list, exp_time_seies_list, _, _, time_points = evolution_forward_sim_rgb_data(
        init_cell, model, init_time, end_time, spatial_dim=spatial_dim, delta_t = delta_t)


    if adata.obsm['X_spatial_aligned'].shape[1] == 2:
        animation_html_show = plot_2d_video_sim_rgb_data(exp_time_seies_list, spatial_time_series_list, time_points, save_path,
                                figsize=figsize, dpi=dpi, fps=fps, time_unit=time_unit, 
                                time_init_adjust=time_init_adjust, show_or_save=show_or_save,xlim=xlim,ylim=ylim)
        return animation_html_show
    elif adata.obsm['X_spatial_aligned'].shape[1] == 3:
        # TODO
        pass