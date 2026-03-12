import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import math
import anndata as ad

from .utils import *
from .plot_3d_utils import *


def interpolate(adata, 
                model_path,
                st_classifier_path,
                label_to_cell_type_map,
                interpolation_time,
                ae_model_path = None,
                init_time_index = None,
                delta_t = 0.1,
                init_cell_type = None,
                save_path = 'interpolate.png', 
                show_or_save='show',
                cell_type_color_map = None,
                cell_type_colors_key = None,
                figsize=(6,4),
                dpi = 300,
                time_unit = 'hour',
                text_kwargs = {"font_size": 7*4},
                outline_kwargs = {"font_size": 7*4},
                legend_kwargs = {"label_font_size": 7*4,  "title_font_size": 7*4},
                ):
    time_init_adjust = np.min(adata.obs['time'])
    interpolation_time = interpolation_time - time_init_adjust

    if cell_type_color_map is None and cell_type_colors_key in adata.uns.keys() and cell_type_colors_key[0:-7] in adata.obs.keys():
        cell_type_color_map = generate_annotation_colors_map(adata, 
                                                             annotation_key=cell_type_colors_key[0:-7], 
                                                             annotation_colors_key=cell_type_colors_key)

    if init_time_index is not None:
        init_time = adata.obs['time_input'].unique()[init_time_index]
    else:
        init_time = np.unique(adata.obs['time_input'][adata.obs['time_input'] < interpolation_time])[-1]

    init_cell = torch.cat((torch.from_numpy(adata[adata.obs.time_input == init_time, :].obsm['X_spatial_aligned']), 
                           torch.from_numpy(adata[adata.obs.time_input == init_time, :].obsm['X_gene_input'])), 
                           dim=1).type(torch.float32)
    model = torch.load(model_path, map_location='cpu')
    model_anno = torch.load(st_classifier_path, map_location='cpu')
    spatial_dim = adata.obsm['X_spatial_aligned'].shape[1]

    spatial_time_series_list, exp_time_series_list, cell_type_time_series_list, _, _ = evolution_forward(
        init_cell, model, model_anno, init_time, interpolation_time, label_to_cell_type_map, 
        spatial_dim=spatial_dim, init_cell_type=init_cell_type, delta_t = delta_t)
    
    interpolation_spatial, interpolation_exp, interpolation_cell_type = spatial_time_series_list[-1], exp_time_series_list[-1], cell_type_time_series_list[-1]
    if ae_model_path is not None:
        interpolation_exp_ae = interpolation_exp.copy()
        ae_model = torch.load(ae_model_path, map_location='cpu')
        interpolation_exp = ae_model.decode_mlp1(torch.from_numpy(interpolation_exp))
    interpolation_adata = ad.AnnData(X=interpolation_exp.detach().numpy())
    interpolation_adata.obsm['X_spatial_aligned'] = interpolation_spatial
    interpolation_adata.obsm['X_ae'] = interpolation_exp_ae
    categories = cell_type_color_map.keys()
    interpolation_adata.obs['Annotation'] = pd.Series(pd.Categorical(interpolation_cell_type, categories=categories),
                                                      index=interpolation_adata.obs.index)
    interpolation_adata.uns['Annotation_colors'] = cell_type_color_map.values()
    interpolation_adata.obs['time'] = interpolation_time + time_init_adjust

    if adata.obsm['X_spatial_aligned'].shape[1] == 2:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        sc.pl.scatter(interpolation_adata, basis='spatial_aligned', color='Annotation', title=f'{interpolation_time+time_init_adjust} '+time_unit,
                show=False, frameon=False, ax=ax)
        if save_path and show_or_save == 'save_and_show':
            plt.savefig(save_path, dpi=dpi)
            # plt.show()
            return fig, interpolation_adata
        elif save_path and show_or_save == 'save':
            plt.savefig(save_path, dpi=dpi)
            return interpolation_adata
        elif show_or_save == 'show':
            return fig, interpolation_adata
        
    elif adata.obsm['X_spatial_aligned'].shape[1] == 3:
        output_plotter =plot_from_adata_3d(interpolation_adata, save_path, type_key='Annotation', subtype=None, spatial_key='X_spatial_aligned',
                    cell_type_color_map=cell_type_color_map, cpo='iso', text_kwargs=text_kwargs, outline_kwargs=outline_kwargs, legend_kwargs=legend_kwargs,
                    show_legend=True, window_size=(512, 512), add_text=f'{interpolation_time+time_init_adjust} '+time_unit)
        if show_or_save == 'show' or show_or_save == 'save_and_show':
            return output_plotter, interpolation_adata
        elif show_or_save == 'save':
            return interpolation_adata
        