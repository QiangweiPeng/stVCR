#!/usr/bin/env python 
# -*- coding:utf-8 -*-

# import spateo as spt
# import pyvista as pv
# from spateo.plotting.static.three_d_plot.three_dims_plots import wrap_to_plotter

from typing import List, Optional, Union
import matplotlib as mpl

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
# from pyvista import MultiBlock, Plotter, PolyData, UnstructuredGrid

# from spateo.plotting.static.three_d_plot.three_dims_plotter import (
#     _set_jupyter,
#     add_legend,
#     add_model,
#     add_outline,
#     add_text,
#     create_plotter,
#     output_plotter,
#     save_plotter,
# )

def wrap_to_plotter(
    plotter: Plotter,
    model: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: Union[str, list] = None,
    background: str = "white",
    cpo: Union[str, list] = "iso",
    colormap: Optional[Union[str, list]] = None,
    ambient: Union[float, list] = 0.2,
    opacity: Union[float, np.ndarray, list] = 1.0,
    model_style: Union[Literal["points", "surface", "wireframe"], list] = "surface",
    model_size: Union[float, list] = 3.0,
    show_legend: bool = True,
    legend_kwargs: Optional[dict] = None,
    show_outline: bool = False,
    outline_kwargs: Optional[dict] = None,
    text: Optional[str] = None,
    text_kwargs: Optional[dict] = None,
):
    """
    What needs to be added to the visualization window.

    Args:
        plotter: The plotting object to display pyvista/vtk model.
        model: A reconstructed model.
        key: The key under which are the labels.
        background: The background color of the window.
        cpo: Camera position of the active render window. Available ``cpo`` are:

                * Iterable containing position, focal_point, and view up.
                    ``E.g.: [(2.0, 5.0, 13.0), (0.0, 0.0, 0.0), (-0.7, -0.5, 0.3)].``
                * Iterable containing a view vector.
                    ``E.g.: [-1.0, 2.0, -5.0].``
                * A string containing the plane orthogonal to the view direction.
                    ``E.g.: 'xy', 'xz', 'yz', 'yx', 'zx', 'zy', 'iso'.``
        colormap: Name of the Matplotlib colormap to use when mapping the scalars.

                  When the colormap is None, use {key}_rgba to map the scalars, otherwise use the colormap to map scalars.
        ambient: When lighting is enabled, this is the amount of light in the range of 0 to 1 (default 0.0) that reaches
                 the actor when not directed at the light source emitted from the viewer.
        opacity: Opacity of the model.

                 If a single float value is given, it will be the global opacity of the model and uniformly applied
                 everywhere, elif a numpy.ndarray with single float values is given, it
                 will be the opacity of each point. - should be between 0 and 1.

                 A string can also be specified to map the scalars range to a predefined opacity transfer function
                 (options include: 'linear', 'linear_r', 'geom', 'geom_r').
        model_style: Visualization style of the model. One of the following:

                * ``model_style = 'surface'``,
                * ``model_style = 'wireframe'``,
                * ``model_style = 'points'``.
        model_size: If ``model_style = 'points'``, point size of any nodes in the dataset plotted.

                    If ``model_style = 'wireframe'``, thickness of lines.
        show_legend: whether to add a legend to the plotter.
        legend_kwargs: A dictionary that will be pass to the ``add_legend`` function.
                       By default, it is an empty dictionary and the ``add_legend`` function will use the
                       ``{"legend_size": None, "legend_loc": None, "legend_size": None, "legend_loc": None,
                       "title_font_size": None, "label_font_size": None, "font_family": "arial", "fmt": "%.2e",
                       "n_labels": 5, "vertical": True}`` as its parameters. Otherwise, you can provide a dictionary
                       that properly modify those keys according to your needs.
        show_outline:  whether to produce an outline of the full extent for the model.
        outline_kwargs: A dictionary that will be pass to the ``add_outline`` function.

                        By default, it is an empty dictionary and the `add_legend` function will use the
                        ``{"outline_width": 5.0, "outline_color": "black", "show_labels": True, "font_size": 16,
                        "font_color": "white", "font_family": "arial"}`` as its parameters. Otherwise,
                        you can provide a dictionary that properly modify those keys according to your needs.
        text: The text to add the rendering.
        text_kwargs: A dictionary that will be pass to the ``add_text`` function.

                     By default, it is an empty dictionary and the ``add_legend`` function will use the
                     ``{ "font_family": "arial", "font_size": 12, "font_color": "black", "text_loc": "upper_left"}``
                     as its parameters. Otherwise, you can provide a dictionary that properly modify those keys
                     according to your needs.
    """

    bg_rgb = mpl.colors.to_rgb(background)
    cbg_rgb = (1 - bg_rgb[0], 1 - bg_rgb[1], 1 - bg_rgb[2])

    # Add model(s) to the plotter.
    add_model(
        plotter=plotter,
        model=model,
        key=key,
        colormap=colormap,
        ambient=ambient,
        opacity=opacity,
        model_size=model_size,
        model_style=model_style,
    )

    # # Set the camera position of plotter.
    # plotter.camera_position = cpo

    # Add a legend to the plotter.
    if show_legend:
        lg_kwargs = dict(
            title=key if isinstance(key, str) else key[-1],
            legend_size=None,
            legend_loc=None,
            font_color=cbg_rgb,
            title_font_size=15,
            label_font_size=12*4,
            font_family="arial",
            fmt="%.2e",
            n_labels=5,
            vertical=True,
        )
        if not (legend_kwargs is None):
            lg_kwargs.update((k, legend_kwargs[k]) for k in lg_kwargs.keys() & legend_kwargs.keys())

        add_legend(
            plotter=plotter,
            model=model,
            key=key,
            colormap=colormap,
            **lg_kwargs,
        )

    # Add an outline to the plotter.
    if show_outline:
        ol_kwargs = dict(
            outline_width=5.0,
            outline_color=cbg_rgb,
            show_labels=True,
            font_size=16,
            font_color=bg_rgb,
            font_family="arial",
        )
        if not (outline_kwargs is None):
            ol_kwargs.update((k, outline_kwargs[k]) for k in ol_kwargs.keys() & outline_kwargs.keys())
        add_outline(plotter=plotter, model=model, **ol_kwargs)

    # Add text to the plotter.
    if not (text is None):
        t_kwargs = dict(
            font_family="arial",
            font_size=12,
            font_color=cbg_rgb,
            text_loc="upper_left",
        )
        if not (text_kwargs is None):
            t_kwargs.update((k, text_kwargs[k]) for k in t_kwargs.keys() & text_kwargs.keys())
        add_text(plotter=plotter, text=text, **t_kwargs)


def list2adata(cur_spa, cur_cell_type, spatial_key='spatial_input'):
    import anndata as ad
    adata_cur = ad.AnnData(np.zeros((len(cur_cell_type), 2)))
    adata_cur.obsm[spatial_key] = cur_spa
    adata_cur.obs['temp_anno'] = list(cur_cell_type)
    return adata_cur


def plot_3d_video(cell_type_time_series_list, spatial_time_series_list, time_points, save_path,
                cell_type_color_map="rainbow", show_or_save='show', save_image = False,
                fps = 10,
                show_axes = True,
                show_text = True,
                key: Union[str, list] = None,
                jupyter: Union[bool, Literal["panel", "none", "pythreejs", "static", "ipygany"]] = False,
                off_screen: bool = False,
                window_size: tuple = (512, 512),
                background: str = "white",
                cpo: Union[str, list] = "iso",
                colormap: Optional[Union[str, list]] = None,
                ambient: Union[float, list] = 0.2,
                opacity: Union[float, np.ndarray, list] = 1.0,
                model_style: Union[Literal["points", "surface", "wireframe"], list] = "surface",
                model_size: Union[float, list] = 3.0,
                show_legend: bool = True,
                legend_kwargs: Optional[dict] = None,
                show_outline: bool = False,
                outline_kwargs: Optional[dict] = None,
                text: Optional[str] = None,
                text_kwargs: Optional[dict] = None,
                ):

    plotter_kws = dict(
        jupyter=False if jupyter is False else True,
        window_size=window_size,
        background=background,
        show_axes = show_axes,
    )

    model_kwargs = dict(
        background=background,
        colormap=colormap,
        ambient=ambient,
        opacity=opacity,
        model_style=model_style,
        model_size=model_size,
        show_legend=show_legend,
        legend_kwargs=legend_kwargs,
        show_outline=show_outline,
        outline_kwargs=outline_kwargs,
        text=text,
        text_kwargs=text_kwargs,
    )

    # Set jupyter.
    off_screen1, off_screen2, jupyter_backend = _set_jupyter(jupyter=jupyter, off_screen=off_screen)

    # Create a plotting object to display pyvista/vtk model.
    p = create_plotter(off_screen=off_screen1, **plotter_kws)
    if cpo is not None:
        p.camera_position = cpo

    p.open_gif(filename=filename, fps=fps)

    for points, types, time in zip(all_spa, all_cell_type, time_point):

        cur_adata = list2adata(points, types, spatial_key='spatial_input')
        embryo_pc, _ = spt.tdr.construct_pc(adata=cur_adata.copy(), spatial_key='spatial_input', groupby="anno_tissue",
                                            key_added=key, colormap=cell_type_color_map)

        wrap_to_plotter(plotter=p, model=embryo_pc, key=key, cpo=cpo, **model_kwargs)
        
        if show_text:
            p.add_text(f"Time: E{time:.2f}h", name='time-label')

        if save_image:
            save_image_path = filename[:-4] + f'E{time:.2f}h' + '.pdf'
            p.save_graphic(save_image_path)

        p.write_frame()

        p.renderer.clear_actors()

    p.close()
    



# def plot_3d_gif_norm_velocity(all_spa, all_cell_type, all_norm_velocity, time_point, filename='cell_migration_animation.gif',
#                 cell_type_colormap="rainbow", save_image = False,
#                 spatial_key='spatial_input', fps = 5,
#                 show_axes = True,
#                 show_text = True,
#                 key: Union[str, list] = None,
#                 jupyter: Union[bool, Literal["panel", "none", "pythreejs", "static", "ipygany"]] = False,
#                 off_screen: bool = False,
#                 window_size: tuple = (512, 512),
#                 background: str = "white",
#                 cpo: Union[str, list] = "iso",
#                 colormap: Optional[Union[str, list]] = None,
#                 ambient: Union[float, list] = 0.2,
#                 opacity: Union[float, np.ndarray, list] = 1.0,
#                 model_style: Union[Literal["points", "surface", "wireframe"], list] = "surface",
#                 model_size: Union[float, list] = 3.0,
#                 show_legend: bool = True,
#                 legend_kwargs: Optional[dict] = None,
#                 show_outline: bool = False,
#                 outline_kwargs: Optional[dict] = None,
#                 text: Optional[str] = None,
#                 text_kwargs: Optional[dict] = None,
#                 ):
#     """
#     Visualize reconstructed 3D model.

#     Args:
#         model: A reconstructed model.
#         key: The key under which are the labels.
#         filename: Filename of output file. Writer type is inferred from the extension of the filename.

#                 * Output an image file,please enter a filename ending with
#                   ``'.png', '.tif', '.tiff', '.bmp', '.jpeg', '.jpg', '.svg', '.eps', '.ps', '.pdf', '.tex'``.
#                 * Output a gif file, please enter a filename ending with ``.gif``.
#                 * Output a mp4 file, please enter a filename ending with ``.mp4``.
#         jupyter: Whether to plot in jupyter notebook. Available ``jupyter`` are:

#                 * ``'none'`` - Do not display in the notebook.
#                 * ``'pythreejs'`` - Show a pythreejs widget
#                 * ``'static'`` - Display a static figure.
#                 * ``'ipygany'`` - Show an ipygany widget
#                 * ``'panel'`` - Show a panel widget.
#         off_screen: Renders off-screen when True. Useful for automated screenshots.
#         window_size: Window size in pixels. The default window_size is ``[512, 512]``.
#         background: The background color of the window.
#         cpo: Camera position of the active render window. Available ``cpo`` are:

#                 * Iterable containing position, focal_point, and view up.
#                     ``E.g.: [(2.0, 5.0, 13.0), (0.0, 0.0, 0.0), (-0.7, -0.5, 0.3)].``
#                 * Iterable containing a view vector.
#                     ``E.g.: [-1.0, 2.0, -5.0].``
#                 * A string containing the plane orthogonal to the view direction.
#                     ``E.g.: 'xy', 'xz', 'yz', 'yx', 'zx', 'zy', 'iso'.``
#         colormap: Name of the Matplotlib colormap to use when mapping the scalars.

#                   When the colormap is None, use {key}_rgba to map the scalars, otherwise use the colormap to map scalars.
#         ambient: When lighting is enabled, this is the amount of light in the range of 0 to 1 (default 0.0) that reaches
#                  the actor when not directed at the light source emitted from the viewer.
#         opacity: Opacity of the model.

#                  If a single float value is given, it will be the global opacity of the model and uniformly applied
#                  everywhere, elif a numpy.ndarray with single float values is given, it
#                  will be the opacity of each point. - should be between 0 and 1.

#                  A string can also be specified to map the scalars range to a predefined opacity transfer function
#                  (options include: 'linear', 'linear_r', 'geom', 'geom_r').
#         model_style: Visualization style of the model. One of the following:

#                 * ``model_style = 'surface'``,
#                 * ``model_style = 'wireframe'``,
#                 * ``model_style = 'points'``.
#         model_size: If ``model_style = 'points'``, point size of any nodes in the dataset plotted.

#                     If ``model_style = 'wireframe'``, thickness of lines.
#         show_legend: whether to add a legend to the plotter.
#         legend_kwargs: A dictionary that will be pass to the ``add_legend`` function.
#                        By default, it is an empty dictionary and the ``add_legend`` function will use the
#                        ``{"legend_size": None, "legend_loc": None,  "legend_size": None, "legend_loc": None,
#                        "title_font_size": None, "label_font_size": None, "font_family": "arial", "fmt": "%.2e",
#                        "n_labels": 5, "vertical": True}`` as its parameters. Otherwise, you can provide a dictionary
#                        that properly modify those keys according to your needs.
#         show_outline:  whether to produce an outline of the full extent for the model.
#         outline_kwargs: A dictionary that will be pass to the ``add_outline`` function.

#                         By default, it is an empty dictionary and the `add_legend` function will use the
#                         ``{"outline_width": 5.0, "outline_color": "black", "show_labels": True, "font_size": 16,
#                         "font_color": "white", "font_family": "arial"}`` as its parameters. Otherwise,
#                         you can provide a dictionary that properly modify those keys according to your needs.
#         text: The text to add the rendering.
#         text_kwargs: A dictionary that will be pass to the ``add_text`` function.

#                      By default, it is an empty dictionary and the ``add_legend`` function will use the
#                      ``{ "font_family": "arial", "font_size": 12, "font_color": "black", "text_loc": "upper_left"}``
#                      as its parameters. Otherwise, you can provide a dictionary that properly modify those keys
#                      according to your needs.

#     Returns:

#         cpo: List of camera position, focal point, and view up.
#              Returned only if filename is None or filename ending with
#              ``'.png', '.tif', '.tiff', '.bmp', '.jpeg', '.jpg', '.svg', '.eps', '.ps', '.pdf', '.tex'``.

#         img: Numpy array of the last image.
#              Returned only if filename is None or filename ending with
#              ``'.png', '.tif', '.tiff', '.bmp', '.jpeg', '.jpg', '.svg', '.eps', '.ps', '.pdf', '.tex'``.
#     """


#     def list2adata_norm_velocity(cur_spa, cur_cell_type, cur_norm_velocity, spatial_key='spatial_input'):
#         import anndata as ad
#         adata_cur = ad.AnnData(np.zeros((len(cur_cell_type), 2)))
#         adata_cur.obsm[spatial_key] = cur_spa
#         adata_cur.obs['anno_tissue'] = list(cur_cell_type)
#         adata_cur.obs['norm_velocity'] = cur_norm_velocity
#         return adata_cur


#     plotter_kws = dict(
#         jupyter=False if jupyter is False else True,
#         window_size=window_size,
#         background=background,
#         show_axes = show_axes,
#     )

#     model_kwargs = dict(
#         background=background,
#         colormap=colormap,
#         ambient=ambient,
#         opacity=opacity,
#         model_style=model_style,
#         model_size=model_size,
#         show_legend=show_legend,
#         legend_kwargs=legend_kwargs,
#         show_outline=show_outline,
#         outline_kwargs=outline_kwargs,
#         text=text,
#         text_kwargs=text_kwargs,
#     )

#     # Set jupyter.
#     off_screen1, off_screen2, jupyter_backend = _set_jupyter(jupyter=jupyter, off_screen=off_screen)

#     # Create a plotting object to display pyvista/vtk model.
#     p = create_plotter(off_screen=off_screen1, **plotter_kws)
#     if cpo is not None:
#         p.camera_position = cpo

#     p.open_gif(filename=filename, fps=fps)

#     for points, types, norm_velocity, time in zip(all_spa, all_cell_type, all_norm_velocity, time_point):
# # all_norm_velocity
#         cur_adata = list2adata_norm_velocity(points, types, list(norm_velocity), spatial_key=spatial_key)
#         print(cur_adata.obs.norm_velocity)
#         embryo_pc, _ = spt.tdr.construct_pc(adata=cur_adata.copy(), spatial_key=spatial_key,
#                                             groupby='norm_velocity', colormap = 'Blues',
#                                             key_added=key)
#         # colormap=cell_type_colormap
#         print(embryo_pc[f'{key}_rgba'])

#         wrap_to_plotter(plotter=p, model=embryo_pc, key=key, cpo=cpo, **model_kwargs)
        
# #         if show_text:
# #             p.add_text(f"Time: E{time:.2f}h", name='time-label')
        

#         if save_image:
#             save_image_path = filename[:-4] + f'E{time:.2f}h' + '.pdf'
#             p.save_graphic(save_image_path)

#         p.write_frame()

#         p.renderer.clear_actors()

#     p.close()


def plot_from_adata(adata, filename, type_key='anno_tissue', subtype=None, spatial_key='spatial_input',
                    cell_type_colormap="rainbow", cpo='iso', text_kwargs=None, outline_kwargs=None, legend_kwargs=None,
                    show_legend=False, window_size=(512, 512)):
    embryo_pc,_ = spt.tdr.construct_pc(adata=adata.copy(), spatial_key=spatial_key, groupby=type_key,
                                       key_added=type_key, colormap=cell_type_colormap)
    if subtype is not None:
        embryo_pc = spt.tdr.three_d_pick(model=embryo_pc, key=type_key, picked_groups=subtype)[0]
    spt.pl.three_d_plot(
        model=embryo_pc,
        key=type_key,
        model_style='points',
        jupyter="static",
        model_size=10,
        filename=filename,
        cpo=cpo,
        text_kwargs=text_kwargs,
        outline_kwargs=outline_kwargs,
        legend_kwargs=legend_kwargs,
        window_size=window_size,
        show_legend=show_legend,
    )