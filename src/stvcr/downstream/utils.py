import torch
import numpy as np

def generate_annotation_colors_map(adata, annotation_key='Annotation', annotation_colors_key='Annotation_colors'):
    '''Generate annotation colors map'''
    cell_types = adata.obs[annotation_key]
    unique_types = np.array(list(cell_types.cat.categories))
    if annotation_colors_key is not None and annotation_colors_key in adata.uns.keys():
        cell_type_color_map = dict(zip(unique_types, adata.uns[annotation_colors_key]))
    return cell_type_color_map

# Time series classifier
def get_cell_type(cur_cell, model_anno, t, label_to_cell_type_map):
    model_anno.eval()
    with torch.no_grad():
        time = torch.tensor(t.repeat(cur_cell.shape[0]), dtype=torch.float32).unsqueeze(1)
        inputs = torch.cat((cur_cell, time), dim=1)
        outputs = model_anno(inputs)
        cur_cell_label = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        # cur_cell_type = list(label_to_cell_type_map[cell_type_index])
        cur_cell_type = [label_to_cell_type_map[cell_label_i.item()] for cell_label_i in cur_cell_label]
    return cur_cell_type

def evolution_forward(init_cell, model, model_anno, init_time, end_time, label_to_cell_type_map, 
                      spatial_dim=2, init_cell_type=None, fix_cell_type=False, delta_t = 0.1, other_time_points=None):
    def growth(x, delta_t, model, t, cur_cell_id, sigma_d=0.00001):
        cell_number = x.shape[0]
        x_growth = []
        new_cell_id = []
        g = model.growth_rate_net(torch.tensor(t, dtype=torch.float32), x)
        for i in range(cell_number):
            g_i = g[i, :]
            temp = np.random.rand()
            if g_i > 0 and temp < (g_i * delta_t):
                x_growth.append(x[i, :])
                new_cell_id.append(cur_cell_id[i])
                
                new_cell = x[i, :] + sigma_d * torch.tensor(np.random.randn(x[i, :].shape[0]))
                # new_cell[0:3] = torch.maximum(new_cell[0:3], torch.tensor(0.0))
                x_growth.append(new_cell)
                new_cell_id.append(cur_cell_id[i])
            elif g_i > 0 and temp > (g_i * delta_t):
                x_growth.append(x[i, :])
                new_cell_id.append(cur_cell_id[i])
            elif g_i < 0 and temp > (-g_i * delta_t):
                x_growth.append(x[i, :])
                new_cell_id.append(cur_cell_id[i])
        cur_cell = torch.vstack(x_growth)
        cur_cell = torch.tensor(cur_cell, dtype=torch.float32)
        return cur_cell, new_cell_id
    
    time_point = np.arange(init_time, end_time, delta_t)
    time_point = np.union1d(time_point, end_time)
    if other_time_points is not None:
        time_point = np.union1d(time_point, other_time_points)
    time_point = np.round(time_point, 2)
    time_point = np.unique(time_point)
    time_step = time_point[1:] - time_point[0:-1]
    
    all_time_point = list(time_point)
    all_exp = []
    all_spa = []
    all_cell_type = []
    all_cell_id = []
    
    cur_cell = init_cell
    cur_cell_id = np.arange(cur_cell.shape[0])
    if init_cell_type is None:
        cur_cell_type = get_cell_type(cur_cell, model_anno, time_point[0], label_to_cell_type_map)
    else:
        cur_cell_type = [init_cell_type]*cur_cell.shape[0]
    
    for i in range(len(time_point)):
        all_spa.append(cur_cell[:, 0:spatial_dim:].detach().numpy())
        all_exp.append(cur_cell[:, spatial_dim:].detach().numpy())
        all_cell_id.append(cur_cell_id)

        if i > 0:
            if fix_cell_type:
                cur_cell_type = [init_cell_type]*cur_cell.shape[0]
            else:
                cur_cell_type = get_cell_type(cur_cell, model_anno, time_point[i], label_to_cell_type_map)
    
        all_cell_type.append(cur_cell_type)
    
        if i < len(time_step):
            cur_cell, cur_cell_id = growth(cur_cell, time_step[i], model, time_point[i], cur_cell_id)
            cur_cell[:, 0:spatial_dim] = cur_cell[:, 0:spatial_dim] + time_step[i] * model.spatial_velocity_net(
                torch.tensor(time_point[i], dtype=torch.float32), cur_cell)
            cur_cell[:, spatial_dim:] = cur_cell[:, spatial_dim:] + time_step[i] * model.gene_velocity_net(
                torch.tensor(time_point[i], dtype=torch.float32), cur_cell)
    return all_spa, all_exp, all_cell_type, all_cell_id, all_time_point


def evolution_forward_sim_rgb_data(init_cell, model, init_time, end_time,
                      spatial_dim=2, delta_t = 0.1, other_time_points=None):
    def growth(x, delta_t, model, t, cur_cell_id, sigma_d=0.00001):
        cell_number = x.shape[0]
        x_growth = []
        new_cell_id = []
        g = model.growth_rate_net(torch.tensor(t, dtype=torch.float32), x)
        for i in range(cell_number):
            g_i = g[i, :]
            temp = np.random.rand()
            if g_i > 0 and temp < (g_i * delta_t):
                x_growth.append(x[i, :])
                new_cell_id.append(cur_cell_id[i])
                
                new_cell = x[i, :] + sigma_d * torch.tensor(np.random.randn(x[i, :].shape[0]))
                # new_cell[0:3] = torch.maximum(new_cell[0:3], torch.tensor(0.0))
                x_growth.append(new_cell)
                new_cell_id.append(cur_cell_id[i])
            elif g_i > 0 and temp > (g_i * delta_t):
                x_growth.append(x[i, :])
                new_cell_id.append(cur_cell_id[i])
            elif g_i < 0 and temp > (-g_i * delta_t):
                x_growth.append(x[i, :])
                new_cell_id.append(cur_cell_id[i])
        cur_cell = torch.vstack(x_growth)
        cur_cell = torch.tensor(cur_cell, dtype=torch.float32)
        return cur_cell, new_cell_id
    
    time_point = np.arange(init_time, end_time, delta_t)
    time_point = np.union1d(time_point, end_time)
    if other_time_points is not None:
        time_point = np.union1d(time_point, other_time_points)
    time_point = np.round(time_point, 2)
    time_point = np.unique(time_point)
    time_step = time_point[1:] - time_point[0:-1]
    
    all_time_point = list(time_point)
    all_exp = []
    all_spa = []
    all_cell_type = []
    all_cell_id = []
    
    cur_cell = init_cell
    cur_cell_id = np.arange(cur_cell.shape[0])
    
    for i in range(len(time_point)):
        all_spa.append(cur_cell[:, 0:spatial_dim:].detach().numpy())
        all_exp.append(cur_cell[:, spatial_dim:].detach().numpy())
        all_cell_id.append(cur_cell_id)
    
        if i < len(time_step):
            cur_cell, cur_cell_id = growth(cur_cell, time_step[i], model, time_point[i], cur_cell_id)
            cur_cell[:, 0:spatial_dim] = cur_cell[:, 0:spatial_dim] + time_step[i] * model.spatial_velocity_net(
                torch.tensor(time_point[i], dtype=torch.float32), cur_cell)
            cur_cell[:, spatial_dim:] = cur_cell[:, spatial_dim:] + time_step[i] * model.gene_velocity_net(
                torch.tensor(time_point[i], dtype=torch.float32), cur_cell)
    return all_spa, all_exp, all_cell_type, all_cell_id, all_time_point