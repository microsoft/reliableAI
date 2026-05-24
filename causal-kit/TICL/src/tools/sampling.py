from pgmpy.models.BayesianModel import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling



def sample_data_from_bn(graph, n_samples, do=None, virtual_intervention=None, synthetic=False):
    df = graph.simulate(n_samples=n_samples, do=do, virtual_intervention=virtual_intervention, seed=44, show_progress=False)
    
    if synthetic:
        df = df[[str(i) for i in range(len(df.columns))]]
    else:
        node_state_idx_dict = graph.states
        for column_name in node_state_idx_dict.keys():
            node_state_idx_dict[column_name] = {name: idx for idx, name in enumerate(node_state_idx_dict[column_name])} # replace node states with numbers
        df = df.replace(node_state_idx_dict)
    df_reordered = df.reindex(columns=graph.nodes)  # IMPORTANT! Reorder columns to match the order of the nodes
    
    return df_reordered.to_numpy().astype(int)