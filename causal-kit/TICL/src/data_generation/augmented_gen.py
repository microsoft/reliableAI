import os
import shutil
import pickle
import math
import random
import numpy as np
from data_generation.intervention_gen import InterventionalDataGenerator


class AugmentedDatasetGenerator:
    def __init__(self, args):
        self._unpack_basic_paramters(args)
        self.folder_dir = f"../datasets/experiment/original/{self.exp_number}"
        self.origin_adj_matrix = self._get_adj_matrix(self.bn)
        self.generator = InterventionalDataGenerator(self.bn, self.intervention_sample, self.observation_sample, self.intervention_type, self.unknown_type)
    
    def _unpack_basic_paramters(self, args):
        self.benchmark_name = args.benchmark_name
        self.bn = args.bn
        self.support_type = args.support_type
        self.intervention_type = args.intervention_type
        self.unknown_type = args.unknown_type
        self.observation_sample = args.observation_sample
        self.intervention_sample = args.intervention_sample
        self.intervention_size = args.intervention_size
        self.flat_rate  = args.flat_rate
        self.exp_number = args.exp_number
    
    def _get_adj_matrix(self, bn):
        adj_matrix = np.zeros((len(bn.nodes), len(bn.nodes)), dtype=int)
        node_name_to_index_dict = {node: idx for idx, node in enumerate(list(bn.nodes))}
        for edge in list(bn.edges):
            adj_matrix[node_name_to_index_dict[edge[0]], node_name_to_index_dict[edge[1]]] = 1
        return adj_matrix
    
    def _choose_randomly_intervention_nodes(self, nodes, adj_matrix, intervention_size, num_nodes=1, on_child_only=True):
        """Pick randomly nodes

        Args:
            num_nodes(int): number of nodes to intervene on
            on_child_only(boolean): if True (always), exclude root nodes from the selection
        Returns:
            list: contains the indices of the chosen nodes
        """
        if on_child_only:
            num_parent_idx = np.sum(adj_matrix, axis=0) > 0
            avail_nodes = list(np.array(nodes)[num_parent_idx])
        else:
            avail_nodes = nodes
        # assert num_nodes <= len(avail_nodes) and intervention_size <= math.comb(len(avail_nodes), num_nodes), "num_nodes has to been smaller or equal to the total number of avail nodes"
        assert len(avail_nodes) >= 3, "num_nodes has to been smaller or equal to the total number of avail nodes"
        intervention_tartget_list = [sorted(np.random.choice(avail_nodes, num_nodes, replace=False)) for _ in range(intervention_size)]
        return intervention_tartget_list
    
    def get_augmented_bn_dataset(self, logger):
        # Generate intervention target families and corresponding datasets
        if self.support_type == 'single':
            self.targets_list = self._choose_randomly_intervention_nodes(list(self.bn.nodes), self.origin_adj_matrix, self.intervention_size, 1)
        elif self.support_type == 'multiple':
            self.targets_list = self._choose_randomly_intervention_nodes(list(self.bn.nodes), self.origin_adj_matrix, self.intervention_size, random.randint(1,3))
        self.generator.generate_intervention_dataset(self.targets_list, self.flat_rate, logger)
    
    def save_augdataset_and_groundtruth(self, logger):
        # First, form a augmented dataset and raw dataset list
        aug_dataset_list = []
        raw_dataset_list = []
        for idx, one_bn_dict in enumerate(self.generator.all_bn_list):
            sys_dataset = one_bn_dict['data']
            env_dataset = np.zeros((len(sys_dataset), self.intervention_size), dtype=int)
            if idx != 0:
                env_dataset[:, idx-1] = 1
            raw_dataset_list.append(sys_dataset)
            aug_dataset_list.append(np.append(sys_dataset, env_dataset, axis=1))
        aug_dataset = np.vstack(aug_dataset_list)
        
        # Second, form a whole augmented graph and intervention targets
        aug_adj_matrix = np.zeros((len(self.bn.nodes)+self.intervention_size, len(self.bn.nodes)+self.intervention_size))
        node_name_to_index_dict = {node: idx for idx, node in enumerate(list(self.bn.nodes))}
        print(node_name_to_index_dict)
        
        for edge in list(self.bn.edges):
            aug_adj_matrix[node_name_to_index_dict[edge[0]], node_name_to_index_dict[edge[1]]] = 1
                
        transform_target_list = [[]]
        for idx, target in enumerate(self.targets_list):
            transform_targets = []
            for node in target:
                aug_adj_matrix[len(self.bn.nodes)+idx, node_name_to_index_dict[node]] = 1
                transform_targets.append(node_name_to_index_dict[node])
            transform_target_list.append(transform_targets)
        logger.info(f'transform intervention targets lists: {transform_target_list}')
        
        # Third, save the aug_dataset, aug_adj_matrix, raw_dataset, and intervention targets
        os.makedirs(f'{self.folder_dir}/aug_samples', exist_ok=True)
        with open(f'{self.folder_dir}/aug_samples/{self.benchmark_name}_aug_dataset.npy', 'wb') as f:
            np.save(f, aug_dataset)
        
        os.makedirs(f'{self.folder_dir}/aug_graphs', exist_ok=True)
        with open(f'{self.folder_dir}/aug_graphs/{self.benchmark_name}_aug_graph.txt', 'wb') as f:
            np.savetxt(f, aug_adj_matrix, fmt='%i')
        
        os.makedirs(f'{self.folder_dir}/raw_samples', exist_ok=True)
        with open(f'{self.folder_dir}/raw_samples/{self.benchmark_name}_raw_dataset.pkl', 'wb') as f:
            pickle.dump(raw_dataset_list, f)
        
        os.makedirs(f'{self.folder_dir}/int_targets', exist_ok=True)
        with open(f'{self.folder_dir}/int_targets/{self.benchmark_name}_int_targets.pkl', 'wb') as f:
            pickle.dump(transform_target_list, f)
        
        logger.info(f'Finshed augmented datasets genetation on {self.benchmark_name}!\n')