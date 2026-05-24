import os
import copy
import time
import pickle
import numpy as np
from itertools import product
from pathos.multiprocessing import Pool, cpu_count
from base.dag import DiGraph
from base.pdag import MixedGraph
from base.dataloader import Dataset
from base.citester import CITester
from tools.utils import Kernel_Embedding, meanstdmaxmin


class DirectionFeatureExtractor:
    def __init__(self, train_dir, infer_dir, feature_dir) -> None:
        self.train_dir = train_dir
        self.infer_dir = infer_dir
        self.feature_dir = feature_dir
    

    def _get_one_trainset(self, sibling_name):
        one_data_path = f"{self.train_dir}/{sibling_name}.npy"
        one_graph_path = f"{self.train_dir}/{sibling_name}.txt"

        # get all tfork features
        dataset = Dataset(one_data_path)
        cit = CITester(dataset.IndexedDataT, maxCountOfSepsets=50)
        
        dag = MixedGraph(graph_path=one_graph_path)
        tforks = dag.tforks
        PC_of_nodes = {node: dag.getPC(node) for node in dag.NodeIDs}
                
        ke = Kernel_Embedding()
        
        all_features = []
        for tf in tforks:
            T, X, Y = tf
            PCT, PCX, PCY = PC_of_nodes[T], PC_of_nodes[X], PC_of_nodes[Y]
            PCX_minus_T = PCX - {T}
            PCY_minus_T = PCY - {T}
            PCX_PCY_pairs = [_ for _ in {tuple(sorted(tp)) for tp in product(PCX_minus_T, PCY_minus_T)} if _[0] != _[1] and not dag.adjacent_in_mixed_graph(_[0], _[1])]  # sort and set to remove repeat, add this restrict: no adjacent pair
            Xpcypairs = [_ for _ in product({X}, PCY_minus_T) if not dag.adjacent_in_mixed_graph(_[0], _[1])]
            Ypcxpairs = [_ for _ in product({Y}, PCX_minus_T) if not dag.adjacent_in_mixed_graph(_[0], _[1])]
            raw_feature = cit.ExtractTForkFeatureBasedOnPC(T, X, Y, PCT, PCX, PCY, Xpcypairs, Ypcxpairs, PCX_PCY_pairs)
            
            scalings_overlaps = raw_feature[:12] # 12 = 5 scalings + 7 overlaps
            est_conds = raw_feature[12:]
            unitary_XY_T = list(est_conds[0][0][0])
            processed_fts = copy.copy(scalings_overlaps + unitary_XY_T)
            
            unitary_flag = True
            for condlist in est_conds:
                for estlist in condlist:
                    if unitary_flag: # jump the first unitary, no need to percentile
                        unitary_flag = False; continue
                    pvals = [_[0] for _ in estlist]
                    svrts = [_[1] for _ in estlist]
                    
                    meanstdmaxmin_values = [len(pvals)] + meanstdmaxmin(pvals) + meanstdmaxmin(svrts) # length=1+4+4=9
                    pval_embd = ke.get_empirical_embedding(pvals) # length=15
                    svrt_embd = ke.get_empirical_embedding(svrts) # length=15
                    processed_fts.extend(meanstdmaxmin_values + pval_embd + svrt_embd)
            
            all_features.append([T, X, Y] + processed_fts)
        
        one_sibling_features = np.array(all_features)
        
        # get all tfork labels
        one_sibling_labels = np.array([tuple(tf) in dag.vstrucs for tf in tforks])

        feature_path =  f"{self.feature_dir}/{sibling_name}.pkl"
        with open(feature_path, 'wb') as fp:
            pickle.dump((one_sibling_features, one_sibling_labels), fp)
    
    
    def get_trainsets(self, logger):
        all_siblings_names = [file_path.split('.')[0] for file_path in os.listdir(self.train_dir) if file_path.endswith('.npy')][:200]  # TODO speed up!
        
        logger.info(f"Start feature extraction...")
        start_time = time.time()
        with Pool(cpu_count()) as p:
            p.map(self._get_one_trainset, all_siblings_names)
        end_time = time.time()
        logger.info(f"UT feature extraction total time is {end_time - start_time}.")
        
    
    def get_all_tfork_features(self, data_path, graph_path):
        dataset = Dataset(data_path)
        cit = CITester(dataset.IndexedDataT, maxCountOfSepsets=50)

        skeleton = MixedGraph(graph_path=graph_path)
        tforks = skeleton.tforks
        PC_of_nodes = {node: skeleton.getPC(node) for node in skeleton.NodeIDs}
                
        ke = Kernel_Embedding()
        
        all_features = []
        for tf in tforks:
            T, X, Y = tf
            PCT, PCX, PCY = PC_of_nodes[T], PC_of_nodes[X], PC_of_nodes[Y]
            PCX_minus_T = PCX - {T}
            PCY_minus_T = PCY - {T}
            PCX_PCY_pairs = [_ for _ in {tuple(sorted(tp)) for tp in product(PCX_minus_T, PCY_minus_T)} if _[0] != _[1] and not skeleton.adjacent_in_mixed_graph(_[0], _[1])]  # sort and set to remove repeat, add this restrict: no adjacent pair
            Xpcypairs = [_ for _ in product({X}, PCY_minus_T) if not skeleton.adjacent_in_mixed_graph(_[0], _[1])]
            Ypcxpairs = [_ for _ in product({Y}, PCX_minus_T) if not skeleton.adjacent_in_mixed_graph(_[0], _[1])]
            raw_feature = cit.ExtractTForkFeatureBasedOnPC(T, X, Y, PCT, PCX, PCY, Xpcypairs, Ypcxpairs, PCX_PCY_pairs)
            
            scalings_overlaps = raw_feature[:12] # 12 = 5 scalings + 7 overlaps
            est_conds = raw_feature[12:]
            unitary_XY_T = list(est_conds[0][0][0])
            processed_fts = copy.copy(scalings_overlaps + unitary_XY_T)

            unitary_flag = True
            for condlist in est_conds:
                for estlist in condlist:
                    if unitary_flag: # jump the first unitary, no need to percentile
                        unitary_flag = False; continue
                    pvals = [_[0] for _ in estlist]
                    svrts = [_[1] for _ in estlist]
                    meanstdmaxmin_values = [len(pvals)] + meanstdmaxmin(pvals) + meanstdmaxmin(svrts) # length=1+4+4=9
                    pval_embd = ke.get_empirical_embedding(pvals) # length=15
                    svrt_embd = ke.get_empirical_embedding(svrts) # length=15
                    processed_fts.extend(meanstdmaxmin_values + pval_embd + svrt_embd)
            
            all_features.append([T, X, Y] + processed_fts)
        
        return np.array(all_features)

