import os
import shutil
import warnings
import argparse
import json
import random
import numpy as np
from tqdm import tqdm
from data_generation.augmented_gen import AugmentedDatasetGenerator
from proxy_generation.sibling_gen import SiblingGraphGenerator
from skeleton_learning.skeleton_learner import SkeletonTrainer
from direction_learning.direction_learner import DirectionTrainer
from tools.utils import getLogger
from pgmpy.readwrite import BIFReader


def parser_args():
    parser = argparse.ArgumentParser(description='iSCL')
    
    ################ config of basic setup ####################
    parser.add_argument("--benchmark_names_list", type=list, default=['01earthquake', '02survey', '03asia', '04sachs',  '05child', '06insurance', '07water', '08mildew', '09alarm', '10barley', '11hailfinder', '12hepar2', '13win95pts', '14pathfinder'], help="")
    parser.add_argument("--exp_number", type=str, default="exp_200", help="")
    parser.add_argument("--seed", type=int, default="42", help="")
    
    ################ config of stage1_data_generate ####################
    parser.add_argument("--support_type", type=str, default="single", help="single / multiple")
    parser.add_argument("--intervention_type", type=str, default="soft", help="hard / soft")
    parser.add_argument("--unknown_type", type=bool, default=True, help="unknow(True) / know(False)")
    parser.add_argument("--observation_sample", type=int, default=10000, help="")
    parser.add_argument("--intervention_sample", type=int, default=10000, help="")
    parser.add_argument("--intervention_size_ratio", type=float, default=0.2, help="")
    parser.add_argument("--flat_rate", type=float, default=0.2, help="")
    
    ################ config of stage2_proxy_generate ###################
    parser.add_argument("--purely_random", type=bool, default=False, help="purely random / IS-MCMC")
    parser.add_argument("--proxy_type", type=str, default="vicinal", help="starting poing with vicinal / random")
    parser.add_argument("--use_int_bais", type=bool, default=True, help="")
    parser.add_argument("--mcmc_step", type=int, default=20, help="")
    
    parser.add_argument("--proxy_model", type=str, default="blip", help="jci-blip / jci-hc / jci-pc")
    parser.add_argument("--case_num", type=int, default=2000, help="100 / 200 / 400 / 800 / 1600")
    parser.add_argument("--proxy_degree", type=float, default=0.2, help="")
    parser.add_argument("--mixed_samples", type=int, default=20000, help="2k / 5k / 10k / 20k")
    
    ################ config of stage3_skeleton_learn ###################
    parser.add_argument("--contain_env", type=bool, default=True, help="")
    parser.add_argument("--skeleton_train_type", type=str, default="together", help="together / separate")
    parser.add_argument("--decision_thres", type=float, default=0.5, help="")
    
    ################ config of stage4_direction_learn ##################
    parser.add_argument("--edge_threshold", type=float, default=0.1, help="")
    
    args = parser.parse_args()
    return args


def stage1_data_generate(args, logger):
    logger.info(f"{'-'*50} Start Stage1: {'-'*50} \n")
    data_generator = AugmentedDatasetGenerator(args)
    data_generator.get_augmented_bn_dataset(logger)
    data_generator.save_augdataset_and_groundtruth(logger)


def stage2_proxy_generate(args, logger):
    logger.info(f"{'-'*50} Start Stage2: {'-'*50} \n")
    generator = SiblingGraphGenerator(args)
    generator.generate_sibling(logger)


def stage3_skeleton_learn(args, logger):
    logger.info(f"{'-'*50} Start Stage3: {'-'*50} \n")
    skeleton_trainer = SkeletonTrainer(args)
    skeleton_trainer.process_pipeline(logger)


def stage4_direction_learn(args, logger):
    logger.info(f"{'-'*50} Start Stage4: {'-'*50} \n")
    direction_trainer = DirectionTrainer(args)
    direction_trainer.process_pipeline(logger)


def main(args, logger):

    # Fixed random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # # Clean up remaining experimental files
    if os.path.exists(f'../datasets/experiment/siblings/{args.exp_number}'):
        shutil.rmtree(f'../datasets/experiment/siblings/{args.exp_number}')
    
    for idx in tqdm(range(len(args.benchmark_names_list))[0:]):
        # Fixed specific parameters
        args.benchmark_name = args.benchmark_names_list[idx]
        
        logger.info('*'*100 + str(args.benchmark_name) + '*'*100 + '\n')
        
        args.bn = BIFReader(path=f"../datasets/raw/bif/{args.benchmark_name}.bif").get_model()
        args.intervention_size = round(len(args.bn.nodes) * args.intervention_size_ratio)
        args.mixed_samples = args.observation_sample + args.intervention_sample * args.intervention_size
        
        # start iml4c
        stage1_data_generate(args, logger)
        stage2_proxy_generate(args, logger)
        stage3_skeleton_learn(args, logger)
        stage4_direction_learn(args, logger)


if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")
    args = parser_args()
    logger = getLogger(args.exp_number)
    logger.info(f"The parameter configuration for this experiment is as follows:\n{json.dumps(vars(args), indent=4)}\n")
    if args.purely_random == True:
        args.proxy_type = 'purelyrandom'
    main(args, logger)
