import numpy as np
from tools.sampling import sample_data_from_bn
from pgmpy.factors.discrete import TabularCPD


class InterventionalDataGenerator:
    def __init__(self, bn, intervention_sample, observation_sample, intervention_type, unknown_type):
        self.bn = bn
        self.intervention_sample = intervention_sample
        self.observation_sample = observation_sample
        self.intervention_type = intervention_type
        self.unknown_type = unknown_type
        self.all_bn_list = []
    
    
    def generate_intervention_dataset(self, targets_list, flat_rate, logger):
        """
        Applying interventions to obtain a list of observations and intervention families combined
        Args:
            targets_list: a list of different targets case
            flat_rate: flat rate controls how flat the intervened cpd will be
            logger: record Log
        Returns:
            None
        """
        self.all_bn_list.append({'targets': [], 'bn': self.bn, 'data':sample_data_from_bn(self.bn, self.observation_sample)})        
        
        for targets in targets_list:
            logger.info(f"Apply one {self.intervention_type} intervention with targets = {targets}")
            
            hard_intervention, soft_intervention = {}, []
            for var in targets:
                if self.intervention_type == 'hard':
                    hard_intervention[var] = self.bn.states[var][np.random.randint(0, self.bn.get_cardinality(var))]
                elif self.intervention_type == 'soft':
                    cardinality=self.bn.get_cardinality(var)
                    values = [[x] for x in np.round(np.random.dirichlet(np.ones(cardinality) * 0.2), 2)]
                    state_names={var: self.bn.states[var]}
                    soft_intervention.append(TabularCPD(variable=var, variable_card=cardinality, values=values, state_names=state_names))
        
            if self.intervention_type == 'hard':
                intervention_data = sample_data_from_bn(self.bn, self.intervention_sample, do=hard_intervention, synthetic=False)
            elif self.intervention_type == 'soft':
                intervention_data = sample_data_from_bn(self.bn, self.intervention_sample, virtual_intervention=soft_intervention, synthetic=False)
                        
            targets_label = [] if self.unknown_type else targets
            self.all_bn_list.append({'targets': targets_label, 'bn': None, 'data':intervention_data})
        