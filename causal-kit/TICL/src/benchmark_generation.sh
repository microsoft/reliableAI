# Real-World:

## the easiest (default)
python main.py --exp_number exp_1 --seed 1 --intervention_size_ratio 0.2 --support_type single --intervention_type hard --observation_sample 5000 --intervention_sample 200

## most difficult
python main.py --exp_number exp_2 --seed 1 --intervention_size_ratio 0.2 --support_type multiple --intervention_type soft --observation_sample 5000 --intervention_sample 200


# Effect of the sample size

### Limited observational data sample sizes
python main.py --exp_number exp_3 --seed 1 --intervention_size_ratio 0.2 --support_type single --intervention_type hard --observation_sample 500 --intervention_sample 200
python main.py --exp_number exp_4 --seed 1 --intervention_size_ratio 0.2 --support_type single --intervention_type hard --observation_sample 1000 --intervention_sample 200
python main.py --exp_number exp_5 --seed 1 --intervention_size_ratio 0.2 --support_type single --intervention_type hard --observation_sample 2000 --intervention_sample 200


### Limited interventional data sample sizes
python main.py --exp_number exp_6 --seed 1 --intervention_size_ratio 0.2 --support_type single --intervention_type hard --observation_sample 5000 --intervention_sample 20
python main.py --exp_number exp_7 --seed 1 --intervention_size_ratio 0.2 --support_type single --intervention_type hard --observation_sample 5000 --intervention_sample 50
python main.py --exp_number exp_8 --seed 1 --intervention_size_ratio 0.2 --support_type single --intervention_type hard --observation_sample 5000 --intervention_sample 100


### Limited sample sizes
python main.py --exp_number exp_9 --seed 1 --intervention_size_ratio 0.2 --support_type single --intervention_type hard --observation_sample 500 --intervention_sample 10

### Large sample size
python main.py --exp_number exp_10 --seed 1 --intervention_size_ratio 0.2 --support_type single --intervention_type hard --observation_sample 50000 --intervention_sample 2000


# Effect of the intervention ratio

python main.py --exp_number exp_11 --seed 1 --intervention_size_ratio 0.4 --support_type single --intervention_type hard --observation_sample 5000 --intervention_sample 200
python main.py --exp_number exp_12 --seed 1 --intervention_size_ratio 0.6 --support_type single --intervention_type hard --observation_sample 5000 --intervention_sample 200
python main.py --exp_number exp_13 --seed 1 --intervention_size_ratio 0.8 --support_type single --intervention_type hard --observation_sample 5000 --intervention_sample 200
python main.py --exp_number exp_14 --seed 1 --intervention_size_ratio 1.0 --support_type single --intervention_type hard --observation_sample 5000 --intervention_sample 200

