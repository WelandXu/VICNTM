#!/bin/bash
seeds=(2022 41 42 100 200 1 2 2023 2024 2025)
# seeds=(2022)
pyfile='./run_scholar_vic.py'
input1='data/20ng_min_df_100'
input2='data/IMDb_min_df_100'
input3='data/wiki_min_df_100'


##################################20NG

for seed in ${seeds[@]};
do
    python $pyfile $input1 --seed $seed --o output/20ng/vic/bs50 --sim_coeff 12.0 --std_coeff 12.0 --cov_coeff 2.0 --batch-size 50
done
python test_npmi_vic.py 20ng --sim_coeff 12.0 --std_coeff 12.0 --cov_coeff 2.0 --o output/20ng/vic/bs50

# for seed in ${seeds[@]};
# do
#     python $pyfile $input1 --seed $seed --o output/20ng/vic/K200/bs50 --sim_coeff 12.0 --std_coeff 12.0 --cov_coeff 2.0 --batch-size 50 --alpha 0.2 -k 200
# done
# python test_npmi_vic.py 20ng --sim_coeff 12.0 --std_coeff 12.0 --cov_coeff 2.0 --o output/20ng/vic/K200/bs50 -k 200
##################################

################################## IMDb
# for seed in ${seeds[@]};
# do
#     python $pyfile $input2 --seed $seed --o output/IMDb/vic/bs1000 --sim_coeff 12.0 --std_coeff 12.0 --cov_coeff 2.0 --batch-size 1000
# done
# python test_npmi_vic.py IMDb --sim_coeff 12.0 --std_coeff 12.0 --cov_coeff 2.0 --o output/IMDb/vic/bs1000

# for seed in ${seeds[@]};
# do
#     python $pyfile $input2 --seed $seed --o output/IMDb/vic/K200/bs1000 --sim_coeff 12.0 --std_coeff 12.0 --cov_coeff 2.0 --batch-size 1000 --alpha 0.5 -k 200
# done
# python test_npmi_vic.py IMDb --sim_coeff 12.0 --std_coeff 12.0 --cov_coeff 2.0 --o output/IMDb/vic/K200/bs1000 -k 200

################################## Wiki
# for seed in ${seeds[@]};
# do
#     python $pyfile $input3 --seed $seed --o output/wiki/vic/bs250 --sim_coeff 12.0 --std_coeff 12.0 --cov_coeff 2.0 --batch-size 250 --alpha 0.01
# done
# python test_npmi_vic.py wiki --sim_coeff 12.0 --std_coeff 12.0 --cov_coeff 2.0 --o output/wiki/vic/bs250

# for seed in ${seeds[@]};
# do
#     python $pyfile $input3 --seed $seed --o output/wiki/vic/K200/bs250/a02 --sim_coeff 12.0 --std_coeff 12.0 --cov_coeff 2.0 --batch-size 250 --alpha 0.2 -k 200
# done
# python test_npmi_vic.py wiki --sim_coeff 12.0 --std_coeff 12.0 --cov_coeff 2.0 --o output/wiki/vic/K200/bs250/a02 -k 200

# for seed in ${seeds[@]};
# do
#     python $pyfile $input3 --seed $seed --o output/simpleTA/wiki/vic/K200/bs250 --sim_coeff 12.0 --std_coeff 12.0 --cov_coeff 2.0 --batch-size 250 --positive aug_texts.pt --alpha 0.1 -k 200
# done
# python test_npmi_vic.py wiki --sim_coeff 12.0 --std_coeff 12.0 --cov_coeff 2.0 --o output/simpleTA/wiki/vic/K200/bs250/a01 -k 200

