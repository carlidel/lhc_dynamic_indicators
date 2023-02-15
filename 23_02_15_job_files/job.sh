#!/bin/bash

# let's try this cvmfs distribution
source /cvmfs/sft.cern.ch/lcg/views/LCG_102b_cuda/x86_64-centos7-gcc8-opt/setup.sh
# now let's also source the local venv
source /afs/cern.ch/work/c/camontan/public/lhc_dynamic_indicators/myenv/bin/activate

# print the python path
which python

# echo the 4 arguments received by the script
echo $1
echo $2
echo $3
echo $4

# run the simulation
python3 /afs/cern.ch/work/c/camontan/public/lhc_dynamic_indicators/run_sim_normed.py --mask $1 --tracking $2 --kind $3 --output $3 --zeta $4

# copy the output to eos
eos cp *.h5 /eos/user/c/camontan/lhc_dynamic_data

# remove the output
rm *.h5