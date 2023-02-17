#!/bin/bash

export EOS_MGM_URL=root://eosuser.cern.ch
export MYPYTHON=/afs/cern.ch/work/c/camontan/public/anaconda3

unset PYTHONHOME
unset PYTHONPATH
source $MYPYTHON/bin/activate
export PATH=$MYPYTHON/bin:$PATH

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
