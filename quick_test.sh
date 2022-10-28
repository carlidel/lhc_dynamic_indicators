#!/bin/bash

python run_sim.py --mask configs/mask1.json --tracking configs/quick_tracking.json --kind stability --zeta min --output stability &\
python run_sim.py --mask configs/mask1.json --tracking configs/quick_tracking.json --kind rem --zeta min --output rem &\
python run_sim.py --mask configs/mask1.json --tracking configs/quick_tracking.json --kind stability --zeta min --output stability