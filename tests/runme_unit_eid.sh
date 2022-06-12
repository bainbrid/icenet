#!/bin/sh
#
# Execute training and evaluation for electron ID
#
# Run with: source runme.sh

DATAPATH="./travis-stash/input/iceid"

# Use * or other glob wildcards for filenames
#CONFIG="unit_rw_none"
#python analysis/eid_train.py --config $CONFIG --datapath $DATAPATH --datasets "output_{0,1,2,3,4}" #,1 #,2,3,4,5,6
#python analysis/eid_eval.py  --config $CONFIG --datapath $DATAPATH --datasets "output_0" #,1 #,2,3,4,5,6

CONFIG="unit_rw_background"
python analysis/eid_train.py --config $CONFIG --datapath $DATAPATH --datasets "output_{0,1,2,3,4}.root" #,1 #,2,3,4,5,6
python analysis/eid_eval.py  --config $CONFIG --datapath $DATAPATH --datasets "output_0.root" #,1 #,2,3,4,5,6

CONFIG="unit_rw_signal"
python analysis/eid_train.py --config $CONFIG --datapath $DATAPATH --datasets "output_{0,1,2,3,4}.root" #,1 #,2,3,4,5,6
python analysis/eid_eval.py  --config $CONFIG --datapath $DATAPATH --datasets "output_0.root" #,1 #,2,3,4,5,6

#python analysis/eid_visual_tensors.py --config $CONFIG --datapath $DATAPATH --datasets "output_0" #,1 #,2,3,4,5,6
