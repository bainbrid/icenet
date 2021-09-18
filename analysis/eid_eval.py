# Electron ID [EVALUATION] code
#
# m.mieskolainen@imperial.ac.uk, 2021


# icenet system paths
import _icepaths_

import numpy as np

# icenet
from icenet.tools import aux
from icenet.tools import process

# iceid
from iceid import common
from iceid import graphio


def ele_mva_classifier(data, data_tensor=None, data_kin=None, data_graph=None, args=None):
    """
    External classifier directly from the root tree
    """

    varname = 'ele_mva_value_depth15'

    print(f'\nEvaluate <{varname}> classifier ...')
    try:
        y    = np.array(data.tst.y, dtype=np.float)
        yhat = np.array(data.tst.x[:, data.ids.index(varname)], dtype=np.float)

        return aux.Metric(y_true = y, y_soft = yhat)
    except:
        print(__name__ + 'Variable not found')


# Main function
#
def main() :

    ## Get input
    data, args, features = common.init()

    # Evaluate external classifier
    met_elemva = ele_mva_classifier(data=data)
    
    # Add to the stack
    process.roc_mstats.append(met_elemva)
    process.roc_labels.append('elemva15')


    ## Parse graph network data
    data_graph = {}
    if args['graph_on']:
        data_graph['tst'] = graphio.parse_graph_data(X=data.tst.x, Y=data.tst.y, ids=data.ids,
            features=features, global_on=args['graph_param']['global_on'], coord=args['graph_param']['coord'])

    ## Split and factor data
    data, data_tensor, data_kin = common.splitfactor(data=data, args=args)

    ## Evaluate classifiers
    process.evaluate_models(outputname='eid', \
        data=data, data_tensor=data_tensor, data_kin=data_kin, data_graph=data_graph, args=args)
    
    print(__name__ + ' [Done]')


if __name__ == '__main__' :

    main()
