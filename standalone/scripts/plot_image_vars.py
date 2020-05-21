# python 2 and 3 compatibility, pip install future and six
from __future__ import print_function
from future.utils import raise_with_traceback
import future
import builtins
import past
import six

from argparse import ArgumentParser
import os
import uproot
import numpy as np
import pandas as pd

################################################################################
# -100. --> nan
# charge*phi

# matplotlib imports 
import matplotlib
matplotlib.use('Agg') # choose backend before doing anything else with pyplot! 
#matplotlib.use('macosx')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=False)
#rc('text', usetex=True)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.font_manager import FontProperties
import mplhep as hep
#plt.style.use(hep.style.CMS)

################################################################################
print("##### Command line args #####")

parser = ArgumentParser()
parser.add_argument('--verbose',action='store_true')
parser.add_argument('--nevents',default=1,type=int)
args = parser.parse_args()
print("Command line args:",vars(args))

################################################################################
print("##### Define inputs #####")

print(os.getcwd())
assert os.getcwd().endswith("icenet/standalone/scripts"), print("You must execute this script from within the 'icenet/standalone/scripts' dir!")

# I/O directories
input_data='../data'
print("input_data:",input_data)
input_base=os.getcwd()+"/../input"
output_base=os.getcwd()+"/../output"
if not os.path.isdir(input_base) : 
   raise_with_traceback(ValueError('Could not find input_base "{:s}"'.format(input_base)))
print("input_base:",input_base)
if not os.path.isdir(output_base) : 
   os.makedirs(output_base)
print("output_base:",output_base)
   
files = [input_data+'/test.root']

################################################################################
print("##### Define features #####")

features = [
   'eid_rho',
   'eid_ele_pt',
   'eid_sc_eta',
   'eid_shape_full5x5_sigmaIetaIeta',
   'eid_shape_full5x5_sigmaIphiIphi',
   'eid_shape_full5x5_circularity',
   'eid_shape_full5x5_r9',
   'eid_sc_etaWidth',
   'eid_sc_phiWidth',
   'eid_shape_full5x5_HoverE',
   'eid_trk_nhits',
   'eid_trk_chi2red',
   'eid_gsf_chi2red',
   'eid_brem_frac',
   'eid_gsf_nhits',
   'eid_match_SC_EoverP',
   'eid_match_eclu_EoverP',
   'eid_match_SC_dEta',
   'eid_match_SC_dPhi',
   'eid_match_seed_dEta',
   'eid_sc_E',
   'eid_trk_p',
   'gsf_bdtout1'
]

additional = [
   'gen_pt','gen_eta', 
   'trk_pt','trk_eta','trk_charge','trk_dr',
   'gsf_pt','gsf_eta','gsf_dr','gsf_bdtout2',
   'ele_pt','ele_eta','ele_dr','ele_mva_value','ele_mva_id',
   'evt','weight'
]

labelling = [
   'is_e','is_egamma',
   'has_trk','has_seed','has_gsf','has_ele',
   'seed_trk_driven','seed_ecal_driven'
]

columns = features + additional + labelling
columns = list(set(columns))

################################################################################
print("##### Load files #####")

file = files[0]
tree = uproot.open(file).get('ntuplizer/tree')
print("tree.numentries:",tree.numentries)

import collections
events = tree.arrays(['is_e','image_*'], outputtype=collections.namedtuple)

print('\n'.join([str(key) for key in tree.keys() if key.startswith(b'image_')]))

################################################################################
# reduce to [-pi,pi]
def limit_phi(phi) :
   my_func = np.vectorize( lambda x : x 
                           if x <= np.pi 
                           else x - round(x/(2.*np.pi))*2.*np.pi )
   return my_func(phi)

is_e = (events.is_e == True)
is_gsf = events.image_gsf_pt_inner > 0.
mask = is_e & is_gsf

eta_inner  = events.image_gsf_eta_inner
eta_proj   = events.image_gsf_eta_proj
eta_atcalo = events.image_gsf_eta_atcalo
eta_atcalo_tmp = np.where(eta_atcalo < -np.pi,           # if value is -10.
                          np.full(eta_atcalo.shape,10.), # replace with +10.
                          eta_atcalo)                    # or keep original

image_gsf_eta_min_ = np.minimum(eta_inner,eta_proj)
image_gsf_eta_max_ = np.maximum(eta_inner,eta_proj)
image_gsf_eta_del_ = image_gsf_eta_max_ - image_gsf_eta_min_

image_gsf_eta_min = np.minimum(image_gsf_eta_min_,eta_atcalo_tmp)
image_gsf_eta_max = np.maximum(image_gsf_eta_max_,eta_atcalo)
image_gsf_eta_del = image_gsf_eta_max - image_gsf_eta_min

print('eta') 
print([ "{:5.2f} ".format(x) for x in eta_inner[mask][:5]],"inner") 
print([ "{:5.2f} ".format(x) for x in eta_proj[mask][:5]],"proj") 
print([ "{:5.2f} ".format(x) for x in eta_atcalo[mask][:5]],"atcalo") 
print([ "{:5.2f} ".format(x) for x in image_gsf_eta_min_[mask][:5]],"min_")
print([ "{:5.2f} ".format(x) for x in image_gsf_eta_max_[mask][:5]],"max_")
print([ "{:5.2f} ".format(x) for x in image_gsf_eta_del_[mask][:5]],"max_")
print([ "{:5.2f} ".format(x) for x in image_gsf_eta_min[mask][:5]],"min")
print([ "{:5.2f} ".format(x) for x in image_gsf_eta_max[mask][:5]],"max")
print([ "{:5.2f} ".format(x) for x in image_gsf_eta_del[mask][:5]],"max")

mask = (events.is_e == True) & (events.image_gsf_pt_inner > 0.)
phi_inner  = events.image_gsf_phi_inner
phi_proj   = events.image_gsf_phi_proj
phi_atcalo = events.image_gsf_phi_atcalo
phi_atcalo_tmp = np.where(phi_atcalo < -np.pi,           # if value is -10.
                          np.full(phi_atcalo.shape,10.), # replace with +10.
                          phi_atcalo)                    # or keep original

image_gsf_phi_min_ = np.minimum(phi_inner,phi_proj)
image_gsf_phi_max_ = np.maximum(phi_inner,phi_proj)
image_gsf_phi_del_ = limit_phi(image_gsf_phi_max_ - image_gsf_phi_min_)

image_gsf_phi_min = np.minimum(image_gsf_phi_min_,phi_atcalo_tmp)
image_gsf_phi_max = np.maximum(image_gsf_phi_max_,phi_atcalo)
image_gsf_phi_del = limit_phi(image_gsf_phi_max - image_gsf_phi_min)

print('phi') 
print([ "{:5.2f} ".format(x) for x in phi_inner[mask][:10]],"inner") 
print([ "{:5.2f} ".format(x) for x in phi_proj[mask][:10]],"proj") 
print([ "{:5.2f} ".format(x) for x in phi_atcalo[mask][:10]],"atcalo") 
print([ "{:5.2f} ".format(x) for x in image_gsf_phi_min_[mask][:10]],"min_") 
print([ "{:5.2f} ".format(x) for x in image_gsf_phi_max_[mask][:10]],"max_") 
print([ "{:5.2f} ".format(x) for x in image_gsf_phi_del_[mask][:10]],"del_") 
print([ "{:5.2f} ".format(x) for x in image_gsf_phi_min[mask][:10]],"min") 
print([ "{:5.2f} ".format(x) for x in image_gsf_phi_max[mask][:10]],"max") 
print([ "{:5.2f} ".format(x) for x in image_gsf_phi_del[mask][:10]],"del") 

################################################################################
# all image_* variables

cols = [key.decode("utf-8") for key in tree.keys() if key.startswith(b'image_')]
for col in cols :
   print("histogram:",col)
   counts,edges = np.histogram( tree[col].array()[is_e&is_gsf].flatten(), bins=100 )
   f, ax = plt.subplots()
   hep.histplot(counts,edges)
   plt.xlabel(col)
   plt.ylabel('Counts/bin')
   plt.yscale('log')
   plt.ylim(0.5, counts.max()*2.)
   # hep.cms.text("Internal")
   # hep.mpl_magic()
   plt.savefig('../output/plots_image/{:s}.pdf'.format(col))
   plt.close()

################################################################################
# min, max, and diff or eta and phi
histos = {
   "image_gsf_eta_max_":image_gsf_eta_max_[is_e&is_gsf],
   "image_gsf_eta_min_":image_gsf_eta_min_[is_e&is_gsf],
   "image_gsf_eta_del_":image_gsf_eta_del_[is_e&is_gsf],
   "image_gsf_phi_max_":image_gsf_phi_max_[is_e&is_gsf],
   "image_gsf_phi_min_":image_gsf_phi_min_[is_e&is_gsf],
   "image_gsf_phi_del_":image_gsf_phi_del_[is_e&is_gsf],
   "image_gsf_eta_max":image_gsf_eta_max[is_e&is_gsf],
   "image_gsf_eta_min":image_gsf_eta_min[is_e&is_gsf],
   "image_gsf_eta_del":image_gsf_eta_del[is_e&is_gsf],
   "image_gsf_phi_max":image_gsf_phi_max[is_e&is_gsf],
   "image_gsf_phi_min":image_gsf_phi_min[is_e&is_gsf],
   "image_gsf_phi_del":image_gsf_phi_del[is_e&is_gsf],
   }

for col,val in histos.items() :
   print("histogram:",col)
   counts,edges = np.histogram( val, bins=100 )
   f, ax = plt.subplots()
   hep.histplot(counts,edges)
   plt.xlabel(col)
   plt.ylabel('Counts/bin')
   plt.yscale('log')
   plt.ylim(0.5, counts.max()*2.)
   # hep.cms.text("Internal")
   # hep.mpl_magic()
   plt.savefig('../output/plots_image/{:s}.pdf'.format(col))
   plt.close()

################################################################################
# histograms 2d

def histo_2d(x,y,cut,xlabel,ylabel,title,xlim=(None,None),ylim=(None,None)) :
   #print("histogram:",x,"VS",y,"CUT",cut)
   f, ax = plt.subplots()
   ax.scatter(x[cut],y[cut]);
   plt.xlim(xlim)
   plt.ylim(ylim)
   plt.xlabel(xlabel)
   plt.ylabel(ylabel)
   plt.title(title)
   # hep.cms.text("Internal")
   # hep.mpl_magic()
   plt.savefig('../output/plots_image/2d/{:s}_VS_{:s}_CUT_{:s}.pdf'.format(xlabel,ylabel,title))
   plt.close()

histos2d = {
   ( "image_gsf_pt_inner",
     "image_gsf_phi_del_",
     "is_e&is_gsf&(events.image_gsf_pt_inner<1.0)" ) :
      ( events.image_gsf_pt_inner,
        image_gsf_phi_del_,
        is_e&is_gsf&(events.image_gsf_pt_inner<1.0) ),
   ( "image_gsf_pt_inner",
     "image_gsf_phi_del", 
     "np.invert(is_e)&is_gsf&(events.image_gsf_pt_inner<1.0)" ) :
      ( events.image_gsf_pt_inner,
        image_gsf_phi_del,
        np.invert(is_e)&is_gsf&(events.image_gsf_pt_inner<1.0) ),
   }

for (xlabel,ylabel,title),(x,y,cut) in histos2d.items() :
   histo_2d(x,y,cut,xlabel,ylabel,title)
