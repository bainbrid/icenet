from __future__ import print_function
from __future__ import absolute_import
import builtins
import future
from future.utils import raise_with_traceback
import past
import six

import matplotlib
matplotlib.use('Agg') # choose backend before doing anything else with pyplot! ('macosx')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=False)
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.font_manager import FontProperties

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

################################################################################
# 
def miniaod(dir,test,egamma,has_pfgsf_branches=True,AxE=True) :
   print('##### MINIAOD ##########################################################')

   #############
   # ROC CURVE #
   #############

   plt.figure(figsize=(6,6))
   ax = plt.subplot(111)
   plt.title('Efficiency and mistag rate w.r.t. GSF tracks')
   plt.xlim(1.e-3,1.1)
   plt.ylim([0., 1.02])
   plt.xlabel('FPR')
   plt.ylabel('TPR')
   ax.tick_params(axis='x', pad=10.)
   plt.gca().set_xscale('log')
   plt.grid(True)

   ########################################
   # "by chance" line
   plt.plot(np.arange(0.,1.,plt.xlim()[0]),np.arange(0.,1.,plt.xlim()[0]),'k--',lw=0.5)

   ########################################
   # Low-pT GSF tracks + ROC curves

   # pT > 0.5 GeV, VL WP for Seed BDT 
   has_gsf = (test.has_gsf) & (test.gsf_pt>0.5) & (np.abs(test.gsf_eta)<2.5)
   denom = has_gsf&test.is_e; numer = has_gsf&denom;
   gsf_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_gsf&(~test.is_e); numer = has_gsf&denom;
   gsf_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   plt.plot([gsf_fr], [gsf_eff],
            marker='o', markerfacecolor='none', markeredgecolor='red', 
            markersize=8, linestyle='none',
            label='Low-pT GSF track, pT > 0.5 GeV, VLoose Seed',
            )

   # pT > 1.0 GeV, Tight WP for Seed BDT 
   has_gsf_T = has_gsf & (test.gsf_pt>1.0) & (test.gsf_bdtout1>3.05) # Used for standard CMS data sets 
   denom = has_gsf&test.is_e; numer = has_gsf_T&denom;
   gsf_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_gsf&(~test.is_e); numer = has_gsf_T&denom;
   gsf_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   plt.plot([gsf_fr], [gsf_eff],
            marker='o', markerfacecolor='none', markeredgecolor='blue', 
            markersize=8, linestyle='none',
            label='Low-pT GSF track, pT > 1.0 GeV, Tight Seed',
            )

   ########################################
   # Low-pT GSF electrons + ROC curves

   # pT > 0.5 GeV, VL WP for Seed BDT 
   has_gsf = (test.has_gsf) & (test.gsf_pt>0.5) & (np.abs(test.gsf_eta)<2.5)
   has_ele = (test.has_ele) & (test.ele_pt>0.5) & (np.abs(test.ele_eta)<2.5)
   denom = has_gsf&test.is_e; numer = has_ele&denom;
   ele_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_gsf&(~test.is_e); numer = has_ele&denom;
   ele_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   plt.plot([ele_fr], [ele_eff],
            marker='o', markerfacecolor='red', markeredgecolor='red', 
            markersize=8,linestyle='none',
            label='Low-pT GSF electron, pT > 0.5 GeV, VLoose Seed',
            )

   # pT > 1.0 GeV, Tight WP for Seed BDT 
   has_ele = has_ele & (test.gsf_pt>1.0) & (test.gsf_bdtout1>3.05) # Used for standard CMS data sets 
   denom = has_gsf&test.is_e; numer = has_ele&denom;
   ele_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_gsf&(~test.is_e); numer = has_ele&denom;
   ele_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   plt.plot([ele_fr], [ele_eff],
            marker='o', markerfacecolor='blue', markeredgecolor='blue', 
            markersize=8, linestyle='none',
            label='Low-pT GSF electron, pT > 1.0 GeV, Tight Seed',
            )

   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_ele],test['training_out'][has_ele])
   id_auc = roc_auc_score(test.is_e[has_ele],test['training_out'][has_ele])
   plt.plot(id_fpr*ele_fr, 
            id_tpr*ele_eff,
            linestyle='solid', color='black', linewidth=1.0,
            label='ID, AUC={:.3f}'.format(id_auc))

   ########################################
   # EGamma GSF tracks and PF GSF electrons

   has_gsf = (egamma.has_gsf) & (egamma.gsf_pt>0.5) & (np.abs(egamma.gsf_eta)<2.5)
   has_pfgsf = (egamma.has_pfgsf) & (egamma.pfgsf_pt>0.5) & (np.abs(egamma.pfgsf_eta)<2.5)
   denom = has_gsf&egamma.is_e; numer = has_pfgsf&denom
   eg_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_gsf&(~egamma.is_e); numer = has_pfgsf&denom
   eg_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   plt.plot([eg_fr], [eg_eff],
            marker='o', color='green', 
            markersize=8, linestyle='none',
            label='EGamma GSF track')

   has_ele = (egamma.has_ele) & (egamma.ele_pt>0.5) & (np.abs(egamma.ele_eta)<2.5)
   denom = has_gsf&egamma.is_e; numer = has_ele&denom
   pf_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_gsf&(~egamma.is_e); numer = has_ele&denom
   pf_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   plt.plot([pf_fr], [pf_eff],
            marker='o', color='purple', 
            markersize=8, linestyle='none',
            label='PF GSF electron')

   #################
   # Working points

   id_ELE = np.abs(id_fpr*ele_fr-pf_fr).argmin()
   same_fr = test['training_out']>id_score[id_ELE]

   x,y = id_fpr[id_ELE]*ele_fr,id_tpr[id_ELE]*ele_eff
   plt.plot([x], [y], marker='o', markerfacecolor='none', markeredgecolor='black', markersize=4)
   plt.text(x, y+0.03, "T", fontsize=10, ha='center', va='center', color='black' )

   id_ELE = np.abs(id_fpr*ele_fr-pf_fr*2.).argmin()
   double_fr = test['training_out']>id_score[id_ELE]

   x,y = id_fpr[id_ELE]*ele_fr,id_tpr[id_ELE]*ele_eff
   plt.plot([x], [y], marker='o', markerfacecolor='none', markeredgecolor='black', markersize=4)
   plt.text(x, y+0.03, "L", fontsize=10, ha='center', va='center', color='black' )
   
   ##########
   # Finish up ... 
   plt.legend(loc='upper left',framealpha=None,frameon=False)
   plt.tight_layout()
   plt.savefig(dir+'/roc.pdf')
   plt.clf()

   ##############
   # EFF CURVES #
   ##############

   # Binning 
   bin_edges = np.linspace(0., 4., 8, endpoint=False)
   bin_edges = np.append( bin_edges, np.linspace(4., 8., 4, endpoint=False) )
   bin_edges = np.append( bin_edges, np.linspace(8., 10., 2, endpoint=True) )
   bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
   bin_widths = (bin_edges[1:] - bin_edges[:-1])
   bin_width = bin_widths[0]
   bin_widths /= bin_width
   #print("bin_edges",bin_edges)
   #print("bin_centres",bin_centres)
   #print("bin_widths",bin_widths)
   #print("bin_width",bin_width)

   plt.figure()
   ax = plt.subplot(111)

   has_gsf = (test.has_gsf) & (test.gsf_pt>0.5) & (np.abs(test.gsf_eta)<2.5)
   has_ele = (test.has_ele) & (test.ele_pt>0.5) & (np.abs(test.ele_eta)<2.5)
   has_ele_T = has_ele & (test.gsf_pt>1.0) & (test.gsf_bdtout1>3.05) # Used for standard CMS data sets
   has_gsf_ = (egamma.has_gsf) & (egamma.gsf_pt>0.5) & (np.abs(egamma.gsf_eta)<2.5)
   has_ele_ = (egamma.has_ele) & (egamma.ele_pt>0.5) & (np.abs(egamma.ele_eta)<2.5)
   curves = [
      {"label":"Open","var":test.gsf_pt,"mask":(test.is_e)&(has_gsf),"condition":(has_ele),"colour":"red","fill":True,"size":7,},
      {"label":"Tight Seed","var":test.gsf_pt,"mask":(test.is_e)&(has_gsf),"condition":(has_ele_T),"colour":"blue","fill":True,"size":7,},
      {"label":"PF ELE","var":egamma.gsf_pt,"mask":(egamma.is_e)&(has_gsf_),"condition":(has_ele_),"colour":"purple","fill":True,"size":7,},
      {"label":"ID (Tight)","var":test.gsf_pt,"mask":(test.is_e)&(has_gsf),"condition":(has_ele_T)&(same_fr),"colour":"black","fill":True,"size":7,},
      {"label":"ID (Loose)","var":test.gsf_pt,"mask":(test.is_e)&(has_gsf),"condition":(has_ele_T)&(double_fr),"colour":"black","fill":False,"size":7,},
      ]
             
   for idx,curve in enumerate(curves) :
      #print("label:",curve["label"])
      his_total,_ = np.histogram(curve["var"][curve["mask"]],bins=bin_edges)
      his_passed,_ = np.histogram(curve["var"][curve["mask"]&curve["condition"]],bins=bin_edges)
      ax.errorbar(x=bin_edges[:-1], #bin_centres,
                  y=[ x/y if y > 0 else 0. for x,y in zip(his_passed,his_total) ],
                  yerr=[ np.sqrt(x/y*(1-x/y)/y) if y > 0 else 0. for x,y in zip(his_passed,his_total) ],
                  #color=None,
                  label='{:s} (mean={:5.3f})'.format(curve["label"],
                                                     float(his_passed.sum())/float(his_total.sum()) \
                                                        if his_total.sum() > 0 else 0.),
                  marker='o',
                  color=curve["colour"],
                  markerfacecolor = curve["colour"] if curve["fill"] else "white",
                  markersize=curve["size"],
                  linewidth=0.5,
                  elinewidth=0.5)

   ##########
   # Finish up ... 
   plt.title('Efficiency as a function of GSF track pT')
   plt.xlabel('Transverse momentum (GeV)')
   plt.ylabel('Efficiency')
   ax.set_xlim(bin_edges[0],bin_edges[-1])
   plt.ylim([0., 1.])
   plt.legend(loc='best')
   plt.tight_layout()
   plt.savefig(dir+'/eff.pdf')
   plt.clf()

   #################
   # MISTAG CURVES #
   #################

   plt.figure()
   ax = plt.subplot(111)

   has_gsf = (test.has_gsf) & (test.gsf_pt>0.5) & (np.abs(test.gsf_eta)<2.5)
   has_ele = (test.has_ele) & (test.ele_pt>0.5) & (np.abs(test.ele_eta)<2.5)
   has_ele_T = has_ele & (test.gsf_pt>1.0) & (test.gsf_bdtout1>3.05) # Used for standard CMS data sets
   has_gsf_ = (egamma.has_gsf) & (egamma.gsf_pt>0.5) & (np.abs(egamma.gsf_eta)<2.5)
   has_ele_ = (egamma.has_ele) & (egamma.ele_pt>0.5) & (np.abs(egamma.ele_eta)<2.5)
   curves = [
      {"label":"Open","var":test.gsf_pt,"mask":(~test.is_e)&(has_gsf),"condition":(has_ele),"colour":"red","fill":True,"size":7,},
      {"label":"Tight Seed","var":test.gsf_pt,"mask":(~test.is_e)&(has_gsf),"condition":(has_ele_T),"colour":"blue","fill":True,"size":7,},
      {"label":"PF ELE","var":egamma.gsf_pt,"mask":(~egamma.is_e)&(has_gsf_),"condition":(has_ele_),"colour":"purple","fill":True,"size":7,},
      {"label":"ID (Tight)","var":test.gsf_pt,"mask":(~test.is_e)&(has_gsf),"condition":(has_ele_T)&(same_fr),"colour":"black","fill":True,"size":7,},
      {"label":"ID (Loose)","var":test.gsf_pt,"mask":(~test.is_e)&(has_gsf),"condition":(has_ele_T)&(double_fr),"colour":"black","fill":False,"size":7,},
      ]
   
   for idx,curve in enumerate(curves) :
      his_total,_ = np.histogram(curve["var"][curve["mask"]],bins=bin_edges)
      his_passed,_ = np.histogram(curve["var"][curve["mask"]&curve["condition"]],bins=bin_edges)
      ax.errorbar(x=bin_edges[:-1], #bin_centres,
                  y=[ x/y if y > 0. else 0. for x,y in zip(his_passed,his_total) ],
                  yerr=[ np.sqrt(x)/y if y > 0. else 0. for x,y in zip(his_passed,his_total) ],
                  label='{:s} (mean={:6.4f})'.format(curve["label"],
                                                     float(his_passed.sum())/float(his_total.sum()) \
                                                        if his_total.sum() > 0 else 0.),
                  marker='o',
                  color=curve["colour"],
                  markerfacecolor = curve["colour"] if curve["fill"] else "white",
                  markersize=curve["size"],
                  linewidth=0.5,
                  elinewidth=0.5)
      
   ##########
   # Finish up ... 
   plt.title('Mistag rate as a function of GSF track pT')
   plt.xlabel('Transverse momentum (GeV)')
   plt.ylabel('Mistag rate')
   plt.gca().set_yscale('log')
   ax.set_xlim(bin_edges[0],bin_edges[-1])
   ax.set_ylim([0.0001, 1.])
   plt.legend(loc='best')
   plt.tight_layout()
   plt.savefig(dir+'/mistag.pdf')
   plt.clf()
