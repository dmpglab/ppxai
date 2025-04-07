#Date: Dec 11, 2024
#Author: Sonal Allana
#Purpose: Calculation of different XAI scores 
import quantus
import numpy as np
import faithfulness_correlation
import faithfulness_estimate

class xai_metrics():
    scores_faithfulnesscorrelation = 0
    scores_faithfulnessestimate = 0
    scores_sufficiency = 0
    
    def __init__(self):
        self.scores_faithfulnesscorrelation = 0
    
    def calculateXaiMetrics(self,baselinemodel,X_adv_test,Y_adv_test,attributions_test):
        #Faithfulness metrics
        self.getFaithfulnessCorrelation(baselinemodel,X_adv_test,Y_adv_test,attributions_test)
        self.getFaithfulnessEstimate(baselinemodel,X_adv_test,Y_adv_test,attributions_test)  
        self.getSufficiency(baselinemodel,X_adv_test,Y_adv_test,attributions_test) #Works
        
       
    #Faithfulness Metrics
    def getFaithfulnessCorrelation(self,baselinemodel,X_adv_test,Y_adv_test,attributions_test):
        self.scores_faithfulnesscorrelation = faithfulness_correlation.FaithfulnessCorrelation(
                return_aggregate=True,
                subset_size=1,
                           disable_warnings=False,
                           display_progressbar=True,
                           abs=True, normalise=True,
                           perturb_baseline="mean"
                           )(model=baselinemodel,
                             x_batch=X_adv_test,
                             y_batch=Y_adv_test,
                             a_batch=attributions_test)
        print("Faithfulness Correlation - aggregate score: ", self.scores_faithfulnesscorrelation)
    
    def getFaithfulnessEstimate(self,baselinemodel,X_adv_test,Y_adv_test,attributions_test):
        self.scores_faithfulnessestimate = faithfulness_estimate.FaithfulnessEstimate(
                   return_aggregate=True,
                   disable_warnings=False,
                   display_progressbar=True,
                   abs=True, normalise=True,
                   perturb_baseline="mean"
                   )(model=baselinemodel,
                     x_batch=X_adv_test,
                     y_batch=Y_adv_test,
                     a_batch=attributions_test)
        print("Faithfulness Estimate - aggregate score: ", self.scores_faithfulnessestimate)

    def getSufficiency(self,baselinemodel,X_adv_test,Y_adv_test,attributions_test):
        self.scores_sufficiency = quantus.Sufficiency(
                       return_aggregate=True,
                       disable_warnings=False,
                       display_progressbar=True,
                       )(model=baselinemodel,
                         x_batch=X_adv_test,
                         y_batch=Y_adv_test,
                         a_batch=attributions_test)
        print("Sufficiency - aggregate score: ", self.scores_sufficiency)
        
#For creating a batch of samples
def getSamples(x_adv_test):
    batch_size = 128
    indices = np.arange(np.size(x_adv_test,0)) 
    samples=[]
    np.random.shuffle(indices) 
    for i in indices:
        samples.append(i)
        if len(samples)==batch_size:
            break
    return samples
