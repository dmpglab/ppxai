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
    scores_monotonicity = 0
    scores_sufficiency = 0
    scores_relativeInputStability = 0
    scores_relativeOutputStability = 0
    scores_relativeRepresentationStability = 0
    scores_complexity = 0
    scores_avgSensitivity = 0
    
    def __init__(self):
        self.scores_faithfulnesscorrelation = 0
    
    def calculateXaiMetrics(self,baselinemodel,X_adv_test,Y_adv_test,attributions_test):
        #Faithfulness metrics
        #indexing error on faithfulness correlation 
        self.getFaithfulnessCorrelation(baselinemodel,X_adv_test,Y_adv_test,attributions_test)
        #indexing error on faithfulness estimate
        self.getFaithfulnessEstimate(baselinemodel,X_adv_test,Y_adv_test,attributions_test)  
        #self.getMonotonicity(baselinemodel,X_adv_test,Y_adv_test,attributions_test)  
        self.getSufficiency(baselinemodel,X_adv_test,Y_adv_test,attributions_test) #Works
        
        #Complexity metrics
        #self.getComplexity(baselinemodel,X_adv_test,Y_adv_test,attributions_test)   #Works

    
#    def printXaiMetrics(self):
#        print("Faithfulness Correlation - aggregate score: ", self.scores_faithfulnesscorrelation)
#        #print("Sufficiency - aggregate score: ", self.scores_sufficiency)
#        #print("Complexity - aggregate score: ", self.scores_complexity)
       
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
    
    #gives value of 1.0 
    def getMonotonicity(self,baselinemodel,X_adv_test,Y_adv_test,attributions_test):    
        self.scores_monotonicity = monotonicity.Monotonicity(
                   return_aggregate=True,
                   disable_warnings=True,
                   display_progressbar=True,
                   abs=True, normalise=True
                   )(model=baselinemodel,
                     x_batch=X_adv_test,
                     y_batch=Y_adv_test,
                     a_batch=attributions_test)
        print("Monotonicity - aggregate score: ", self.scores_monotonicity)
    
    #some calculated values are nan
    def getSensitivityN(self,baselinemodel,X_adv_test,Y_adv_test,attributions_test):
        self.scores_sensitivityN = sensitivity_n.SensitivityN(
                       return_aggregate=True,
                       disable_warnings=False,
                       display_progressbar=True,
                       perturb_func=quantus.perturb_func.uniform_noise
                       )(model=baselinemodel,
                         x_batch=X_adv_test,
                         y_batch=Y_adv_test,
                         a_batch=attributions_test)
        print("SensitivityN - aggregate score: ", self.scores_sensitivityN)


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

    #Robustness Metrics   
#    #Error
#    def getConsistency(self,baselinemodel,X_adv_test,Y_adv_test,attributions_test):
#        self.scores_consistency = quantus.Consistency(
#                       return_aggregate=True,
#                       disable_warnings=False,
#                       display_progressbar=True,
#                       )(model=baselinemodel,
#                         x_batch=X_adv_test,
#                         y_batch=Y_adv_test,
#                         a_batch=attributions_test)
#        print("Consistency - aggregate score: ", self.scores_consistency)    
#
#    def getContinuity(self,baselinemodel,X_adv_test,Y_adv_test,attributions_test):
#        self.scores_continuity = quantus.Continuity(
#                       return_aggregate=True,
#                       disable_warnings=False,
#                       display_progressbar=True,
#                       )(model=baselinemodel,
#                         x_batch=X_adv_test,
#                         y_batch=Y_adv_test,
#                         a_batch=attributions_test,
#                         explain_func=quantus.explain,
#                         explain_func_kwargs={"method": "IntegratedGradients", "reduce_axes": ()})
#        print("Continuity - aggregate score: ", self.scores_continuity)   
#        
#    #Error
#    def getRelativeInputStability(self,baselinemodel,X_adv_test,Y_adv_test,attributions_test):
#        self.scores_relativeInputStability = quantus.RelativeInputStability(
#                       return_aggregate=True,
#                       disable_warnings=False,
#                       display_progressbar=True,
#                       )(model=baselinemodel,
#                         x_batch=X_adv_test,
#                         y_batch=Y_adv_test,
#                         a_batch=attributions_test)
#        print("Relative Input Stability - aggregate score: ", self.scores_relativeInputStability)
        
    #Error
    def getAvgSensitivity(self,baselinemodel,X_adv_test,Y_adv_test,attributions_test):
        self.scores_avgSensitivity = quantus.AvgSensitivity(
                       nr_samples=2,
                       lower_bound=0.2,
                       norm_numerator=quantus.norm_func.fro_norm,
                       norm_denominator=quantus.norm_func.fro_norm,
                       perturb_func=quantus.perturb_func.uniform_noise,
                       similarity_func=quantus.similarity_func.difference,
                       abs=True,
                       normalise=False,
                       aggregate_func=np.mean,
                       return_aggregate=True,
                       disable_warnings=False,
                       display_progressbar=True,
                       )(model=baselinemodel,
                         x_batch=X_adv_test,
                         y_batch=Y_adv_test,
                         a_batch=attributions_test,
                         explain_func=quantus.explain,
                         explain_func_kwargs={"method": "IntegratedGradients", "reduce_axes": ()})
        print("Average Sensitivity - aggregate score: ", self.scores_avgSensitivity)
              
#    #Error
#    def getRelativeOutputStability(self,baselinemodel,X_adv_test,Y_adv_test,attributions_test):
#        self.scores_relativeOutputStability = quantus.RelativeOutputStability(
#                       return_aggregate=True,
#                       disable_warnings=False,
#                       display_progressbar=True,
#                       )(model=baselinemodel,
#                         x_batch=X_adv_test,
#                         y_batch=Y_adv_test,
#                         a_batch=attributions_test)
#        print("Relative Output Stability - aggregate score: ", self.scores_relativeOutputStability)
#  
#    #Error
#    def getRelativeRepresentationStability(self,baselinemodel,X_adv_test,Y_adv_test,attributions_test):
#        self.scores_relativeRepresentationStability = quantus.RelativeRepresentationStability(
#                       return_aggregate=True,
#                       disable_warnings=False,
#                       display_progressbar=True,
#                       )(model=baselinemodel,
#                         x_batch=X_adv_test,
#                         y_batch=Y_adv_test,
#                         a_batch=attributions_test)
#        print("Relative Representation Stability - aggregate score: ", self.scores_relativeRepresentationStability)
#     
#    #Error
#    def getLocalLipschitzEstimate(self,baselinemodel,X_adv_test,Y_adv_test,attributions_test):
#        self.scores_localLipschitzEstimate = quantus.LocalLipschitzEstimate(
#                       return_aggregate=True,
#                       disable_warnings=False,
#                       display_progressbar=True,
#                       )(model=baselinemodel,
#                         x_batch=X_adv_test,
#                         y_batch=Y_adv_test,
#                         a_batch=attributions_test,
#                         explain_func=quantus.explain,
#                         explain_func_kwargs={"method": "IntegratedGradients", "reduce_axes": ()})
#        print("Local Lipschitz Estimate (or Stability) - aggregate score: ", self.scores_localLipschitzEstimate)
        
    #Complexity Metrics
    def getComplexity(self,baselinemodel,X_adv_test,Y_adv_test,attributions_test):
        self.scores_complexity = quantus.Complexity(return_aggregate=True,
                   disable_warnings=False,
                   display_progressbar=True,)(model=baselinemodel,
                     x_batch=X_adv_test,
                     y_batch=Y_adv_test,
                     a_batch=attributions_test)
        print("Complexity - aggregate score: ", self.scores_complexity)
        
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