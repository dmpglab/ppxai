#Date: Dec 11, 2024
#Author: Sonal Allana
#Purpose: Launching the attribute inference attack and running metrics for its effectiveness
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, precision_score,f1_score, roc_curve, precision_recall_curve, confusion_matrix


class miattack_metrics():
    precision = 0; recall = 0; f1_score = 0; fpr = 0; attack_advtg = 0; attack_succ = 0
    
    def __init__(self,precision, recall, f1_score, fpr, attack_advtg, attack_succ):
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.fpr = fpr
        self.attack_advtg = attack_advtg
        self.attack_succ = attack_succ
        
    def printMetrics(self):
        print("Precision: {:.4f}".format(self.precision))
        print("Recall: {:.4f}".format(self.recall))
        print("F1-Score: {:.4f}".format(self.f1_score))
        print("FPR: {:.4f}".format(self.fpr))
        print("Attacker's advantage: {:.4f}".format(self.attack_advtg))
        print("Attack success: {:.2f}%".format(self.attack_succ))
        
        
def miattack_explanations(attributions_train,attributions_test,Z_adv_train,Z_adv_test):
    #Following attack is adapted from Duddu, V., & Boutet, A. (2022, October). 
    attack_model = MLPClassifier(solver='adam',alpha=1e-3, hidden_layer_sizes=(64,128,32,),verbose=0,max_iter=500,random_state=1337)
    
    attack_model.fit(attributions_train, Z_adv_train)
    Z_pred_prob = attack_model.predict_proba(attributions_train) 
    Z_pred_prob = Z_pred_prob[:,1]
    precision, recall, threshold = precision_recall_curve(Z_adv_train, Z_pred_prob)
    f1_score_val = (2 * precision * recall) /(precision + recall)
    best_f1_score = np.max(f1_score_val) 
    best_threshold = threshold[np.argmax(f1_score_val)]  #get best threshold that maximises F1-score
    Z_pred_prob_test = attack_model.predict_proba(attributions_test)
    Z_pred_prob_test = Z_pred_prob_test[:,1]
    Z_pred = Z_pred_prob_test > best_threshold
    #end citation
    
    #Attack metrics calculation
    precision = precision_score(Z_adv_test,Z_pred)
    recall = recall_score(Z_adv_test,Z_pred)
    f1_score_val = f1_score(Z_adv_test,Z_pred)
    tn, fp, fn, tp = confusion_matrix(Z_adv_test,Z_pred).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    attacker_advt = tpr - fpr
    attack_success = (tp + tn) / (tp + fp + tn + fn) * 100
    
    #Print attack metrics
    modinvobj = miattack_metrics(precision,recall,f1_score_val,fpr,attacker_advt,attack_success)
    
    return modinvobj 