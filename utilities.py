#Date: Dec 11, 2024
#Author: Sonal Allana
#Purpose: Supporting methods for preprocessing, loading and splitting datasets, creating model architectures
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.models import Model
import tensorflow_privacy as tp
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdagradOptimizer


#loads appropriate synthetic dataset as required
def loadSynDataset(dataset_name,syndataType):
    fname = '../datasets/{0}_sdv_{1}.csv'.format(dataset_name,syndataType)
    X = pd.read_csv(fname,sep=',', engine='python', na_values='?')
    print(X.columns)
    if dataset_name == "adult":          
        Y = np.array(X['class'])
        X = np.array(X.drop('class', axis=1))
    elif dataset_name == "credit":
        Y = np.array(X['default.payment.next.month'])
        X = np.array(X.drop(['default.payment.next.month'], axis=1))
    elif dataset_name == "compas":
        Y = np.array(X['decile_score'])
        X = np.array(X.drop(['decile_score'], axis=1))
    elif dataset_name == "hospital": 
        #return as dataframe instead of np.array
        Y = X['readmitted']
        X = X.drop(['readmitted'], axis=1)
    return X, Y    


def preprocess(dataset_name):
    if dataset_name == "adult":
        X,Y = preprocess_adult()
    elif dataset_name == "credit":
        X,Y = preprocess_credit()
    elif dataset_name == "compas":
        X,Y = preprocess_compas()
    elif dataset_name == "hospital":
        X,Y = preprocess_hospital()
    return X, Y
    
#The preprocessing of adult dataset is adapted from [1]
def preprocess_adult():
    headers = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss', 'hours-per-week','native-country','class']
    fname = '../datasets/adult.data'
    adult = pd.read_csv(fname, sep=',', engine='python', na_values='?',names=headers) 

    #Remove spaces in data
    adult = adult.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    # Drop all records with missing values
    adult.dropna(inplace=True) 
    adult.reset_index(drop=True, inplace=True)
    
    # Drop fnlwgt, not interesting for ML 
    adult.drop('fnlwgt', axis=1, inplace=True) 
    adult.drop('education', axis=1, inplace=True)
    
    #Set sensitive attributes to binary
    adult['sex'] = (adult['sex']=="Male").astype(int)
    adult['race'] = (adult['race']=="White").astype(int)

    # Convert objects to categories
    obj_columns = adult.select_dtypes(['object']).columns
    adult[obj_columns] = adult[obj_columns].astype('category')
    num_columns = adult.select_dtypes(['int64']).columns 
    adult[num_columns] = adult[num_columns].astype('float64') 
    for c in num_columns:
        adult[c] /= (adult[c].max()-adult[c].min())

    adult['class'] = adult['class'].cat.codes
    obj_columns = adult.select_dtypes(['category']).columns
    adult.replace(['Divorced',
                   'Married-AF-spouse',
                   'Married-civ-spouse',
                   'Married-spouse-absent',
                   'Never-married',
                   'Separated',
    'Widowed'
                  ],
                  ['not married',
                   'married',
                   'married',
                   'married',
                   'not married',
                   'not married',
                   'not married'
                   ], inplace = True)

    adult = pd.get_dummies(adult, columns=obj_columns)
    print(adult)
    #adult.to_csv('../datasets/adult_preprocessed.csv',index=False) #source for creating synthetic dataset (run once)
    print(adult.columns)
    X = np.array(adult.drop('class', axis=1))
    Y = np.array(adult['class'])
    return X, Y
    
#The preprocessing of credit dataset is adapted from [2]
def preprocess_credit():
    fname = '../datasets/UCI_Credit_Card.csv'
    
    credit = pd.read_csv(fname,sep=',', engine='python', na_values='?') 
    # Drop all records with missing values
    credit.dropna(inplace=True) 
    credit.reset_index(drop=True, inplace=True)

    credit = credit.rename(columns={'PAY_0': 'PAY_1'})
    credit['LIMIT_BAL'] = credit['default.payment.next.month'] + np.random.normal(scale=0.5, size=credit.shape[0])
    credit.loc[credit['SEX'] == 2, 'LIMIT_BAL'] = np.random.normal(scale=0.5, size=credit[credit['SEX'] == 2].shape[0])
    Y = np.array(credit["default.payment.next.month"])
    credit['AGE'] = (credit['AGE'] < 40).astype(int)  
    credit['SEX'] = credit['SEX'] - 1
    credit.drop(['ID'],axis=1,inplace=True)
    #credit.to_csv("../datasets/credit_preprocessed.csv",index=False) #source for creating synthetic dataset (run once)
    X = np.array(credit.drop(['default.payment.next.month'], axis=1))
    print(credit.columns)
    return X, Y


#Preprocessing of compas dataset for sensitive attribute conversin to binary,
# update of target variable (i.e., decile_score) into binary,
# and inclusion of selected features from raw data from Propublica
def preprocess_compas():
    fname = '../datasets/compas-scores-two-years.csv'
    df = pd.read_csv(fname,sep=',', engine='python', na_values='?') 
    
    #Set sensitive attributes to binary
    df['sex'] = (df['sex']=="Male").astype(int)
    df['race'] = (df['race']=="Caucasian").astype(int)
    
    #Set decile scores 5 and above as 1, remaining 0
    df['decile_score'] = (df['decile_score'] > 5).astype(int)  

    X = pd.DataFrame()

    # Numerical variables that we can pull directly
    X = df.loc[
    :, 
    [
        'sex',  
        'age',  
        'race', 
        'juv_fel_count', 
        'juv_misd_count', 
        'juv_other_count', 
        'priors_count',
        'days_b_screening_arrest',
        'is_recid',
        'decile_score'
    ]]

    categorical_var_names = [
    'c_charge_degree', 
    ]
    
    for categorical_var_name in categorical_var_names:
        categorical_var = pd.Categorical(
            df.loc[:, categorical_var_name])

        # Just have one dummy variable if it's boolean
        if len(categorical_var.categories) == 2:
            drop_first = True
        else:
            drop_first = False

        dummies = pd.get_dummies(
            categorical_var, 
            prefix=categorical_var_name,
            drop_first=drop_first)

        X = pd.concat([X, dummies], axis=1)
 
    X.dropna(inplace=True)
    X.reset_index(drop=True, inplace=True)
 
    ### Set the Y labels
    Y = X.decile_score
    #X.to_csv('../datasets/compas_preprocessed.csv',index=False) #source for creating synthetic dataset (run once)
    
    X.drop('decile_score',axis=1,inplace=True)
    
    print(X.columns)
    
    return np.array(X), np.array(Y)



def preprocess_hospital():    
    fname = '../datasets/hospital_data.csv'
    df = pd.read_csv(fname, sep=',', engine='python', na_values='?') 

    df.dropna(subset=["race"], inplace=True) #drop records with NA values for race
    df.drop(df[df.gender == "Unknown/Invalid"].index, inplace=True)

    df['gender'] = (df['gender']=="Male").astype(int)
    df['race'] = (df['race']=="Caucasian").astype(int)

    df['readmitted'].replace(['>30',
            '<30',
            'NO'                  
           ],
           [1,1,0
            ], inplace = True)


    print(df['gender'].unique())
    print(df['race'].unique())
    
    #The following processing is adapted from [3]
    #Source: https://github.com/kohpangwei/influencerelease/blob/master/scripts/hospital_readmission.ipynb

    # Convert categorical variables into numeric ones

    X = pd.DataFrame()

    # Numerical variables that we can pull directly
    X = df.loc[
    :, 
    [
        'race',
        'gender',
        'time_in_hospital',
        'num_lab_procedures',
        'num_procedures',
        'num_medications',
        'number_outpatient',
        'number_emergency',
        'number_inpatient',
        'number_diagnoses',
        'readmitted'
    ]]

    categorical_var_names = [
    'age', 
    'discharge_disposition_id',
    'max_glu_serum',
    'A1Cresult',
    'metformin',
    'repaglinide',
    'nateglinide',
    'chlorpropamide',
    'glimepiride',
    'acetohexamide',
    'glipizide',
    'glyburide',
    'tolbutamide',
    'pioglitazone',
    'rosiglitazone',
    'acarbose',
    'miglitol',
    'troglitazone',
    'tolazamide',
    'examide',
    'citoglipton',
    'insulin',
    'glyburide-metformin',
    'glipizide-metformin',
    'glimepiride-pioglitazone',
    'metformin-rosiglitazone',
    'metformin-pioglitazone',
    'change',
    'diabetesMed'
    ]
    for categorical_var_name in categorical_var_names:
        categorical_var = pd.Categorical(
            df.loc[:, categorical_var_name])

        # Just have one dummy variable if it's boolean
        if len(categorical_var.categories) == 2:
            drop_first = True
        else:
            drop_first = False

        dummies = pd.get_dummies(
            categorical_var, 
            prefix=categorical_var_name,
            drop_first=drop_first)

        X = pd.concat([X, dummies], axis=1)
 
    X.dropna(inplace=True)
    X.reset_index(drop=True, inplace=True)
 
    ### Set the Y labels
    Y = X.readmitted
    #X.to_csv('../datasets/hospital_preprocessed.csv',index=False) #source for creating synthetic dataset (run once)
    print(X.columns)
    X.drop('readmitted',axis=1,inplace=True)
    #end citation 
    
    return X,Y
 
#The following technique for train/test balancing of classes is adapted from [3]
#Source: #https://github.com/kohpangwei/influencerelease/blob/master/scripts/hospital_readmission.ipynb
def getTrainTestSets(X,Y):
    ### Split into training and test sets. 
    # For convenience, we balance the training set to have 10k positives and 10k negatives.

    np.random.seed(2)
    num_examples = len(Y)
    assert X.shape[0] == num_examples
    num_train_examples = 20000
    num_train_examples_per_class = int(num_train_examples / 2)
    #original expression commented below
    #num_test_examples = num_examples - num_train_examples
    #updated expression to retrieve test samples per class
    num_test_examples_per_class = int(0.33/0.67 * num_train_examples/2)
    assert num_test_examples_per_class > 0

    pos_idx = np.where(Y == 1)[0]
    neg_idx = np.where(Y == 0)[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    assert len(pos_idx) + len(neg_idx) == num_examples

    train_idx = np.concatenate((pos_idx[:num_train_examples_per_class], neg_idx[:num_train_examples_per_class]))
    test_idx = np.concatenate((pos_idx[num_train_examples_per_class:num_train_examples_per_class+num_test_examples_per_class],neg_idx[num_train_examples_per_class:num_train_examples_per_class+num_test_examples_per_class]))
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    X_train = np.array(X.iloc[train_idx, :], dtype=np.float32)
    Y_train = Y[train_idx]

    X_test = np.array(X.iloc[test_idx, :], dtype=np.float32)
    Y_test = Y[test_idx]
    return X_train, X_test, Y_train, Y_test
    
#The following network is adapted from [1]
def create_nn(dataset_name, input_shape):
    input_data = Input(shape = input_shape, name="input_layer")
    if dataset_name != "hospital":
        x = Dense(40, activation='relu', name="dense1")(input_data) 
        x = Dense(40, activation='relu', name = "dense2")(x) 
    else:
    #additional layers and neurons 
    #number of neurons adapted from [4]
        x = Dense(1024, activation='relu', name="dense1")(input_data) 
        x = Dense(512, activation='relu', name = "dense2")(x)
        x = Dense(256, activation='relu', name = "dense3")(x) 
        x = Dense(100, activation='relu', name = "dense4")(x)         
    output = Dense(1)(x)
    model = Model(input_data, output)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True) 
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy']) 
    return model

#The following network is adapted from [1]
def create_dp_nn(dataset_name, input_shape, noise_multiplier, l2_norm_clip, microbatches,learning_rate): 
    input_data = Input(shape = input_shape, name="input_layer")
    if dataset_name != "hospital":
        x = Dense(40, activation='relu', name="dense1")(input_data) 
        x = Dense(40, activation='relu', name = "dense2")(x) 
    else:
    #additional layers and neurons 
    #number of neurons adapted from [4]
        x = Dense(1024, activation='relu', name="dense1")(input_data) 
        x = Dense(512, activation='relu', name = "dense2")(x)
        x = Dense(256, activation='relu', name = "dense3")(x) 
        x = Dense(100, activation='relu', name = "dense4")(x)         
    output = Dense(1)(x)
    model = Model(input_data, output)
    optimizer = DPKerasAdamOptimizer(
                       l2_norm_clip=l2_norm_clip,
                       noise_multiplier=noise_multiplier,
                       num_microbatches=microbatches,
                       learning_rate=learning_rate)    
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.losses.Reduction.NONE)
    model.compile(optimizer=optimizer,
             loss=loss,
             metrics=['accuracy']) 
    return model

'''
References:
[1] A. Blanco-Justicia, D. Sánchez, J. Domingo-Ferrer, and K. Muralidhar, “A Critical Review on the Use (and Misuse) of Differential Privacy in Machine Learning,” ACM Comput. Surv., vol. 55, no. 8, pp. 1–16, Aug. 2023, doi: 10.1145/3547139.

[2] V. Duddu and A. Boutet, “Inferring Sensitive Attributes from Model Explanations,” in Proceedings of the 31st ACM International Conference on Information & Knowledge Management (CIKM ’22), New York, NY, USA: Association for Computing Machinery, Oct. 2022, pp. 416–425. doi: 10.1145/3511808.3557362.

[3] P. W. Koh and P. Liang, “Understanding Black-box Predictions via Influence Functions,” in Proceedings of Machine Learning Research, Jul. 2017, pp. 1885–1894. [Online]. Available: https://proceedings.mlr.press/v70/koh17a

[4] R. Shokri, M. Strobel, and Y. Zick, “On the Privacy Risks of Model Explanations,” in Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics, and Society, Virtual Event USA: ACM, Jul. 2021, pp. 231–241. doi: 10.1145/3461702.3462533.
'''