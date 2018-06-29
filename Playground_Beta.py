############################################################################################################
# High Risk Deals - beta model
# Checked-in: 06/29/2018 -- 11:00 am (UTC-7)
# GOAL:
# Build ML model to classify and assign probability of high risk, flag, low risk, for each deal, 
# -- ultimately scoring 'unseen' deals and providing interpretability of results.
#
# STEP 0: Characterize the problem in order to better understand the goal(s) of the ML experiment and tasks
# STEP L: Load needed libraries
# STEP A: Analyze data using descriptive statistics and other visual tools to check data
# STEP 1: Load data 
# STEP 2: Use data transforms in order to structure the prediction problem to modeling ML algorithms  
# STEP 3: Evaluate competing ML algorithms on the data, selecting top performers according to clear criteria 
# STEP 4: Improve model, i.e. given best algorithm, hyperparameter tuning (and/or ensemble methods) 
#         getting the most of best performer algorithm
# STEP 5: Finalize model, make predictions, summary results, and then pickle model
# STEP G - For investigation (checking) purposes only
#
# STEP L
# Loading necessary libraries
# location: C:\Users\v-rostan\source\repos\Playground_Beta\Playground_Beta
# at VM C:\Users\v-rostan\ds\beta_model

print(__doc__)
import scipy
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import statsmodels.api as sm
import seaborn as sns
import itertools
import re
#import graphviz
#from imblearn.combine import SMOTEENN
from scipy import stats 
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils import class_weight
from os import listdir
from collections import Counter
from treeinterpreter import treeinterpreter as ti
from pygam import LogisticGAM
from pygam.utils import generate_X_grid
#from IPython.display import HTML
import sys
sys.path.insert(0, "C:\\Users\\v-rostan\\ds")
from beta_model import Feature_Intel as INTEL

sns.set_palette('colorblind')
blue, green, red, purple, yellow, cyan = sns.color_palette('colorblind')

############################################################################################
# Helper function to plot confusion matrix normalized
# author: v-rostan
# date:
# note: confusion matrix is our means to evaluate the quality of our classifier's output
# The higher the diagonal values of the confusion matrix the better given the many 
# correct predictions, yet there'll be a tradeff between precision and recall, and then
# the interpretability of the classifier model.
#############################################################################################

def plot_confusion_matrix(cm, classes, normalize=False, title='HRD confusion matrix', cmap=plt.cm.Blues):

    '''
    # This function prints and plots the confusion matrix where normalization can be applied by setting
    # 'normalized=True'
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix without normalization')
    
    print (cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



############################################################################################
# Custom scorer function for HRD -----------------------------------------------------------
# name: hrd_custom_loss_func
# author: v-rostan
# date: 
# This function defines a custom scoring strategy for HRD in order to quantify the quality
# of predictions for the purpose of evaluating classifier (model) configurations.
#############################################################################################
custom_recall = make_scorer(recall_score, average=None)

def hrd_custom_loss_func(ground_truth, predictions):

    #r_scores = recall_score(ground_truth, predictions, average=None)
    #f_scores = f1_score(ground_truth, predictions, average=None)
    fb_score = fbeta_score(ground_truth, predictions, beta=2, average=None)
    #return r_scores[1] # this is high risk recall
    return fb_score[1]
#
#
score = make_scorer(hrd_custom_loss_func, greater_is_better = True)
loss = make_scorer(hrd_custom_loss_func, greagter_is_better = False)
#
############################################################################################

############################################################################################
# STEP 1
# Load dataset
# ------------ SQL Query ------------------
# select * from [Zeus_Intermediate].[DS].[Fact_HRDEA_DS] where [First Billed Quarter] >= 'FY16-Q4'
# 
# data frozen on 05/18/2018 and available at G:\Datascience\Fact_HRDEA_DS_05182018.csv  
# 
############################################################################################

# ML model to create: 0 (Greedy), 1 (Intelligible), 2 (Ethical)
ML_MODEL = 1

if ML_MODEL == 0: # Greedy

    ML_MODEL_NAME = 'Greedy'

elif ML_MODEL == 1: # Intelligible

    ML_MODEL_NAME = 'Intelligible'

elif ML_MODEL == 2: # Ethical

    ML_MODEL_NAME = 'Ethical'

source_file = "C:\\Users\\v-rostan\\ds\\Fact_HRDEA_DS_05182018.csv"
data = read_csv(source_file, encoding = "ISO-8859-1")

# Load source data as pandas data frame
df = pd.DataFrame(data)

# Potential features from raw data
df = df[['Enrollment/Term', 'First Billed Date', 'First Billed Month', 'First Billed Quarter', 'Quarter', 
		 'Flagged', 'SystemFlagged', 'ManuallyFlagged', 
		 'Subsidiary', 'Sub-Segment', 'Cluster', 'Area', 'Region', 'Sub-Region', 'Sector',    
         'GeoRiskTier', 'Customer TPID', 'Reseller TPID', 'Product',
		 'Account Manager', 'Industry', 'Vertical', 'Government Entity', 'SOE Business Contry', 'Super RSD', 
		 'Contract Value', 'Discount Amount', 'Discount Percentage', 
		 'Enrollment Contract Value', 'Enrollment Discount Amount', 'Enrollment Discount Percentage', 
		 'Cloud vs On-Prem', 'Renewal vs New', 'Platform vs Component', 'SOE Flag', 
		 'Partner GOV SOE Conc', 'Partner SOE Total Conc', 'Partner Max Previous 8 qtr Conc',  
		 'Average Risk Score', 'Geo Risk Score', 'SOE Risk Score', 'Partner Risk Score', 'Large Discount Risk Score', 
         'Discount Trend Risk Score', 'Flagged In HRDD', 'Manually Flagged for Review', 'PEP_FLAG', 
		 'Deal Revenue Tier', 'Partner Concentration Score', 'SOE_Revenue%', 'Total Percent Per Reseller', 
		 'ECIF Percent of EA Deal', 'ECIF in last 15 days of Quarter', 'ECIF % Of Deal', 'ECIF EOQ Days', 
		 'All ECIF via Same Partner', 'ECIF Num Of Projects for Same Partner', 'ECIF in Same Month as Sales Score', 
		 'ECIF in Same Month as Sales', 'ECIF for Same Product as Sales Deal Score', 'ECIF for Same Product as Sales Deal', 
		 'AssociatedECIFProjects', 'ECIF Count', 'ECIF Points', 'ECIF Risk Score', 'ECIF Flag', 
		 'CTE Length of Term Score', 'CTE Days', 'CTE Approval Amount Score', 'CTE Amount', 'CTE Amount % per Partner', 
		 'CTE ReOccurrence Score', 'CTE ReOccurrence', 'CTE Points', 'CTE Flag', 'CTE Risk Score', 
		 'CTE_AmountperPARTNER', 'Flag_Count_PerPARTNER', 'Flag_Count_PerCUSTOMER', 
		 'QuoteID', 'RevisionID', 'HRDD QuoteState', 'HRDD QuoteStatus', 'HRDD IsFinalizeRevision', 'HRDD Flagged For Review', 
		 'HRDD Manually Flagged for Review', 'HRDD ReviewState', 
		 'IsFrameWorkAvailale', 'FrameWorkID', 'FrameWorkDiscount', 'DiscountIsLessOrEquvaltoFramework', 
		 'Total SOE Partner Revenue', 'Total SOE Country Revenue', '% SOE Revenue (All)', '% SOE Revenue (Trend)', 
		 'Total SOE Partner Deals', 'Total SOE Country Deals', '% of SOE Deals (All)', '% of SOE Deals (Trend)', 
		 'Concentration: % SOE Revenue (All)', 'Concentration: % SOE Revenue (Trend)', 'Concentration: % of SOE Deals (All)', 
		 'Concentration: % of SOE Deals (Trend)', 'Partner - Total Flagged', 'Area - Total Flagged', '% Flagged Deals', 
		 'Partner - Total Escalated Deals', 'EOQ', 'IsQuotehashasOnlyOneRevision', 'AvgDaysBetweenRevisions', 'DaysBetweenFirstandLastRevision', 
		 'Escalated', 'LCC', 'OLC', 'Gaming', 'Devices', 'MCB', 'Developer Tools', 'Inactive', 'MPN', 'Dynamics', 
		 'Enterprise Services', 'Corp', 'Windows', 'Online Advertising', 'CSS', 'Server and CAL', 'LEX', 'MS Research', 'CSV', 
		 'N/A', 'Skype', 'OPG', 'Retail Stores', 
		 'Returned Revenue', 'Revenue', 'Returned Revenue %', 'PartnerOneID', 'PartnerOneName', 
		 'ATUName', 'ATUManager', 'SalesGroupName', 'SalesUnitName', 'SalesTeamName', 'Deal Fully Compliant?', 
		 'Type', 'Pull Forward Type', 'Lower of Pull Forward Value or Billed Amount', 'Pull Forward Period', 
		 'Issue - Quota Adjustment', 'Issue CBT', 'Issue - Additional Approvals', 'Issue - MOPET Approvals', 'Issue - Amendment', 
		 'Addition of a Pool', 'Change Fulfillment Duration', 'Change Initial Duration', 'Change Microsoft Sales Affiliate', 
		 'Change of Anniversary Date', 'Change of Channel Partner', 'Change of Level', 'Change of Pricing Terms', 
		 'Change of Requirement Threshold', 'Change of Terms and Conditions', 'Change Renewal Duration', 'Compliance Re-level',
		 'Ending a Pool', 'Initial Set up Incorrect', 'Non-Specific', 'Operating Entity Agreement Number', 
		 'Type_Unknown', 'Applied', 'Reject', 'Review', 'Status_Unknown', 'Standard Amendment', 'Customized Amendment']]

# Pre-processing dataset

## Turn First Billed Month as Month only info
## regex and find are too slow; need to bring a more efficient way to extract substr
##for i in range(len(df)):
##	idx = df['First Billed Month'].iloc[i].find(',')
##	df['First Billed Month'].iloc[i] = df['First Billed Month'].iloc[i][:idx]

# Additional pre-processing steps
df.fillna(value=0, inplace=True)
df.replace(('Yes', 'No'), (1, 0), inplace=True)
df.replace(('YES', 'NO'), (1, 0), inplace=True)
# df['Renewal vs New'].replace(('Renewal', 'New'), (1, 0), inplace=True)

# Discount features should be 0 to 100 bound
df.loc[ (df['Discount Amount'] < 0), 'Discount Amount'] = 0
df.loc[ (df['Discount Percentage'] < 0), 'Discount Percentage'] = 0
df.loc[ (df['Discount Percentage'] > 100), 'Discount Percentage'] = 100
df.loc[ (df['Contract Value'] < 0), 'Contract Value'] = 0
df.loc[ (df['Enrollment Discount Amount'] < 0), 'Enrollment Discount Amount'] = 0
df.loc[ (df['Enrollment Discount Percentage'] < 0), 'Enrollment Discount Percentage'] = 0
df.loc[ (df['Enrollment Discount Percentage'] > 100), 'Enrollment Discount Percentage'] = 100
df.loc[ (df['Enrollment Contract Value'] < 0), 'Enrollment Contract Value'] = 0

# Duplicate features or simply of no use for classification ----------------------------

df.drop(['First Billed Date'], axis=1, inplace=True)
df.drop(['First Billed Month'], axis=1, inplace=True)
df.drop(['First Billed Quarter'], axis=1, inplace=True)
# df.drop(['Flagged'], axis=1, inplace=True) -----	OBS.: we'll drop this feature later
df.drop(['SystemFlagged'], axis=1, inplace=True)
df.drop(['ManuallyFlagged'], axis=1, inplace=True)

# Features that are applicable to Greedy ML model and shouldn't be dropped
if ML_MODEL != 0:

    df.drop(['Area'], axis=1, inplace=True)
    df.drop(['Sub-Segment'], axis=1, inplace=True)
    df.drop(['Cluster'], axis=1, inplace=True)
    df.drop(['Region'], axis=1, inplace=True)
    df.drop(['Sub-Region'], axis=1, inplace=True)
    df.drop(['Customer TPID'], axis=1, inplace=True)
    df.drop(['Reseller TPID'], axis=1, inplace=True)
    df.drop(['Account Manager'], axis=1, inplace=True) # OBS.: Ethical ML should replace name with tenure and level HR db
    df.drop(['Vertical'], axis=1, inplace=True)
    df.drop(['Super RSD'], axis=1, inplace=True)
    df.drop(['Contract Value'], axis=1, inplace=True)
    df.drop(['Discount Amount'], axis=1, inplace=True)
    df.drop(['Discount Percentage'], axis=1, inplace=True)
    df.drop(['CTE_AmountperPARTNER', 'Flag_Count_PerPARTNER', 'Flag_Count_PerCUSTOMER'], axis=1, inplace=True)
    df.drop(['Total SOE Partner Revenue'], axis=1, inplace=True)
    df.drop(['Total SOE Country Revenue'], axis=1, inplace=True)
    df.drop(['% SOE Revenue (Trend)'], axis=1, inplace=True)
    df.drop(['Total SOE Partner Deals'], axis=1, inplace=True)
    df.drop(['Total SOE Country Deals'], axis=1, inplace=True)
    df.drop(['% of SOE Deals (Trend)'], axis=1, inplace=True) 
    df.drop(['Concentration: % SOE Revenue (All)'], axis=1, inplace=True) 
    df.drop(['Concentration: % SOE Revenue (Trend)'], axis=1, inplace=True) 
    df.drop(['Concentration: % of SOE Deals (All)'], axis=1, inplace=True)
    df.drop(['Concentration: % of SOE Deals (Trend)'], axis=1, inplace=True)
    df.drop(['N/A', 'Skype', 'OPG', 'Retail Stores'], axis=1, inplace=True)
    df.drop(['ATUManager'], axis=1, inplace=True)
    df.drop(['SalesGroupName', 'SalesUnitName', 'SalesTeamName'], axis=1, inplace=True)

df.drop(['Average Risk Score'], axis=1, inplace=True)
df.drop(['Geo Risk Score'], axis=1, inplace=True)
df.drop(['SOE Risk Score'], axis=1, inplace=True)
df.drop(['Partner Risk Score'], axis=1, inplace=True)
df.drop(['Large Discount Risk Score'], axis=1, inplace=True)
df.drop(['Discount Trend Risk Score'], axis=1, inplace=True)
df.drop(['Partner Concentration Score'], axis=1, inplace=True)
df.drop(['ECIF Percent of EA Deal'], axis=1, inplace=True)
df.drop(['ECIF in Same Month as Sales Score'], axis=1, inplace=True)
df.drop(['ECIF for Same Product as Sales Deal Score'], axis=1, inplace=True)
df.drop(['CTE Length of Term Score'], axis=1, inplace=True)
df.drop(['ECIF Count'], axis=1, inplace=True)
df.drop(['ECIF Points'], axis=1, inplace=True)
df.drop(['ECIF Risk Score'], axis=1, inplace=True)
df.drop(['ECIF Flag'], axis=1, inplace=True)
df.drop(['CTE Points'], axis=1, inplace=True)
df.drop(['CTE Risk Score'], axis=1, inplace=True)
df.drop(['CTE Flag'], axis=1, inplace=True)
df.drop(['CTE Days', 'CTE Approval Amount Score', 'CTE Amount % per Partner'], axis=1, inplace=True) 
df.drop(['CTE ReOccurrence Score', 'CTE ReOccurrence'], axis=1, inplace=True)
df.drop(['QuoteID', 'RevisionID', 'HRDD QuoteState', 'HRDD QuoteStatus', 'HRDD IsFinalizeRevision', 'HRDD Flagged For Review'], axis=1, inplace=True) 
df.drop(['HRDD ReviewState'], axis=1, inplace=True)
df.drop(['Revenue'], axis=1, inplace=True)
df.drop(['Gaming', 'Devices', 'MCB', 'Developer Tools', 'Inactive', 'MPN'], axis=1, inplace=True)  
df.drop(['Enterprise Services', 'Corp', 'Windows', 'Online Advertising', 'CSS', 'MS Research', 'CSV'], axis=1, inplace=True) 
df.drop(['PartnerOneName'], axis=1, inplace=True)

# Framework feature engineering 
df['Framework'] = 0
df.loc[ (df['IsFrameWorkAvailale'] == 1), 'Framework'] = 1
df['Framework Discount Risk'] = 0
df.loc[ (df['DiscountIsLessOrEquvaltoFramework'] == 0) & (df['IsFrameWorkAvailale']== 1), 
       'Framework Discount Risk'] = 1

# Defining Target
who_is_target = 'MultiClass' # 'Escalated'
df[who_is_target] = 'Empty'

#df.loc[ (df['OLC'] == 1), 'MultiClass'] = 'High' # OBS.: Post-MLADS def. of High Risk
#df.loc[ (df['LCC'] == 1), 'MultiClass'] = 'High' # OBS.: Post-MLADS def. of High Risk
df.loc[ (df['Escalated'] == 1), 'MultiClass'] = 'High'
df.loc[ (df['Flagged'] == 1) & (df['MultiClass'] == 'Empty'), 'MultiClass'] = 'Med'
df.loc[ (df['MultiClass'] == 'Empty'), 'MultiClass'] = 'Low'
df.loc[ (df['Flagged In HRDD'] == 1) & (df['Escalated'] != 1), 'MultiClass'] = 'Med'
df.loc[ (df['HRDD Manually Flagged for Review'] == 1) & (df['Escalated'] != 1), 'MultiClass'] = 'Med'

# We can now drop OLC, LLC, Escalated and Flagged features from data frame
df.drop(['OLC'], axis=1, inplace=True)
df.drop(['LCC'], axis=1, inplace=True)
df.drop(['Escalated'], axis=1, inplace=True)
df.drop(['Flagged'], axis=1, inplace=True)
df.drop(['IsFrameWorkAvailale'], axis=1, inplace=True)
df.drop(['FrameWorkID'], axis=1, inplace=True)
df.drop(['FrameWorkDiscount'], axis=1, inplace=True)
df.drop(['DiscountIsLessOrEquvaltoFramework'], axis=1, inplace=True)
df.drop(['Flagged In HRDD'], axis=1, inplace=True)
df.drop(['Manually Flagged for Review'], axis=1, inplace=True)

# More pre-processing, turning categorical features into dummy variables
print('df shape before creating dummies: ', df.shape)

#for x in df:
#	if 'Low' in df[x].values:
#		print(x)

if ML_MODEL != 0:

    columns_ = ['Quarter', 'Subsidiary', 'Sector', 'GeoRiskTier', 'Industry', 'Product', 
								   'Deal Revenue Tier', 'Renewal vs New', 'Cloud vs On-Prem', 
								   'ATUName', 'Type', 'Pull Forward Type', 'Pull Forward Period',
								   'Platform vs Component', 'SOE Flag',
								   'Government Entity', 'PartnerOneID']
else: # Greedy ML model

    columns_ = ['Quarter', 'Subsidiary', 'Sector', 'GeoRiskTier', 'Industry', 'Product', 
								   'Deal Revenue Tier', 'Renewal vs New', 'Cloud vs On-Prem', 
								   'ATUName', 'Type', 'Pull Forward Type', 'Pull Forward Period',
								   'Platform vs Component', 'SOE Flag',
								   'Government Entity', 'PartnerOneID', 
                                   'Sub-Segment', 'Cluster', 'Area', 'Region', 'Sub-Region',
                                   'Reseller TPID', 'Account Manager', 'ATUManager', 'Vertical', 
                                   'Super RSD','SalesGroupName', 'SalesUnitName', 'SalesTeamName']

df = pd.get_dummies(df, columns = columns_)

print('df shape after dummies: ', df.shape)
colnames = df.columns.values

df['Target'] = df[who_is_target].copy()
print('Target as %s %s:' % (who_is_target, Counter(df['Target'])))

############################################################################################
# STEP 2
# Train, Test dataset split
# Synthetic sampling in order to boost minority class and reduce majority class
# SMOTEENN
# Setting options for cross validation and scoring metric e.g. accuracy, recall
############################################################################################
#train_cols = df.columns[1:-1, df.columns != 'MultiClass'] # skip Enrollment/Term, and Target
train_cols = df.loc[:, df.columns != 'MultiClass'].columns[1:-1]

X = df[train_cols]
Y = df['Target']

# 80/20 train, test split
validation_size = 0.20

# Random seed to keep results reproducible
seed = 7

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                    Y, 
                                                    test_size=validation_size, 
                                                    random_state=seed
                                                   )
# Synthetic sampling
#smote_enn = SMOTEENN(random_state=seed, n_jobs=2)
#X_train_resampled, Y_train_resampled = smote_enn.fit_sample(X_train, Y_train)

X_train_resampled = X_train
Y_train_resampled = Y_train

# Default test options for cross validation and evaluation (socring) metric
num_folds = 10
n_features = 100

#scoring = 'recall_micro' #make_scorer(hrd_custom_loss_func) # 'recall' 'recall_macro'
class_weight = class_weight.compute_class_weight('balanced', np.unique(Y_train_resampled), Y_train_resampled)

if ML_MODEL == 0: # Greedy
    scoring = 'recall_macro'
    n_features = 1000
    param_grid = {'class_weight': ['balanced'], 'min_samples_leaf': [10]} # Model Config ID 38 (a.k.a MLADS 2018)
    model = DecisionTreeClassifier(random_state=seed)

elif ML_MODEL == 1: # Intelligible
    scoring = 'recall_micro'
    n_features = 1000
    param_grid = {'class_weight': ['balanced'], 'min_samples_leaf': [10]} # Model Config ID 18 (a.k.a MLADS 2018)
    model = DecisionTreeClassifier(random_state=seed)

elif ML_MODEL == 2: # Ethical
    scoring = make_scorer(hrd_custom_loss_func)
    n_features = 100
    param_grid = {'class_weight': ['balanced'], 'min_samples_leaf': [10]} # Model Config ID 47 (a.k.a MLADS 2018)
    model = RandomForestClassifier(random_state=seed)

#print("% of +class in original data: {}%".format(100*sum(df['Target'])/float(len(df['Target']))))
#print("% of +class in resampled training: {}%".format(100*sum(Y_train_resampled)/float(len(Y_train_resampled))))
print(sorted(Counter(Y_train_resampled).items()))

#####################################################################################################
# STEP 3
# Model
# Algorithm tuning for best scaled ML candidate
# Best candidate so far: ScaledRFC -- now searching for best configuration
#####################################################################################################
#for iii in range(5):
# Tune scaled top model performer
scaler = MinMaxScaler(feature_range=(0, 1)).fit(X_train_resampled)
rescaledX = scaler.transform(X_train_resampled)
selector = SelectKBest(f_classif, k=n_features).fit(rescaledX, Y_train_resampled)
rescaledX = selector.transform(rescaledX)
mask = selector.get_support()
selected_features = df[train_cols].columns[mask]

# Hyper-parameter tuning for grid search

#param_grid = {'n_estimators': [100], 'class_weight': [{'Med': 10, 'High': 100, 'Low': 1}]}
#param_grid = {'class_weight':[{'Med': 10, 'High': 100, 'Low': 1}, 
#                              {'Med': 6.7, 'High': 95.9, 'Low': 0.35}, 
#                              'balanced', None], 
#                              'min_samples_leaf':[10]}
#param_grid = {'class_weight':[{'Med': 10, 'High': 100, 'Low': 1}, {'Med': 6.7, 'High': 95.9, 'Low': 0.35}, 'balanced', None], 
#              'min_samples_leaf':[5, 10, 20]}
#param_grid = {'class_weight': ['balanced'], 'min_samples_leaf': [10]}
# {'Med': 20, 'High': 100, 'Low': 1}, {'Med': 5, 'High': 200, 'Low': 1}, {'Med': 20, 'High': 100, 'Low': 0.5}, 
                              

#model = DecisionTreeClassifier(random_state=seed)
#model = RandomForestClassifier()

kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train_resampled)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

if ML_MODEL == 0: # Greedy
    
    model = DecisionTreeClassifier(**grid_result.best_params_, random_state=seed)

elif ML_MODEL == 1: # Intelligible

    model = DecisionTreeClassifier(**grid_result.best_params_, random_state=seed)

elif ML_MODEL == 2: # Ethical

    model = RandomForestClassifier(**grid_result.best_params_, random_state=seed)

#model = RandomForestClassifier(**grid_result.best_params_)
#model = DecisionTreeClassifier(**grid_result.best_params_, random_state=seed) #(class_weight='balanced', min_samples_leaf=10)

model.fit(rescaledX, Y_train_resampled)

#model_.fit(rescaledX, Y_train_resampled)
#print('Confusion matrix based on Training set:')
#predictions = model.predict(rescaledX)
#print(confusion_matrix(Y_train_resampled, predictions))
#print(classification_report(Y_train_resampled, predictions))
#print('End of Training set report -----------------------------------------')

# Estimate accuracy on validation (test 'unseen') dataset
# print('iii: ' + str(iii))
print(ML_MODEL_NAME + ' Confusion matrix based on Unseen (test) set: ----------------------------')
rescaledValidationX = scaler.transform(X_test)
rescaledValidationX = selector.transform(rescaledValidationX)
predictions = model.predict(rescaledValidationX)
print('Accuracy score: ' + str(accuracy_score(Y_test, predictions)))
cm = confusion_matrix(Y_test, predictions, labels=['High', 'Med', 'Low'])
print(cm)
print(classification_report(Y_test, predictions, labels=['High', 'Med', 'Low']))
print('Recall score (micro): ' + str(recall_score(Y_test, predictions, average='micro')))
print('Recall score (macro): ' + str(recall_score(Y_test, predictions, average='macro')))
print('Recall score (weighted): ' + str(recall_score(Y_test, predictions, average='weighted')))
print('Recall score (None): ' + str(recall_score(Y_test, predictions, average=None)))

##
# Saving results to file
v_df = data.loc[Y_test.index]
v_df['Prediction'] = predictions
v_df = v_df.merge(pd.DataFrame(Y_test), how='inner', validate='one_to_one', left_index=True, right_index=True)
# Output - saving to file

v_df.to_csv("C:\\Users\\v-rostan\\" + ML_MODEL_NAME + "_beta_df.csv")
#
#df.to_csv("C:\\Users\\v-rostan\\beta_df.csv")
#X_test.to_csv("C:\\Users\\v-rostan\\beta_X_test.csv")

# LIME
explainer = INTEL.lime.lime_tabular.LimeTabularExplainer(rescaledX, feature_names=selected_features, 
                                                   class_names=['High', 'Low', 'Med'], discretize_continuous=False)
e_id = '81276600-1' #'67140707-1' #'64606482-1' #'81276600-1' #'7505327-1' #'83500580-1'      
observation_index = df.loc[ df['Enrollment/Term']  == e_id ].index.values.astype(int)[0]
rescaledValidationX_one = scaler.transform(df[train_cols].iloc[observation_index:observation_index+1])
rescaledValidationX_one = selector.transform(rescaledValidationX_one)
class_result = model.predict(rescaledValidationX_one)
class_scores = model.predict_proba(rescaledValidationX_one)

exp = explainer.explain_instance(rescaledValidationX_one[0], model.predict_proba, num_features=10, top_labels=3)
exp.show_in_notebook(show_table=True, show_all=False) #IPython.core.display.HTML
#print(exp.local_exp)
exp.save_to_file("C:\\Users\\v-rostan\\" + ML_MODEL_NAME + "_exp_.html")
#plt.show(exp.show_in_notebook(show_table=True, show_all=False))
####################################################################################################
# Summarizing the number of manually flagged deals that the classifier was able to properly identify
# note: should turn this into a function ---

list_ = []
c = 0
for i in range(len(Y_test)):
    if Y_test.iloc[i] == 'Med':
        if Y_test.iloc[i] == predictions[i]:
            c = c+1
            list_.append(Y_test.index[i])

print('Number of ManuallyFlagged deals in the unseen Test data set:')
print(Counter(data['ManuallyFlagged'].iloc[Y_test.index]))
print('Number of properly identified manually flagged deals from Test data set:')
print(Counter(data.loc [ data.index[list_], 'ManuallyFlagged'] ))
print('End of Unseen (test) set report -----------------------------------------')

np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=['High', 'Med', 'Low'], normalize=True,
                      title=ML_MODEL_NAME + ' HRD normalized confusion matrix')
plt.show()
        
######################################################################################################
# Step 4
# Feature importance
#
#
######################################################################################################
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
mask = selector.get_support()
selected_features = X_train.columns[mask]
features = selected_features

# Feature importance selection and plot
m_id = ML_MODEL_NAME + ' Model: '

if isinstance(model, DecisionTreeClassifier):
    ML_MODEL_TYPE = 'DecisionTreeClassifier'
elif isinstance(model, RandomForestClassifier):
    ML_MODEL_TYPE = 'RandomForestClassifier'

k = 15; plt.title(m_id + 'Feature Importances: ' + ML_MODEL_TYPE + ' - no synthetic sampling - ' + str(n_features) + ' top features : f_classif')
plt.barh(range(len(indices[:k])), importances[indices[:k]], color='b', align='center')
plt.yticks(range(len(indices[:k])), features[indices[:k]])
plt.xlabel('relative importance')
plt.show()

print('Stop 1 of 5')

##########################################################################################
# Generalized Additive Model
# note: playground for GAM, not required
#
#
##########################################################################################
'''
df_GAM = df[features[indices[:k]]].copy()
Y_GAM = df['Target'].copy()

Y_GAM.replace(to_replace='Low', value=0, inplace=True)
Y_GAM.replace(to_replace='High', value=1, inplace=True)
Y_GAM.replace(to_replace='Med', value=1, inplace=True)

gam = LogisticGAM().fit(df_GAM, Y_GAM)

print(gam.summary())
print(gam.accuracy(df_GAM, Y_GAM))

XX = generate_X_grid(gam)

# Top six features for now

fig, axs = plt.subplots(1, 6)
titles = df_GAM[0:5].columns

for i, ax in enumerate(axs):
    pdep, confi = gam.partial_dependence(XX, feature=i+1, width=.95)
    ax.plot(XX[:, i], pdep, c='r')
    ax.plot(XX[:, i], confi[0], c='blue', ls='--')
    ax.set_title(titles[i])

plt.show()

X_train_GAM, X_test_GAM, y_train_GAM, y_test_GAM = train_test_split(df_GAM, Y_GAM, test_size=0.2, random_state=7) 

gam = LogisticGAM().gridsearch(X_train_GAM, y_train_GAM)

print(gam.accuracy(X_train_GAM, y_train_GAM))

predictions_GAM = gam.predict(X_test_GAM)
probas = gam.predict_proba(X_test_GAM)
print("Accuracy: {} ".format(accuracy_score(y_test_GAM, predictions_GAM)))
print("Confusion matrix GAM: ")
print(confusion_matrix(y_test_GAM, predictions_GAM))
print("Classificaiton report GAM: ")
print(classification_report(y_test_GAM, predictions_GAM))
'''
##########################################################################################

print('Stop 2 of 5')

##########################################################################################
# Interpretability ---
#
# Integer representing which observation we would like to examine feature contributions
# for classification
# 
#e_id = '81276600-1' #'67140707-1' #'64606482-1' #'81276600-1' #'7505327-1' #'83500580-1'      
observation_index = df.loc[ df['Enrollment/Term']  == e_id ].index.values.astype(int)[0]
#observation_index = 1669 #1397 #425

rescaledValidationX_one = scaler.transform(df[train_cols].iloc[observation_index:observation_index+1])
rescaledValidationX_one = selector.transform(rescaledValidationX_one)
class_result = model.predict(rescaledValidationX_one)
class_scores = model.predict_proba(rescaledValidationX_one)

if class_result[0] == 'High':
    class_index = 0
elif class_result[0] == 'Med':
    class_index = 1
else:
    class_index = 2

scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
rescaledX_Total = scaler.transform(X)
selector = SelectKBest(f_classif, k=n_features).fit(rescaledX_Total, Y)
rescaledX_Total = selector.transform(rescaledX_Total)

dt_multi_pred, dt_multi_bias, dt_multi_contrib = ti.predict(model, rescaledX_Total) #pd.Series(Y['Target']
INTEL.plot_obs_feature_contrib(model, dt_multi_contrib, X[selected_features], Y, 
						 index=observation_index, class_index=class_index, num_features=15, 
                         order_by='contribution', violin=True, class_scores=class_scores, e_id=e_id)

#plt.show()
print('Stop 3 of 5')

# Single observation from top of panda dataframe

#rX = scaler.transform(X_train.iloc[0:1])
#rX = selector.transform(rX)
#dt_m_p, dt_m_b, dt_m_c = ti.predict(model, rX)
#plot_obs_feature_contrib(model, dt_multi_contrib, X_test[selected_features].iloc[0:1], Y_test,
#                         index=0, class_index=0, num_features=20, order_by='contribution', violin='True')

plt.show()
print('Stop 4 of 5')

colours = [red, green, blue]
class_names = ['Risk = {}'.format(s) for s in ('High', 'Low', 'Med')]
#fig, ax = plt.subplots(1, 3, sharey=True)
#fig.set_figwidth(20)

# Name of feature for examining all its values in relatio to its importances
# for classifications
feat_name_ = 'Enrollment Contract Value' #'Customized Amendment' #'SOE_Revenue%' #'Enrollment Contract Value' #'Partner - Total Escalated Deals'

#for i in range(len(colours)):
#    plot_single_feat_contrib(feat_name_, dt_multi_contrib, X_test[selected_features],
#                             class_index=i, class_name=class_names[i],
#                             c=colours[i], ax=ax[i])
    
#plt.tight_layout()
#plt.show() #savefig('plots/shell_weight_contribution_by_sex_dt.png')

fig, ax = plt.subplots(1, 3, sharey=True)
fig.set_figwidth(20)

dt_multi_pred, dt_multi_bias, dt_multi_contrib = ti.predict(model, rescaledValidationX)

for i in range(len(colours)):
    INTEL.plot_single_feat_contrib(feat_name_, dt_multi_contrib, X_test[selected_features],
                             class_index=i, class_name=class_names[i],
                             add_smooth=True, c=colours[i], ax=ax[i])
    
plt.tight_layout()
plt.show()
print('Stop 5 of 5')

# Initialize js visualization code
#shap.initjs()

# Explain the model's predictions using SHAP values
# (same syntax works for LightGBM and scikit-learn models)
#shap_values = shap.TreeExplainer(model_).shap_values(X)

# visualize the first prediction's explanation
#shap.force_plot(shap_values[0,:], X.iloc[0,:])
