############################################################################################################
# High Risk Deals - beta model
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
import shap
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
from statsmodels.nonparametric.smoothers_lowess import lowess
from pygam import LogisticGAM
from pygam.utils import generate_X_grid

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
# Plot feature(s) contributions to a given particular deal i.e. one observation or record.
# author: gregtam from github codebase - intepreting random forests and other trees
# reviewer: v-rostan (adapted code) 
############################################################################################
sns.set_palette('colorblind')
blue, green, red, purple, yellow, cyan = sns.color_palette('colorblind')

def plot_obs_feature_contrib(clf, contributions, features_df, labels, index, 
                             class_index=0, num_features=None,
                             order_by='natural', violin=False, **kwargs):

    """Plots a single observation's feature contributions.
    Inputs:
    clf - A Decision Tree or Random Forest classifier object
    contributions - The contributions from treeinterpreter
    features_df - A Pandas DataFrame with the features
    labels - A Pandas Series of the labels
    index - An integer representing which observation we would like to
            look at
    class_index - The index of which class to look at (Default: 0)
    num_features - The number of features we wish to plot. If None, then
                   plot all features (Default: None)
    order_by - What to order the contributions by. The default ordering
               is the natural one, which takes the original feature
               ordering. (Options: 'natural', 'contribution')
    violin - Whether to plot violin plots (Default: False)
    Returns:
    obs_contrib_df - A Pandas DataFrame that includes the feature values
                     and their contributions
    """
    def _extract_contrib_array():
        # If regression tree
        if len(contributions.shape) == 2:
            if class_index > 0:
                raise ValueError('class_index cannot be positive for regression.')

            contrib_array = contributions[index]

        # If classification tree

        elif len(contributions.shape) == 3:

            if class_index >= contributions.shape[2]:

                raise ValueError('class_index exceeds number of classes.')

            contrib_array = contributions[index, :, class_index]

        else:

            raise ValueError('contributions is not the right shape.')

        return contrib_array

    def _plot_contrib():
        """
        Plot contributions for a given observation. Also plot violin
        plots for all other observations if specified.
        """
        if violin:

            # Get contributions for the class

            if len(contributions.shape) == 2:

                contrib = contributions

            elif len(contributions.shape) == 3:

                contrib = contributions[:, :, class_index]

            contrib_df = pd.DataFrame(contrib, columns=features_df.columns)

            if has_ax:

                ax.violinplot([contrib_df[w] for w in obs_contrib_tail.index],

                              vert=False,

                              positions=np.arange(len(obs_contrib_tail))

                             )

                ax.scatter(obs_contrib_tail.contrib,

                           np.arange(obs_contrib_tail.shape[0]),

                           color=red,

                           s=100

                          )

                ax.set_yticks(np.arange(obs_contrib_tail.shape[0]))

                ax.set_yticklabels(obs_contrib_tail.index)

            else:

                # Plot a violin plot using only variables in obs_contrib_tail

                plt.violinplot([contrib_df[w] for w in obs_contrib_tail.index],

                               vert=False,

                               positions=np.arange(len(obs_contrib_tail))

                              )

                plt.scatter(obs_contrib_tail.contrib,

                            np.arange(obs_contrib_tail.shape[0]),

                            color=red,

                            s=100

                           )

                plt.yticks(np.arange(obs_contrib_tail.shape[0]),

                           obs_contrib_tail.index

                          )
        else:

            obs_contrib_tail['contrib'].plot(kind='barh', ax=ax)

        if has_ax:

            ax.axvline(0, c='black', linestyle='--', linewidth=2)

        else:

            plt.axvline(0, c='black', linestyle='--', linewidth=2)

        x_coord = ax.get_xlim()[0]

        for y_coord, feat_val in enumerate(obs_contrib_tail['feat_val']):

            if has_ax:

                t = ax.text(x_coord, y_coord, feat_val)

            else:

                t = plt.text(x_coord, y_coord, feat_val)

            t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor=blue))

    def _edit_axes():

        if has_ax:

            ax.set_xlabel('Contribution of feature')

        else:

            plt.xlabel('Contribution of feature')

        true_label = labels.iloc[index]

        if isinstance(clf, DecisionTreeClassifier)\
                or isinstance(clf, RandomForestClassifier):

            scores = clf.predict_proba(features_df.iloc[index:index+1])[0]

            scores = [float('{:1.3f}'.format(i)) for i in scores]

            # v-rostan: fixing a bug -- making sure it displays all probability scores accordingly
            # including classified class and complement classes
            # based on scores from predict_proba scikit-learn 
            # ---
            if has_ax:
                ax.set_title('True Value: {}\nScores: Flag ({}), High ({}), Low ({})'
                                 .format(true_label, scores[0], scores[1], scores[2])) # scores[class_index]
            else:

                plt.title('True Value: {}\nScores: Flag ({}), High ({}), Low ({})'

                              .format(true_label, scores[0], scores[1], scores[2])) # scores[class_index]

            # Returns obs_contrib_df (flipped back), true labels, and scores 

            return obs_contrib_df.iloc[::1], true_label, scores

        elif isinstance(clf, DecisionTreeRegressor)\
                or isinstance(clf, RandomForestRegressor):

            pred = clf.predict(features_df.iloc[index:index+1])[0]

            if has_ax:

                ax.set_title('True Value: {}\nPredicted Value: {:1.3f}'

                                 .format(true_label, pred))

            else:

                plt.title('True Value: {}\nPredicted Value: {:1.3f}'

                              .format(true_label, pred))

            # Returns obs_contrib_df (flipped back), true labels, and scores 
            return obs_contrib_df.iloc[::-1], true_label, pred

    if 'ax' in kwargs:

        has_ax = True

        ax = kwargs['ax']

    else:

        has_ax = False

        fig, ax = plt.subplots()

    feature_array = features_df.iloc[index]
    contrib_array = _extract_contrib_array()

    obs_contrib_df = pd.DataFrame({'feat_val': feature_array,

                                   'contrib': contrib_array

                                  })

    # Flip DataFrame vertically to plot in same order
    obs_contrib_df = obs_contrib_df.iloc[::-1]

    obs_contrib_df['abs_contrib'] = np.abs(obs_contrib_df['contrib'])

    if order_by == 'contribution':

        obs_contrib_df.sort_values('abs_contrib', inplace=True)

    # Trim the contributions if num_features is specified

    if num_features is not None:

        obs_contrib_tail = obs_contrib_df.tail(num_features).copy()
        print(obs_contrib_tail)

    else:

        obs_contrib_tail = obs_contrib_df.copy()

    _plot_contrib()

    return _edit_axes()


def plot_single_feat_contrib(feat_name, contributions, features_df,

                             class_index=0, class_name='', add_smooth=False,

                             frac=2/3, **kwargs):

    """
    Plots a single feature's values across all observations against
    their corresponding contributions.

    Inputs:
    feat_name - The name of the feature
    contributions - The contributions from treeinterpreter
    features_df - A Pandas DataFrame with the features
    class_index - The index of the class to plot (Default: 0)
    class_name - The name of the class being plotted (Default: '')
    add_smooth - Add a lowess smoothing trend line (Default: False)
    frac - The fraction of data used when estimating each y-value
           (Default: 2/3)
    """

    # Create a DataFrame to plot the contributions

    def _get_plot_df():

        """Gets the feature values and their contributions."""

        if len(contributions.shape) == 2:

            contrib_array = contributions[:, feat_index]

        elif len(contributions.shape) == 3:

            contrib_array = contributions[:, feat_index, class_index]

        else:

            raise Exception('contributions is not the right shape.')

        plot_df = pd.DataFrame({'feat_value': features_df[feat_name].tolist(),

                                'contrib': contrib_array

                               })

        return plot_df

    def _get_title():

        # Set title according to class_

        if class_name == '':

            return 'Contribution of {}'.format(feat_name)

        else:

            return 'Conribution of {} ({})'.format(feat_name, class_name)

    def _plot_contrib():

        # If a matplotlib ax is specified in the kwargs, then set ax to it
        # so we can overlay multiple plots together.

        if 'ax' in kwargs:

            ax = kwargs['ax']

            # If size is not specified, set to default matplotlib size

            if 's' not in kwargs:

                kwargs['s'] = 40

            plot_df\
                .sort_values('feat_value')\
                .plot(x='feat_value', y='contrib', kind='scatter', **kwargs)
            ax.axhline(0, c='black', linestyle='--', linewidth=2)

            ax.set_title(title)

            ax.set_xlabel(feat_name)

            ax.set_ylabel('Contribution')

        else:

            plt.scatter(plot_df.feat_value, plot_df.contrib, **kwargs)

            plt.axhline(0, c='black', linestyle='--', linewidth=2)

            plt.title(title)

            plt.xlabel(feat_name)

            plt.ylabel('Contribution')

    def _plot_smooth():

        # Gets lowess fit points

        x_l, y_l = lowess(plot_df.contrib, plot_df.feat_value, frac=frac).T

        # Overlays lowess curve onto data

        if 'ax' in kwargs:

            ax = kwargs['ax']

            ax.plot(x_l, y_l, c='black')

        else:

            plt.plot(x_l, y_l, c='black')

    # Get the index of the feature

    feat_index = features_df.columns.get_loc(feat_name)

    # Gets the DataFrame to plot

    plot_df = _get_plot_df()

    title = _get_title()

    _plot_contrib()

    if add_smooth:

        _plot_smooth()

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
# select * from [Zeus_Intermediate].[dbo].[vw_Fact_HRDEA_DS] 
# 
############################################################################################
source_file = "G:\\Datascience\\target_clean_FBQ.csv"
data = read_csv(source_file, encoding = "ISO-8859-1")

df = pd.DataFrame(data)
df.fillna(value=0, inplace=True)
colnames = df.columns.values

# Pre-processing dataset
who_is_target = 'MultiClass' # 'Escalated'

df['Target'] = df[who_is_target].copy()
df.loc[ (df['Discount Amount'] < 0), 'Discount Amount'] = 0
df.loc[ (df['Discount Percentage'] < 0), 'Discount Percentage'] = 0
df.loc[ (df['Discount Percentage'] > 100), 'Discount Percentage'] = 100
df.loc[ (df['Contract Value'] < 0), 'Contract Value'] = 0

df['Partner GOV SOE Conc'].fillna(value=0, inplace=True)
df['Partner Max Previous 8 qtr Conc'].fillna(value=0, inplace=True)
df['CTE Length of Term Score'].fillna(value=0, inplace=True)
df['CTE_AmountperPARTENR'].fillna(value=0, inplace=True)
df['AssociatedECIFProjects'].fillna(value=0, inplace=True)
df['SOE Business Contry'].fillna(value=0, inplace=True)
df['Partner SOE Total Conc'].fillna(value=0, inplace=True)
df['ECIF in last 15 days of Quarter'].fillna(value=0, inplace=True)
df['All ECIF via Same Partner'].fillna(value=0, inplace=True)
df['ECIF Percent of EA Deal'].fillna(value=0, inplace=True)
df['ECIF in Same Month as Sales'].fillna(value=0, inplace=True)
df['ECIF for Same Product as Sales Deal'].fillna(value=0, inplace=True)

df['Framework'] = 0
df.loc[ (df['IsFrameWorkAvailable'] == 'Yes'), 'Framework'] = 1
df['Framework Discount Risk'] = 0
df.loc[ (df['DiscountIsLessOrEquvaltoFramework'] == 0) & (df['IsFrameWorkAvailable']== 'Yes'), 
       'Framework Discount Risk'] = 1

df = df[['Enrollment/Term', 'Target', 'Escalated', 'OLC', 'LCC', 'Flagged', 'SystemFlagged', 'ManuallyFlagged', 
         'Average Risk Score', 'Geo Risk Score', 'SOE Risk Score', 'Partner Risk Score', 'Large Discount Risk Score', 
         'Discount Trend Risk Score', 'First Billed Quarter', 'Sub-Segment', 'Sector', 'Cluster', 'Area', 'Sub-Region', 
         'GeoRiskTier', 'Industry', 'Vertical', 'Government Entity', 'Reseller TPID', 'Partner GOV SOE Conc', 
         'SOE Business Contry', 'Partner SOE Total Conc','Partner Max Previous 8 qtr Conc', 'Super RSD','Contract Value', 
         'Discount Amount', 'Discount Percentage' , 'Deal Revenue Tier', 'Renewal vs New', 'Cloud vs On-Prem', 
         'CTE Length of Term Score', 'CTE_AmountperPARTENR', 'AssociatedECIFProjects', 'ECIF in last 15 days of Quarter', 
         'All ECIF via Same Partner', 'ECIF Percent of EA Deal', 'ECIF in Same Month as Sales', 
         'ECIF for Same Product as Sales Deal', 'Framework', 'Framework Discount Risk', 'Total SOE Partner Revenue', 
         'Total SOE Country Revenue', '% SOE Revenue (All)', '% SOE Revenue (Trend)', 'Total SOE Partner Deals', 
         'Total SOE Country Deals', '% of SOE Deals (All)', '% of SOE Deals (Trend)', 'Concentration: % SOE Revenue (All)',
         'Concentration: % SOE Revenue (Trend)', 'Concentration: % of SOE Deals (All)', 'Concentration: % of SOE Deals (Trend)', 
         'Partner - Total Flagged', 'Area - Total Flagged', '% Flagged Deals', 'Partner - Total Escalated Deals', 
         'EOQ', 'IsQuotehashasOnlyOneRevision', 'AvgDaysBetweenRevisions', 'DaysBetweenFirstandLastRevision', 'Gaming', 
         'Devices', 'MCB', 'Developer Tools', 'Inactive', 'MPN', 'Dynamics', 'Enterprise Services', 'Corp', 'Windows', 
         'Online Advertising', 'CSS', 'Server and CAL', 'LEX', 'MS Research', 'CSV', 'Skype', 'MOD', 'Retail Stores', 
         'Returned Revenue', 'Revenue', 'Returned Revenue %', 'PartnerOneID', 'PartnerOneName', 'ATUName', 'ATUManager', 
         'SalesGroupName', 'SalesUnitName', 'SalesTeamName', 'First Billed Month']]

# Removing non-features e.g. RiskScores, Flagged, OLC, LCC

df.drop(['OLC'], axis=1, inplace=True)
df.drop(['LCC'], axis=1, inplace=True)
df.drop(['Flagged'], axis=1, inplace=True)
df.drop(['SystemFlagged'], axis=1, inplace=True)
df.drop(['ManuallyFlagged'], axis=1, inplace=True)
df.drop(['Average Risk Score'], axis=1, inplace=True)
df.drop(['Geo Risk Score'], axis=1, inplace=True)
df.drop(['SOE Risk Score'], axis=1, inplace=True)
df.drop(['Partner Risk Score'], axis=1, inplace=True)
df.drop(['Large Discount Risk Score'], axis=1, inplace=True)
df.drop(['Discount Trend Risk Score'], axis=1, inplace=True)
df.drop(['CTE Length of Term Score'], axis=1, inplace=True)
df.drop(['Escalated'], axis=1, inplace=True)
df.drop(['Total SOE Partner Revenue', 'Total SOE Country Revenue', '% SOE Revenue (Trend)', 'Total SOE Partner Deals', 'Total SOE Country Deals', '% of SOE Deals (Trend)', 'Concentration: % SOE Revenue (All)', 'Concentration: % SOE Revenue (Trend)', 'Concentration: % of SOE Deals (All)', 'Concentration: % of SOE Deals (Trend)', 'Partner - Total Flagged', 'Area - Total Flagged', '% Flagged Deals', 'Partner - Total Escalated Deals'], axis=1, inplace=True)

df.drop(['ATUManager'], axis=1, inplace=True)
df.drop(['PartnerOneName'], axis=1, inplace=True)
df.drop(['Revenue'], axis=1, inplace=True)

print('Target as %s %s:' % (who_is_target, Counter(df['Target'])))

# More pre-processing, turning categorical features into dummy variables
print('df shape before creating dummies: ', df.shape)

# Discount percentages should be 100 bound
df.loc[ (df['Discount Percentage'] > 100), 'Discount Percentage' ] = 100

df = pd.get_dummies(df, columns = ['Sub-Segment', 'Sector', 'Cluster', 'Area', 'Sub-Region', 
      'GeoRiskTier', 'Industry', 'Vertical', 'Government Entity', 'Reseller TPID',  
      'Super RSD', 'Deal Revenue Tier', 'Renewal vs New', 'Cloud vs On-Prem', 'PartnerOneID',
      'ATUName', 'SalesGroupName', 'SalesUnitName', 'SalesTeamName', 'First Billed Quarter',
      'First Billed Month'])

print('df shape after dummies: ', df.shape)

############################################################################################
# STEP 2
# Train, Test dataset split
# Synthetic sampling in order to boost minority class and reduce majority class
# SMOTEENN
# Setting options for cross validation and scoring metric e.g. accuracy, recall
############################################################################################
train_cols = df.columns[2:] # skip Enrollment/Term, Target, Escalated, First Billed Quarter

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

# Test options for cross validation and evaluation (socring) metric
num_folds = 10
scoring = 'recall_macro' #make_scorer(hrd_custom_loss_func) # 'recall' 'recall_macro'
class_weight = class_weight.compute_class_weight('balanced', np.unique(Y_train_resampled), Y_train_resampled)

#print("% of +class in original data: {}%".format(100*sum(df['Target'])/float(len(df['Target']))))
#print("% of +class in resampled training: {}%".format(100*sum(Y_train_resampled)/float(len(Y_train_resampled))))
print(sorted(Counter(Y_train_resampled).items()))

#####################################################################################################
# STEP 3
# Model
# Algorithm tuning for best scaled ML candidate
# Best candidate so far: ScaledRFC -- now searching for best configuration
#####################################################################################################
n_features = 1000

# Tune scaled top model performer
scaler = MinMaxScaler(feature_range=(0, 1)).fit(X_train_resampled)
rescaledX = scaler.transform(X_train_resampled)
selector = SelectKBest(f_classif, k=n_features).fit(rescaledX, Y_train_resampled)
rescaledX = selector.transform(rescaledX)
mask = selector.get_support()
selected_features = df[train_cols].columns[mask]

# Hyper-parameter tuning for grid search

#param_grid = {'n_estimators': [100], 'class_weight': ['balanced']}
#param_grid = {'class_weight':[{'Flag': 10, 'High': 100, 'Low': 1}, 
#                              {'Flag': 6.7, 'High': 95.9, 'Low': 0.35}, 
#                              'balanced', None], 
#                              'min_samples_leaf':[10]}
param_grid = {'class_weight':['balanced'], 'min_samples_leaf':[10]}

# {'Flag': 20, 'High': 100, 'Low': 1}, {'Flag': 5, 'High': 200, 'Low': 1}, {'Flag': 20, 'High': 100, 'Low': 0.5}, 
                              

model = DecisionTreeClassifier()
#model_ = RandomForestClassifier()

kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train_resampled)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

#model_ = RandomForestClassifier(n_estimators=10, class_weight='balanced') #(**grid_result.best_params_)
model = DecisionTreeClassifier(**grid_result.best_params_) #(class_weight='balanced', min_samples_leaf=10)
model.fit(rescaledX, Y_train_resampled)
#model_.fit(rescaledX, Y_train_resampled)
#print('Confusion matrix based on Training set:')
#predictions = model.predict(rescaledX)
#print(confusion_matrix(Y_train_resampled, predictions))
#print(classification_report(Y_train_resampled, predictions))
#print('End of Training set report -----------------------------------------')

# Estimate accuracy on validation (test 'unseen') dataset
print('Confusion matrix based on Unseen (test) set: ----------------------------')
rescaledValidationX = scaler.transform(X_test)
rescaledValidationX = selector.transform(rescaledValidationX)
predictions = model.predict(rescaledValidationX)
print('Accuracy score: ' + str(accuracy_score(Y_test, predictions)))
cm = confusion_matrix(Y_test, predictions)
print(cm)
print(classification_report(Y_test, predictions))
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
v_df.to_csv("C:\\Users\\v-rostan\\beta_testset.csv")
#

####################################################################################################
# Summarizing the number of manually flagged deals that the classifier was able to properly identify
# note: should turn this into a function ---

list_ = []
c = 0
for i in range(len(Y_test)):
    if Y_test.iloc[i] == 'Flag':
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
plot_confusion_matrix(cm, classes=['Flag', 'High', 'Low'], normalize=True,
                      title='HRD normalized confusion matrix')
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
k = 15; plt.title('Feature Importances: DecisionTreeClassifier - no synthetic sampling - 1K top features : f_classif')
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
Y_GAM.replace(to_replace='Flag', value=1, inplace=True)

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
observation_index = 425

class_result = model.predict(df[selected_features].iloc[observation_index:observation_index+1])

if class_result[0] == 'Flag':
    class_index = 0
elif class_result[0] == 'High':
    class_index = 1
else:
    class_index = 2

dt_multi_pred, dt_multi_bias, dt_multi_contrib = ti.predict(model, rescaledValidationX)
plot_obs_feature_contrib(model, dt_multi_contrib, df[selected_features], pd.Series(df['Target']), 
                         index=observation_index, class_index=class_index, num_features=20, order_by='contribution', violin=True)

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

colours = [blue, red, green]
class_names = ['Risk = {}'.format(s) for s in ('Flag', 'High', 'Low')]
#fig, ax = plt.subplots(1, 3, sharey=True)
#fig.set_figwidth(20)

# Name of feature for examining all its values in relatio to its importances
# for classifications
feat_name_ = 'Discount Amount'

#for i in range(len(colours)):
#    plot_single_feat_contrib(feat_name_, dt_multi_contrib, X_test[selected_features],
#                             class_index=i, class_name=class_names[i],
#                             c=colours[i], ax=ax[i])
    
#plt.tight_layout()
#plt.show() #savefig('plots/shell_weight_contribution_by_sex_dt.png')

fig, ax = plt.subplots(1, 3, sharey=True)
fig.set_figwidth(20)

for i in range(len(colours)):
    plot_single_feat_contrib(feat_name_, dt_multi_contrib, X_test[selected_features],
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
