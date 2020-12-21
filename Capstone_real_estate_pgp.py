from numpy.core.fromnumeric import shape
from numpy.lib.utils import info
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.extmath import density
import sweetviz as sv
import plotly.graph_objs as go
###########################################################
# Data Pre-processing #

df= pd.read_csv('train.csv')
df_test= pd.read_csv('test.csv')
df.shape
df_test.shape
df.head(5)
df_test.head(5)
df.isnull().sum()


''' 
There seem to be many columns with null values. 
We need to explore further to identify a fill strategy 

'''

# using sweetviz package for EDA

eda_report= sv.analyze(df)
eda_report.show_html()

#dropping BLOCKID, SUMLEVEL & primary 
df.drop(['BLOCKID', 'SUMLEVEL', 'primary'], inplace=True, axis=1)
df_test.drop(['BLOCKID', 'SUMLEVEL', 'primary'], inplace=True, axis= 1)

#setting UID as index
df.set_index('UID',inplace=True)
df_test.set_index('UID', inplace=True)
df.head(5)
df_test.head(5)

#Imputation strategy = column mean

def my_imputer(dframe):
    for col in dframe.columns:
        if ((dframe[col].isnull().sum() >0) & (dframe[col].dtype != 'object')):
            dframe.replace(np.nan, dframe[col].mean(), inplace=True)
            

my_imputer(df)
my_imputer(df_test)

df.isnull().sum()
df_test.isnull().sum()

#############################################################
# EDA #

# selecting percentage ownership > 10% & percent households with second mortgage < 50%
df_debt= df[(df['pct_own']> 0.1) & (df['second_mortgage'] < 0.5)]\
    [['city', 'place', 'lat', 'lng', 'second_mortgage']]
df_debt.shape

# trimming for top 2500 locations only
df_debt_top= df_debt.nlargest(2500, 'second_mortgage')
df_debt_top

# visulaizing as a geo map
fig = go.Figure(data=go.Scattergeo(lat= df_debt_top['lat'], lon= df_debt_top['lng']))
fig.update_layout(geo= dict(scope= 'north america'))
fig.write_html('geomap.html', auto_open=True)

# eliminating bad data
df['debt'].value_counts()
df['bad_debt'].value_counts()
df['second_mortgage'].value_counts()

df.drop(df.loc[df['debt'] >=1].index, inplace=True)
df.head()

# Debt analysis
#function to add bad_debt column
def bd(dfrm):
    dfrm['bad_debt']= dfrm['second_mortgage']+dfrm['home_equity']-dfrm['home_equity_second_mortgage']

bd(df)
df.head()

# Making pie charts to analyze Debt and Bad debt 
df['category']= pd.cut(df['bad_debt'], \
    bins=[0,0.1,0.2,1], labels=['less than 10%', '10-20%', 'more than 20%'])
df.groupby('category').size().plot(kind='pie', autopct= '%1.2f%%',subplots= True)
plt.title('Break up of Bad Debt')
plt.show()

df['category_debt']= pd.cut(df['debt'], \
    bins=[0,0.5,1], labels=['less than 50%', 'more than 50%'])
df.groupby('category_debt').size().plot(kind='pie', autopct= '%1.2f%%',subplots= True)
plt.title('Debt analysis')
plt.show()

#Create Box and Whisker plots to analyze distribution for 2nd mortgage, home equity, 
#good debt and bad debt for different cities

df_city= df[['city', 'type', 'second_mortgage', 'home_equity', 'debt', 'bad_debt']]

sns.boxplot(data= df_city, x='second_mortgage', y='type', palette='pastel', )
plt.title('Distribution of Second Mortgage')
plt.show()

sns.boxplot(data= df_city, x='home_equity', y='type', palette='pastel', )
plt.title('Distribution of Home Equity')
plt.show()

sns.boxplot(data= df_city, x='debt', y='type', palette='pastel', )
plt.title('Distribution of Overall Debt')
plt.show()

sns.boxplot(data= df_city, x='bad_debt', y='type', palette='pastel', )
plt.title('Distribution of Bad Debt')
plt.xticks(ticks=[0,0.2,0.4,0.6,0.8,1.0])
plt.show()

# Collated income distribution #

df_id= df[['hi_mean', 'family_mean', 'hi_samples', 'family_samples','bad_debt']]
df_id['remaining_income'] = df_id['family_mean'] - df_id['hi_mean']
df_id

income_analysis = sv.analyze(df_id)
income_analysis.show_html()

g= sns.jointplot(data= df_id, x= 'hi_mean', y= 'family_mean')
g.plot_joint(sns.kdeplot, shade=True)
plt.show()

f= sns.jointplot(data= df_id, x= 'remaining_income', y= 'family_mean')
f.plot_joint(sns.kdeplot, shade=True)
plt.show()

#######################################################################################
''' WEEK 2 '''

# EDA

df_pop = df[['type','ALand', 'pop', 'male_pop', 'female_pop', 'male_age_mean','female_age_mean', 'married', 'separated', 'divorced']]
df_pop['pop_density']= df['pop']/df['ALand']

# found odd bad date in age columns, removing those
#######################################################
df_pop.drop(df_pop.loc[df_pop['female_age_mean'] > 1055.129032].index, inplace= True)
df_pop.drop(df_pop.loc[df_pop['male_age_mean'] == 1055.129032].index, inplace= True)


df_pop.female_age_mean.max()
df_pop.male_age_mean.max()
#######################################################

sns.relplot(x= 'male_age_mean', y= 'female_age_mean', data=df_pop, hue='type')
plt.show()

# Creating a new column median age
df_med= df[['male_age_median', 'female_age_median', 'male_pop', 'female_pop']]
df_med['median_age']=((df['male_age_median']*df['male_pop'])+(df['female_age_median']*df['female_pop']))/(df['male_pop']+df['female_pop'])
df_med.head()

sns.distplot(df_med['median_age'],bins=5, hist=True)
plt.show()

# Creating bins for the population data

df_pop['bins_pop']= pd.cut(df_pop['pop'], bins= 5, labels=['sparse', 'low', 'medium', 'high', 'dense'])
df_pop.head()
df_pop['bins_pop'].value_counts(normalize=True)

df_pop.head()
# Analyse the married, spearated and Divorced population for these population brackets
df_cut= df_pop.groupby('bins_pop')[['married','separated','divorced']].agg(np.mean)

sns.lineplot(data=df_cut)
plt.show()

########################################################

# Observations on rent

df_rent = df[['rent_mean','family_mean','state']]
df_rent['rent_perc_income'] = df['rent_mean']/df['family_mean']
df_rent['rent_perc_income'].mean()

# Puerto Rico is not a state in the US and the model is being created for US hence dropping it
df_rent.drop(df_rent[df_rent['state'] == 'Puerto Rico'].index, inplace= True)
df_rent.groupby('state')['rent_perc_income'].agg(np.mean).nlargest(20)

# Correlation analysis using Heat map

df_cor= pd.merge(df_rent, df, how='inner')
df_cor.columns

df_corr= df_cor[['city','type','zip_code','family_mean','rent_mean', 'second_mortgage','home_equity',\
   'debt','hs_degree','pct_own','married','separated', 'divorced','bad_debt','home_equity_second_mortgage',\
       'hc_mortgage_mean']].corr()

sns.heatmap(df_corr, annot=True)
plt.show()

#################################################################################################
'''Week 3'''
# Factor Analysis #

from factor_analyzer import FactorAnalyzer
fact= FactorAnalyzer(n_factors=2,rotation='promax')

df_cor= pd.merge(df_cor, df_med)
a_data= df_cor[['hs_degree', 'median_age', 'second_mortgage', 'pct_own', 'bad_debt']]

fact.fit_transform(a_data)
ev, v= fact.get_eigenvalues()

plt.plot(ev)
plt.xticks(range(len(a_data.columns)),labels=['1','2','3','4','5'])
plt.show()

plt.plot(fact.loadings_)
plt.xticks(range(len(a_data.columns)),labels=['1','2','3','4','5'])
plt.show()

fact.get_communalities()

##################################################################################################
'''Week 4'''
# Regression Analysis #

from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

df_reg= df_cor[['area_code','type', 'pop', 'family_mean','second_mortgage',\
    'bad_debt','pct_own','median_age', 'hc_mortgage_mean', 'rent_mean']]

df_reg['density']= df_cor['pop']/df_cor['ALand']

le= LabelEncoder()
asdf= le.fit_transform(df_reg['type'])
df_reg['type']= pd.Series(asdf)
df_reg.head()
df_reg.dropna()

X_train = df_reg.drop('hc_mortgage_mean',axis=1)
y_train = df_reg['hc_mortgage_mean']

#####################################################
# adding median_age column to test data set
df_test['median_age']=((df_test['male_age_median']*df_test['male_pop'])+\
    (df_test['female_age_median']*df_test['female_pop']))/(df_test['male_pop']+df_test['female_pop'])

#calling function to add bad_debt column
bd(df_test)
df_test.head()
####################################################

df_test_reg= df_test[['area_code','type', 'pop', 'family_mean','second_mortgage',\
    'bad_debt','pct_own','median_age','hc_mortgage_mean','rent_mean']]

df_test_reg['density']= df_test['pop']/df_test['ALand']
df_test_reg.density.head()

df_test_reg.drop(df_test.loc[df_test['pct_own'] > 1].index, inplace= True, axis=1)
df_test_reg.head()

asdf_test= le.fit_transform(df_test_reg['type'])
df_test_reg['type'] = asdf_test
df_test_reg.head()
df_test_reg.dropna()

X_test = df_test_reg[['area_code','type', 'pop', 'family_mean', 'second_mortgage',\
    'bad_debt','pct_own','median_age', 'hc_mortgage_mean','rent_mean','density']]
X_test.dropna(inplace=True)
X_test.shape
X_test.isnull().sum()

#Scaling
Scaled_X_train= StandardScaler().fit_transform(X_train)
Scaled_X_test= StandardScaler().fit_transform(X_test.drop('hc_mortgage_mean', axis= 1))

y_test = X_test['hc_mortgage_mean']

#####################################################

lr= LinearRegression()
lr.fit(Scaled_X_train,y_train)
y_pred=lr.predict(Scaled_X_test)

print("The R2 score is: {}".format(r2_score(y_test, y_pred)))
print("The RMSE is: {}".format(np.sqrt(mean_squared_error(y_test, y_pred))))


residuals= y_pred - y_test
sns.residplot(y_pred, y_test)
plt.show
sns.distplot(residuals)
plt.show()
##################################################

#Export for Tableaux dashboards
