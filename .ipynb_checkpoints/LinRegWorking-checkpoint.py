"""
Author: Josh Edward
Created: 16/09/26 10:30am
Last Modified: 16/09/26 01:45pm

TLDR: The best OLS model is a standard additive model (no interaction terms), assumptions are satisfied. Below is the table
containing the relevant info:

==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const        320.2934      5.412     59.183      0.000     309.678     330.909
Week          -0.4015      0.313     -1.281      0.200      -1.016       0.213
Price         -8.5218      0.416    -20.488      0.000      -9.338      -7.706
Display       23.5248      1.802     13.053      0.000      19.990      27.060
Media         28.3744      2.071     13.699      0.000      24.312      32.437

Description: A rough working document for the assignment. My goal is try and figure out the model. I don't think the code from this
should be used for the Assignment (Maybe the code cleaning). Feel free to copy any parts (or download the file) if you find them helpful.
"""

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import patsy
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv("retail_dynamics_weekly_sales.csv")

# print(df.head())

# 2 Cleaning Tasks need to happening, standardising the display feature and dropping StoreID
# Note to self: Features are "Week": int, "Price": float, "Display": str with binary values, "Media": binary int. Y is "Sales" int

y = df["Sales"]
df = df.drop(columns="StoreID")

# print(df.head())

# Next is to make a function to standardise "Display", I want to convert a No to 0 and a yes to 1
def StandardisingDisplay(row):
    stringToBeConverted = row["Display"]
    if stringToBeConverted[0] == "N" or stringToBeConverted == "n":
        return 0
    else:
        return 1

df["Display"] = df.apply(StandardisingDisplay, axis=1)
X = df[["Week", "Price", "Display", "Media"]]

# print(df["Display"])
# print(df.head())

# Cleaning complete

# Ok, now I want to understand the data. I am going to take an inference approach to try and understand what is happening.
# I am going to start with simple linear regression, drop any factors that don't seem to have any relation and check for
# any non linear relationships. I doubt any will be completely unrelated so then I will use multiple linear regression
# with no assumptions. Depending on the ability of that model I will then consider including interaction terms. From there
# we will then reason about which model should make the best predictions and make our Ridge, Lasso and final OLS model. 

# Simple LR
X_Week = sm.add_constant(df["Week"])
X_Price = sm.add_constant(df["Price"])
X_Display = sm.add_constant(df["Display"])
X_Media = sm.add_constant(df["Media"])

lr_week = sm.OLS(y, X_Week).fit()
# print(lr_week.summary())
# R-Squard of 0.003, explains no variance

lr_Price = sm.OLS(y, X_Price).fit()
# print(lr_Price.summary())
# R-Squared of 0.182, explains some variance

lr_Display = sm.OLS(y, X_Display).fit()
# print(lr_Display.summary())
# R-Squared of 0.074, explains little variance

lr_Media = sm.OLS(y, X_Media).fit()
# print(lr_Media.summary())
# R-Squared of 0.085, explains little variance

# Ok so I already would like to drop week, it doesn't seem to explain any variance. 
figure, axes = plt.subplots(1,4)


axes[0].scatter(df["Week"], y)
axes[0].plot(range(df["Week"].min(), df["Week"].max() + 1), lr_week.params[0] + lr_week.params[1] * range(df["Week"].min(), df["Week"].max() + 1))

axes[1].scatter(df["Price"], y)
axes[1].plot(np.arange(df["Price"].min(), df["Price"].max() + 1), lr_Price.params[0] + lr_Price.params[1] * np.arange(df["Price"].min(), df["Price"].max() + 1))

axes[2].scatter(df["Display"], y)
axes[2].plot(range(-1,2), lr_Display.params[0] + lr_Display.params[1] * range(-1, 2))

axes[3].scatter(df["Media"], y)
axes[3].plot(range(-1,2), lr_Media.params[0] + lr_Media.params[1] * range(-1,2))

# plt.show()
# Yeah none of them look very good. I forgot that using simple linear regression rarely makes sense for binary variables.
# Price is going to be our strongest predictor. Neither Price or week show any non-linearity.

# I am going to quickly make some boxplots to sus out what our binary variables look like
figure2, axes_2 = plt.subplots(2,2)

df_Media_1 = df[df["Media"] == 1]
df_Media_1_y = df_Media_1["Sales"]
df_Media_0 = df[df["Media"] == 0]
df_Media_0_y = df_Media_0["Sales"]

df_Display_1 = df[df["Display"] == 1]
df_Display_1_y = df_Display_1["Sales"]
df_Display_0 = df[df["Display"] == 0]
df_Display_0_y = df_Display_0["Sales"]

axes_2[0,0].boxplot(df_Media_0_y)
axes_2[0,0].set_title("When Media = 0")

axes_2[0,1].boxplot(df_Media_1_y)
axes_2[0,1].set_title("When Media = 1")

axes_2[1,0].boxplot(df_Display_0_y)
axes_2[1,0].set_title("When Display = 0")

axes_2[1,1].boxplot(df_Display_1_y)
axes_2[1,1].set_title("When Display = 1")


# plt.show()
# Not so quick, realised I know how to do this in ggplot but not python :(
# It really wasn't so bad. Its pretty simple. :)
# It seems display has stronger effect but both do have an effect. 
# Time for the Multiple linear regression model

X_lr = sm.add_constant(X)
lr_no_interaction = sm.OLS(y, X_lr).fit()
# print(lr_no_interaction.summary())
# Adj R-Squared of 0.341 which is pretty meh. Week is a non significant factor (P>0.05)
# Time to check assumptions
figure, axes_3 = plt.subplots(3,2)
# Linearity, we need to check that each component and the entire model has no trends in residual vs fitted
# Doubles up as working for constant variance
axes_3[0,0].scatter(lr_week.fittedvalues, lr_week.resid)
axes_3[0,0].axhline(0)
axes_3[0,0].set_title("Fitted vs Residuals: Week")

axes_3[0,1].scatter(lr_Price.fittedvalues, lr_Price.resid)
axes_3[0,1].axhline(0)
axes_3[0,1].set_title("Fitted vs Residuals: Price")

axes_3[1,0].scatter(lr_Display.fittedvalues, lr_Display.resid)
axes_3[1,0].axhline(0)
axes_3[1,0].set_title("Fitted vs Residuals: Display")


axes_3[1,1].scatter(lr_Media.fittedvalues, lr_Media.resid)
axes_3[1,1].axhline(0)
axes_3[1,1].set_title("Fitted vs Residuals: Media")


axes_3[2,0].scatter(lr_no_interaction.fittedvalues, lr_no_interaction.resid)
axes_3[2,0].axhline(0)
axes_3[2,0].set_title("Fitted vs Residuals: Multiple")

# Normality
sm.qqplot(lr_no_interaction.resid, line='s', ax=axes_3[2,1])

# Our independence data is violated as it is time series data however it seems reasonable to continue as week does not have a 
# strong effect on the model
# plt.show()

# Everything looks good. Now checking for multi-collinearity
x_vif = df[['Week', 'Price', 'Media', 'Display']].assign(const=1)
vif = pd.DataFrame({
    'Feature': x_vif.columns,
    'Vif': [variance_inflation_factor(x_vif.values, i) for i in range(x_vif.shape[1])]
})
# print(vif[vif['Feature'] != 'const'].to_string(index=False))

# All vif values are ~1. So all assumptions are all good. Now looking for influential points. May need to read over my
# Linear regression notes to get these tests and thresholds.

# Influential points = cooks distance > 1, Leverage points is kinda complicated, gonna look for a python package to help with that
# Will also look at standardised and studentised residuals, if > 2 then its an outlier. 

# So very conveniently cooks distance is built into the lr object in statsmodule. 
influence = lr_no_interaction.get_influence()
cooks_distance, p_values = influence.cooks_distance
df["Cooks_Distance"] = cooks_distance
# print(len(df[df["Cooks_Distance"] > 1]))
# We have 0 influential values

leverage_values = influence.hat_matrix_diag
# A point is a leverage point if its value > 2(k+1) / n where k is the number of predictors and n is the number of measurements
k = len(X.columns)
n = len(y)
leverage_threshold = (2 * (k + 1)) / n
df["Leverage_Values"] = leverage_values
# print(len(df[df["Leverage_Values"] > leverage_threshold]))
# Here we get 21 values, we will print these and have a look
# print(df[df["Leverage_Values"] > leverage_threshold])
# Nothing obviously out of the obvious with any of these, going to see if any of these are flagged as outliers aswell

# StatsModule has studentised residules built in which is easy. 
studentised_residuals = influence.resid_studentized_internal
df["Studentised_Residuals"] = studentised_residuals
# print(len(df[abs(df["Studentised_Residuals"]) > 2]))
# 69 values!

# Lets look at observations that are both a leverage point and an outlier
# print(len(df[(abs(df["Studentised_Residuals"]) > 2) & (df["Leverage_Values"] > leverage_threshold)]))
# We get 0 observations that are both leverage points and studentised residuals. 

# I have only just learnt the below is a thing
# print(influence.summary_frame())
# Its not helpful, just a huge dataframe with all these diagnostics in it. 

# Ok so lets try adding interaction terms. 
# As there are alot of possible interaction terms I am going to use stepwise regression to reduce the number of interaction terms in the 
# final model

# Something that is kinda rude is that python/sm does not have stepwise regression built into it which is pretty cringe. Going to have
# to write the logic myself. 

def AICStepwiseSelector(df):
    formula = "Sales ~ Week * Price * Media * Display"
    y, X = patsy.dmatrices(formula, data=df, return_type='dataframe')
    # print(X.columns)
    best_model = sm.OLS(y, X).fit()
    best_aic =  best_model.aic
    while True:
        features = [i for i in X.columns if ":" in i ]
        if len(features) == 0:
            break
        feature_to_drop = None
        for feature in features:
            X_temp = X.drop(columns=feature)
            temp_model = sm.OLS(y, X_temp).fit()
            if temp_model.aic < best_aic:
                best_model = temp_model
                best_aic = temp_model.aic
                feature_to_drop = feature
        if feature_to_drop != None:
            X = X.drop(columns=feature_to_drop)
        else:
            break

    return best_model


        

best_lr = AICStepwiseSelector(df)
# print(best_lr.summary())
# print(lr_no_interaction.rsquared_adj)

# This also has an adj r-squared of 0.344 (no interaction model is 0.342). That seems to be as good as we are going to get. It also dropped the media column because
# the code above isn't correct but its good enough to demonstrate the point. The best model does not include interactions.

print(lr_no_interaction.summary())
formula = "Sales ~ Week + Price + Media + Display + Week:Display + Price:Display + Week:Price:Display"
yt, Xt = patsy.dmatrices(formula, data=df, return_type='dataframe')
test_model = sm.OLS(yt, Xt).fit()
# print(test_model.summary())

# Ok so the above is actually my favourite model, it contains only statistically significant interactions (p<0.05) from the best AIC model 
# while only sacrificing 0.007 in the adj rsquared. I think this is our best understanding of the data. There are some interesting ideas 
# here Display becomes more important the higher the week and so does the same is true about price.

# I want to see how close each predicted value is to each actual value
f, a = plt.subplots(2,1)
a[0].scatter(lr_no_interaction.fittedvalues, lr_no_interaction.resid)
a[0].axhline(0)
a[0].set_title("Fitted vs Residuals: No Interaction")

a[1].scatter(test_model.fittedvalues, test_model.resid)
a[1].axhline(0)
a[1].set_title("Fitted vs Residuals: Interactions")

# plt.show()

# The best OLS model is a model with no interaction terms

