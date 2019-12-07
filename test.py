#!/usr/bin/env python
# coding: utf-8

from io import BytesIO

import numpy as np
import pandas as pd
import holoviews as hv
import hvplot.pandas
import panel as pn
import datetime as dt
import sklearn as sk
import math

from holoviews import dim, opts

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (brier_score_loss, precision_score, recall_score, precision_recall_curve,
                             f1_score, confusion_matrix, roc_curve, roc_auc_score)

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split

import statsmodels.api as sm

from scipy.stats import gamma
from scipy.stats import poisson
from scipy.stats import expon
from scipy.stats import norm
from scipy.stats import nbinom

hv.extension('bokeh')
pn.extension()

import warnings
warnings.filterwarnings("ignore")

##################################THIS IS THE TREE FUNCTION THAT JASON WROTE

def uVal(x, row, transform, best):
    transform = float(transform)
    e = 2.718281828
   
    a = 1/(1-e**((-(best-transform))/row))
   
    uValue = a - a * (np.exp((-(x-transform)/row)))
    return uValue;


def ceVal(u, row, transform, best):
   
   
    a = 1/(1-np.exp(-(best-transform)/row))
    ceVal = -((np.log(1-(u/a))*row)+ transform)
    return ceVal;

def treeFunctionNoClair (thinBottlePrice = 20,
                         bottleNowPrice = 28.5,
                         bottleMoldPrice = 80,
                         bulkGrapesPrice = 10,
                         bottle25Price = 35,
                         bottle20Price = 30,
                         bottle19Price = 25,
                         bottleNoSell = 0,
                         numCases = 1000,
                         numBottlesPerCase = 12,
                         concRainNoMold = 1.075,
                         concMold = 0.70,
                         concOther = 1,
                         moldBonus = 150000, ###update 
                         advertisingCost = 250000, ###update
                         dataCost =1000,
                         sporesCost = 10000, ###update 
                         probAcidityGreater = 0.8,
                         probAcidityLower = .2,
                         probSugar25 = 0.5,
                         probSuger20 = .5,
                         probSuger19 = 1,
                         probMold = 0.4,
                         probNoMold = .6,
                         probRainYag = 2.0/3.0,
                         probNoRainYag = 1.0/3.0,
                         PROB_RAIN_DATA_ANAL = 0.667,
                         PROB_NO_RAIN_DATA_ANAL = 0.333,
                         row = 72000,
                         detectSensitivity = 0.9,
                         detectNotSensitivity = 0.1,
                         detectSpecificity = 0.9,
                         detectNotSpecificity = 0.1):
    
    numBottles = (numBottlesPerCase * numCases)
    #all outcomes to find worst outcome we would do

    #outcomes no buying
    harvestNowNoBuyingRaw = (numBottles *bottleNowPrice* concOther)
    outcome20NoBuyingRaw = (numBottles *bottle20Price* concOther)
    outcome25NoBuyingRaw = numBottles * bottle25Price* concOther
    outcome19NoBuyingRaw =(numBottles * bottle19Price * concOther)
    outcomeNoMoldBulkNoBuyingRaw = (numBottles* bulkGrapesPrice * concRainNoMold)
    outcomeNoMoldThinNoBuyingRaw = ((numBottles* thinBottlePrice * concRainNoMold)- advertisingCost)
    outcomeNoSellNoBuyingRaw = bottleNoSell* numBottles
    outcomeMoldNoBuyingRaw = (bottleMoldPrice *numBottles * concMold) + moldBonus
   

    listForBestDecisionNoMoldNoBuying = [outcomeNoMoldThinNoBuyingRaw,outcomeNoMoldBulkNoBuyingRaw,
                                        outcomeNoSellNoBuyingRaw]
    bestDecisionNoMoldNoBuying = np.argmax(listForBestDecisionNoMoldNoBuying)
   
   
    #buying data
    harvestNowBuyingDataRaw = (numBottles *bottleNowPrice* concOther)- dataCost
    outcome20BuyingDataRaw = (numBottles *bottle20Price* concOther) - dataCost
    outcome25BuyingDataRaw = (numBottles * bottle25Price* concOther)- dataCost
    outcome19BuyingDataRaw = (numBottles * bottle19Price * concOther)- dataCost
    outcomeNoMoldBulkBuyingDataRaw = (numBottles* bulkGrapesPrice * concRainNoMold)- dataCost
    outcomeNoMoldThinBuyingDataRaw =(numBottles* thinBottlePrice * concRainNoMold) -advertisingCost - dataCost
    outcomeNoSellBuyingDataRaw = -dataCost
    outcomeMoldBuyingDataRaw = (bottleMoldPrice *numBottles * concMold +moldBonus)- dataCost
   
   
    listForDecisionNoMoldDataRaw =[outcomeNoMoldThinBuyingDataRaw,outcomeNoMoldBulkBuyingDataRaw,outcomeNoSellBuyingDataRaw]
    bestDecisionNoMoldData = np.argmax(listForDecisionNoMoldDataRaw)
 
   
    #buyingDataAndSpores

    harvestNowBuyingDataSporesRaw =  (numBottles *bottleNowPrice* concOther)-dataCost-sporesCost
    outcome20BuyingDataSporesRaw = (numBottles *bottle20Price* concOther) - dataCost - sporesCost
   
    outcome25BuyingDataSporesRaw = (numBottles * bottle25Price* concOther)- dataCost - sporesCost
    outcome19BuyingDataSporesRaw = (numBottles * bottle19Price * concOther)- dataCost - sporesCost

    outcomeMoldBuyingDataSporesRaw = (bottleMoldPrice *numBottles * concMold) + moldBonus - dataCost- sporesCost
 
    #buyingSpores

    harvestNowBuyingSporesRaw = (numBottles *bottleNowPrice* concOther)- sporesCost
    outcome20BuyingSporesRaw = (numBottles *bottle20Price* concOther) - sporesCost
    outcome25BuyingSporesRaw = (numBottles * bottle25Price* concOther)- sporesCost
    outcome19BuyingSporesRaw = (numBottles * bottle19Price * concOther)- sporesCost
    outcomeMoldBuyingSporesRaw = (bottleMoldPrice *numBottles * concMold +moldBonus)- sporesCost

    listOfAllOutComesRaw=[ outcomeMoldBuyingSporesRaw,
                          outcome25BuyingSporesRaw,
                          outcome20BuyingSporesRaw,
                          outcome19BuyingSporesRaw,
                          harvestNowBuyingSporesRaw,

                            harvestNowBuyingDataRaw,
                          outcomeMoldBuyingDataRaw,
                          listForDecisionNoMoldDataRaw[bestDecisionNoMoldData],
                          outcome25BuyingDataRaw,
                        outcome20BuyingDataRaw,
                        outcome19BuyingDataRaw,

                        outcomeMoldBuyingDataSporesRaw,
                        outcome25BuyingDataSporesRaw,
                        outcome20BuyingDataSporesRaw,
                        outcome19BuyingDataSporesRaw,
                        harvestNowBuyingDataSporesRaw,

                        outcomeMoldNoBuyingRaw,
                        listForBestDecisionNoMoldNoBuying[bestDecisionNoMoldNoBuying],
                        outcome25NoBuyingRaw,
                        outcome20NoBuyingRaw,
                        outcome19NoBuyingRaw,
                        harvestNowNoBuyingRaw,
                       ]

   

    worstRawOutcome = listOfAllOutComesRaw[np.argmin(listOfAllOutComesRaw)]


    #list of worst decisions we could make
    anotherListOfBadDecisions=[outcomeNoMoldThinNoBuyingRaw,outcomeNoMoldBulkNoBuyingRaw,
                        outcomeNoSellNoBuyingRaw,outcomeNoMoldThinBuyingDataRaw,
                        outcomeNoMoldBulkBuyingDataRaw,outcomeNoSellBuyingDataRaw]
   

    bestOutcomeValue = listOfAllOutComesRaw[np.argmax(listOfAllOutComesRaw)]
   
    worstDecidedOutcome = np.argmin(listOfAllOutComesRaw)
    anotherWorstDecisionOutcome =np.argmin(anotherListOfBadDecisions)
   
    potentialTransform = listOfAllOutComesRaw[worstDecidedOutcome]
    anotherPotentialTransform = anotherListOfBadDecisions[anotherWorstDecisionOutcome]
   
    transformValue= 0
   
    if(potentialTransform < anotherPotentialTransform):
        transformValue = potentialTransform
   
    else:
        transformValue = anotherPotentialTransform
       
   
    #outcome values  buying spores u values
     #basic structure: number of bottles * price * concentration
   
    outcomeMoldBuyingSpores =uVal((listOfAllOutComesRaw[0]),row, transformValue, bestOutcomeValue)
    outcome25BuyingSpores = uVal((listOfAllOutComesRaw[1]),row, transformValue, bestOutcomeValue)
    outcome20BuyingSpores = uVal(listOfAllOutComesRaw[2], row, transformValue, bestOutcomeValue)
    outcome19BuyingSpores = uVal(listOfAllOutComesRaw[3], row, transformValue, bestOutcomeValue)
    outcomeHarvestNowBuyingSpores = uVal((listOfAllOutComesRaw[4]),row, transformValue, bestOutcomeValue)

    #outcome values  buying data
     #basic structure: number of bottles * price * concentration

    outcomeHarvestNowBuyingData = uVal(listOfAllOutComesRaw[5],row, transformValue, bestOutcomeValue)
    outcomeMoldBuyingData =uVal(listOfAllOutComesRaw[6],row, transformValue, bestOutcomeValue)
    outcomeNoMoldDataBestDecision = uVal(listOfAllOutComesRaw[7],row, transformValue, bestOutcomeValue)
    outcome25BuyingData = uVal(listOfAllOutComesRaw[8],row,transformValue, bestOutcomeValue)
    outcome20BuyingData =uVal(listOfAllOutComesRaw[9],row, transformValue, bestOutcomeValue)
    outcome19BuyingData = uVal(listOfAllOutComesRaw[10],row, transformValue, bestOutcomeValue)


        #outcome values  buying data and spores
     #basic structure: number of bottles * price * concentration

    outcomeMoldBuyingDataSpores = uVal(listOfAllOutComesRaw[11],row, transformValue, bestOutcomeValue)
    outcome25BuyingDataSpores = uVal(listOfAllOutComesRaw[12], row, transformValue, bestOutcomeValue)
    outcome20BuyingDataSpores = uVal(listOfAllOutComesRaw[13], row, transformValue, bestOutcomeValue)
    outcome19BuyingDataSpores = uVal(listOfAllOutComesRaw[14], row, transformValue, bestOutcomeValue)
    outcomeHarvestNowBuyingDataSpores = uVal(listOfAllOutComesRaw[15], row, transformValue, bestOutcomeValue)


    #outcome values no buying
     #basic structure: number of bottles * price * concentration

    outcomeMoldNoBuying = uVal((listOfAllOutComesRaw[16]),row,transformValue,bestOutcomeValue)
    outcomeNoMoldNoBuyingBestDecision = uVal((listOfAllOutComesRaw[17]),row,transformValue,bestOutcomeValue)
    outcome25NoBuying = uVal((listOfAllOutComesRaw[18]),row,transformValue,bestOutcomeValue)
    outcome20NoBuying =uVal((listOfAllOutComesRaw[19]),row,transformValue,bestOutcomeValue)
    outcome19NoBuying = uVal((listOfAllOutComesRaw[20]),row,transformValue,bestOutcomeValue)
    harvestNowNoBuying = uVal((listOfAllOutComesRaw[21]),row,transformValue,bestOutcomeValue)
    listOfAllEndUValues=[
        outcomeMoldBuyingSpores,
        outcome25BuyingSpores,
        outcome20BuyingSpores,
        outcome19BuyingSpores,
        outcomeHarvestNowBuyingSpores,

        outcomeHarvestNowBuyingData,
        outcomeMoldBuyingData,
        outcomeNoMoldDataBestDecision,
        outcome25BuyingData,
        outcome20BuyingData,
        outcome19BuyingData,

        outcomeMoldBuyingDataSpores,
        outcome25BuyingDataSpores ,
        outcome20BuyingDataSpores,
        outcome19BuyingDataSpores,
        outcomeHarvestNowBuyingDataSpores,

        outcomeMoldNoBuying,
        outcomeNoMoldNoBuyingBestDecision,
        outcome25NoBuying,
        outcome20NoBuying,
        outcome19NoBuying,
        harvestNowNoBuying    
    ]
   

    #RESOLVE A DECISION of Wait, Harvest on Spores

    acidityGreaterSporesUValue= probSugar25*outcome25BuyingDataSpores  + probSuger20*outcome20BuyingSpores
    acidityLessSporesUValue = outcome19BuyingSpores

    noRainSporesUValue = probAcidityGreater*acidityGreaterSporesUValue + probAcidityLower*acidityLessSporesUValue
    rainSporesUValue = outcomeMoldBuyingSpores

    waitSporesUValue = probRainYag*rainSporesUValue + probNoRainYag*noRainSporesUValue
    harvestNowSporesUValue = outcomeHarvestNowBuyingSpores

   

    #RESOLVE A DECISION of Wait, Harvest on Data

    acidityGreaterDataUValue= probSugar25*outcome25BuyingData  + probSuger20*outcome20BuyingData
    acidityLessDataUValue = outcome19BuyingData

    noRainDataUValue = probAcidityGreater*acidityGreaterDataUValue + probAcidityLower*acidityLessDataUValue

    rainDataUValue = probMold*outcomeMoldBuyingData + (probNoMold*outcomeNoMoldDataBestDecision)

    waitDataUValue = PROB_RAIN_DATA_ANAL*rainDataUValue +PROB_NO_RAIN_DATA_ANAL*noRainDataUValue
    harvestNowDataUValue = outcomeHarvestNowBuyingData;
   



    #RESOLVE A DECISION of Wait, Harvest on Data And Spores

    acidityGreaterDataSporesUValue = probSugar25*outcome25BuyingDataSpores + probSuger20*outcome20BuyingDataSpores
   
    acidityLowerDataSporesUValue = outcome19BuyingDataSpores
    noRainDataSporesUValue = probAcidityGreater*acidityGreaterDataSporesUValue + probAcidityLower*acidityLowerDataSporesUValue
    rainDataSporesUValue = outcomeMoldBuyingDataSpores

    waitDataSporesUValue = PROB_RAIN_DATA_ANAL*rainDataSporesUValue + PROB_NO_RAIN_DATA_ANAL*noRainDataSporesUValue
    harvestNowDataSporesUValue =outcomeHarvestNowBuyingDataSpores
 


    #RESOLVE A DECISION of Wait, Harvest on no Buying

   
    acidityGreaterNoBuyinguVal = probSugar25*outcome20NoBuying + probSuger20 *outcome20NoBuying
    acidityLowerNoBuyinguVal = outcome19NoBuying
    noRainNoBuyinguVal = (probAcidityGreater*acidityGreaterNoBuyinguVal) +(probAcidityLower *acidityLowerNoBuyinguVal)
    rainNoBuyinguVal = (probMold *outcomeMoldNoBuying) + (probNoMold*outcomeNoMoldNoBuyingBestDecision)
    waitNoBuyingUVal = probRainYag*rainNoBuyinguVal + probNoRainYag*noRainNoBuyinguVal
    harvestNowNoBuyingUValue = harvestNowNoBuying
   



    #RESOLVE FINAL DECSION

    listOfAllWaitsAndHarvestUValues = [
                                         waitDataUValue,
                                         harvestNowDataUValue,

                                         waitDataSporesUValue,
                                         harvestNowDataSporesUValue,

                                         waitNoBuyingUVal,
                                         harvestNowNoBuyingUValue,
                                         
                                         waitSporesUValue,
                                         harvestNowSporesUValue
                                        ]
   
    bestOverallDecision = np.argmax(listOfAllWaitsAndHarvestUValues)
   
    listOfCEVals = np.zeros(8)

    for i in range (8):
        listOfCEVals[i] = ceVal(listOfAllWaitsAndHarvestUValues[i], row, transformValue , bestOutcomeValue)
    bestFinalDecisionIndex = np.argmax(listOfCEVals)

    #the best outcome has a value of listOfCEVals[bestFinalDecisionIndex]

   
    best = bestFinalDecisionIndex
    money= listOfCEVals[bestFinalDecisionIndex]
    money = float(money)
   
    decision_vec = ['Buy Data / Wait', 'Buy Data / Harvest Now', 
                   'Buy Data & Spores / Wait', 'Buy Data & Spores / Harvest Now',
                   'No Data & No Spores / Wait', 'No Data & No Spores / Harvest Now',
                   'Buy Spores Only / Wait', 'Buy Spores Only / Harvest Now']
    
    decision = "because this will give you a situation valued at: " + str(round(money,2))
    
    return best, money, decision, decision_vec, listOfCEVals

########################################################GOD HELP US ALL


def get_data():
    file = 'case_study.csv'
    data = pd.read_csv(file)
    data['DATE'] = pd.to_datetime(data['DATE'])
    return data

def get_weather():
    file = 'case_study.csv'
    return pd.read_csv(file, index_col='DATE', parse_dates=True)

def fldname_convert(full_name, weather_columns):
    full = np.array(['Precipitation', 'Weekly Sum Precip', \
          'Max Temperature', 'Min Temperature', \
          'Mean Temperature', 'Max Temp Daily Delta', \
          'Min Temp Daily Delta', 'Mean Temp Daily Delta']) 
    loc = full_name==full
    result = np.where(loc == True)[0][0]
    
    return weather_columns.tolist()[result]

full = ['Precipitation', 'Weekly Sum Precip', \
          'Max Temperature', 'Min Temperature', \
          'Mean Temperature', 'Max Temp Daily Delta', \
          'Min Temp Daily Delta', 'Mean Temp Daily Delta']

weather = get_weather()
data = get_data()

text1 = """
#  Some Grapetomization

Lets optimize these grapes, Mick!


1. Either leave default model inputs as is, or fill them in yourself.
2. Click the button, sit back, and watch the money roll in.

"""

n_samples = pn.widgets.IntSlider(name='Random samples', value=5000, start=1000, end=10000, step=1000)
button = pn.widgets.Button(name='Run Simulation')

n_precip = pn.widgets.FloatSlider(name='Minimum Daily Precipitation', value=0.1, start=0.0, end=10.0, step=0.01)
n_heat = pn.widgets.FloatSlider(name='Maximum Daily Temperature', value=80, start=45, end=100.0, step=0.1)
forecast_range = pn.widgets.IntSlider(name='Days in Advance', value=7, start=1, end=100, step=1)

dateslide = pn.widgets.DateRangeSlider(name='Date Range', value=(dt.date(1948, 1, 1), \
        dt.date(2017, 12, 14)), start=dt.date(1948, 1, 1), end=dt.date(2017, 12, 14))
select = pn.widgets.Select(name='Select Field', options=['PRCP', 'PRCP_ROLLINGSUM', 'TMAX', 
                                'TMIN', 'T_MEAN', 'TMAX_DELT','TMIN_DELT', 'T_MEAN_DELT'])
interval = pn.widgets.IntSlider(name='Select Week to Predict Weather',
                                     start=1, end=52, step=1)

n_cases = pn.widgets.IntSlider(name='Cases', value=1000, start=10, end=100000, step=10)
n_bottles = pn.widgets.IntSlider(name='Bottles', value=12, start=1, end=50, step=1)

now = pn.widgets.FloatSlider(name='Harvest Now Price', value=28.5, start=0.1, end=100.0, step=0.1)
thin = pn.widgets.FloatSlider(name='Thin Wine Price', value=10, start=0.1, end=100.0, step=0.1)
mold = pn.widgets.FloatSlider(name='Botrytis Price', value=80, start=0.1, end=500.0, step=0.1)
grapes = pn.widgets.FloatSlider(name='Bulk Grapes Price', value=10, start=0.1, end=100.0, step=0.1)
bottle_25 = pn.widgets.FloatSlider(name='25% Sugar Price', value=35, start=0.1, end=100.0, step=0.1)
bottle_20 = pn.widgets.FloatSlider(name='20% Sugar Price', value=30, start=0.1, end=100.0, step=0.1)
bottle_19 = pn.widgets.FloatSlider(name='19% Sugar Price', value=25, start=0.1, end=100.0, step=0.1)

p_rain = pn.widgets.FloatSlider(name='Probability of Storm', value=2.0/3.0, start=0.0, end=1.0, step=0.01)
p_sug = pn.widgets.FloatSlider(name='Probability of Sugar Content > 25%', value=1.0/2.0, start=0.0, 
                               end=1.0, step=0.01)
p_acid = pn.widgets.FloatSlider(name='Probability of Acidity', value=8.0/10.0, start=0.0, 
                               end=1.0, step=0.01)
p_mold = pn.widgets.FloatSlider(name='Probability of Mold', value=4.0/10.0, start=0.0, 
                               end=1.0, step=0.01)

strength = pn.widgets.Select(name='How strong do you feel about this belief?', 
                             options=['Meh, sorta feel ok about it',
                                      'It will probably happen',
                                      'THIS IS DEFINITELY HAPPENING!'])

risk_tol = pn.widgets.IntSlider(name='Risk Tolerance', value=72000, start=0, end=1000000, step=1000)


widgets = pn.WidgetBox(
    pn.panel(text1, margin=(0, 10)),
    pn.layout.Spacer(height=20),
    pn.panel('You can select your model parameters down here:', margin=(0, 10)),
    pn.layout.Spacer(height=5),
    pn.panel('In how many days in advance will you harvest?', margin=(0, 5)),
    forecast_range,
    pn.panel('What week will you harvest in?', margin=(0, 5)),
    interval,
    pn.layout.Spacer(height=5),
    pn.panel('How do you define rain? You can specificy criteria below.', margin=(0, 5)),
    n_precip,
    n_heat,
    pn.layout.Spacer(height=5),
    pn.panel('Click the button to run the model.', margin=(0, 5)),
    button,
    pn.layout.Spacer(height=5),
    pn.panel('Now lets get down to business...', margin=(0, 10)),
    pn.layout.Spacer(height=20),
    pn.panel('How many cases/bottles are you selling?', margin=(0, 5)),
    n_cases,
    n_bottles,
    pn.layout.Spacer(height=5),
    pn.panel('How much does each bottle sell for?', margin=(0, 5)),
    now,
    thin,
    mold,
    grapes,
    bottle_25,
    bottle_20,
    bottle_19,
    pn.layout.Spacer(height=5),
    pn.panel('and now, some personal items...', margin=(0, 5)),
    p_sug,
    p_acid,
    p_mold,
    p_rain,
    strength,
    pn.layout.Spacer(height=5)
)

@pn.depends(n_precip, n_heat, forecast_range)
def get_design(data, n_precip, n_heat, forecast_range):
    
    LAG_MONTH = data.MONTH.shift(periods=forecast_range.value, fill_value=np.min(data.MONTH))
    LAG_WEEK = data.WEEK.shift(periods=forecast_range.value, fill_value=np.min(data.WEEK))
    LAG_PRCP = data.PRCP.shift(periods=forecast_range.value, fill_value=np.mean(data.PRCP))
    LAG_PRCP_ROLLINGSUM = data.PRCP_ROLLINGSUM.shift(periods=forecast_range.value, 
                                                     fill_value=np.mean(data.PRCP_ROLLINGSUM))
    LAG_TMAX = data.TMAX.shift(periods=forecast_range.value, fill_value=np.mean(data.TMAX))
    LAG_TMIN = data.TMIN.shift(periods=forecast_range.value, fill_value=np.mean(data.TMIN))
    LAG_T_MEAN = data.T_MEAN.shift(periods=forecast_range.value, fill_value=np.mean(data.T_MEAN))
    LAG_TMAX_DELT = data.TMAX_DELT.shift(periods=forecast_range.value, fill_value=np.mean(data.TMAX_DELT))
    LAG_TMIN_DELT = data.TMIN_DELT.shift(periods=forecast_range.value, fill_value=np.mean(data.TMIN_DELT))
    LAG_T_MEAN_DELT = data.T_MEAN_DELT.shift(periods=forecast_range.value, fill_value=np.mean(data.T_MEAN_DELT))

    lagged = np.matrix([LAG_MONTH, LAG_WEEK, LAG_PRCP, LAG_PRCP_ROLLINGSUM, LAG_TMAX, 
                    LAG_TMIN, LAG_T_MEAN, LAG_TMAX_DELT, LAG_TMIN_DELT, LAG_T_MEAN_DELT])

    lagged_col = ['LAG_MONTH', 'LAG_WEEK', 'LAG_PRCP', 'LAG_PRCP_ROLLINGSUM', 'LAG_TMAX', 'LAG_TMIN',
              'LAG_T_MEAN', 'LAG_TMAX_DELT', 'LAG_TMIN_DELT', 'LAG_T_MEAN_DELT']
    df1 = pd.DataFrame(lagged.T, columns=lagged_col)
    
    prcp_condition = data.PRCP >= n_precip.value
    tmax_condition = data.TMAX < n_heat.value
    condition = (prcp_condition == True) & (tmax_condition == True)
    df2 = pd.DataFrame(np.matrix([prcp_condition*1, tmax_condition*1 ,condition*1]).T ,\
                   columns=['PRCP_Cond','TMAX_Cond','Cond'])
    df3 = pd.DataFrame(np.array((df2.Cond.rolling(7, min_periods=1).sum() > 3)*1), columns=['Storm'])
    design = pd.concat((data, df1, df2, df3), axis=1)
    
    storm_lag = design.Storm.shift(periods=1, fill_value=0)
    stop = np.array(storm_lag-design.Storm)
    storm_stop = pd.DataFrame(stop.T, columns=['Stop'])
    
    design = pd.concat((design,storm_stop), axis=1)
    
    return design

blop = get_design(data, n_precip, n_heat, forecast_range)

@pn.depends(dateslide, select)
def get_ts_hist(dateslide, select):
    start_date = dateslide[0] 
    end_date = dateslide[1] 
    mask = (data['DATE'] > start_date) & (data['DATE'] <= end_date)
    filtered = data.loc[mask]
    x = filtered.hvplot.scatter('TMAX', select, grid=True)+ \
        filtered.hvplot.scatter('TMIN', select, grid=True)
    
    return x


@pn.depends(dateslide, select)
def get_ts_plot(dateslide, select):
    start_date = dateslide[0] 
    end_date = dateslide[1] 
    mask = (data['DATE'] > start_date) & (data['DATE'] <= end_date)
    filtered = data.loc[mask]
    
    return filtered.hvplot.line('DATE', select, grid=True)+filtered.hvplot.scatter('PRCP', select, grid=True)

def get_tschart(weather, field):
    tra = weather[field]
    result = sm.tsa.seasonal_decompose(tra, freq=365)
    chart = hv.Curve(result.observed, label='Observations')+ \
        hv.Curve(result.trend, label='Trend')+ \
        hv.Curve(result.seasonal, label='Seasonality')+ \
        hv.Curve(result.resid, label='Residuals')
    
    return chart

@pn.depends(dateslide, select)
def get_ts_hist(dateslide, select):
    start_date = dateslide[0] 
    end_date = dateslide[1] 
    mask = (data['DATE'] > start_date) & (data['DATE'] <= end_date)
    filtered = data.loc[mask]
    x = filtered.hvplot.scatter('TMAX', select, grid=True)+ \
        filtered.hvplot.scatter('TMIN', select, grid=True)
    
    return x

x_fields = ['LAG_MONTH','LAG_WEEK','LAG_PRCP','LAG_PRCP_ROLLINGSUM','LAG_TMAX','LAG_TMIN',
            'LAG_T_MEAN','LAG_TMAX_DELT','LAG_TMIN_DELT','LAG_T_MEAN_DELT']
y_fields = 'Storm'

def fit_models(design):
    
    X, y = design[x_fields], design[y_fields]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    
    gbm = GradientBoostingClassifier(n_estimators=100)
    lr = LogisticRegression(max_iter=1000,solver='lbfgs')
    gnb = GaussianNB()
    rfc = RandomForestClassifier(n_estimators=100)

    models, thingers = [gbm, lr, gnb, rfc], []
    
    for clf in models:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        thingers.append(np.matrix(confusion_matrix(y_test, y_pred)))
    
    return X_train, X_test, y_train, y_test, models, thingers 

X_train, X_test, y_train, y_test, models, thingers = fit_models(blop)


def get_sens_spec(thingers):
    
    abbrev_labels = ['GBM', 'LR', 'NB', 'RF']
    tp = [thingers[i][0,0] for i in range(4)]
    fp = [thingers[i][0,1] for i in range(4)]
    tn = [thingers[i][1,1] for i in range(4)]
    fn = [thingers[i][1,0] for i in range(4)]

    sens = np.array([tp[i]/(tp[i]+fn[i]) for i in range(4)])
    spec = np.array([tn[i]/(tn[i]+fp[i]) for i in range(4)])

    hist1 = hv.Bars([(abbrev_labels[i], sens[i]) for i in range(4)], \
                    label='Sensitivity').opts(width=500, height=500)
    hist2 = hv.Bars([(abbrev_labels[i], spec[i]) for i in range(4)], \
                    label='Specificity').opts(width=500, height=500)

    both = (hist1*hist2).opts(xlabel='Models', ylabel='Rates', \
           legend_position='top_right').redim(x=hv.Dimension('x', range=(0.0, 1)), \
                                              y=hv.Dimension('y', range=(0.0, 1)))
    
    both = both.opts(opts.Bars(alpha=0.5)).redim(x=hv.Dimension('x', range=(0.0, 1)), \
                                              y=hv.Dimension('y', range=(0.0, 1)))
    
    return sens, spec, both

def model_comp(models):
    labels = ['Gradient Boosting: ', 'Logistic Regression','Naive Bayes', 'Random Forest']
    predicted_y_probs = [clf.predict_proba(X_test)[:,0] for clf in models]

    thresholds = np.linspace(0,1,100) # or however many points you want

    sens = [[recall_score(y_test, predicted_y_probs[i] >= t) for t in thresholds] for i in range(4)]
    prec = [[precision_score(y_test, predicted_y_probs[i] >= t) for t in thresholds] for i in range(4)]

    x = [(hv.Distribution(sens[i], label='Sensitivity')\
          *hv.Distribution(prec[i], label='Precision')) for i in range(4)]
    x = [x[i].opts(xlabel=labels[i]) for i in range(4)]

    return x[0],x[1],x[2],x[3]


timeseries1 = get_tschart(weather, 'PRCP')
timeseries11 = get_tschart(weather, 'TMAX')

sens, spec, spss = get_sens_spec(thingers)
x1, x2, x3, x4 = model_comp(models)

new_design = blop.groupby(['WEEK']).mean()[x_fields]
week_probs = [clf.predict_proba(new_design)[:,1] for clf in models]
    
@pn.interact(Week=(0, 52, 1))
def get_pbars(Week):
        abbrev_labels = ['GBM', 'LR', 'NB', 'RF']
        mean = round(np.mean([week_probs[i][Week] for i in range(4)])*100,2)
        res = hv.Bars([(abbrev_labels[i], week_probs[i][Week]) for i in range(4)],
                        label='Rain Probability: '+ str(mean)+'%').opts(width=500, height=500)

        res.opts(xlabel='Models', ylabel='Probability', \
                legend_position='top_right').redim(x=hv.Dimension('x', range=(0.0, 1)), \
                                                      y=hv.Dimension('y', range=(0.0, 1)))
        res = res.opts(opts.Bars(alpha=0.5))      
        return res

@pn.depends(n_precip, n_heat, interval, p_rain, strength)
def simulator(data, n_precip, n_heat, interval, p_rain, strength, samples):
    if strength == 'Meh, sorta feel ok about it':
        mult = 5.0
    elif strength == 'It will probably happen':
        mult = 25.0
    else:
        mult = 50.0
    prior_a = 1.0*mult
    prior_b = (1.0/p_rain.value)*mult
    
    sample = data[data.WEEK==interval.value]
    years = np.max(sample.YEAR) - np.min(sample.YEAR)
    a, b = int(prior_a + np.sum(sample.RAIN)), int(prior_b + years)
    
    if np.isnan(a)==True: a = 0
    if np.isnan(b)==True: b = 1
    
    gam = gamma.rvs(a=a, scale=1/b, size=samples)
    
    rain_mu = a/b
    mu, sigma = norm.fit(sample.TMAX)
    l, s, = expon.fit(sample.PRCP)
    
    raindays = poisson.rvs(rain_mu, size=samples)
    storm = np.zeros(samples)
    t_vec = np.zeros(samples)
    rf_vec = np.zeros(samples)
    
    for i in range(len(raindays)):
        if raindays[i] > 0:
            if raindays[i] > 7:
                days = np.random.randint(3,7)
            else:
                days = raindays[i]
            t_max = norm.rvs(mu, sigma, size=days)
            rainfall = expon.rvs(l, s, days)
            temp = np.zeros(days)
            for j in range(len(t_max)):
                if rainfall[j] >= n_precip.value and t_max[j] < n_heat.value:
                    temp[j] = 1
            t_vec[i] = np.max(t_max)
            rf_vec[i] = np.sum(rainfall)
            if np.sum(temp) >= 3:
                storm[i] = 1
    
    return gam, storm, raindays, t_vec, rf_vec 

@pn.depends(n_precip, n_heat,interval, p_rain, strength)
def other_simulator(data, n_precip, n_heat, interval, p_rain, strength, samples):
    if strength == 'Meh, sorta feel ok about it':
        mult = 5.0
    elif strength == 'It will probably happen':
        mult = 25.0
    else:
        mult = 50.0
    prior_a = 1.0*mult
    prior_b = (1.0/p_rain.value)*mult
    
    sample = data[data.WEEK==interval]
    years = np.max(sample.YEAR) - np.min(sample.YEAR)
    a, b = prior_a + np.sum(sample.RAIN), prior_b + years
    
    if np.isnan(a)==True: a = 0
    if np.isnan(b)==True: b = 1
    
    gam = gamma.rvs(a=a, scale=1/b, size=samples)
    
    rain_mu = a/b
    mu, sigma = norm.fit(sample.TMAX)
    l, s, = expon.fit(sample.PRCP)
    
    raindays = poisson.rvs(rain_mu, size=samples)
    storm = np.zeros(samples)
    t_vec = np.zeros(samples)
    rf_vec = np.zeros(samples)
    
    for i in range(len(raindays)):
        if raindays[i] > 0:
            if raindays[i] > 7:
                days = np.random.randint(3,7)
            else:
                days = raindays[i]
            t_max = norm.rvs(mu, sigma, size=days)
            rainfall = expon.rvs(l, s, days)
            temp = np.zeros(days)
            for j in range(len(t_max)):
                if rainfall[j] >= n_precip.value and t_max[j] < n_heat.value:
                    temp[j] = 1
            t_vec[i] = np.max(t_max)
            rf_vec[i] = np.sum(rainfall)
            if np.sum(temp) >= 3:
                storm[i] = 1
    
    return gam, storm, raindays, t_vec, rf_vec 

@pn.interact(Week=(0, 52, 1))
def get_simcharts(Week):
        gam, stormy, rain, t_vec, prc = other_simulator(data, n_precip, n_heat, Week, p_rain, strength, 1000)
        new_p = np.sum(stormy)/len(stormy)
        p1 = pd.DataFrame(gam).hvplot.kde(alpha=.5, xlabel='Lambda', shared_axes=False)
        p2 = pd.DataFrame(rain).hvplot.hist(alpha=.5, xlabel='Simulated Days of Rain', shared_axes=False)
        p3 = pd.DataFrame(prc).hvplot.kde(alpha=.5, xlabel='Simulated Precipitation', shared_axes=False)
        p4 = pd.DataFrame(t_vec[t_vec>0]).hvplot.kde(alpha=.5, xlabel='Simulated Temperature', shared_axes=False)

        return pn.Column(pn.Row('#Simulated Rain Probability: {}'.format(new_p*100)),
                         pn.Row(p1,p2), pn.Row(p3,p4), sizing_mode='stretch_width')

def wrapped(now, 
                thin, 
                mold, 
                grapes, 
                bottle_25, 
                bottle_20, 
                bottle_19,
                n_cases,
                n_bottles,
                p_sug,
                p_acid,
                p_mold,
                p_rain,
                sens,
                spec,
                interval):
        rec, CE, reasoning, decisions, CEs = treeFunctionNoClair (thinBottlePrice = thin,
                         bottleNowPrice = now,
                         bottleMoldPrice = mold,
                         bulkGrapesPrice = grapes,
                         bottle25Price = bottle_25,
                         bottle20Price = bottle_20,
                         bottle19Price = bottle_19,
                         bottleNoSell = 0,
                         numCases = n_cases,
                         numBottlesPerCase = n_bottles,
                         concRainNoMold = 1.075,
                         concMold = 0.70,
                         concOther = 1,
                         moldBonus = 150000, ###update 
                         advertisingCost = 250000, ###update
                         dataCost =1000,
                         sporesCost = 10000, ###update 
                         probAcidityGreater = p_acid,
                         probAcidityLower = 1-p_acid,
                         probSugar25 = p_sug,
                         probSuger20 = 1-p_sug,
                         probSuger19 = 1,
                         probMold = p_mold,
                         probNoMold = 1-p_mold,
                         probRainYag = p_rain,
                         probNoRainYag = 1-p_rain,
                         PROB_RAIN_DATA_ANAL = round(np.mean([week_probs[i][interval] 
                                                              for i in range(4)]),2),
                         PROB_NO_RAIN_DATA_ANAL = 1-round(np.mean([week_probs[i][interval] 
                                                              for i in range(4)]),2),
                         row = 72000,
                         detectSensitivity = np.mean(sens),
                         detectNotSensitivity = 1-np.mean(sens),
                         detectSpecificity = np.mean(spec),
                         detectNotSpecificity = 1-np.mean(spec))
        
        return rec, CE, reasoning, decisions, CEs

@pn.depends(button.param.clicks)
def analysis(_):
    data = get_data()
    blop = get_design(data, n_precip, n_heat, forecast_range)
    X_train, X_test, y_train, y_test, models, thingers = fit_models(blop)
    #feature_chart = get_feature_chart(models)
    sens, spec, spss = get_sens_spec(thingers)
    #rochart = get_rochart(models, X_test, y_test)
    #prchart = get_prchart(models, X_test, y_test)
    #reliability = get_reliability(models, X_test, y_test)
    x1, x2, x3, x4 = model_comp(models)

    new_design = blop.groupby(['WEEK']).mean()[x_fields]
    week_probs = [clf.predict_proba(new_design)[:,1] for clf in models]
    
    gam, stormy, rain, t_vec, prc = simulator(data, n_precip, n_heat, interval, p_rain, strength, 1000)
    new_p = np.sum(stormy)/len(stormy)
    
    @pn.depends(now, 
                thin, 
                mold, 
                grapes, 
                bottle_25, 
                bottle_20, 
                bottle_19,
                n_cases,
                n_bottles,
                p_sug,
                p_acid,
                p_mold,
                p_rain,
                interval)
    def get_CEchart(now, 
                thin, 
                mold, 
                grapes, 
                bottle_25, 
                bottle_20, 
                bottle_19,
                n_cases,
                n_bottles,
                p_sug,
                p_acid,
                p_mold,
                p_rain,
                interval):
        a,b,c,d,e = wrapped(now, 
                thin, 
                mold, 
                grapes, 
                bottle_25, 
                bottle_20, 
                bottle_19,
                n_cases,
                n_bottles,
                p_sug,
                p_acid,
                p_mold,
                p_rain,
                sens,
                spec,
                interval)
        ahh = pd.DataFrame(pd.DataFrame(e).T)
        ahh.columns=d
        p5 = ahh.hvplot.bar(alpha=.5,invert=True,height=600,width=600,
                            xlabel='Decisions',ylabel='Monetary Value')
        return pn.Column(pn.Row('#Optimal Decision: {}'.format(d[a]), sizing_mode='stretch_width'), 
                         pn.Row('#Monetary Value: {:,.2f}'.format(b)),
                         pn.Row(p5, sizing_mode='stretch_width'),
                         sizing_mode='stretch_width')
    
    return pn.Tabs(
        ('Exploration', pn.Column(
            pn.Row(pn.Column(dateslide, select, get_ts_plot), sizing_mode='stretch_width'),
            pn.Row(get_ts_hist, sizing_mode='stretch_width'),
            pn.Row(timeseries1, sizing_mode='stretch_width'),
            pn.Row(timeseries11, sizing_mode='stretch_width'), 
            sizing_mode='stretch_width')),
        ('Prediction', pn.Row(pn.Column(
            pn.Row(spss, sizing_mode='stretch_width'),
            pn.Row(get_pbars, sizing_mode='stretch_height')),
            pn.Column(pn.Row(x1), pn.Row(x2), pn.Row(x3), pn.Row(x4), sizing_mode='stretch_width'),
            sizing_mode='stretch_width')),
        ('Simulation', get_simcharts),
        ('Decision Analytics', 
            pn.Row(get_CEchart, sizing_mode='stretch_width'))
        )

pn.Row(pn.Column(widgets), pn.layout.Spacer(width=20), analysis).servable()
