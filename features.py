#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 20:25:43 2018

@author: philipwidegren

"""

import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.rand(1000,3),columns=['mid','bid','ask'])
"""
    FILTERS
"""
def sma(x,n):
    return None

def ema(x,n):
    return None
    
def wma(x,n):
    return None

"""
    TECHNICAL OVERLAYS
"""
def bollingerBands(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:bollinger_bands
    w = x.rolling(n)
    mu = w.mean()
    std = w.std()
    
    return mu, mu-std*2, mu+std*2
    
def chandelierExit(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chandelier_exit
    return None

def ichimokuClouds(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ichimoku_cloud
    return None    

def KAMA(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:kaufman_s_adaptive_moving_average
    return None

def keltnerChannels(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:keltner_channels
    return None   
    
def movingAveragesEnvelope(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_average_envelopes
    return None
    
def parabolicSAR(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:parabolic_sar
    return None

def pivotPoints(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:pivot_points
    return None

def priceChannels(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:price_channels
    return None

def volumeByPrice(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:volume_by_price
    return None
    
def vwap(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:vwap_intraday
    return None
    
def zigZag(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:zigzag
    return None
    
    
"""
    TECHNICAL INDICATORS
"""
def accumulationDistributionLine(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:accumulation_distribution_line

# NEED VOLUME
    return None

def aroon(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:aroon
    w = x.rolling(n)
    return (w.apply(np.argmax))/n,(w.apply(np.argmin))/n

def aroonOscillator(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:aroon_oscillator
    z = aroon(x,n)
    return z[0]-z[1]
    
def averageDirectionalIndex(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx    
    return None
    

def trueRange(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_true_range_atr
    w = x.rolling(n)
    minimum = w.min()
    maximum = w.max()
    
    return maximum-minimum

def averageTrueRange(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_true_range_atr
    z = np.zeros((len(x),1))
    for i in range(0,len(x)):
        if i == 0:
            z[0] = x[0]
        else:
            z[i] = (z[i-1]*(n-1)+x[i])/n
    
    return z    
    
def bollingerBandWidth(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:bollinger_band_width    
    z = bollingerBands(x,n)  
    
    return (z[2]-z[1])/z[0]

def percentageBIndicator(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:bollinger_band_perce
    z = bollingerBands(x,n) 
    return (x-z[1])/(z[2]-z[1])
    
def chaikinMoneyFlow(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf

# NEED VOLUME
    return None
    
def chandeTrendMeter(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chande_trend_meter
    return None
    
def commodityChannelIndex(x,high,low,n,c=0.15):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:commodity_channel_index_cci
    w = x.rolling(n)
    return None
    
def coppockCurve(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:coppock_curve    
    return None

def correlationCoefficient(x,y,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:correlation_coeffici
    return pd.rolling_corr(x,y,window=n)

def decisionPointPriceMomentumOscillator(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:dppmo
    return None
    
def detrendedPriceOscillator(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:detrended_price_osci
    return None

def easeOfMovement(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ease_of_movement_emv
    return None
    
def forceIndex(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:force_index
    return None
    
def massIndex(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:mass_index
    return None
    
def macd(x,n1,n2):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_average_convergence_divergence_macd
    z = pd.ewma(x, span=n1)-pd.ewma(x, span=n2)
    return z
    
def macdHistogram(x,n1,n2,n3):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_average_convergence_divergence_macd
    z1 = macd(x,n1,n2)
    z2 = pd.ewma(z1, span=n3)
    
    return z1-z2
    
def moneyFlowIndex(x,n):
    return None
    
def negativeVolumeIndex(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:negative_volume_inde

# NEED VOLUME
    return None 

def onBalanceVolume(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:on_balance_volume_obv

# NEED VOLUME
    return None
    
def priceOscillators(x,n1,n2,n3):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:price_oscillators_ppo
    ppo = (pd.ewma(x, span=n1)-pd.ewma(x, span=n2))/pd.ewma(x, span=n2)
    signalLine = pd.ewma(ppo,span=n3)
    ppo_histogram = ppo-signalLine

    return ppo, signalLine, ppo_histogram
    
def percentageVolumeOscillator(x,n1,n2,n3):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:percentage_volume_oscillator_pvo
    pvo = (pd.ewma(x, span=n1)-pd.ewma(x, span=n2))/pd.ewma(x, span=n2)
    signalLine = pd.ewma(pvo,span=n3)
    pvo_histogram = pvo-signalLine

    return pvo, signalLine, pvo_histogram
    
def priceRelative(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:price_relative    
    return None
    
def knowSureThing(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:know_sure_thing_kst    
    return None
    
def pringsSpecialK(x,n):  
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:pring_s_special_k   
    return None
  
    
    
def rateOfChange(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:rate_of_change_roc_and_momentum
    w = x.rolling(n)
    return w[-1]/w[0]-1




z4 = rateOfChange(df['mid'],1)
z3=df['mid'].rolling(5)

z3.apply(np.min)

"""
    SOURCES:
        - http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators
"""
def chandelierExit(x,n)
Chandelier Exit (long) = 22-day High - ATR(22) x 3 
Chandelier Exit (short) = 22-day Low + ATR(22) x 3
    
z = bollingerBands(df['mid'],20)

def maximum(mid,bid,ask):
    
    return np.max(mid)
 
    
z2 = pd.rolling_apply(df, func = maximum, window=2, min_periods=None, args=(df['mid'],df['mid']))
"""