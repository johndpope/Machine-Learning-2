#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 20:25:43 2018

@author: philipwidegren

"""

import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.rand(1000,3),columns=['mid','bid','ask'])

def rollingMin(x,n):
    return None
    
def rollingMax(x,n):
    return None
    
def drawdown(x,n):
    return None
    
def maximumDrawdown(x,n):
    return None
"""
    HELP FCNS
"""
def positive(x):
    return x>0
def negative(x):
    return x<0
    
def positive_values(x):
    return x*(x>0)
def negative_values(x):
    return x*(x<0)    
    
"""
    FILTERS
"""

df = pd.DataFrame(np.random.rand(1000,3),columns=['mid','bid','ask'])

def mult(x):
    return x*range(0,2)
df.rolling(2).apply(mult)

def sma(x,n):
    return x.rolling(n).mean()

def ema(x,n):
    return pd.ewma(x, span = n)
    
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
  
def chaikinMoneyFlow(close,high,low,volume,n=20):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
    z_multiplier = ((close-low)-(high-close))/(high-low)
    z_volume = z_multiplier*volume

    return z_volume.rolling(n).sum()/volume.rolling(n).sum()
    
def chandeTrendMeter(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chande_trend_meter
    return None
  
def commodityChannelIndex(close,high,low,n=20,c=0.15):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:commodity_channel_index_cci
    tp = (high+low+close)/3

    tp_sma = tp.rolling(n).mean()
    
    md = (tp-tp_sma).abs()
    md = md.rolling(n).mean()

    return (tp - tp_sma) / (c * md)
    
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

z = df_tmp['mid']
chaikinMoneyFlow(z,z,z-1,z,10)  
def easeOfMovement(high,low,volume,n=14):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ease_of_movement_emv
    prior_high, prior_low = high.shift(1), low.shift(1)
    dm = ((high + low)/2 - (prior_high + prior_low)/2) 
    br = ((volume/100000000)/(high - low))
    emv = dm / br

    return emv.rolling(n).mean()
    
def forceIndex(close,volume,n=13):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:force_index
    prior_close = close.shift(1)
    fi = (close-prior_close)*volume

    return pd.ewma(fi, span = n)
    
def massIndex(high,low,n1=9,n2=9,n3=25):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:mass_index
    sEMA = pd.ewma(high-low, span = n1)
    dEMA = pd.ewma(sEMA, span = n2)
    ratio = sEMA/dEMA

    return ratio.rolling(n3).sum()
    
def macdLine(x,n1=12,n2=26):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_average_convergence_divergence_macd
    z = pd.ewma(x, span=n1)-pd.ewma(x, span=n2)
    return z

def macdSignalLine(x,n1=12,n2=26,n3=9):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_average_convergence_divergence_macd
    z1 = macd(x,n1,n2)
    z2 = pd.ewma(z1, span=n3)
    
    return z2
    
def macdHistogram(x,n1=12,n2=26,n3=9):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_average_convergence_divergence_macd
    z1 = macd(x,n1,n2)
    z2 = pd.ewma(z1, span=n3)
    
    return z1-z2
    
z.apply(positive).rolling(10).sum()  
print z.sum()
    
def moneyFlowIndex(close,high,low,volume,n=14):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:money_flow_index_mfi    
    tp = (high + low + close)/3
    rmf = tp*volume
    mfr = (rmf.apply(positive_values).rolling(10).sum()/rmf.apply(positive).rolling(10).sum())
    mfr /= (rmf.apply(negative_values).rolling(10).sum()/rmf.apply(negative).rolling(10).sum())
        
    return 100-100/(1+mfr)
    
def negativeVolumeIndex(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:negative_volume_inde

# NEED VOLUME
    return None 

def onBalanceVolume(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:on_balance_volume_obv

# NEED VOLUME
    return None
    
def priceOscillators(x,n1=12,n2=26,n3=9):
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
    
z.shift(1)    
def knowSureThing(close, n0=9, n1 = (10,10), n2 =(10,15), n3 =(10,20), n4 =(15,30)):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:know_sure_thing_kst    

    sma_1 = (close/close.shift(n1[0])-1).rolling(n1[1]).sum()
    sma_2 = (close/close.shift(n2[0])-1).rolling(n2[1]).sum()
    sma_3 = (close/close.shift(n3[0])-1).rolling(n3[1]).sum()
    sma_4 = (close/close.shift(n4[0])-1).rolling(n4[1]).sum()

    kst = sma_1*1 + sma_2*2 + sma_3*3 + sma_4*4

    return kst.rolling(n0).mean()
    
def pringsSpecialK(x,n):  
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:pring_s_special_k   
    return None
    
def rateOfChange(close,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:rate_of_change_roc_and_momentum
    return close/close.shift(n)-1

def relativeStrengthIndex(close,n=14):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:relative_strength_index_rsi
    rsi = (close.apply(positive_values).rolling(n).sum()/close.apply(positive).rolling(n).sum())
    rsi /= (close.apply(negative_values).rolling(n).sum()/close.apply(negative).rolling(n).sum())
        
    return 100-100/(1+rsi)

   
def rrgRelativeStrength(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:rrg_relative_strength
    return None
    
def stockChartsTechnicalRank(close,n1=(200,125), n2=(50,20), n3=(3,14)):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:sctr    
    lt_ema = (close/pd.ewma(close, span=n1[0])-1)
    lt_roc = (close/close.shift(n1[1])-1)
    
    mt_ema = (close/pd.ewma(close, span=n2[0])-1)
    mt_roc = (close/close.shift(n1[1])-1)
    
    st_ppo = priceOscillators(close).rolling(n3[0]).last()/priceOscillators(close).rolling(n3[0]).first()
    st_rsi = relativeStrengthIndex(close,n3[1])
    
    return lt_pct*0.3+lt_ema*0.3+mt_ema*0.15+mt_roc*0.15+st_ppo*0.05+st_rsi*0.05

def slope(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:slope
    return None

def std(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:standard_deviation_volatility
    return x.rolling(n).std()

def stochasticOscillator(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:stochastic_oscillator_fast_slow_and_full
    return None

def stochRsi(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:stochrsi
    rsi = relativeStrengthIndex(close,n=14)

    return (rsi-rsi.rolling(n).min())/(rsi.rolling(n).max()-rsi.rolling(n).min())

def trix(x,n=(15,15,15)):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:trix
    sEMA = pd.ewma(x, span=n[0])
    dEMA = pd.ewma(x, span=n[1])
    tEMA  = pd.ewma(x, span=n[2])

    return tEMA/tEMA.shift(1)-1

def trueStrengthIndex(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:true_strength_index
    pc = x-x.shift(1)
    fs = pd.ewma(pc, span=25)
    ss = pd.ewma(fs, span=13)
    
    apc = pc.abs()
    afs = pd.ewma(apc, span=25)
    ass = pd.ewma(fs, span=13) 
    
    return 100*(ss/ass)

def ulcerIndex(close,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ulcer_index

    pdd = (close/close.rolling(14)-1)*100
    sqd = (pdd**2).rolling(14).mean()
    return sqa**(1/2.0)

def ultimateOscillator(close,high,low,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ultimate_oscillator
    bp = close-np.min(low,close.shift(1))

    tr = np.max(high,close.shift(1))-np.min(low,close.shift(1))
 
    av07 = bp.rolling(7).sum()/tr.rolling(7).sum()
    av14 = bp.rolling(14).sum()/tr.rolling(14).sum()
    av28 = bp.rolling(28).sum()/tr.rolling(28).sum()

    return 100*(4*Av07+2*Av14+1*Av28)/(4+2+1)

def vortexIndicator(x,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:vortex_indicator
    return None

def williamsR(close,high,low,n):
#http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:williams_r

    return (high.rolling(14).max()-close)/(high.rolling(14).max()-low.rolling(14).min())*-100    
    
    
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