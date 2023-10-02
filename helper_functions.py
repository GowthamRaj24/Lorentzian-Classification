import math 
# import ta
import pandas_ta as ta

def series_from(feature_string, _close, _high, _low, _hlc3, f_paramA, f_paramB):
    if feature_string == "RSI":
        return ta.momentum.RSIIndicator(close=_close)
        return ml.n_rsi(_close, f_paramA, f_paramB)
    elif feature_string == "WT":
        return ml.n_wt(_hlc3, f_paramA, f_paramB)
    elif feature_string == "CCI":
        return ml.n_cci(_close, f_paramA, f_paramB)
    elif feature_string == "ADX":
        return ml.n_adx(_high, _low, _close, f_paramA)

def get_lorentzian_distance(i, featureCount, featureSeries, featureArrays):
    distance = 0
    
    for j in range(1, featureCount + 1):
        diff = featureSeries['f{}'.format(j)] - featureArrays['f{}'.format(j)][i]
        distance += math.log(1 + abs(diff))
    
    return distance
