import math 
import talib

def series_from(feature_string, _close, _high, _low, _hlc3, f_paramA, f_paramB):
    if feature_string == "RSI":
        return ml.n_rsi(_close, f_paramA, f_paramB)
    elif feature_string == "WT":
        return ml.n_wt(_hlc3, f_paramA, f_paramB)
    elif feature_string == "CCI":
        return ml.n_cci(_close, f_paramA, f_paramB)
    elif feature_string == "ADX":
        return ml.n_adx(_high, _low, _close, f_paramA)

def get_lorentzian_distance(i, featureCount, featureSeries, featureArrays):
    if featureCount == 5:
        return (math.log(1 + abs(featureSeries.f1 - featureArrays.f1[i])) +
                math.log(1 + abs(featureSeries.f2 - featureArrays.f2[i])) +
                math.log(1 + abs(featureSeries.f3 - featureArrays.f3[i])) +
                math.log(1 + abs(featureSeries.f4 - featureArrays.f4[i])) +
                math.log(1 + abs(featureSeries.f5 - featureArrays.f5[i])))
    elif featureCount == 4:
        return (math.log(1 + abs(featureSeries.f1 - featureArrays.f1[i])) +
                math.log(1 + abs(featureSeries.f2 - featureArrays.f2[i])) +
                math.log(1 + abs(featureSeries.f3 - featureArrays.f3[i])) +
                math.log(1 + abs(featureSeries.f4 - featureArrays.f4[i])))
    elif featureCount == 3:
        return (math.log(1 + abs(featureSeries.f1 - featureArrays.f1[i])) +
                math.log(1 + abs(featureSeries.f2 - featureArrays.f2[i])) +
                math.log(1 + abs(featureSeries.f3 - featureArrays.f3[i])))
    elif featureCount == 2:
        return (math.log(1 + abs(featureSeries.f1 - featureArrays.f1[i])) +
                math.log(1 + abs(featureSeries.f2 - featureArrays.f2[i])))
