class Settings:
    def __init__(self, source, neighborsCount, maxBarsBack, featureCount, colorCompression, showExits, useDynamicExits):
        self.source = source
        self.neighborsCount = neighborsCount
        self.maxBarsBack = maxBarsBack
        self.featureCount = featureCount
        self.colorCompression = colorCompression
        self.showExits = showExits
        self.useDynamicExits = useDynamicExits

class Label:
    def __init__(self, long, short, neutral):
        self.long = long
        self.short = short
        self.neutral = neutral

class FeatureArrays:
    def __init__(self, f1, f2, f3, f4, f5):
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.f4 = f4
        self.f5 = f5

class FeatureSeries:
    def __init__(self, f1, f2, f3, f4, f5):
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.f4 = f4
        self.f5 = f5

class MLModel:
    def __init__(self, firstBarIndex, trainingLabels, loopSize, lastDistance, distancesArray, predictionsArray, prediction):
        self.firstBarIndex = firstBarIndex
        self.trainingLabels = trainingLabels
        self.loopSize = loopSize
        self.lastDistance = lastDistance
        self.distancesArray = distancesArray
        self.predictionsArray = predictionsArray
        self.prediction = prediction

class FilterSettings:
    def __init__(self, useVolatilityFilter, useRegimeFilter, useAdxFilter, regimeThreshold, adxThreshold):
        self.useVolatilityFilter = useVolatilityFilter
        self.useRegimeFilter = useRegimeFilter
        self.useAdxFilter = useAdxFilter
        self.regimeThreshold = regimeThreshold
        self.adxThreshold = adxThreshold

class Filter:
    def __init__(self, volatility, regime, adx):
        self.volatility = volatility
        self.regime = regime
        self.adx = adx

class FeatureSettings:
    def __init__(self, title, options, defval, inline, tooltip, group):
        self.title = title
        self.options = options
        self.defval = defval
        self.inline = inline
        self.tooltip = tooltip
        self.group = group