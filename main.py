import math
import numpy as np

from classes import Settings, Label,FeatureArrays,FeatureSeries,FilterSettings,Filter,MLModel,FeatureSettings
from helper_functions import series_from
# Helper Functions

settings = Settings(
    input.source(title='Source', defval="close", group="General Settings", tooltip="Source of the input data"),
    input.int(title='Neighbors Count', defval=8, group="General Settings", minval=1, maxval=100, step=1, tooltip="Number of neighbors to consider"),
    input.int(title="Max Bars Back", defval=2000, group="General Settings"),
    input.int(title="Feature Count", defval=5, group="Feature Engineering", minval=2, maxval=5, tooltip="Number of features to use for ML predictions."),
    input.int(title="Color Compression", defval=1, group="General Settings", minval=1, maxval=10, tooltip="Compression factor for adjusting the intensity of the color scale."),
    input.bool(title="Show Default Exits", defval=False, group="General Settings", tooltip="Default exits occur exactly 4 bars after an entry signal. This corresponds to the predefined length of a trade during the model's training process.", inline="exits"),
    input.bool(title="Use Dynamic Exits", defval=False, group="General Settings", tooltip="Dynamic exits attempt to let profits ride by dynamically adjusting the exit threshold based on kernel regression logic.", inline="exits")
)

# Trade Stats Settings
# Note: The trade stats section is NOT intended to be used as a replacement for proper backtesting. It is intended to be used for calibration purposes only.
showTradeStats = True  # Boolean value
useWorstCase = False  # Boolean value


filterSettings = FilterSettings(
    input.bool(title="Use Volatility Filter", defval=True, tooltip="Whether to use the volatility filter.", group="Filters"),
    input.bool(title="Use Regime Filter", defval=True, group="Filters", inline="regime"),
    input.bool(title="Use ADX Filter", defval=False, group="Filters", inline="adx"),
    input.float(title="Threshold", defval=-0.1, minval=-10, maxval=10, step=0.1, tooltip="Whether to use the trend detection filter. Threshold for detecting Trending/Ranging markets.", group="Filters", inline="regime"),
    input.int(title="Threshold", defval=20, minval=0, maxval=100, step=1, tooltip="Whether to use the ADX filter. Threshold for detecting Trending/Ranging markets.", group="Filters", inline="adx")
)

filter = Filter(
    ml.filter_volatility(1, 10, filterSettings.useVolatilityFilter), 
    ml.regime_filter(ohlc4, filterSettings.regimeThreshold, filterSettings.useRegimeFilter),
    ml.filter_adx(settings.source, 14, filterSettings.adxThreshold, filterSettings.useAdxFilter)
)


# Feature Variables: User-Defined Inputs for calculating Feature Series.


f1_settings = FeatureSettings(
    title="Feature 1",
    options=["RSI", "WT", "CCI", "ADX"],
    defval="RSI",
    inline="01",
    tooltip="The first feature to use for ML predictions.",
    group="Feature Engineering"
)

f1_paramA = input.int(title="Parameter A", tooltip="The primary parameter of feature 1.", defval=14, inline="02", group="Feature Engineering")
f1_paramB = input.int(title="Parameter B", tooltip="The secondary parameter of feature 2 (if applicable).", defval=1, inline="02", group="Feature Engineering")

f2_settings = FeatureSettings(
    title="Feature 2",
    options=["RSI", "WT", "CCI", "ADX"],
    defval="WT",
    inline="03",
    tooltip="The second feature to use for ML predictions.",
    group="Feature Engineering"
)

f2_paramA = input.int(title="Parameter A", tooltip="The primary parameter of feature 2.", defval=10, inline="04", group="Feature Engineering")
f2_paramB = input.int(title="Parameter B", tooltip="The secondary parameter of feature 2 (if applicable).", defval=11, inline="04", group="Feature Engineering")

f3_settings = FeatureSettings(
    title="Feature 3",
    options=["RSI", "WT", "CCI", "ADX"],
    defval="CCI",
    inline="05",
    tooltip="The third feature to use for ML predictions.",
    group="Feature Engineering"
)

f3_paramA = input.int(title="Parameter A", tooltip="The primary parameter of feature 3.", defval=20, inline="06", group="Feature Engineering")
f3_paramB = input.int(title="Parameter B", tooltip="The secondary parameter of feature 3 (if applicable).", defval=1, inline="06", group="Feature Engineering")

f4_settings = FeatureSettings(
    title="Feature 4",
    options=["RSI", "WT", "CCI", "ADX"],
    defval="ADX",
    inline="07",
    tooltip="The fourth feature to use for ML predictions.",
    group="Feature Engineering"
)

f4_paramA = input.int(title="Parameter A", tooltip="The primary parameter of feature 4.", defval=20, inline="08", group="Feature Engineering")
f4_paramB = input.int(title="Parameter B", tooltip="The secondary parameter of feature 4 (if applicable).", defval=2, inline="08", group="Feature Engineering")

f5_settings = FeatureSettings(
    title="Feature 5",
    options=["RSI", "WT", "CCI", "ADX"],
    defval="RSI",
    inline="09",
    tooltip="The fifth feature to use for ML predictions.",
    group="Feature Engineering"
)

f5_paramA = input.int(title="Parameter A", tooltip="The primary parameter of feature 5.", defval=9, inline="10", group="Feature Engineering")
f5_paramB = input.int(title="Parameter B", tooltip="The secondary parameter of feature 5 (if applicable).", defval=1, inline="10", group="Feature Engineering")


featureSeries = FeatureSeries(
    series_from(f1_settings, close, high, low, hlc3, f1_paramA, f1_paramB),  # f1
    series_from(f2_settings, close, high, low, hlc3, f2_paramA, f2_paramB),  # f2
    series_from(f3_settings, close, high, low, hlc3, f3_paramA, f3_paramB),  # f3
    series_from(f4_settings, close, high, low, hlc3, f4_paramA, f4_paramB),  # f4
    series_from(f5_settings, close, high, low, hlc3, f5_paramA, f5_paramB)  # f5
)

# FeatureArrays Variables: Storage of Feature Series as Feature Arrays Optimized for ML
# Note: These arrays cannot be dynamically created within the FeatureArrays Object Initialization and thus must be set up in advance.

f1Array = np.array(featureSeries.f1)
f2Array = np.array(featureSeries.f2)
f3Array = np.array(featureSeries.f3)
f4Array = np.array(featureSeries.f4)
f5Array = np.array(featureSeries.f5)


featureArrays = FeatureArrays(
    f1Array,  # f1
    f2Array,  # f2
    f3Array,  # f3
    f4Array,  # f4
    f5Array  # f5
)

direction = Label(
    long=1,
    short=-1,
    neutral=0
)


# Derived from General Settings
max_bars_back_index = last_bar_index if last_bar_index >= settings.maxBarsBack else 0

# EMA Settings
use_ema_filter = input.bool(title="Use EMA Filter", defval=False, group="Filters", inline="ema")
ema_period = input.int(title="Period", defval=200, minval=1, step=1, group="Filters", inline="ema", tooltip="The period of the EMA used for the EMA Filter.")
is_ema_uptrend = close > ta.ema(close, ema_period) if use_ema_filter else True
is_ema_downtrend = close < ta.ema(close, ema_period) if use_ema_filter else True

use_sma_filter = input.bool(title="Use SMA Filter", defval=False, group="Filters", inline="sma")
sma_period = input.int(title="Period", defval=200, minval=1, step=1, group="Filters", inline="sma", tooltip="The period of the SMA used for the SMA Filter.")
is_sma_uptrend = close > ta.sma(close, sma_period) if use_sma_filter else True
is_sma_downtrend = close < ta.sma(close, sma_period) if use_sma_filter else True

# Nadaraya-Watson Kernel Regression Settings
use_kernel_filter = input.bool(True, "Trade with Kernel", group="Kernel Settings", inline="kernel")
show_kernel_estimate = input.bool(True, "Show Kernel Estimate", group="Kernel Settings", inline="kernel")
use_kernel_smoothing = input.bool(False, "Enhance Kernel Smoothing", tooltip="Uses a crossover based mechanism to smoothen kernel color changes. This often results in fewer color transitions overall and may result in more ML entry signals being generated.", inline='1', group='Kernel Settings')
h = input.int(8, 'Lookback Window', minval=3, tooltip='The number of bars used for the estimation. This is a sliding value that represents the most recent historical bars. Recommended range: 3-50', group="Kernel Settings", inline="kernel")
r = input.float(8., 'Relative Weighting', step=0.25, tooltip='Relative weighting of time frames. As this value approaches zero, the longer time frames will exert more influence on the estimation. As this value approaches infinity, the behavior of the Rational Quadratic Kernel will become identical to the Gaussian kernel. Recommended range: 0.25-25', group="Kernel Settings", inline="kernel")
x = input.int(25, "Regression Level", tooltip='Bar index on which to start regression. Controls how tightly fit the kernel estimate is to the data. Smaller values are a tighter fit. Larger values are a looser fit. Recommended range: 2-25', group="Kernel Settings", inline="kernel")
lag = input.int(2, "Lag", tooltip="Lag for crossover detection. Lower values result in earlier crossovers. Recommended range: 1-2", inline='1', group='Kernel Settings')

# Display Settings
show_bar_colors = input.bool(True, "Show Bar Colors", tooltip="Whether to show the bar colors.", group="Display Settings")
show_bar_predictions = input.bool(defval=True, title="Show Bar Prediction Values", tooltip="Will show the ML model's evaluation of each bar as an integer.", group="Display Settings")
use_atr_offset = input.bool(defval=False, title="Use ATR Offset", tooltip="Will use the ATR offset instead of the bar prediction offset.", group="Display Settings")
bar_predictions_offset = input.float(0, "Bar Prediction Offset", minval=0, tooltip="The offset of the bar predictions as a percentage from the bar high or close.", group="Display Settings")

# Next Bar Classification
# This model specializes in predicting the direction of price action over the next 4 bars.
# To avoid complications with the ML model, this value is hardcoded to 4 bars, but support for other training lengths may be added in the future.
src = settings.source
y_train_series = [direction.short if src[i + 4] < src[i] else direction.long if src[i + 4] > src[i] else direction.neutral for i in range(len(src))]
y_train_array = []

# Variables used for ML Logic
predictions = []
prediction = 0.0
signal = direction.neutral
distances = []


# =========================
# ====  Core ML Logic  ====
# =========================

last_distance = -1.0
size = min(settings.maxBarsBack - 1, len(y_train_array) - 1)
size_loop = min(settings.maxBarsBack - 1, size)

if bar_index >= max_bars_back_index:
    for i in range(size_loop + 1):
        d = get_lorentzian_distance(i, settings.featureCount, featureSeries, featureArrays)
        if d >= last_distance and i % 4 != 0:
            last_distance = d
            distances.append(d)
            predictions.append(round(y_train_array[i]))
            if len(predictions) > settings.neighborsCount:
                last_distance = distances[round(settings.neighborsCount * 3 / 4)]
                distances.pop(0)
                predictions.pop(0)

    prediction = sum(predictions)


# User Defined Filters: Used for adjusting the frequency of the ML Model's predictions
filter_all = filter.volatility and filter.regime and filter.adx

# Filtered Signal: The model's prediction of future price movement direction with user-defined filters applied
if prediction > 0 and filter_all:
    signal = direction.long
elif prediction < 0 and filter_all:
    signal = direction.short
else:
    signal = signal[1]

# Bar-Count Filters: Represents strict filters based on a pre-defined holding period of 4 bars
bars_held = 0
if ta.change(signal):
    bars_held = 0
else:
    bars_held += 1
is_held_four_bars = bars_held == 4
is_held_less_than_four_bars = 0 < bars_held < 4

# Fractal Filters: Derived from relative appearances of signals in a given time series fractal/segment with a default length of 4 bars
is_different_signal_type = ta.change(signal)
is_early_signal_flip = ta.change(signal) and (ta.change(signal[1]) or ta.change(signal[2]) or ta.change(signal[3]))
is_buy_signal = signal == direction.long and isEmaUptrend and isSmaUptrend
is_sell_signal = signal == direction.short and isEmaDowntrend and isSmaDowntrend
is_last_signal_buy = signal[4] == direction.long and isEmaUptrend[4] and isSmaUptrend[4]
is_last_signal_sell = signal[4] == direction.short and isEmaDowntrend[4] and isSmaDowntrend[4]
is_new_buy_signal = is_buy_signal and is_different_signal_type
is_new_sell_signal = is_sell_signal and is_different_signal_type

# Kernel Regression Filters: Filters based on Nadaraya-Watson Kernel Regression using the Rational Quadratic Kernel
# For more information on this technique refer to my other open source indicator located here: 
# https:#www.tradingview.com/script/AWNvbPRM-Nadaraya-Watson-Rational-Quadratic-Kernel-Non-Repainting/
c_green = '#009988'
c_red = '#CC3311'
transparent = '#000000'
yhat1 = kernels.rationalQuadratic(settings.source, h, r, x)
yhat2 = kernels.gaussian(settings.source, h-lag, x)
kernel_estimate = yhat1
# Kernel Rates of Change
was_bearish_rate = yhat1[2] > yhat1[1]
was_bullish_rate = yhat1[2] < yhat1[1]
is_bearish_rate = yhat1[1] > yhat1
is_bullish_rate = yhat1[1] < yhat1
is_bearish_change = is_bearish_rate and was_bullish_rate
is_bullish_change = is_bullish_rate and was_bearish_rate
# Kernel Crossovers
is_bullish_cross_alert = ta.crossover(yhat2, yhat1)
is_bearish_cross_alert = ta.crossunder(yhat2, yhat1) 
is_bullish_smooth = yhat2 >= yhat1
is_bearish_smooth = yhat2 <= yhat1
# Kernel Colors
color_by_cross = c_green if is_bullish_smooth else c_red
color_by_rate = c_green if is_bullish_rate else c_red
plot_color = color_by_cross if showKernelEstimate else transparent
plot(kernel_estimate, color=plot_color, linewidth=2, title="Kernel Regression Estimate")
# Alert Variables
alert_bullish = is_bullish_cross_alert if useKernelSmoothing else is_bullish_change
alert_bearish = is_bearish_cross_alert if useKernelSmoothing else is_bearish_change
# Bullish and Bearish Filters based on Kernel
is_bullish = is_bullish_smooth if useKernelFilter else True if useKernelFilter else True
is_bearish = is_bearish_smooth if useKernelFilter else True if useKernelFilter else True

# Entry Conditions: Booleans for ML Model Position Entries
start_long_trade = isNewBuySignal and is_bullish and isEmaUptrend and isSmaUptrend
start_short_trade = isNewSellSignal and is_bearish and isEmaDowntrend and isSmaDowntrend

# Dynamic Exit Conditions: Booleans for ML Model Position Exits based on Fractal Filters and Kernel Regression Filters
last_signal_was_bullish = ta.barssince(start_long_trade) < ta.barssince(start_short_trade)
last_signal_was_bearish = ta.barssince(start_short_trade) < ta.barssince(start_long_trade)
bars_since_red_entry = ta.barssince(start_short_trade)
bars_since_red_exit = ta.barssince(alertBullish)
bars_since_green_entry = ta.barssince(start_long_trade)
bars_since_green_exit = ta.barssince(alertBearish)
is_valid_short_exit = bars_since_red_exit > bars_since_red_entry
is_valid_long_exit = bars_since_green_exit > bars_since_green_entry
end_long_trade_dynamic = (is_bearish_change and is_valid_long_exit[1])
end_short_trade_dynamic = (is_bullish_change and is_valid_short_exit[1])

# Fixed Exit Conditions: Booleans for ML Model Position Exits based on a Bar-Count Filters
end_long_trade_strict = ((is_held_four_bars and is_last_signal_buy) or (is_held_less_than_four_bars and isNewSellSignal and is_last_signal_buy)) and start_long_trade[4]
end_short_trade_strict = ((is_held_four_bars and is_last_signal_sell) or (is_held_less_than_four_bars and isNewBuySignal and is_last_signal_sell)) and start_short_trade[4]
is_dynamic_exit_valid = not useEmaFilter and not useSmaFilter and not useKernelSmoothing
end_long_trade = settings.useDynamicExits and is_dynamic_exit_valid ? end_long_trade_dynamic : end_long_trade_strict 
end_short_trade = settings.useDynamicExits and is_dynamic_exit_valid ? end_short_trade_dynamic : end_short_trade_strict

# Plotting Labels
plotshape(start_long_trade ? low : na, 'Buy', shape.labelup, location.belowbar, color=ml.color_green(prediction), size=size.small, offset=0)
plotshape(start_short_trade ? high : na, 'Sell', shape.labeldown, location.abovebar, ml.color_red(-prediction), size=size.small, offset=0)
plotshape(end_long_trade and settings.showExits ? high : na, 'StopBuy', shape.xcross, location.absolute, color='#3AFF17', size=size.tiny, offset=0)
plotshape(end_short_trade and settings.showExits ? low : na, 'StopSell', shape.xcross, location.absolute, color='#FD1707', size=size.tiny, offset=0)


# Separate Alerts for Entries and Exits
alertcondition(start_long_trade, title='Open Long ?', message='LDC Open Long ? | {{ticker}}@{{close}} | ({{interval}})')
alertcondition(end_long_trade, title='Close Long ?', message='LDC Close Long ? | {{ticker}}@{{close}} | ({{interval}})')
alertcondition(start_short_trade, title='Open Short ?', message='LDC Open Short  | {{ticker}}@{{close}} | ({{interval}})')
alertcondition(end_short_trade, title='Close Short ?', message='LDC Close Short ? | {{ticker}}@{{close}} | ({{interval}})')

# Combined Alerts for Entries and Exits
alertcondition(start_short_trade or start_long_trade, title='Open Position ??', message='LDC Open Position ?? | {{ticker}}@{{close}} | ({{interval}})')
alertcondition(end_short_trade or end_long_trade, title='Close Position ??', message='LDC Close Position  ?? | {{ticker}}@[{{close}}] | ({{interval}})')

# Kernel Estimate Alerts
alertcondition(condition=alertBullish, title='Kernel Bullish Color Change', message='LDC Kernel Bullish ? | {{ticker}}@{{close}} | ({{interval}})')
alertcondition(condition=alertBearish, title='Kernel Bearish Color Change', message='LDC Kernel Bearish ? | {{ticker}}@{{close}} | ({{interval}})')

# Display Signals
atrSpaced = useAtrOffset ? ta.atr(1) : na
compressionFactor = settings.neighborsCount / settings.colorCompression
c_pred = (color.green if prediction > 0 else color.red) if prediction != 0 else color.gray
c_label = c_pred if showBarPredictions else color.gray
c_bars = color.new(c_pred, 50) if showBarColors else color.gray
x_val = bar_index
y_val = (high + atrSpaced) if prediction > 0 else (low - atrSpaced)
label.new(x_val, y_val, str.tostring(prediction), xloc.bar_index, yloc.price, color.new(color.white, 100), style=label.style_label_up, textcolor=c_label, size=size.normal, textalign=text.align_left)
barcolor(c_bars)

# Backtesting
backTestStream = switch(
    startLongTrade, 1,
    endLongTrade, 2,
    startShortTrade, -1,
    endShortTrade, -2,
    0
)
plot(backTestStream, "Backtest Stream", display=display.none)

# Display real-time trade stats
def init_table():
    c_transparent = color.new(color.black, 100)
    tbl = table.new(position.top_right, columns=2, rows=7, frame_color=c_transparent, frame_width=1, border_width=1, border_color=c_transparent)

def update_table(tbl, tradeStatsHeader, totalTrades, totalWins, totalLosses, winLossRatio, winRate, stopLosses):
    c_transparent = color.new(color.black, 100)
    table.cell(tbl, 0, 0, tradeStatsHeader, text_halign=text.align_center, text_color=color.gray, text_size=size.normal)
    table.cell(tbl, 0, 1, 'Winrate', text_halign=text.align_center, bgcolor=c_transparent, text_color=color.gray, text_size=size.normal)
    table.cell(tbl, 1, 1, str.tostring(totalWins / totalTrades, '#.#%'), text_halign=text.align_center, bgcolor=c_transparent, text_color=color.gray, text_size=size.normal)
    table.cell(tbl, 0, 2, 'Trades', text_halign=text.align_center, bgcolor=c_transparent, text_color=color.gray, text_size=size.normal)
    table.cell(tbl, 1, 2, str.tostring(totalTrades, '#') + ' (' + str.tostring(totalWins, '#') + '|' + str.tostring(totalLosses, '#') + ')', text_halign=text.align_center, bgcolor=c_transparent, text_color=color.gray, text_size=size.normal)
    table.cell(tbl, 0, 5, 'WL Ratio', text_halign=text.align_center, bgcolor=c_transparent, text_color=color.gray, text_size=size.normal)
    table.cell(tbl, 1, 5, str.tostring(totalWins / totalLosses, '0.00'), text_halign=text.align_center, bgcolor=c_transparent, text_color=color.gray, text_size=size.normal)
    table.cell(tbl, 0, 6, 'Early Signal Flips', text_halign=text.align_center, bgcolor=c_transparent, text_color=color.gray, text_size=size.normal)
    table.cell(tbl, 1, 6, str.tostring(totalEarlySignalFlips, '#'), text_halign=text.align_center, bgcolor=c_transparent, text_color=color.gray, text_size=size.normal)

if showTradeStats:
    tbl = init_table()
    if barstate.islast:
        update_table(tbl, tradeStatsHeader, totalTrades, totalWins, totalLosses, winLossRatio, winRate, totalEarlySignalFlips)
