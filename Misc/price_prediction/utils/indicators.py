def get_sma(prices, window):
    """
    Return Simple Moving Average given prices and window    
    :param prices: Series of Prices  			  		 			     			  	 
    """
    sma = prices.rolling(window).mean()
    return sma

def get_rsi(prices, window=14):

    momentum = prices.diff()[1:]
    price_up, price_down = momentum.copy(), momentum.copy()
    price_up[price_up < 0] = 0
    price_down[price_down > 0] = 0
    roll_up = get_sma(price_up,window)
    roll_down = -1 * get_sma(price_down,window)

    return 100 * roll_up / (roll_up+roll_down)

def get_momentum(prices, window=14):
    """
    Momentum measures the rate of the rise/fall in stock prices.
    
    :param prices: Series of Prices
    :type prices: pandas.Series
    :param window: look-back window
    :type window: int
    """
    momentum = (prices / prices.shift(window)) - 1

    return momentum


def get_macd_signal(prices, short_window=12, long_window=26, sig_window=9):

    ema_12 = prices.ewm(span=short_window, adjust=False).mean()
    ema_26 = prices.ewm(span=long_window, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=sig_window, adjust=False).mean()
    macd_hist = macd - signal

    return (macd,signal,macd_hist)