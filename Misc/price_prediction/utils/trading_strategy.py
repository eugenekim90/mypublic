def trade(predicted_price, current_price, capital, quantity, trading_history):
    margin = 0.0005
    buy_threshold = (1+margin)*current_price
    sell_threshold = (1-margin)*current_price
    if predicted_price >= buy_threshold:
        if quantity < 0 :
            print("COVER SHORT SIGNAL!: Price will go up, covering short")
            cover_cost = abs(quantity) * current_price
            capital -= cover_cost 
            trading_history.append({'action': 'Cover Short', 'price': current_price, 'quantity': abs(quantity), 'capital': capital})
            quantity = 0
        else:
            print('BUY SIGNAL!!!: Price will go up')
            if capital > 0:
                quantity += capital / current_price
                trading_history.append({'action': 'Buy', 'price': current_price, 'quantity': quantity, 'capital': 0})
                capital = 0
    elif predicted_price <= sell_threshold:
        if quantity > 0:
            print('SELL SIGNAL!!: Price will go down')
            capital += quantity * current_price
            trading_history.append({'action': 'Sell', 'price': current_price, 'quantity': quantity, 'capital': capital})
            quantity = 0
        else:
            print('SHORT SELL SIGNAL!!: Price will go down, initiating short sell')
            short_sell_quantity = capital / current_price
            quantity -= short_sell_quantity
            capital += short_sell_quantity * current_price
            trading_history.append({'action': 'Short Sell', 'price': current_price, 'quantity': short_sell_quantity, 'capital': capital})
    else:
        print('No signal!!')
    return capital, quantity, trading_history
