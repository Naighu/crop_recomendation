import statsmodels.api as sm


def predict_price(crop_file,steps):
    results = sm.load_pickle(crop_file)
        # Get forecast 20 steps ahead in future
    pred_uc = results.get_forecast(steps=steps)

    # Get confidence intervals of forecasts
    pred_ci = pred_uc.conf_int()
    return pred_ci

# print(predict_price("mango.pkl",5))