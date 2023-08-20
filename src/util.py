import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from scipy.stats import t
import plotly.express as px
import plotly.graph_objects as go


ycol_name_mapper = {
    'PctChange1':'First Day Return',
    'Close_Open':'Close-to-Open Return',
    'Open_Ret':'Opening Return',
    '20d_ret':'Next 20 Days Return',
}

def load_ipo_data(include_ace = True):
    sector_df = pd.read_excel("mk_sector.xlsx")
    fdf = pd.read_excel("IPO_2.xlsx")
    fdf['Code'] = fdf['Code'].apply(lambda x: str(x).zfill(4))
    sector_df['Code'] = sector_df['ID_ISIN'].apply(lambda x: str(x)[3:7])
    fdf = pd.merge(fdf, sector_df[['Code', 'EXCH_MKT_GRP', 'GICS_SECTOR_NAME']], left_on="Code", right_on="Code", how="left")
    fdf['IPO_Price'] = fdf['Close'] - fdf['PriceChange1']
    fdf['Open_Ret'] = fdf['Open']/fdf['IPO_Price'] - 1
    fdf['Close_Open'] = fdf['Close'] - fdf['Open']

    cols = ['Code', 'Date', 'Stock', 'StockLongName',
            'EXCH_MKT_GRP', 'GICS_SECTOR_NAME',
            'Open', 'Close', 'Volume', 'NumShares',
            'PriceChange1', 'PctChange1', 'Open_Ret', 'Close_Open', '20d_ret', 
            'Oversubscription_Rate', 'Total_Applicants']
    fdf2 = fdf[cols].sort_values("Date").copy()

    enc = OneHotEncoder(handle_unknown='ignore', drop='first')
    X = fdf2[['EXCH_MKT_GRP', 'GICS_SECTOR_NAME']]
    enc.fit(X)
    new_feats = enc.transform(X).toarray()

    xcols = ['Oversubscription_Rate', 'Total_Applicants']
    for j in range(new_feats.shape[1]):
        fdf2[f"C{j}"] = new_feats[:, j].astype(int)
        xcols += [f"C{j}"]
    u1 = fdf2[['Total_Applicants', 'Oversubscription_Rate']].isna().mean(axis=1) ==  0
    fdf3 = fdf2.loc[u1].copy()
    if not include_ace:
        u2 = fdf3['EXCH_MKT_GRP'].isin(['ACE'])
        fdf3 = fdf3.loc[~u2].copy()
    return fdf3


class Model():
    def __init__(self, fdf3, ycol, selected_stocks=None):
        # ycol = 'PctChange1'
        xcols = ['Oversubscription_Rate','Total_Applicants',]
        

        df = fdf3.copy()
        X,y = df[xcols], df[ycol]
        if selected_stocks is None:
            X_train = X.iloc[:-1]
            X_test = X.iloc[-1:]
            y_train = y.iloc[:-1]
            y_test = y.iloc[-1:]
            names = df.iloc[-1:]['Stock']
        else:
            u1 = df['Stock'].isin(selected_stocks)
            X_test,y_test = df.loc[u1, xcols], df.loc[u1, ycol]
            X_train,y_train = df.loc[~u1, xcols], df.loc[~u1, ycol]
            names = df.loc[u1,'Stock']
        

        # fdf3.shape,X_train.shape, X_test.shape
        model_ = HuberRegressor()
        model_.fit(X_train,y_train)
        self.model_ = model_
        y_pred_train = model_.predict(X_train)
        y_pred_test = model_.predict(X_test)

        X_train_2 = sm.add_constant(X_train)  # Adding a constant term for the intercept
        X_test_2 = sm.add_constant(X_test)  # Adding a constant term for the intercept
        model = sm.OLS(y_train, X_train_2).fit()
        self.model = model
        predictions = model.predict(X_test_2)

        # Getting predictions and standard errors
        std_errors = model.get_prediction(X_test_2).se_mean
        mae = mean_absolute_error(y_train, y_pred_train)

        # Calculate prediction intervals
        alpha = 0.05  # significance level
        degrees_freedom = model.df_resid  # degrees of freedom
        t_value = np.abs(t.ppf(alpha / 2, df=degrees_freedom))  # t-score for two-tailed test
        # print(std_errors)
        # lower_bound_test = y_pred_test - t_value * std_errors
        # upper_bound_test = y_pred_test + t_value * std_errors
        # print(mae, y_pred_test)
        lower_bound_test = y_pred_test - mae
        upper_bound_test = y_pred_test + mae
        err_vals = [t_value*j for j in std_errors]
        
        # Create a scatter plot
        xp = fdf3[xcols[0]]
        yp = fdf3[ycol]
        yname = ycol_name_mapper[ycol]
        fig = px.scatter(x=xp, y=yp, title=f"{yname} vs Oversubscription Rate", color_discrete_sequence=['lightblue'])

        # Add a linear regression line using plotly.graph_objects
        regression_line = np.polyfit(xp, yp, 1)
        line_x = np.linspace(min(xp), max(xp), 50)
        line_y = np.polyval(regression_line, line_x)
        fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', line=dict(color='lightcoral'), name='Linear Regression'))

        # Calculate the coefficients for the regression equation
        slope, intercept = regression_line
        regression_equation = f'y = {slope:.2f}x + {intercept:.2f}'

        # Add the regression equation as an annotation
        fig.add_annotation(
            x=line_x[len(line_x) // 2]+np.max(xp)/3,  # X position for the annotation
            y=line_y[len(line_y) // 2],  # Y position for the annotation
            text=regression_equation,    # Text for the annotation
            showarrow=False,             # No arrow for the annotation
            font=dict(size=12)           # Font size for the annotation text
        )
        for err_val, x, y_act, y, lb, ub, name in zip(err_vals, X_test['Oversubscription_Rate'], y_test, y_pred_test, lower_bound_test,upper_bound_test, names):
            # error_y = [lb, ub]  # Error of [1, 2] for the 50th point
            # error_y = [y-mae, y+mae]  # Error of [1, 2] for the 50th point
            # print(name, error_y)
            error_y = [mae]
            error_y = [err_val]
            highlight_trace = go.Scatter(x=[x], y=[y], mode='markers', error_y=dict(type='data', array=error_y, visible=True) , marker=dict(color='darkblue'), name=f'{name} (Prediction)')
            fig.add_trace(highlight_trace)

            trace_2 = go.Scatter(x=[x], y=[y_act], mode='markers', marker=dict(color='darkred'), name=f'{name} (Actual)')
            fig.add_trace(trace_2)

        # Show the plot
        fig.update_traces(showlegend=False, selector=dict(name="Linear Regression"))
        # fig.update_layout(y_title='as', template="plotly_white")
        fig.update_layout(xaxis_title="Oversubscription Rate", yaxis_title=yname, xaxis_tickformat=',.1f', yaxis_tickformat=',.1%')
        self.fig = fig
    
    def predict(self, x1, x2):
        y_pred = self.model_.predict([[x1, x2]])
        std_errors = self.model.get_prediction([[1.0, x1, x2]]).se_mean

        # Calculate prediction intervals
        alpha = 0.05  # significance level
        degrees_freedom = self.model.df_resid  # degrees of freedom
        alpha = 0.05
        t_value = np.abs(t.ppf(alpha / 2, df=degrees_freedom))  # t-score for two-tailed test
        lower_bound = y_pred - t_value * std_errors
        upper_bound = y_pred + t_value * std_errors
        return [lower_bound[0], y_pred[0], upper_bound[0]]
        
        