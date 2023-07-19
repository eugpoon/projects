import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from plotly import express as px, graph_objs as go, io as pio
import json
from sklearn.linear_model import *
from sklearn.model_selection import *
pio.renderers.default = 'notebook'

def plotDist(df, col, title, x):
    fig, axs = plt.subplots(ncols=2, figsize=(10, 5))

    sns.distplot(df[col[0]], color='c', ax=axs[0],
             hist_kws={'linewidth':0, 'alpha':0.4}, 
             kde_kws={'color':'blue'}, label='pre')

    sns.distplot(df[col[1]], color='pink', ax=axs[0],
             hist_kws={'linewidth':0, 'alpha':0.4}, 
             kde_kws={'color':'red'}, label='post')

    sns.distplot(df.change, color='grey', ax=axs[1],
             hist_kws={'linewidth':0, 'alpha':0.4}, 
             kde_kws={'color':'black'}, label='difference')

    fig.suptitle(title)
    fig.supxlabel(x)
    axs[0].set(xlabel='')
    axs[1].set(xlabel='', ylabel='')
    axs[0].legend()
    axs[1].legend()
    plt.show()


def plotChoro(df, col, var):
    centroid = df.dissolve().centroid[0]  # center of map
    visible = np.array(col)  # chosen col
    traces = []
    buttons = []

    def name(s: str):
        if 'PRE' in s:
            return 'Pre-Lockdown Average ' + var
        elif 'POS' in s:
            return 'Post-Lockdown Average ' + var
        elif s == 'change':
            return 'Difference in Average ' + var

    for value in col:
        traces.append(go.Choroplethmapbox(
            geojson=json.loads(df.geometry.to_json()),
            locations=df.index,
            z=df[value].astype(float),  # color
            visible=True if value == col[0] else False,
            colorbar_title=var,
            colorscale=[[0, 'rgba(255,255,255, 0.7)'], [
                1, 'rgba(63, 0, 255, 0.7)']],
        )
        )
        buttons.append(dict(label=value, method='update',
                            args=[{'visible': list(visible == value)},
                                  {'title': name(value)}]))

    updatemenus = [{'active': 0, 'buttons': buttons}]

    fig = go.Figure(data=traces, layout=dict(updatemenus=updatemenus))
    fig.update_layout(height=700, width=700,
                      title=name(col[0]), title_x=0.5,
                      mapbox={'center': {'lat': centroid.y,
                                         'lon': centroid.x}, 'zoom': 8.5},
                      mapbox_style='carto-positron',
                      )
    fig.show()


def ts_to_np(df: pd.DataFrame, T: int, n_val: int):
    '''Convert 'time series' into train-validate splits, in numpy
    Assume dataframe contains a 'day' column starting from 1 and are consecutive.

    :param df: dataframe with dimensions n x d
        n: number of samples (census tracts)
        d: number of dimensions (days)
        values are the speeds
    :param T: number of days to include in each training sample
    :param n_val: number of days to hold out for the validation set
    :return: Set of 4 numpy arrays - X_train, y_train, X_val, y_val - where
        X_* arrays are (n, T) and y_* arrays are (n,).
    '''
    X_train = pd.DataFrame()
    for t in np.arange(df.shape[1]-T-n_val):
        x_t = df.iloc[:, t:t+T].values
        X_train = pd.concat([X_train, pd.DataFrame(x_t)])

    X_val = pd.DataFrame()
    for t in np.arange(df.shape[1]-T-n_val, df.shape[1]-T):
        x_v = df.iloc[:, t:t+T].values
        X_val = pd.concat([X_val, pd.DataFrame(x_v)])

    y_train = df.iloc[:, T:df.shape[1]-n_val].to_numpy().T.flatten()
    y_val = df.iloc[:, df.shape[1]-n_val:].to_numpy().T.flatten()

    return X_train.to_numpy(), y_train, X_val.to_numpy(), y_val


def remove_nans(X: np.array, y: np.array):
    '''Remove all nans from the provided (X, y) pair.

    :param X: (n, T) array of model inputs
    :param y: (n,) array of labels
    :return:  (X, y)
    '''
    if not len(X):
        return X, y
    df = pd.DataFrame(X)
    df['y'] = y
    df = df.dropna()
    return df.iloc[:, :X.shape[1]].to_numpy(), df.iloc[:, X.shape[1]].to_numpy()


def ts_to_ds(time_series: pd.DataFrame, T: int, n_val: int):
    '''Convert 'time series' dataframe to a numpy dataset.

    Uses utilites above 'ts_to_np' and 'remove_nans'

    For description of arguments, see 'ts_to_np' docstring.
    '''
    answer = ts_to_np(time_series, T, n_val)
    train = remove_nans(answer[0], answer[1])
    val = remove_nans(answer[2], answer[3])
    return train[0], train[1], val[0], val[1]


def trainTS(data, model=None, plotT=False, plotV=False):
    X_train, y_train, X_val, y_val = data
    if model == None:
        model = LinearRegression().fit(X_train, y_train)
    
    if plotT==True:
        print(f'test r2: {model.score(X_train, y_train):.3f}')
        X = X_train
        Y = y_train
        title = 'Test: '
    if plotV==True:
        print(f'train r2: {model.score(X_train, y_train):.3f}')
        print(f'val r2: {model.score(X_val, y_val):.3f}')
        X = X_val
        Y = y_val
        title = 'Val: '
        
    if (plotT == True) or (plotV == True): 
        sns.regplot(x=model.predict(X), y=Y, marker='.', color='#592693', 
            scatter_kws={'s': 10, 'alpha':0.4}, 
            line_kws={'lw':1, 'color': 'black'})
        plt.title(title +'Predicted vs Actual Speed Averages')
        plt.xlabel('Predicted Speed Averages')
        plt.ylabel('Actual Speed Averages')
        plt.show()
    return model



def performance(model, ts, start):
    scores = []
    for i in np.arange(start, 31):
        prev_five = ts.iloc[:, i-5:i+1].dropna()
        scr = model.score(prev_five.iloc[:, :5], prev_five.iloc[:, 5])
        scores += [scr]

    plt.plot(np.arange(start+1, 32), scores)
    plt.title('Model Performance')
    plt.xlabel('Day of Month')
    plt.ylabel('Score')
    plt.show()

# +
# fig = px.choropleth_mapbox(pre_post,
#                            geojson=pre_post.geometry,
#                            locations=pre_post.index,
#                            color='speedPRE',
#                            center={'lat': centroid.y, 'lon':centroid.x},
#                            mapbox_style='carto-positron',
#                           )
# fig.show()

# fig, axs = plt.subplots(ncols=2, figsize=(20, 10))
# cx.add_basemap(pre_post.to_crs(epsg=3857).plot(column='speedPRE',
#                legend_kwds={'label': 'Average Traffic Speed (mph)'},
#                legend=True, alpha=0.7, figsize=(10, 10), ax=axs[0]))
# cx.add_basemap(pre_post.to_crs(epsg=3857).plot(column='change',
#                legend_kwds={'label': 'Average Traffic Speed Difference (mph)'},
#                legend=True, alpha=0.7, figsize=(10, 10), ax=axs[1]))
# axs[0].set(title='Pre-Lockdown Average Traffic Speed By Census Tract',
#            xlabel='Longitude', ylabel='Latitude')
# axs[1].set(title='Average Traffic Speed Difference By Census Tract',
#            xlabel='Longitude', ylabel='Latitude')
# plt.show()

# fig, axs = plt.subplots(ncols=2, figsize=(20, 10))
# cx.add_basemap(pre_time_geo.to_crs(epsg=3857).plot(column='Mean Travel Time (Seconds)',
#                legend_kwds={'label': 'Average Travel Time (seconds)'},
#                legend=True, alpha=0.7, figsize=(10, 10), ax=axs[0]))
# cx.add_basemap(pre_pos_time_geo.to_crs(epsg=3857).plot(column='post-pre diff',
#                legend_kwds={
#                    'label': 'Average Travel Time Difference (seconds)'},
#                legend=True, alpha=0.7, figsize=(10, 10), ax=axs[1]))
# axs[0].set(title='Pre-Lockdown Average Travel Time By Census Tract',
#            xlabel='Longitude', ylabel='Latitude')
# axs[1].set(title='Average Travel Time Difference Pre and Post-lockdown By Cluster',
#            xlabel='Longitude', ylabel='Latitude')
