import os
import warnings
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler


warnings.filterwarnings('ignore')

target = 'contest-tmp2m-14d__tmp2m'


def downloadData():
    # download from kaggle
    # setup: https://github.com/Kaggle/kaggle-api
    # #     !kaggle competitions download -c widsdatathon2023
    z = ZipFile('widsdatathon2023.zip')
    train = pd.read_csv(z.open('train_data.csv'), parse_dates=[
                        "startdate"]).drop('index', axis=1)
    test = pd.read_csv(z.open('test_data.csv'), parse_dates=[
                       "startdate"]).drop('index', axis=1)
    submit = pd.read_csv(z.open('sample_solution.csv'))
    train.to_pickle('train_data.pkl')
    test.to_pickle('test_data.pkl')
    submit.to_pickle('sample_solution.pkl')

    z = ZipFile('data/widsdatathon2023pkl.zip', 'w', ZIP_DEFLATED)
    for path in ['train_data.pkl', 'test_data.pkl', 'sample_solution.pkl']:
        z.write(path, os.path.basename(path))
    z.close()
# #     !rm widsdatathon2023.zip *.pkl


def fillna(train):
    for meanStr in list(train.columns[train.columns.str.contains('mean')])[:-1]:
        temp = train.loc[:, train.columns.str.contains(
            meanStr.split('__', 1)[0])]
        cc = temp.columns[temp.columns.str.contains('ccsm3')][0]
        train[cc] = 9*temp[meanStr] - temp.drop(columns=[cc, meanStr]).sum(1)

    t = ['cancm30', 'cancm40', 'ccsm40', 'cfsv20',
         'gfdlflora0', 'gfdlflorb0', 'gfdl0', 'nasa0']
    train['ccsm30'] = 9*train['nmme0mean'] - train[t].sum(1)
    return train


def drop_outliers(df, thres=3):  # drop rows with outliers
    col = df.drop(columns=['lat', 'lon', 'startdate',
                           'climateregions__climateregion'],
                  errors='ignore').columns
    return df[(np.abs(stats.zscore(df[col])) < thres).all(axis=1)]


def corr(df, thres=0.85):  # find highly correlated columns
    matrix = df.drop(columns=['lat', 'lon', 'startdate',
                              'climateregions__climateregion'],
                     errors='ignore').corr().abs()
    matrix = matrix.where(np.triu(np.ones(matrix.shape), k=1).astype(bool))
    drop = [c for c in matrix.columns if any(matrix[c] > thres)]
    if target in drop:
        drop.remove(target)
    return drop


def xy(df):
    X = df.drop(columns=target)
    y = df[target]
    return X.reset_index(drop=True), y.reset_index(drop=True)


def split(df, date):
    train = df[df.year < date]
    test = df[df.year >= date]
    X_train, y_train = xy(train)
    X_test,  y_test = xy(test)
    return X_train, y_train, X_test, y_test


def feature_engineering(train, test):
    train1 = fillna(train)
    df = pd.concat([train1, test[train1.columns.intersection(test.columns)]],
                   ignore_index=True)

    df['coor'] = df.groupby(['lat', 'lon']).ngroup()
    df['climateregions__climateregion'] = LabelEncoder(
    ).fit_transform(df['climateregions__climateregion'])

    df['year'] = df['startdate'].dt.year
    df['month'] = df['startdate'].dt.month
    df['day'] = df['startdate'].dt.day
#     df = df.drop(columns=['startdate', 'lat', 'lon'])
    df = df.drop(columns=['lat', 'lon'])
    X, y, X_test, y_test = split(df, date=2020)
    X = X.drop(columns='year')
    X_test = X_test.drop(columns='year')

    return X, y, X_test, y_test


def transform(train, test, ss=StandardScaler()):
    train = train.drop(columns='startdate', errors='ignore')
    test = test.drop(columns='startdate', errors='ignore')
    train_sc = pd.DataFrame(ss.fit_transform(train),
                            index=train.index, columns=train.columns)
    test_sc = pd.DataFrame(ss.transform(test),
                           index=test.index, columns=test.columns)
    return train_sc, test_sc


def rmse(actual, predicted):
    return mean_squared_error(actual, predicted, squared=False)

'''
USELESS CODE
'''
# # %%time
# from scipy.ndimage import shift

# def addShiftDict(df, c):
#     keys = df[c].unique()
#     values = shift(df[c].unique(), 1, cval=np.nan)
#     return dict(zip(keys, values))

# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer

# def imputer(train, test): # multivariate imputer: fill missing values
#     temp = train.select_dtypes(include=np.number)
#     imp = IterativeImputer(n_nearest_features=10, skip_complete=True,
#                            random_state=100)
#     train[temp.columns] = imp.fit_transform(temp)
#     test[temp.columns] = imp.transform(test[temp.columns])
#     return train, test

# TT, tt = [], []

# for i in X.coor.unique()[:1]:
# #     print(i, end=' ')
#     xTT, xtt = X[X.coor == i][tmp2m].copy(), X_test[X_test.coor == i][tmp2m].copy()
#     for c in xTT.columns:
#         xTT[c] = xTT[c].replace(addShiftDict(xTT, c))
#         xtt[c] = xtt[c].replace(addShiftDict(xtt, c))

#     TT.append(xTT)
#     tt.append(xtt)

# TT, tt = (pd.concat(TT).add_suffix('_shift'), pd.concat(tt).add_suffix('_shift'))

# shifted = list(TT.columns[TT.columns.str.contains('_shift')])
# XS = pd.concat([X[['coor', 'climateregions__climateregion']+useful], TT[shifted]], axis=1).dropna()
# X_testS = pd.concat([X_test[['coor', 'climateregions__climateregion']+useful], tt[shifted]], axis=1).dropna()
# usefulS = useful + shifted
# -


# %%time
# from sklearn.feature_selection import SelectFromModel

# groups = list(pd.Series(non).str.split('-', expand=True).dropna()[range(4)]
#         .agg('-'.join, axis=1).unique())+ ['icec-2010', 'sst-2010']
# groups.extend(pd.Series(X.columns[X.columns.str.contains('prate')])
#               .str.split('__', expand=True)[0].unique())

# selectedCols = list(set(non+oth)-set(X.columns[X.columns.str.contains('|'.join(groups))]))
# selectedCols.remove('startdate')

# for c in groups:
#     cols = list(X.columns[X.columns.str.contains(c)])
#     model = XGBRegressor(objective='reg:squarederror', eval_metric='rmse',
#                        random_state=100)
#     sel = SelectFromModel(model, max_features=3).fit(X[cols], y)
#     selectedCols.extend(X[cols].columns[sel.get_support()])


# cc = list(set(selectedCols) - set(corr(X[selectedCols])))
# X1 = drop_outliers(X.copy()[cc])
# X1_test = X_test.copy()[cc]

# X_S, X_testS = transform(X1, X1_test)
# print(f'train: {X1.shape} | test: {X1_test.shape}')


# X1 = X.copy()
# X1_test = X_test.copy()
# X1, X1_test = transform(X1, X1_test, MinMaxScaler((1, 100)))


# wind = [np.sort(X1.columns[X1.columns.str.contains('uwnd')]),
#         np.sort(X1.columns[X1.columns.str.contains('vwnd')])]
# for u, v in zip(wind[0], wind[1]):
#     if u.replace('-uwnd', '') == v.replace('-vwnd', ''):
#         X1[u.replace('-uwnd', '')+'_uv_ratio'] = X1[u] / X1[v]
#         X1_test[u.replace('-uwnd', '')+'_uv_ratio'] = X1_test[u] / X1_test[v]

# temp = [np.sort(X1.columns[X1.columns.str.contains('nmme-tmp2m-34w')]),
#         np.sort(X1.columns[X1.columns.str.contains('nmme-tmp2m-56w')])]
# for u, v in zip(temp[0], temp[1]):
#     if u.replace('-34w', '') == v.replace('-56w', ''):
#         # print(u.replace('-34w', ''), v.replace('-56w', ''))
#         X1[u.replace('-34w', '')+'_34_56_delta'] = (X1[v] - X1[u]) / X1[u]
#         X1_test[u.replace('-34w', '')+'_34_56_delta'] = ((X1_test[v] - X1_test[u]) /
#                                                          X1_test[u])

# prate = [np.sort(X1.columns[X1.columns.str.contains('prate-34w')]),
#          np.sort(X.columns[X.columns.str.contains('prate-56w')])]
# for u, v in zip(prate[0], prate[1]):
#     if u.replace('-34w', '') == v.replace('-56w', ''):
#         X1[u.replace('-34w', '')+'_34_56_delta'] = (X1[v] - X1[u]) / X1[u]
#         X1_test[u.replace('-34w', '')+'_34_56_delta'] = ((X1_test[v] - X1_test[u]) /
#                                                          X1_test[u])

# drop = (list(np.array(wind).flatten()) + list(np.array(temp).flatten()) +
#         list(np.array(prate).flatten()) + useful+['startdate'])

# drop.extend(corr(X1.drop(columns=set(drop), errors='ignore')))

# X1 = X1.loc[X_G1.index].drop(columns=set(drop), errors='ignore')
# X1_test = X1_test[X1.columns]
# X_S, X_testS = transform(X1, X1_test)
# print(f'train: {X1.shape} | test: {X1_test.shape}')