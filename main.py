import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from datetime import datetime
rcParams['figure.figsize'] = 10, 4

data = pd.read_csv('hotelguests.csv')
print(data.head())
print('\n Tipos de datos:')
print(data.dtypes)

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
# print(dateparse('1962-01'))
data = pd.read_csv('hotelguests.csv', parse_dates=['time'], index_col='time',date_parser=dateparse)
print(data.head())
print(data.index)
ts = data['#guests']
print(ts.head(10))
fig = plt.figure()
plt.plot(ts)
fig.show()
plt.waitforbuttonpress()
