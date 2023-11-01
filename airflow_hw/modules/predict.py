# <YOUR_IMPORTS>
import pandas as pd
import dill
import json
import glob
import os
from datetime import datetime

from pathlib import (Path)
#/home/airflow  cars_pipe_202311010808
def predict():
    # <YOUR_CODE>
    path = os.path.expanduser('/home/airflow/airflow_hw') # 'это путь до папки проекта
    # ...дальше путь внутри папки до модели
    with open(f'{path}/data/models/cars_pipe_202311010808.pkl', 'rb') as file:
        model = dill.load(file)
    print('ok')
    # готовим путь до файлов для теста
    path_files = path + '/data/test/*json'
    df_pred = pd.DataFrame({'id' : [], 'pred' : []})
    # перебираем тестовые файлы из путей файлов
    for json_files_path in glob.iglob(path_files):
        with open(json_files_path) as fin:
            form = json.load(fin)
            df = pd.DataFrame.from_dict([form])
            pred = model.predict(df)
            df_pred = df_pred._append([df['id'], pred])
        df_pred.to_csv(f'{path}/data/predictions/pred_{datetime.now().strftime("%Y%m%d%H%M")}.csv')
            # y = model.predict(df)

if __name__ == '__main__':
    predict()
