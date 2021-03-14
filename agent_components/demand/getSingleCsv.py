import pandas as pd
import os

ML_DATA_FOLDER = 'D:\\Users\\X\\Desktop\\FYP\\codes\\analysis_data\\ml_data\\elbow'
USAGE_PROFILE_SINGLE_CSV_FOLDER = "D:\\Users\\X\\Desktop\\FYP\\codes\\analysis_data\\single_csv\\elbow"

for itemtype in os.listdir(ML_DATA_FOLDER):
    folderPath = os.path.join(ML_DATA_FOLDER, itemtype)
    singleFilePath = os.path.join(folderPath, os.listdir(folderPath)[0])
    df = pd.read_pickle(singleFilePath)
    df = df[:1]
    savingPath = os.path.join(USAGE_PROFILE_SINGLE_CSV_FOLDER, f"{itemtype}.csv")
    df.to_csv(savingPath, index=False)



