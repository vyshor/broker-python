import os

def listAvailableModels(folderPath, modelType="DNN"):
    modelTypesAvailable = os.listdir(folderPath)
    modelsAvailable = {}
    if modelType not in modelTypesAvailable:
        print(f"Error cannot find model type: {modelType}")
        return None
    else:
        for itemtype_path in os.listdir(os.path.join(folderPath, modelType)):
            basetype_name = os.path.basename(itemtype_path)
            itemtype_fullpath = os.path.join(folderPath, modelType, basetype_name)
            last_trained_unix = set()
            for model_path in os.listdir(itemtype_fullpath):
                path_splits = model_path.split("-")
                if len(path_splits) > 1:
                    last_trained_unix.add(path_splits[2])
            last_trained_unix = sorted(list(last_trained_unix))[-1]
            last_trained_epoch = set()
            for model_path in [x for x in os.listdir(itemtype_fullpath) if last_trained_unix in x]:
                last_trained_epoch.add(int(model_path.split("-")[4].split('.')[0]))
            last_trained_epoch = sorted(list(last_trained_epoch))[-1]
            model_path = f'model-start-{last_trained_unix}-cp-{last_trained_epoch}.ckpt'
            MODEL_CKPT_FILE = os.path.join(itemtype_fullpath, model_path)
            modelsAvailable[basetype_name] = (modelType, basetype_name, MODEL_CKPT_FILE)
        return modelsAvailable

def listAvailableScalers(folderPath, scalers_cols):
    scalersAvailable = {}
    for itemtype_path in os.listdir(folderPath):
        basetype_name = os.path.basename(itemtype_path)
        itemtype_fullpath = os.path.join(folderPath, basetype_name)
        scalersAvailable[basetype_name] = {}
        for scaler_col in scalers_cols:
            SCALER_FILE = os.path.join(itemtype_fullpath, scaler_col)
            scalersAvailable[basetype_name][scaler_col] = (scaler_col, basetype_name, f"{SCALER_FILE}.gz")
    return scalersAvailable