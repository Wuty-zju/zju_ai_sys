import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import shutil
import os

# ==================== 数据预处理函数 ====================
train_data = pd.read_csv('./data/train_data.csv')

def preprocess_data(data, imputer=None, scaler=None):
    """
    预处理数据：填补缺失值并归一化数值型列。
    :param data: 待处理的数据
    :param imputer: 缺失值填补器（默认为None，使用均值填补）
    :param scaler: 归一化器（默认为None，使用MinMaxScaler）
    :return: 预处理后的数据、imputer 和 scaler
    """
    numeric_columns = [
        'Year', 'Life expectancy ', 'infant deaths', 'Alcohol',
        'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ',
        'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
        ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years', 
        ' thinness 5-9 years', 'Income composition of resources', 'Schooling'
    ]

    # 删除无关列
    data = data.drop(["Country", "Status"], axis=1)
    
    # 填补缺失值
    if imputer is None:
        imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
        imputer.fit(data[numeric_columns])
    data[numeric_columns] = imputer.transform(data[numeric_columns])

    # 归一化处理
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(data[numeric_columns])
    data[numeric_columns] = scaler.transform(data[numeric_columns])

    # 删除非必要列
    data = data.drop(['Year'], axis=1)

    return pd.DataFrame(data, columns=data.columns), imputer, scaler

# ==================== 模型训练函数 ====================
def model_fit(train_data, model_type="linear"):
    """
    训练指定类型的回归模型并保存至 ./results 目录。
    :param train_data: 训练数据
    :param model_type: 模型类型（默认为线性回归）
    :return: 训练好的模型
    """
    train_y = train_data.iloc[:, -1].values
    train_data = train_data.drop(["Adult Mortality"], axis=1)
    train_data_norm, imputer, scaler = preprocess_data(train_data)
    train_x = train_data_norm.values

    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(n_estimators=500, random_state=42)
    elif model_type == "xgboost":
        model = XGBRegressor(n_estimators=20, random_state=42)
    else:
        raise ValueError("未知模型类型")

    model.fit(train_x, train_y)

    # 创建结果目录并保存模型和预处理器
    os.makedirs('./results', exist_ok=True)
    joblib.dump(model, f"./results/{model_type}_model.pkl")
    joblib.dump(imputer, "./results/imputer.pkl")
    joblib.dump(scaler, "./results/scaler.pkl")

    return model

# ==================== 预测函数 ====================
def predict_model(test_data, model_type="linear"):
    """
    使用训练好的模型对测试数据进行预测。
    :param test_data: 测试数据
    :param model_type: 使用的模型类型
    :return: 预测结果
    """
    loaded_model = joblib.load(f"./results/{model_type}_model.pkl")
    imputer = joblib.load("./results/imputer.pkl")
    scaler = joblib.load("./results/scaler.pkl")

    test_data_norm, _, _ = preprocess_data(test_data, imputer, scaler)
    test_x = test_data_norm.values

    predictions_model = loaded_model.predict(test_x)

    return predictions_model

# ==================== 模型性能评估函数 ====================
def evaluate_model(train_data, model_type="linear"):
    """
    评估模型性能并打印结果。
    :param train_data: 训练数据
    :param model_type: 模型类型
    :return: 模型的 MSE 和 R2 分数
    """
    label = train_data['Adult Mortality']
    train_data = train_data.drop(columns=['Adult Mortality'])
    y_pred = predict_model(train_data, model_type)
    
    mse = mean_squared_error(label, y_pred)
    r2 = r2_score(label, y_pred)
    
    print(f"模型: {model_type}, MSE: {mse}, R2 Score: {r2}")
    
    return mse, r2

# ==================== 训练并选择最佳模型 ====================
best_model_type = None
best_score = float('-inf')

for model_type in ["linear", "random_forest", "gradient_boosting", "xgboost"]:
    model_fit(train_data, model_type)
    mse, r2 = evaluate_model(train_data, model_type)
    
    if r2 > best_score:
        best_score = r2
        best_model_type = model_type

# 复制最佳模型为 best_model.pkl
print(f"最佳模型: {best_model_type}，MSE: {best_score}, R2 Score: {r2}")
best_model_path = f"./results/{best_model_type}_model.pkl"
best_model_destination = "./results/best_model.pkl"
shutil.copy(best_model_path, best_model_destination)
print(f"最佳模型已保存至: {best_model_destination}")

'''
# ==================== 平台测试-预测函数 ====================
def predict(test_data):
    """
    使用训练好的模型对测试数据进行预测。
    :param test_data: 测试数据
    :param model_type: 使用的模型类型
    :return: 预测结果
    """
    loaded_model = joblib.load(f"./results/best_model.pkl")
    imputer = joblib.load("./results/imputer.pkl")
    scaler = joblib.load("./results/scaler.pkl")

    test_data_norm, _, _ = preprocess_data(test_data, imputer, scaler)
    test_x = test_data_norm.values

    predictions = loaded_model.predict(test_x)

    return predictions
'''