import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

# 数据预处理函数
def preprocess_data(data, imputer=None, scaler=None):
    # 数值型列
    numeric_columns = ['Year', 'Life expectancy ', 'infant deaths', 'Alcohol',
                       'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ',
                       'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
                       ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years', 
                       ' thinness 5-9 years', 'Income composition of resources', 'Schooling']

    # 删除无关列
    data = data.drop(["Country", "Status"], axis=1)
    
    # 处理缺失值（使用均值填充）
    if imputer is None:
        imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
        imputer.fit(data[numeric_columns])  # 训练 imputer 只在数值型列上
    data[numeric_columns] = imputer.transform(data[numeric_columns])  # 填充缺失值

    # 数据归一化处理
    if scaler is None:
        scaler = MinMaxScaler()  # 默认使用MinMaxScaler进行归一化
        scaler.fit(data[numeric_columns])
    data[numeric_columns] = scaler.transform(data[numeric_columns])

    # 删除不必要的列
    data = data.drop(['Year'], axis=1)  # 删除年份列（不作为特征使用）

    return pd.DataFrame(data, columns=data.columns), imputer, scaler

# 模型训练函数
def model_fit(train_data, model_type="linear"):
    train_y = train_data.iloc[:, -1].values
    train_data = train_data.drop(["Adult Mortality"], axis=1)
    train_data_norm, imputer, scaler = preprocess_data(train_data)

    train_x = train_data_norm.values

    # 选择不同的模型
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif model_type == "xgboost":
        model = XGBRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError("未知模型类型")

    # 训练模型
    model.fit(train_x, train_y)

    # 保存模型
    joblib.dump(model, f"adult_mortality_prediction/{model_type}_model.pkl")
    joblib.dump(imputer, "adult_mortality_prediction/imputer.pkl")
    joblib.dump(scaler, "adult_mortality_prediction/scaler.pkl")

    return model

# 预测函数
def predict(test_data, model_type="linear"):
    # 加载模型及预处理器
    loaded_model = joblib.load(f"adult_mortality_prediction/{model_type}_model.pkl")
    imputer = joblib.load("adult_mortality_prediction/imputer.pkl")
    scaler = joblib.load("adult_mortality_prediction/scaler.pkl")

    # 预处理测试数据
    test_data_norm, _, _ = preprocess_data(test_data, imputer, scaler)
    test_x = test_data_norm.values

    # 进行预测
    predictions = loaded_model.predict(test_x)

    return predictions

# 模型性能评估函数
def evaluate_model(train_data, model_type="linear"):
    label = train_data['Adult Mortality']
    train_data = train_data.drop(columns=['Adult Mortality'])
    
    y_pred = predict(train_data, model_type)
    
    mse = mean_squared_error(label, y_pred)
    r2 = r2_score(label, y_pred)
    
    print(f"模型: {model_type}")
    print(f"MSE: {mse}")
    print(f"R2 Score: {r2}")
    print("\n")

# 读取训练数据
if __name__ == "__main__":
    train_data = pd.read_csv('adult_mortality_prediction/data/train_data.csv')

    # 训练并评估不同模型
    for model_type in ["linear", "random_forest", "gradient_boosting", "xgboost"]:
        print(f"正在训练模型: {model_type}")
        model_fit(train_data, model_type)
        evaluate_model(train_data, model_type)