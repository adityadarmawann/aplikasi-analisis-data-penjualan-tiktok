import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def clean_data(df):
    # Hapus kolom yang tidak relevan
    df_clean = df.drop(columns=['Order ID', 'Seller SKU', 'Tracking ID', 'Cancelation/Return Type', 'Seller Note', 'Checked Marked by'], errors='ignore')

    # Normalisasi data
    for column in df_clean.columns:
        if df_clean[column].dtype == 'object':
            df_clean[column] = df_clean[column].str.replace('IDR', '', regex=False).str.replace('.', '', regex=False)
            try:
                df_clean[column] = df_clean[column].astype(int)
            except:
                pass
    
    df_clean['Created Time'] = pd.to_datetime(df_clean['Created Time'], errors='coerce')
    return df_clean

def feature_engineering(df):
    df['Total_Price'] = df['Quantity'] * df['SKU Subtotal After Discount']
    df['Year'] = df['Created Time'].dt.year  # Tambahkan kolom tahun
    df['Month'] = df['Created Time'].dt.month  # Tambahkan kolom bulan
    return df

def train_predictive_model(monthly_sales):
    # Pastikan monthly_sales memiliki kolom Tahun dan Bulan
    monthly_sales = monthly_sales.reset_index()
    monthly_sales['Tahun'] = monthly_sales['Created Time'].dt.year
    monthly_sales['Bulan'] = monthly_sales['Created Time'].dt.month

    # Pastikan monthly_sales adalah DataFrame
    if not isinstance(monthly_sales, pd.DataFrame):
        raise ValueError("monthly_sales harus berupa DataFrame")

    X = monthly_sales[['Tahun', 'Bulan']]
    y = monthly_sales['Quantity']  # Pastikan 'Quantity' tersedia di monthly_sales

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_gb.fit(X_train, y_train)

    new_data = pd.DataFrame({'Tahun': [2024, 2024], 'Bulan': [8, 9]})
    predictions_gb = model_gb.predict(new_data)

    # Return both predictions and new data for future months
    new_data['Quantity'] = predictions_gb  # Tambahkan kolom 'Quantity' dengan prediksi
    return predictions_gb, new_data



def gradient_boosting_eval(monthly_sales):
    monthly_sales = monthly_sales.reset_index()
    monthly_sales['Tahun'] = monthly_sales['Created Time'].dt.year
    monthly_sales['Bulan'] = monthly_sales['Created Time'].dt.month

    X = monthly_sales[['Tahun', 'Bulan']]
    y = monthly_sales['Quantity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_gb.fit(X_train, y_train)

    y_pred = model_gb.predict(X_test)

    mse_gb = mean_squared_error(y_test, y_pred)
    rmse_gb = mean_squared_error(y_test, y_pred, squared=False)
    mae_gb = mean_absolute_error(y_test, y_pred)
    r2_gb = r2_score(y_test, y_pred)

    return mse_gb, rmse_gb, mae_gb, r2_gb
