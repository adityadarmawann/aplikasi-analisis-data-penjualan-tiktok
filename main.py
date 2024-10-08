import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
from dateutil.relativedelta import relativedelta
from backend import clean_data, feature_engineering, train_predictive_model, gradient_boosting_eval
import plotly.express as px

st.set_page_config(layout="wide")  # Set layout to wide

st.title("Analisis Penjualan dan Prediksi dengan Gradient Boosting")

# File uploader
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

# Fungsi untuk membatasi nama produk menjadi maksimal 5 kata
def limit_words(product_name, word_limit=5):
    words = product_name.split()
    if len(words) > word_limit:
        return ' '.join(words[:word_limit]) + '...'  # Tambahkan ellipsis jika terpotong
    return product_name

# Jika file di-upload, lakukan pemrosesan
if uploaded_file:
    # Langkah pemrosesan: membaca data dan menjalankan fungsi yang diperlukan
    df = pd.read_csv(uploaded_file)

    # Step 1: Data Cleaning
    df_clean = clean_data(df)

    # Step 2: Feature engineering
    df_features = feature_engineering(df_clean)

    # Step 3: Menghitung total item terjual per item
    item_sales = df_features.groupby('Product Name')['Quantity'].sum().sort_values(ascending=False)

    # Step 5: Total Pendapatan
    sales_per_category = df_features.groupby('Product Category')['SKU Subtotal After Discount'].sum().sort_values(ascending=False)
    total_revenue = df_features['SKU Subtotal After Discount'].sum()

    # Step 4: Total penjualan
    total_sales = item_sales.sum()

    # Step 6: Trend Item - Top 5 Produk Terjual (Nama Produk Dipendekkan)
    top_items = df_features['Product Name'].value_counts().nlargest(5)
    shortened_names = ['Kucing Vol 1', 'O-D-G-J Vol 1', 'Polos Vol 1', 'Punk Vol 1', 'Serdadu Djantjuk Vol 1']

    # Step 10: Trend Metode Pembayaran
    payment_counts = df_features['Payment Method'].value_counts()

    # Step 7: Grafik Penjualan Harian
    daily_sales = df_features.groupby(df_features['Created Time'].dt.date)['Quantity'].sum()

    # Step 8: Grafik Penjualan Mingguan
    weekly_sales = df_features.resample('W', on='Created Time')['Quantity'].sum()

    # Step 9: Grafik Penjualan Bulanan
    monthly_sales = df_features.resample('M', on='Created Time')['Quantity'].sum()

    # Step 11: Prediksi Penjualan (Gradient Boosting)
    predictions_gb, new_data = train_predictive_model(monthly_sales)

    # Evaluasi Gradient Boosting
    mse_gb, rmse_gb, mae_gb, r2_gb = gradient_boosting_eval(monthly_sales)

    # Mulai menampilkan di dua kolom
    col1, col2 = st.columns(2)

    # Menampilkan total item terjual dan total pendapatan di kolom 1
    with col1:
        st.subheader("Total Penjualan per Item")
        st.write(item_sales)

        st.subheader("Total Pendapatan")
        st.metric(label="Total Pendapatan", value=f"Rp. {total_revenue:,.0f}")

    # Menampilkan pendapatan per kategori dan total penjualan di kolom 2
    with col2:
        st.subheader("Pendapatan per Kategori Produk")
        fig_category = px.bar(sales_per_category, title="Pendapatan per Kategori")
        st.plotly_chart(fig_category)
    with col1:
        st.subheader("Total Penjualan Semua Item")
        st.metric(label="Total Penjualan", value=total_sales)

    col3, col4 = st.columns(2)

    # Trend Item - Top 5 Produk Terjual
    with col3:
        st.subheader("Trend Item - Top 5 Produk Terjual")
        
        # Ambil 5 produk teratas
        top_5_items = top_items[:5]  # Membatasi hanya 5 produk teratas
        
        # Batasi nama produk di index menjadi maksimal 5 kata
        limited_names = [limit_words(name) for name in top_5_items.index]
        
        plt.figure(figsize=(6, 4))
        sns.barplot(x=limited_names, y=top_5_items.values, palette='husl')
        
        plt.title('5 Item Paling Banyak Terjual')
        plt.xlabel('Nama Produk')
        plt.ylabel('Jumlah Terjual')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        st.pyplot(plt)

    # Trend Metode Pembayaran
    with  col4:
        st.subheader("Trend Metode Pembayaran")
        plt.figure(figsize=(6, 4))
        sns.barplot(x=payment_counts.index, y=payment_counts.values, palette='husl')
        plt.title('Distribusi Metode Pembayaran')
        plt.xlabel('Metode Pembayaran')
        plt.ylabel('Jumlah')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(plt)

    # Grafik Penjualan Harian
    with col3:
        st.subheader("Grafik Penjualan Harian")
        plt.figure(figsize=(12, 6))
        plt.plot(daily_sales.index, daily_sales.values, marker='o')
        plt.xlabel('Tanggal')
        plt.ylabel('Jumlah Item Terjual')
        plt.title('Pergerakan Penjualan Harian')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)

    # Grafik Penjualan Mingguan
    with col4:
        st.subheader("Grafik Penjualan Mingguan")
        plt.figure(figsize=(12, 6))
        plt.plot(weekly_sales.index, weekly_sales.values, marker='o')
        plt.xlabel('Minggu')
        plt.ylabel('Jumlah Item Terjual')
        plt.title('Pergerakan Penjualan Mingguan')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt)

    # Grafik Penjualan Bulanan
    with  col3:
        st.subheader("Grafik Penjualan Bulanan")
        month_names = monthly_sales.index.strftime('%B')
        plt.figure(figsize=(10, 6))
        plt.plot(month_names, monthly_sales.values, marker='o', color='b')
        plt.xlabel('Bulan')
        plt.ylabel('Jumlah Item Terjual')
        plt.title('Pergerakan Penjualan Bulanan')
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt)

    # Prediksi Penjualan
    # Prediksi Penjualan
    with col4:
        st.subheader("Prediksi Penjualan - 2 Bulan ke Depan")
        all_data_gb = pd.concat([monthly_sales, new_data], ignore_index=True)

        if isinstance(all_data_gb, pd.DataFrame):
            # Isi nilai 'Quantity' di data baru dengan prediksi
            all_data_gb['Quantity'] = all_data_gb['Quantity'].fillna(pd.Series(predictions_gb))

            # Isi nilai NaN di kolom 'Bulan' dengan angka default, misalnya 1 (Januari)
            all_data_gb['Bulan'] = all_data_gb['Bulan'].fillna(1)

            # Ubah angka bulan menjadi nama bulan
            all_data_gb['Bulan'] = all_data_gb['Bulan'].apply(lambda x: calendar.month_name[int(x)])

            # Plot prediksi penjualan
            plt.figure(figsize=(10, 6))
            sns.lineplot(x=all_data_gb['Bulan'], y=all_data_gb['Quantity'], marker='o')
            plt.xlabel('Bulan')
            plt.ylabel('Jumlah Item Terjual')
            plt.title('Prediksi Penjualan (Juni - September 2024) - Gradient Boosting')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True)
            st.pyplot(plt)


