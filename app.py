import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Memuat model dan transformer ---
try:
    with open('catboost_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('power_transformer.pkl', 'rb') as file:
        pt = pickle.load(file)
    with open('standard_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    st.error("File model atau transformer tidak ditemukan. Pastikan semua file .pkl berada dalam folder yang sama dengan app.py.")
    st.stop()


# --- Tampilan Aplikasi ---
st.title('Prediksi Harga Mobil Bekas')
st.write('Aplikasi ini memprediksi harga jual mobil bekas berdasarkan beberapa parameter.')

st.sidebar.header('Input Detail Mobil')

# --- Input pengguna ---
year_input = st.sidebar.number_input('Tahun Mobil', min_value=1990, max_value=2024, value=2015, step=1)
km_driven_input = st.sidebar.number_input('Jarak Tempuh (km)', min_value=0, max_value=1000000, value=50000, step=1000)
fuel_type_input = st.sidebar.selectbox('Jenis Bahan Bakar', ['Diesel', 'Petrol', 'CNG', 'LPG', 'Electric'])
seller_type_input = st.sidebar.selectbox('Tipe Penjual', ['Individual', 'Dealer', 'Trustmark Dealer'])
transmission_type_input = st.sidebar.selectbox('Transmisi', ['Manual', 'Automatic'])
owner_status_input = st.sidebar.selectbox('Jumlah Pemilik Sebelumnya', [
    'Test Drive Car', 'First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'
])

# Daftar merek mobil (autocomplete)
car_brands = [
    "Maruti", "Hyundai", "Mahindra", "Tata", "Honda", "Ford",
    "Chevrolet", "Toyota", "Renault", "Volkswagen", "Nissan",
    "Skoda", "Fiat", "Datsun", "Mercedes-Benz", "Audi", "BMW",
    "Mitsubishi", "Ambassador", "OpelCorsa", "Daewoo", "Force"
]

car_brands = sorted(car_brands)  # urutkan agar mudah dibaca

car_name_input = st.sidebar.selectbox(
    "Nama / Merek Mobil",
    options=car_brands,
    index=car_brands.index("Toyota") if "Toyota" in car_brands else 0
)


# Memuat daftar nama mobil yang valid dari X_train_names.csv
try:
    valid_car_names_df = pd.read_csv('X_train_names.csv')
    valid_car_names = valid_car_names_df['name'].unique().tolist()
except FileNotFoundError:
    st.error("File X_train_names.csv tidak ditemukan. Pastikan sudah dibuat.")
    st.stop()

# --- Logika Prediksi ---
if st.sidebar.button('Prediksi Harga Mobil'):
    # Validasi nama/merek
    if car_name_input not in valid_car_names:
        st.error(f"Nama/Merek mobil '{car_name_input}' tidak dikenali. Silakan masukkan nama yang ada di data training.")
    else:
        st.subheader('Detail Input Anda:')
        input_data = {
            'Tahun': year_input,
            'Jarak Tempuh (km)': km_driven_input,
            'Jenis Bahan Bakar': fuel_type_input,
            'Tipe Penjual': seller_type_input,
            'Transmisi': transmission_type_input,
            'Jumlah Pemilik': owner_status_input,
            'Nama / Merek Mobil': car_name_input
        }
        st.write(pd.DataFrame([input_data]))

        # Konversi kategori
        age = 2025 - year_input
        fuel_encoded = fuel_type_input
        seller_type_encoded = seller_type_input
        transmission_encoded = transmission_type_input
        owner_encoded = owner_status_input
        name_encoded = car_name_input

        # Transformasi PowerTransformer
        # Perlu membuat dummy df untuk km_driven agar pt.transform bisa bekerja sesuai format aslinya
        # Pastikan kolom selling_price_yj juga ada, meskipun nilainya dummy (0)
        data_for_pt = pd.DataFrame([[0.0, km_driven_input]], columns=['selling_price', 'km_driven'])
        transformed_data_for_pt = pt.transform(data_for_pt)
        km_driven_yj = transformed_data_for_pt[0, 1] # Ambil nilai km_driven_yj

        # Transformasi StandardScaler
        data_for_scaler = pd.DataFrame([[km_driven_yj, age]], columns=['km_driven_yj', 'age'])
        scaled_features = scaler.transform(data_for_scaler)

        # Susun DataFrame untuk prediksi
        prediction_df = pd.DataFrame([[fuel_encoded, seller_type_encoded, transmission_encoded, owner_encoded,
                                       scaled_features[0, 0], scaled_features[0, 1], name_encoded]],
                                     columns=['fuel', 'seller_type', 'transmission', 'owner',
                                              'km_driven_yj', 'age', 'name'])

        # Prediksi
        predicted_price_yj = model.predict(prediction_df)[0]

        # Inverse transform ke skala asli
        # Perlu membuat dummy df untuk km_driven_yj agar pt.inverse_transform bisa bekerja
        data_for_inverse_pt = pd.DataFrame([[predicted_price_yj, 0.0]], columns=['selling_price', 'km_driven'])
        original_scale_prediction = pt.inverse_transform(data_for_inverse_pt)
        final_predicted_selling_price = original_scale_prediction[0, 0]

        st.subheader('Hasil Prediksi Harga:')
        st.success(f'Harga Mobil Diprediksi: Rp {final_predicted_selling_price:,.2f}')

        st.markdown("""
        **Catatan Penting:**
        * Prediksi ini merupakan estimasi dan tidak sepenuhnya akurat.
        * Kondisi mobil, lokasi penjualan, dan fitur tambahan dapat mempengaruhi harga sebenarnya.
        * Model dilatih berdasarkan data yang tersedia sehingga kualitas prediksi bergantung pada representasi data tersebut.
        """)
