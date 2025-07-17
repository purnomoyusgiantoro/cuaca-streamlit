import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

st.title("🌤️ Prediksi Cuaca 6 Jam ke Depan (BMKG + AI)")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_cuaca.h5")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

model, scaler, le = load_model()

st.sidebar.header("Input Manual (opsional)")
temp = st.sidebar.slider("Suhu (°C)", -10.0, 40.0, 25.0)
wind_kmph = st.sidebar.slider("Kecepatan Angin (km/jam)", 0.0, 100.0, 10.0)
precip = st.sidebar.slider("Curah Hujan (mm)", 0.0, 50.0, 2.0)

if st.button("Prediksi Manual"):
    wind = wind_kmph / 3.6  # konversi ke m/s
    sample = pd.DataFrame([[temp, temp, precip, wind]],
                          columns=['temp_min', 'temp_max', 'precipitation', 'wind'])
    scaled = scaler.transform(sample)
    pred = model.predict(scaled, verbose=0)[0]
    idx = np.argmax(pred)
    label = le.inverse_transform([idx])[0]
    confidence = pred[idx] * 100
    st.success(f"☁️ Prediksi: **{label}** ({confidence:.1f}%)")

st.markdown("---")
st.subheader("📍 Prediksi Cuaca Otomatis dari BMKG")

kode = st.text_input("Masukkan kode wilayah administratif 4 (ADM4)")
btn = st.button("Prediksi dari BMKG")

if btn and kode:
    try:
        import requests
        from dateutil import parser
        from datetime import datetime, timezone, timedelta

        now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
        now_wib = now_utc.astimezone(timezone(timedelta(hours=7)))
        url = f"https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={kode}"
        r = requests.get(url)
        data = r.json()
        field = data['data'][0]['cuaca']
        cuaca_data = field[0] if isinstance(field[0], list) else field

        def parse_ts(ts):
            dt = parser.isoparse(ts)
            return dt.astimezone(timezone(timedelta(hours=7)))

        results = []
        for item in cuaca_data:
            jam_str = item.get('jamCuaca') or item.get('datetime')
            dt = parse_ts(jam_str)
            if dt >= now_wib:
                temp = float(item['t'])
                wind = float(item['ws']) / 3.6
                tp = float(item['tp'])
                df = pd.DataFrame([[temp, temp, tp, wind]],
                                  columns=['temp_min', 'temp_max', 'precipitation', 'wind'])
                X = scaler.transform(df)
                pred = model.predict(X, verbose=0)[0]
                idx = np.argmax(pred)
                label = le.inverse_transform([idx])[0]
                conf = pred[idx] * 100
                results.append((dt.strftime('%H:%M'), label, conf, temp, wind, tp))

        for res in results[:3]:
            st.info(f"🕒 {res[0]} WIB: {res[1]} ({res[2]:.1f}%) | Suhu: {res[3]}°C | Angin: {res[4]:.1f} m/s | Hujan: {res[5]} mm")
    except Exception as e:
        st.error(f"❌ Gagal ambil data BMKG: {e}")
