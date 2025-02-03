import streamlit as st
import geopandas as gpd
import folium
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import io
from reportlab.pdfgen import canvas
from shapely.geometry import Polygon

# ----- 1. GISデータの生成（疑似データ：愛媛県上島町付近） -----
@st.cache_data(show_spinner=False)
def create_sample_gis_data():
    """
    上島町付近の疑似ポリゴンを作成し、リスク属性付きのGeoDataFrameを生成する。
    """
    coords = [
        (132.495, 33.847),  # 左下
        (132.505, 33.847),  # 右下
        (132.505, 33.853),  # 右上
        (132.495, 33.853)   # 左上
    ]
    poly = Polygon(coords)
    gdf = gpd.GeoDataFrame({'risk': [0.5]}, geometry=[poly], crs="EPSG:4326")
    return gdf

def create_folium_map(gdf):
    """
    GeoDataFrameの重心をマップの中心としたFoliumマップを生成する。
    """
    if not gdf.empty:
        mean_lat = gdf.geometry.centroid.y.mean()
        mean_lon = gdf.geometry.centroid.x.mean()
    else:
        mean_lat, mean_lon = 33.85, 132.50  # デフォルト値（上島町付近）
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=14)
    folium.GeoJson(gdf, name="Kamijima Area").add_to(m)
    folium.LayerControl().add_to(m)
    return m

# ----- 2. 機械学習モデルによる予測（ダミーデータ） -----
@st.cache_resource(show_spinner=False)
def train_dummy_model():
    """
    仮の学習データを用いてRandomForestRegressorを学習する。
    """
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = np.random.rand(100)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

def predict_risk(model, input_features):
    """
    入力された特徴量を元にリスクスコアを予測する。
    """
    risk_score = model.predict(input_features)
    return risk_score[0]

# ----- 3. PDFレポート生成 -----
def generate_pdf_report(risk_score):
    """
    予測されたリスクスコアを含むPDFレポートを生成し、バイナリデータとして返す。
    """
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer)
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, 800, "Kamijima Pipeline Safety Report")
    p.setFont("Helvetica", 12)
    p.drawString(100, 780, "Location: Kamijima, Ehime, Japan")
    p.drawString(100, 760, f"Predicted Risk Score: {risk_score:.2f}")
    p.drawString(100, 740, "Note: This is a test report using sample data.")
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

# ----- 4. Streamlitアプリ本体 -----
def main():
    st.title("Kamijima Pipeline Safety Dashboard")
    
    # サイドバー：入力パラメータ（5特徴量）
    st.sidebar.header("Input Parameters (5 Features)")
    feature_values = []
    for i in range(5):
        val = st.sidebar.slider(f"Feature {i+1}", 0.0, 1.0, 0.5, 0.01)
        feature_values.append(val)
    input_features = np.array(feature_values).reshape(1, -1)
    
    # GISデータの生成とFoliumマップ表示
    st.header("GIS Visualization (Kamijima Area)")
    try:
        with st.spinner("Generating GIS data..."):
            gdf = create_sample_gis_data()
            folium_map = create_folium_map(gdf)
            st.components.v1.html(folium_map._repr_html_(), height=500)
    except Exception as e:
        st.error(f"Failed to display GIS data: {e}")
        return

    # 予測モデルの作成とリスク予測
    st.header("Risk Prediction")
    try:
        with st.spinner("Training dummy model..."):
            model = train_dummy_model()
        with st.spinner("Predicting risk..."):
            risk_score = predict_risk(model, input_features)
        st.write(f"Predicted Risk Score: **{risk_score:.2f}**")
    except Exception as e:
        st.error(f"Failed to predict risk: {e}")
        return

    # PDFレポート生成
    st.header("PDF Report Generation")
    if st.button("Generate PDF Report"):
        try:
            pdf_buffer = generate_pdf_report(risk_score)
            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name="kamijima_pipeline_safety_report.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Failed to generate PDF report: {e}")

if __name__ == "__main__":
    main()
