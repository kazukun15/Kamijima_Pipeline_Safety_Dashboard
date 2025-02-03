import streamlit as st
import geopandas as gpd
import folium
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import io
from reportlab.pdfgen import canvas
from shapely.geometry import Polygon, LineString

# ----- 1. GISデータの生成（疑似データ：上島町中央付近の広域領域と上下水道管） -----
@st.cache_data(show_spinner=False)
def create_sample_gis_data():
    """
    広い領域（例：経度 133.190～133.220、緯度 34.245～34.270）を表す土壌領域ポリゴンと、
    2種類の曲折した上下水道管（Water Supply と Sewer）を生成する。
    """
    try:
        # 土壌領域ポリゴンの作成（全域）
        soil_coords = [
            (133.190, 34.245),  # 左下
            (133.220, 34.245),  # 右下
            (133.220, 34.270),  # 右上
            (133.190, 34.270)   # 左上
        ]
        soil_poly = Polygon(soil_coords)
        gdf_soil = gpd.GeoDataFrame({'soil': ['Sample Soil']}, geometry=[soil_poly], crs="EPSG:4326")
        
        # パイプラインの作成（曲折したライン）
        # 水道管：左下から右上方向へ、上昇しながら曲がる
        water_supply_coords = [
            (133.192, 34.248),
            (133.200, 34.252),
            (133.205, 34.257),
            (133.210, 34.262),
            (133.215, 34.266),
            (133.218, 34.268)
        ]
        water_supply_line = LineString(water_supply_coords)
        
        # 下水管：右下から左上方向へ、下降しながら曲がる
        sewer_coords = [
            (133.218, 34.248),
            (133.213, 34.252),
            (133.208, 34.257),
            (133.203, 34.262),
            (133.198, 34.266),
            (133.193, 34.268)
        ]
        sewer_line = LineString(sewer_coords)
        
        gdf_pipelines = gpd.GeoDataFrame(
            {
                'pipeline_type': ['Water Supply', 'Sewer']
            },
            geometry=[water_supply_line, sewer_line],
            crs="EPSG:4326"
        )
        
        return gdf_soil, gdf_pipelines
    except Exception as e:
        st.error(f"Error in creating sample GIS data: {e}")
        raise

def create_folium_map(gdf_soil, gdf_pipelines):
    """
    土壌領域ポリゴンと上下水道管のGeoDataFrameからFoliumマップを生成する。
    土壌領域は薄い緑、Water Supply は青、Sewer はオレンジで表示する。
    """
    try:
        # 土壌領域の中心をマップの中心に設定
        mean_lat = gdf_soil.geometry.centroid.y.mean()
        mean_lon = gdf_soil.geometry.centroid.x.mean()
        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=14)
        
        # 土壌領域を追加（薄い緑色）
        folium.GeoJson(
            gdf_soil,
            name="Soil Area",
            style_function=lambda feature: {
                'fillColor': 'lightgreen',
                'color': 'green',
                'weight': 2,
                'fillOpacity': 0.3
            }
        ).add_to(m)
        
        # 上下水道管の追加
        def pipeline_style(feature):
            if feature['properties']['pipeline_type'] == 'Water Supply':
                return {'color': 'blue', 'weight': 4}
            else:
                return {'color': 'orange', 'weight': 4}
        
        folium.GeoJson(
            gdf_pipelines,
            name="Pipelines",
            style_function=pipeline_style,
            tooltip=folium.features.GeoJsonTooltip(fields=['pipeline_type'], aliases=["Type:"])
        ).add_to(m)
        
        folium.LayerControl().add_to(m)
        return m
    except Exception as e:
        st.error(f"Error in creating Folium map: {e}")
        raise

# ----- 2. 機械学習モデルによる予測（ダミーデータ） -----
@st.cache_resource(show_spinner=False)
def train_dummy_model():
    try:
        np.random.seed(42)
        X_train = np.random.rand(100, 5)
        y_train = np.random.rand(100)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        st.error(f"Error in training dummy model: {e}")
        raise

def predict_risk(model, input_features):
    try:
        risk_score = model.predict(input_features)
        return risk_score[0]
    except Exception as e:
        st.error(f"Error in predicting risk: {e}")
        raise

# ----- 3. PDFレポート生成 -----
def generate_pdf_report(risk_score):
    try:
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
    except Exception as e:
        st.error(f"Error in generating PDF report: {e}")
        raise

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
            gdf_soil, gdf_pipelines = create_sample_gis_data()
            folium_map = create_folium_map(gdf_soil, gdf_pipelines)
            st.components.v1.html(folium_map._repr_html_(), height=500)
    except Exception as e:
        st.error("Failed to display GIS data.")
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
        st.error("Failed to predict risk.")
        return
    
    # PDFレポート生成
    st.header("PDF Report Generation")
    try:
        if st.button("Generate PDF Report"):
            pdf_buffer = generate_pdf_report(risk_score)
            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name="kamijima_pipeline_safety_report.pdf",
                mime="application/pdf"
            )
    except Exception as e:
        st.error("Failed to generate PDF report.")

if __name__ == "__main__":
    main()
