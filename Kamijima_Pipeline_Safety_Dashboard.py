import streamlit as st
import geopandas as gpd
import folium
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import io
from reportlab.pdfgen import canvas
from shapely.geometry import Polygon, LineString

# ----- 1. GISデータの生成（疑似データ：愛媛県上島町中央付近） -----
@st.cache_data(show_spinner=False)
def create_sample_gis_data():
    """
    上島町中央付近に、疑似的な土壌の性質（左右2領域）とパイプライン（中央を横断するライン）を生成する。
    """
    try:
        # 土壌ポリゴンの作成
        # 土壌ポリゴン A: 左側（例：Sandy）
        coords_A = [
            (132.495, 33.847),  # 左下
            (132.500, 33.847),  # 右下
            (132.500, 33.853),  # 右上
            (132.495, 33.853)   # 左上
        ]
        poly_A = Polygon(coords_A)
        # 土壌ポリゴン B: 右側（例：Clay）
        coords_B = [
            (132.500, 33.847),
            (132.505, 33.847),
            (132.505, 33.853),
            (132.500, 33.853)
        ]
        poly_B = Polygon(coords_B)
        # 土壌GeoDataFrameの作成
        gdf_soil = gpd.GeoDataFrame(
            {'soil_type': ['Sandy', 'Clay']},
            geometry=[poly_A, poly_B],
            crs="EPSG:4326"
        )
        # パイプラインの作成：土壌ポリゴンの中央部を横断するライン
        # ここでは、土壌全体の左端から右端までの水平ラインとする
        pipeline_coords = [(132.495, 33.85), (132.505, 33.85)]
        pipeline_line = LineString(pipeline_coords)
        gdf_pipeline = gpd.GeoDataFrame(
            {'pipeline': ['Main Pipeline']},
            geometry=[pipeline_line],
            crs="EPSG:4326"
        )
        return gdf_soil, gdf_pipeline
    except Exception as e:
        st.error(f"Error in creating sample GIS data: {e}")
        raise

def create_folium_map(gdf_soil, gdf_pipeline):
    """
    土壌とパイプラインのGeoDataFrameからFoliumマップを作成する。
    """
    try:
        # 土壌GeoDataFrameの重心をマップの中心に設定
        mean_lat = gdf_soil.geometry.centroid.y.mean()
        mean_lon = gdf_soil.geometry.centroid.x.mean()
        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=14)
        
        # 土壌情報をGeoJsonレイヤーとして追加（土壌種別に応じた色分け）
        folium.GeoJson(
            gdf_soil,
            name="Soil Properties",
            style_function=lambda feature: {
                'fillColor': 'green' if feature['properties']['soil_type'] == 'Sandy' else 'brown',
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.5
            }
        ).add_to(m)
        
        # パイプライン情報をGeoJsonレイヤーとして追加
        folium.GeoJson(
            gdf_pipeline,
            name="Pipeline",
            style_function=lambda feature: {
                'color': 'red',
                'weight': 3
            }
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
            gdf_soil, gdf_pipeline = create_sample_gis_data()
            folium_map = create_folium_map(gdf_soil, gdf_pipeline)
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
