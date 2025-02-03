import streamlit as st
import geopandas as gpd
import folium
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import io
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from shapely.geometry import Polygon, LineString
import tempfile, zipfile
import os
import pandas as pd

# --- フォント登録 ---
try:
    pdfmetrics.registerFont(TTFont('NotoSansCJKjp', 'NotoSansCJKjp-Regular.otf'))
    FONT_NAME = "NotoSansCJKjp"
except Exception as e:
    st.error("日本語フォントの登録に失敗しました。ファイル 'NotoSansCJKjp-Regular.otf' の配置を確認してください。")
    FONT_NAME = "Helvetica"  # フォールバック

# ----- 1. GISデータの生成（疑似データ：上島町中央付近の広域領域と上下水道管） -----
@st.cache_data(show_spinner=False)
def create_sample_gis_data():
    """
    広い領域（経度 133.190～133.220、緯度 34.245～34.270）を表す土壌領域ポリゴンと、
    2種類の曲折した上下水道管（水道管と下水管）を生成する。
    """
    try:
        # 土壌領域ポリゴンの作成
        soil_coords = [
            (133.190, 34.245),  # 左下
            (133.220, 34.245),  # 右下
            (133.220, 34.270),  # 右上
            (133.190, 34.270)   # 左上
        ]
        soil_poly = Polygon(soil_coords)
        gdf_soil = gpd.GeoDataFrame({'soil': ['サンプル土壌']}, geometry=[soil_poly], crs="EPSG:4326")
        
        # 水道管：左下から右上へ、曲折しながら上昇するライン
        water_supply_coords = [
            (133.192, 34.248),
            (133.200, 34.252),
            (133.205, 34.257),
            (133.210, 34.262),
            (133.215, 34.266),
            (133.218, 34.268)
        ]
        water_supply_line = LineString(water_supply_coords)
        
        # 下水管：右下から左上へ、曲折しながら下降するライン
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
            {'pipeline_type': ['水道管', '下水管']},
            geometry=[water_supply_line, sewer_line],
            crs="EPSG:4326"
        )
        
        return gdf_soil, gdf_pipelines
    except Exception as e:
        st.error(f"GISデータ生成エラー: {e}")
        raise

def create_folium_map(gdf_soil, gdf_pipelines):
    """
    土壌領域ポリゴンと上下水道管のGeoDataFrameからFoliumマップを生成する。
    土壌領域は薄い緑、"水道管"は青、"下水管"はオレンジで表示する。
    """
    try:
        mean_lat = gdf_soil.geometry.centroid.y.mean()
        mean_lon = gdf_soil.geometry.centroid.x.mean()
        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=14)
        
        folium.GeoJson(
            gdf_soil,
            name="土壌領域",
            style_function=lambda feature: {
                'fillColor': 'lightgreen',
                'color': 'green',
                'weight': 2,
                'fillOpacity': 0.3
            }
        ).add_to(m)
        
        def pipeline_style(feature):
            if feature['properties']['pipeline_type'] == '水道管':
                return {'color': 'blue', 'weight': 4}
            else:
                return {'color': 'orange', 'weight': 4}
        
        folium.GeoJson(
            gdf_pipelines,
            name="上下水道管",
            style_function=pipeline_style,
            tooltip=folium.features.GeoJsonTooltip(fields=['pipeline_type'], aliases=["種類："])
        ).add_to(m)
        
        folium.LayerControl().add_to(m)
        return m
    except Exception as e:
        st.error(f"Foliumマップ生成エラー: {e}")
        raise

# ----- 施設データの読み込み用関数 -----
def load_gis_file(uploaded_file):
    """
    アップロードされた施設データ（GeoJSON、JSON、またはZip形式）を読み込み、GeoDataFrameを返す。
    """
    try:
        filename = uploaded_file.name
        if filename.endswith(".zip"):
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, filename)
                with open(zip_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(tmpdir)
                shp_files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".shp")]
                if len(shp_files) > 0:
                    gdf = gpd.read_file(shp_files[0])
                    return gdf
                else:
                    st.error("Zipファイル内に有効なShapefileが見つかりませんでした。")
                    return None
        else:
            gdf = gpd.read_file(uploaded_file)
            return gdf
    except Exception as e:
        st.error(f"施設データ読み込みエラー: {e}")
        return None

# ----- リスク予測用トレーニングデータの読み込み -----
def load_training_data(uploaded_file):
    """
    アップロードされたCSVファイルから、リスク予測用のトレーニングデータを読み込む。
    必要な列: "years", "deterioration", "soil_corrosiveness", "pressure", "flow", "risk"
    """
    try:
        df = pd.read_csv(uploaded_file)
        required_cols = {"years", "deterioration", "soil_corrosiveness", "pressure", "flow", "risk"}
        if not required_cols.issubset(set(df.columns)):
            st.error("CSVファイルに必要な列がありません。必要な列: years, deterioration, soil_corrosiveness, pressure, flow, risk")
            return None, None
        X = df[["years", "deterioration", "soil_corrosiveness", "pressure", "flow"]].values
        y = df["risk"].values
        return X, y
    except Exception as e:
        st.error(f"トレーニングデータ読み込みエラー: {e}")
        return None, None

# ----- 2. 機械学習モデルによるリスク予測 -----
@st.cache_resource(show_spinner=False)
def train_dummy_model():
    """
    ダミーデータを用いてRandomForestRegressorを学習する。
    ※実際の運用時は、トレーニングデータCSVからモデルを学習してください。
    """
    try:
        np.random.seed(42)
        X_train = np.random.rand(100, 5)
        y_train = np.random.rand(100)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        st.error(f"ダミーモデル学習エラー: {e}")
        raise

@st.cache_resource(show_spinner=False)
def train_model_from_data(X, y):
    """
    アップロードされたトレーニングデータを用いてRandomForestRegressorを学習する。
    """
    try:
        model = RandomForestRegressor()
        model.fit(X, y)
        return model
    except Exception as e:
        st.error(f"モデル学習エラー: {e}")
        raise

def predict_risk(model, input_features):
    try:
        risk_score = model.predict(input_features)
        return risk_score[0]
    except Exception as e:
        st.error(f"リスク予測エラー: {e}")
        raise

# ----- 3. PDFレポート生成 -----
def generate_pdf_report(risk_score):
    """
    予測されたリスクスコアを含むPDFレポートを生成する。
    日本語表示のため、登録したフォント（FONT_NAME）を使用する。
    """
    try:
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer)
        p.setFont(FONT_NAME, 16)
        p.drawString(100, 800, "上島町パイプライン安全レポート")
        p.setFont(FONT_NAME, 12)
        p.drawString(100, 780, "所在地：上島町、愛媛県")
        p.drawString(100, 760, f"予測リスクスコア： {risk_score:.2f}")
        p.drawString(100, 740, "注：これは実際のデータを用いたテストレポートです。")
        p.showPage()
        p.save()
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"PDFレポート生成エラー: {e}")
        raise

# ----- 4. Streamlitアプリ本体 -----
def main():
    st.title("上島町パイプライン安全ダッシュボード")
    
    # サイドバー：入力パラメータ（5特徴量）
    # ※以下は正規化された値（0～1）として入力してください。
    st.sidebar.header("入力パラメータ（5特徴量）")
    feature_values = []
    feature_labels = [
        "管の使用年数（正規化値）",
        "管の材質劣化指数（正規化値）",
        "土壌腐食性（正規化値）",
        "運転圧力（正規化値）",
        "流量（正規化値）"
    ]
    for label in feature_labels:
        val = st.sidebar.slider(label, 0.0, 1.0, 0.5, 0.01)
        feature_values.append(val)
    input_features = np.array(feature_values).reshape(1, -1)
    
    # サイドバー：リスク予測用トレーニングデータのアップロード（CSV形式）
    training_file = st.sidebar.file_uploader("リスク予測用のトレーニングデータをアップロードしてください（CSV形式）", type=["csv"])
    
    # サイドバー：施設データのアップロード（GeoJSON, JSON, Zip形式）
    uploaded_file = st.sidebar.file_uploader("施設データをアップロードしてください（GeoJSON、JSON、またはZip形式）", type=["geojson", "json", "zip"])
    
    # サイドバー：表示するGISデータの選択
    if uploaded_file is not None:
        data_option = st.sidebar.radio("表示するGISデータを選択してください", options=["サンプルデータ", "アップロードデータ"])
    else:
        data_option = "サンプルデータ"
    
    # GISデータの生成とFoliumマップ表示
    st.header("GIS表示（上島町エリア）")
    try:
        if data_option == "サンプルデータ":
            with st.spinner("サンプルデータを生成中..."):
                gdf_soil, gdf_pipelines = create_sample_gis_data()
                folium_map = create_folium_map(gdf_soil, gdf_pipelines)
        else:
            with st.spinner("施設データを読み込み中..."):
                gdf_uploaded = load_gis_file(uploaded_file)
                if gdf_uploaded is None:
                    st.error("施設データの読み込みに失敗しました。サンプルデータを表示します。")
                    gdf_soil, gdf_pipelines = create_sample_gis_data()
                    folium_map = create_folium_map(gdf_soil, gdf_pipelines)
                else:
                    folium_map = folium.Map(location=[34.25782859672118, 133.20487255560406], zoom_start=14)
                    folium.GeoJson(
                        gdf_uploaded,
                        name="アップロードデータ",
                        tooltip=folium.features.GeoJsonTooltip(fields=list(gdf_uploaded.columns), aliases=["項目："])
                    ).add_to(folium_map)
                    folium.LayerControl().add_to(folium_map)
        st.components.v1.html(folium_map._repr_html_(), height=500)
    except Exception as e:
        st.error("GISデータの表示に失敗しました。")
        return
    
    # 予測モデルの作成とリスク予測
    st.header("リスク予測")
    try:
        # トレーニングデータがアップロードされていれば、実データからモデルを学習
        if training_file is not None:
            with st.spinner("トレーニングデータを読み込み中..."):
                X_train, y_train = load_training_data(training_file)
            if X_train is not None:
                with st.spinner("モデルを学習中..."):
                    model = train_model_from_data(X_train, y_train)
            else:
                st.error("トレーニングデータの読み込みに失敗しました。ダミーモデルを使用します。")
                model = train_dummy_model()
        else:
            with st.spinner("ダミーモデルを学習中..."):
                model = train_dummy_model()
        with st.spinner("リスクを予測中..."):
            risk_score = predict_risk(model, input_features)
        st.write(f"予測されたリスクスコア： **{risk_score:.2f}**")
    except Exception as e:
        st.error("リスクの予測に失敗しました。")
        return
    
    # PDFレポート生成
    st.header("PDFレポート生成")
    try:
        if st.button("PDFレポートを生成"):
            pdf_buffer = generate_pdf_report(risk_score)
            st.download_button(
                label="PDFレポートをダウンロード",
                data=pdf_buffer,
                file_name="kamijima_pipeline_safety_report.pdf",
                mime="application/pdf"
            )
    except Exception as e:
        st.error("PDFレポートの生成に失敗しました。")

if __name__ == "__main__":
    main()
