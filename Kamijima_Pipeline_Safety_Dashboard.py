import streamlit as st
import base64

# ページ設定（タイトルやレイアウト）
st.set_page_config(page_title="Kamijima_Pipeline_Safety_Dashboard", layout="wide")

# サイドバー：操作パネル
st.sidebar.title("操作パネル")

# アップロード機能：HTMLファイルをアップロード（エンコーディングは UTF-8 を前提）
uploaded_file = st.sidebar.file_uploader("HTMLファイルをアップロード", type=["html"])

if uploaded_file is not None:
    try:
        # アップロードされたファイルをバイナリとして読み込み、UTF-8 でデコード
        file_content = uploaded_file.read().decode("utf-8")
    except Exception as e:
        st.sidebar.error(f"ファイルの読み込みに失敗しました: {e}")
        file_content = ""
else:
    # ファイル未アップロードの場合は初期HTMLを設定
    file_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Kamijima_Pipeline_Safety_Dashboard</title>
</head>
<body>
    <h1>ダッシュボードへようこそ！</h1>
    <p>ここでパイプラインの安全情報を確認できます。</p>
</body>
</html>
"""

# エディタ部分：HTMLコードを編集できるテキストエリア
st.markdown("## HTMLエディタ")
html_content = st.text_area("HTMLコードを編集してください", value=file_content, height=300)

# プレビュー更新ボタン（クリックすると右サイドバーにメッセージ表示）
if st.sidebar.button("プレビュー更新"):
    st.sidebar.success("プレビューを更新しました")

# プレビュー表示：st.components.v1.html で編集内容をレンダリング
st.markdown("## プレビュー")
st.components.v1.html(html_content, height=400, scrolling=True)

# 保存機能：編集したHTML内容をダウンロードできるリンクを生成
def get_download_link(content, filename, link_text):
    """
    HTML内容を Base64 エンコードしてダウンロードリンクを生成する
    """
    b64 = base64.b64encode(content.encode()).decode()  # コンテンツを Base64 エンコード
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

download_link = get_download_link(html_content, "output.html", "HTMLファイルをダウンロード")
st.markdown(download_link, unsafe_allow_html=True)
