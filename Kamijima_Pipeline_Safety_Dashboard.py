import os
import sys
import chardet
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QInputDialog,
    QWidget,
    QToolBar,
    QAction,
)
from PyQt5.QtWebEngineWidgets import QWebEngineView


class Kamijima_Pipeline_Safety_Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kamijima_Pipeline_Safety_Dashboard")
        self.setGeometry(100, 100, 1200, 800)

        # プレビュー用ビュー（HTML表示用）
        self.preview = QWebEngineView()
        self.preview.setHtml(self.get_initial_html())

        self.init_ui()      # ボタンなどのUIを初期化
        self.init_toolbar()  # ツールバーを初期化

    def init_ui(self):
        """編集ボタンやプレビュー画面を配置"""
        layout = QVBoxLayout()

        # ボタン: HTMLを開く
        open_btn = QPushButton("HTMLを開く")
        open_btn.clicked.connect(self.load_html)

        # ボタン: HTMLを保存
        save_btn = QPushButton("HTMLを保存")
        save_btn.clicked.connect(self.save_html)

        # レイアウトにボタンとプレビューを追加
        layout.addWidget(open_btn)
        layout.addWidget(save_btn)
        layout.addWidget(self.preview)

        # メインウィジェットの設定
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def init_toolbar(self):
        """ツールバーを初期化して追加"""
        toolbar = QToolBar("編集ツールバー")
        self.addToolBar(toolbar)

        # 太字
        bold_action = QAction("太字", self)
        bold_action.triggered.connect(self.make_bold)
        toolbar.addAction(bold_action)

        # イタリック
        italic_action = QAction("イタリック", self)
        italic_action.triggered.connect(self.make_italic)
        toolbar.addAction(italic_action)

        # フォントサイズを大きく
        increase_font_action = QAction("フォントサイズを大きく", self)
        increase_font_action.triggered.connect(self.increase_font_size)
        toolbar.addAction(increase_font_action)

        # フォントサイズを小さく
        decrease_font_action = QAction("フォントサイズを小さく", self)
        decrease_font_action.triggered.connect(self.decrease_font_size)
        toolbar.addAction(decrease_font_action)

        # リンク挿入
        add_link_action = QAction("リンクを挿入", self)
        add_link_action.triggered.connect(self.add_link)
        toolbar.addAction(add_link_action)

        # ファイル添付
        attach_file_action = QAction("ファイルを添付", self)
        attach_file_action.triggered.connect(self.add_file_attachment)
        toolbar.addAction(attach_file_action)

        # 編集モード切替
        edit_mode_action = QAction("編集モード切替", self)
        edit_mode_action.triggered.connect(self.toggle_edit_mode)
        toolbar.addAction(edit_mode_action)

    def get_initial_html(self):
        """初期HTMLを返す"""
        return """
        <!DOCTYPE html>
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

    def load_html(self):
        """HTMLファイルを読み込む"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "HTMLファイルを開く", "", "HTML Files (*.html);;All Files (*)"
        )
        if file_name:
            # エンコーディングを自動検出
            with open(file_name, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                detected_encoding = result['encoding'] if result['encoding'] else "utf-8"
            try:
                with open(file_name, 'r', encoding=detected_encoding, errors="replace") as file:
                    html_content = file.read()
                    self.preview.setHtml(html_content)
            except Exception as e:
                print(f"エラー: {e}")

    def save_html(self):
        """プレビュー内容をHTMLファイルとして保存"""
        file_name, _ = QFileDialog.getSaveFileName(
            self, "HTMLファイルを保存", "", "HTML Files (*.html);;All Files (*)"
        )
        if file_name:
            def sanitize_html(html):
                # contentEditable属性を削除して保存
                sanitized_html = html.replace("contentEditable='true'", "")
                # metaタグがなければ追加
                if "<meta charset=" not in sanitized_html:
                    sanitized_html = sanitized_html.replace("<head>", '<head>\n<meta charset="utf-8">')
                return sanitized_html
            self.preview.page().toHtml(lambda html: self.write_to_file(file_name, sanitize_html(html)))

    def write_to_file(self, file_name, content):
        """ファイルにUTF-8エンコーディングで書き込む"""
        try:
            with open(file_name, 'w', encoding='utf-8', errors="replace") as file:
                file.write(content)
            print(f"保存が完了しました: {file_name}")
        except Exception as e:
            print(f"保存中にエラーが発生しました: {e}")

    def make_bold(self):
        """選択したテキストを太字にする"""
        self.preview.page().runJavaScript("document.execCommand('bold');")

    def make_italic(self):
        """選択したテキストをイタリックにする"""
        self.preview.page().runJavaScript("document.execCommand('italic');")

    def increase_font_size(self):
        """フォントサイズを大きくする"""
        self.preview.page().runJavaScript("document.execCommand('fontSize', false, '5');")

    def decrease_font_size(self):
        """フォントサイズを小さくする"""
        self.preview.page().runJavaScript("document.execCommand('fontSize', false, '2');")

    def add_link(self):
        """選択したテキストにリンクを挿入する"""
        url, ok = QInputDialog.getText(self, "リンクを入力", "リンクを入力してください:")
        if ok and url:
            js_code = f"document.execCommand('createLink', false, '{url}');"
            self.preview.page().runJavaScript(js_code)
            print(f"リンクが選択されたテキストに追加されました: {url}")

    def add_file_attachment(self):
        """
        ファイルを添付する。
        ・選択されているテキストがあれば、そのテキストをアンカーでラップします。
        ・選択されていなければ、ファイル名をリンクテキストとしてカーソル位置に挿入します。
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "添付するファイルを選択", "", "All Files (*)")
        if file_path:
            file_name = os.path.basename(file_path)
            js_code = f"""
            (function() {{
                var sel = window.getSelection();
                if (sel.rangeCount > 0) {{
                    var range = sel.getRangeAt(0);
                    if (!range.collapsed) {{
                        // 選択されているテキストをアンカーでラップ
                        var selectedText = range.toString();
                        var a = document.createElement('a');
                        a.href = "file://{file_path}";
                        a.download = "";
                        a.textContent = selectedText;
                        range.deleteContents();
                        range.insertNode(a);
                        range.setStartAfter(a);
                        range.collapse(true);
                        sel.removeAllRanges();
                        sel.addRange(range);
                    }} else {{
                        // 選択がない場合は、カーソル位置にファイル名をリンクとして挿入
                        var a = document.createElement('a');
                        a.href = "file://{file_path}";
                        a.download = "";
                        a.textContent = "{file_name}";
                        range.insertNode(a);
                        range.setStartAfter(a);
                        range.collapse(true);
                        sel.removeAllRanges();
                        sel.addRange(range);
                    }}
                }} else {{
                    document.body.insertAdjacentHTML('beforeend', '<a href="file://{file_path}" download>{file_name}</a>');
                }}
            }})();
            """
            self.preview.page().runJavaScript(js_code)
            print(f"ファイルが添付されました: {file_path}")

    def toggle_edit_mode(self):
        """編集モードを切り替える"""
        self.preview.page().runJavaScript("""
            var isEditable = document.body.isContentEditable;
            document.body.contentEditable = !isEditable;
            alert("編集モード: " + (!isEditable ? "有効" : "無効"));
        """)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Kamijima_Pipeline_Safety_Dashboard()
    window.show()
    sys.exit(app.exec_())
