# coding: utf-8

from flask import Flask, render_template

# Flask をインスタンス化
app = Flask(__name__)

# --- View側の設定 ---
# ルートディレクトリにアクセスした場合の挙動


if __name__ == '__main__':
    app.run()