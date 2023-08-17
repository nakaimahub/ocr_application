from ocr_model import transform, Net

# 必要なモジュールのインポート
import torch
from flask import Flask, request, render_template, redirect
import io
from PIL import Image
import base64

# 学習済みモデルをもとに推論する
def predict(img):
    # ネットワークの準備
    net = Net().cpu().eval()
    # # 学習済みモデルの重みを読み込み
    net.load_state_dict(torch.load('./src/ocr_model.pt', map_location=torch.device('cpu')))
    #　データの前処理
    img = transform(img)
    img =img.unsqueeze(0) # 1次元増やす
    #　推論
    y = torch.argmax(net(img), dim=1).cpu().detach().numpy()
    print(torch.argmax(net(img), dim=1).cpu().detach().numpy())
    return y

CHARACTER_LIST = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                  'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z')

 #　推論したラベル該当の文字を返す
def getName(label): 
   return CHARACTER_LIST[label[0]]
    
# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg'])

#　拡張子が適切かどうかをチェック
def allwed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Flask のインスタンスを作成
app = Flask(__name__)

@app.route('/', methods = ['GET'])
def init():
    return render_template('index.html')

# URL にアクセスがあった場合の挙動の設定
@app.route('/predict', methods = ['POST'])
def predicts():
    # ファイルがなかった場合の処理
    if 'filename' not in request.files:
        return redirect('/')
    # データの取り出し
    file = request.files['filename']
    # ファイルのチェック
    if file and allwed_file(file.filename):

        #　画像ファイルに対する処理
        #　画像書き込み用バッファを確保
        buf = io.BytesIO()
        #image = Image.open(file).convert('RGB')
        image = Image.open(file).convert('L')
        #　画像データをバッファに書き込む
        image.save(buf, 'png')
        #　バイナリデータを base64 でエンコードして utf-8 でデコード
        base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        #　HTML 側の src の記述に合わせるために付帯情報付与する
        base64_data = 'data:image/png;base64,{}'.format(base64_str)

        # 入力された画像に対して推論
        pred = predict(image)
        characterName_ = getName(pred)
        return render_template('result.html', characterName=characterName_, image=base64_data)
    return redirect('/')


# アプリケーションの実行の定義
if __name__ == '__main__':
    app.run(debug=True)