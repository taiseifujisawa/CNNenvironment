# CNNenvironment
## How to use
1. `py -3.7 -m venv {environment_name}`で仮想環境を立ち上げる
1. `pip install -r requirements.txt`で必要なモジュールをインストール(cuda11.0, python3.7でGPU運用を想定、tensorflow, kerasのバージョンは必要に応じて要変更)
1. deep_learning_cnn_classを実行ファイルと同階層に置き、`from deep_learning_cnn_class import DeepLearningCnnClassifier, GradCam, set_seed`でパッケージを実行ファイルにインポートする
1. 学習の設定などはcnnconfig.pyにクラスを定義し実行ファイルにインポートする(ex: `from cnnconfig import Cifar10Config`)
1. cnnの構造はcnn.pyのように関数で定義し実行ファイルにインポートする(ex: `from cnn import cifar10_cnn`)


## example of use
`py -3.7 ./example.py`

## To do
- ~gradcamがまだカラー非対応~
- ~callbackにpathlib使うと3.6ではエラーが出る~
- ~keras.utils.image_utilsは3.6にはない~
- deep_learning_cnn_class/deep_learning.py のloaddataset内でreshapeを行えるようにする(転移学習用)
  - 引数でshapeを指定
