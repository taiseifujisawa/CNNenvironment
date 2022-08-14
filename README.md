# CNNenvironment
## How to use
- deep_learning_cnn_classを実行ファイルと同階層に置く
  - `from deep_learning_cnn_class import DeepLearningCnnClassifier, GradCam, set_seed`でインポートする
- 学習の設定などはcnnconfig.pyにクラスを定義しインポートする(ex: `from cnnconfig import Cifar10Config`)
- cnnの構造はcnn.pyのように関数で定義しインポートする(ex: `from cnn import cifar10_cnn`)

## example of use
`py -m ./example.py`

## To do
- ~gradcamがまだカラー非対応~
- callbackにpathlib使うと3.6ではエラーが出る
- keras.utils.image_utilsは3.6にはない
