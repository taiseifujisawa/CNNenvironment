import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataSet:
    index: int
    arr: np.ndarray
    target: int

class SignClassifier:
    def __init__(self):
        """cnnの諸元の設定
        """
        RANDOM_SEED = 0
        tf.random.set_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        self.cwd = Path.cwd()
        self.truedir = self.cwd / 'dataset/true'
        self.falsedir = self.cwd / 'dataset/false'
        self.model_name = 'my_model'
        self.model_savefile = 'my_model.h5'
        self.last_layername = "last_conv"
        self.train_test_rate = 0.05
        self.validation_rate = 0.05
        self.input_shape = (250, 1000)
        self.outputs = 32
        self.optimizer = 'Adam'
        self.lossfunc = 'sparse_categorical_crossentropy'
        self.epochs = 1
        self.batchsize = 8
        self.loadfrompickle = True

    def loaddataset(self):
        """データの読み込み、成形

        Args:
            pickle (bool): pickleから読みだすか. Defaults to False.
        """
        if self.loadfrompickle:
            print('reading pickle...')
            self.X_train, self.X_test, self.y_train, self.y_test = serialize_read(self.cwd / 'dataset.pkl')
            print('finish!')
        else:
            ds = []

            # trueのラベリングは0から(被験者数)-1まで
            # sortedのkeyには関数を入れる
            print('loading true dataset...')
            for subject, dir in enumerate(tqdm(sorted(self.truedir.iterdir(), key=lambda x: int(x.name)))):
                for i, bmp in enumerate(dir.glob('*.bmp')):
                    ds.append(
                        DataSet(i,
                        cv2.cvtColor(cv2.imread(str(bmp)), cv2.COLOR_BGR2GRAY),
                        subject)
                    )
            print('finish!')

            # falseのラベリングは(被験者数)から2(被験者数)-1まで
            print('loading false dataset...')
            for subject, dir in enumerate(tqdm(sorted(self.falsedir.iterdir(), key=lambda x: int(x.name)))):
                for i, bmp in enumerate(dir.glob('*.bmp')):
                    ds.append(
                        DataSet(
                        i,
                        cv2.cvtColor(cv2.imread(str(bmp)), cv2.COLOR_BGR2GRAY),
                        subject + len(list(self.truedir.iterdir()))
                        )
                    )
            print('finish!')

            print('splitting them...')
            train, test = train_test_split(ds, test_size=self.train_test_rate)
            self.X_train = np.array([t.arr for t in train]) / 255
            self.y_train = np.array([t.target for t in train])
            self.X_test = np.array([t.arr for t in test]) / 255
            self.y_test = np.array([t.target for t in test])
            print('finish!')

            print('writing to pickle...')
            serialize_write(
                (self.X_train, self.X_test, self.y_train, self.y_test)
                , self.cwd / 'dataset.pkl'
                )
            print('finish!')
        print(f'train data : {len(self.y_train)}, test data : {len(self.y_test)}')

    def makecnnmodel(self):
        """モデルの作成(Sequential API)
        """
        self.model = tf.keras.Sequential([
            tf.keras.layers.Reshape(self.input_shape + (1,), input_shape=self.input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name=self.last_layername),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.outputs, activation='softmax')
        ], name=self.model_name)
        self.model.summary()
        self.model.compile(optimizer=self.optimizer, loss=self.lossfunc, metrics=['accuracy'])

    def training(self):
        """訓練
        """
        self.history = self.model.fit(self.X_train, self.y_train,\
            batch_size=self.batchsize, epochs=self.epochs, validation_split=self.validation_rate)
        self.model.save(self.model_savefile)

    def drawlossgraph(self):
        """損失関数の描画
        """
        loss     = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        nb_epoch = len(loss)
        fig = plt.figure()
        plt.plot(range(nb_epoch), loss,     marker='.', label='loss')
        plt.plot(range(nb_epoch), val_loss, marker='.', label='val_loss')
        plt.legend(loc='best', fontsize=10)
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        print("the graph of losses has been stored as \"Loss.png\"")
        fig.savefig("Loss.png")

    def testevaluate(self):
        """テストデータによる評価
        """
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print('Test loss:', test_loss)
        print('Test accuracy:', test_acc)

    def prediction(self):
        """全テストデータで予測、予測結果と間違えたテストデータのインデックスpickle化して保存
        """
        predictions = self.model.predict(self.X_test)
        predictions = [np.argmax(pred) for pred in predictions]
        index_failure = [i for i, (p, t) in enumerate(zip(predictions, self.y_test)) if p != t]
        serialize_write((predictions, index_failure), Path.cwd() / 'predictions.pkl')
        self.predict = predictions
        self.index_failure = index_failure

    @classmethod
    def deeplearning(cls):
        """ディープラーニングを実行

        Returns:
            sign: SignClassifierオブジェクト
        """
        sign = cls()
        sign.loaddataset()
        sign.makecnnmodel()
        sign.training()
        sign.drawlossgraph()
        sign.testevaluate()
        sign.prediction()
        return sign

    @classmethod
    def reconstructmodel(cls):
        """保存済みモデルの再構築

        Raises:
            e: 指定した名前の保存済みモデルがない場合

        Returns:
            sign: SignClassifierオブジェクト
        """
        sign = cls()
        sign.loaddataset()
        try:
            sign.model = tf.keras.models.load_model(sign.model_savefile)
        except OSError as e:
            print("No model exists")
            raise e
        else:
            sign.model.summary()
            sign.testevaluate()
            sign.prediction()
        return sign

    def array2img(self, test_no: int, save_dir: Path, extension='png', target_dpi=None, inverse=False, overwrite=True):
        """テストデータを画像ファイルとして保存

        Args:
            test_no (int): テストデータの何番目か
            save_dir (Path): 保存先ディレクトリ
            extension (str, optional): ファイル拡張子. Defaults to 'png'.
            target_dpi (np.ndarray, optional): リサイズするdpi. Defaults to None.
            inverse (bool, optional): 白黒反転. Defaults to False.
            overwrite (bool, optional): 同名ファイルに上書きするか. Defaults to True.
        """
        output_filename = f'{test_no}.{extension}'
        output_filepath = Path.cwd() / save_dir / output_filename
        dpi = (self.input_shape[1], self.input_shape[0])            # numpyは(row, column)　opencvは(x, y)で逆
        target_dpi = dpi if target_dpi == None else target_dpi
        if overwrite or not output_filepath.exists():
            arr = 1 - self.X_test[test_no] if inverse else self.X_test[test_no]
            resized_img = cv2.resize(arr, target_dpi) * 255
            cv2.imwrite(str(output_filepath), resized_img)
        else:
            print(f'There already exists {output_filepath.name}. Overwrite is not valid.')

def serialize_write(obj, filepath: Path):
    """pickle化

    Args:
        obj (Any): pickle化するオブジェクト
        filepath (Path): 保存するパス
    """
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, protocol=4)

def serialize_read(filepath: Path):
    """非pickle化

    Args:
        filepath (Path): 読みだすpickleファイル

    Returns:
        Any: 非pickle化されたオブジェクト
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def main():
    # ディープラーニング実行
    #sign = SignClassifier.deeplearning()

    # 保存済みモデル再構築
    sign = SignClassifier.reconstructmodel()

    # 0 - 9, failureのディレクトリ作成
    cwd = Path.cwd()
    (cwd / 'failure').mkdir(exist_ok=True)
    for i in range(sign.outputs):
        (cwd / f'{i}').mkdir(exist_ok=True)

    # 間違えたテストデータをpng, csvに出力
    with open('failure.csv', 'w', encoding='utf-8') as f:
        f.write('index,prediction,answer\n')
        for i in sign.index_failure:
            Path(f'failure/{i}').mkdir(exist_ok=True)
            sign.array2img(i, Path(f'failure/{i}'))
            f.write(f'{i},{sign.predict[i]},{sign.y_test[i]}\n')

    # 全テストデータをpngに出力
    for i, y_test in enumerate(tqdm(sign.y_test)):
        Path(f'{y_test}/{i}').mkdir(exist_ok=True)
        sign.array2img(i, Path(f'{y_test}/{i}'))

if __name__ == '__main__':
    main()
