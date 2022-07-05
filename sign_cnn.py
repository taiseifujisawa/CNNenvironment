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
import traceback
import shutil
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint


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
        self.train_test_rate = 0.2          # self.loadfrompickle = Falseなら保存済みのpicklefileのtestdataの割合に書き変わる
        self.validation_rate = 0.2          # self.loadfrompickleによらず一定
        self.input_shape = (250, 1000)      # numpy, (row, column)
        self.outputs = 32
        self.optimizer = Adam(lr=0.001)
        self.lossfunc = 'sparse_categorical_crossentropy'
        self.epochs = 50
        self.batchsize = 32
        self.loadfrompickle = True         # テストデータ数を変えるときはFalseにしてdatasetから読み込みなおす

    def loaddataset(self):
        """データの読み込み、成形

        Args:
            pickle (bool): pickleから読みだすか. Defaults to False.
        """
        print('\n====================\n\nloaddataset\n\n====================\n')
        if self.loadfrompickle:
            self.X_train, self.X_test, self.y_train, self.y_test = serialize_read(self.cwd / 'dataset.pkl')
            self.train_test_rate = len(self.y_test) / (len(self.y_train) + len(self.y_test))
        else:
            ds = []

            # trueのラベリングは0から(被験者数)-1まで
            # sortedのkeyには関数を入れる
            print('\n####################\n\nloading true dataset...\n')
            for subject, dir in enumerate(tqdm(sorted(self.truedir.iterdir(), key=lambda x: int(x.name)))):
                for i, bmp in enumerate(dir.glob('*.bmp')):
                    ds.append(
                        DataSet(i,
                        cv2.cvtColor(cv2.imread(str(bmp)), cv2.COLOR_BGR2GRAY),
                        subject)
                    )
            print('\nfinish!\n\n####################\n')

            # falseのラベリングは(被験者数)から2(被験者数)-1まで
            print('\n####################\n\nloading false dataset...\n')
            for subject, dir in enumerate(tqdm(sorted(self.falsedir.iterdir(), key=lambda x: int(x.name)))):
                for i, bmp in enumerate(dir.glob('*.bmp')):
                    ds.append(
                        DataSet(
                        i,
                        cv2.cvtColor(cv2.imread(str(bmp)), cv2.COLOR_BGR2GRAY),
                        subject + len(list(self.truedir.iterdir()))
                        )
                    )
            print('\nfinish!\n\n####################\n')

            print('\n####################\n\nsplitting them...\n')
            if self.train_test_rate == 1:
                test = ds
                self.X_train = np.array([])
                self.y_train = np.array([])
                self.X_test = np.array([t.arr for t in test]) / 255
                self.y_test = np.array([t.target for t in test])
                print('No train data exists')
            else:
                try:
                    train, test = train_test_split(ds, test_size=self.train_test_rate)
                except ValueError:
                    train = ds
                    self.X_test = np.array([])
                    self.y_test = np.array([])
                    print('The exception below was ignored')
                    print(traceback.format_exc())
                    print('No test data exists')
                else:
                    self.X_test = np.array([t.arr for t in test]) / 255
                    self.y_test = np.array([t.target for t in test])
                finally:
                    self.X_train = np.array([t.arr for t in train]) / 255
                    self.y_train = np.array([t.target for t in train])
            print('\nfinish!\n\n####################\n')

            serialize_write(
                (self.X_train, self.X_test, self.y_train, self.y_test)
                , self.cwd / 'dataset.pkl'
                )
        print(f'# of train data : {len(self.y_train)}, # of test data : {len(self.y_test)}')

    def makecnnmodel(self):
        """モデルの作成(Sequential API)
        """
        print('\n====================\n\nmakecnnmodel\n\n====================\n')
        self.model = tf.keras.Sequential([
            tf.keras.layers.Reshape(self.input_shape + (1,), input_shape=self.input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name=self.last_layername),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            #tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            #tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.outputs, activation='softmax')
        ], name=self.model_name)
        self.model.summary()
        self.model.compile(optimizer=self.optimizer, loss=self.lossfunc, metrics=['accuracy'])

    def training(self):
        """訓練
        """
        print('\n====================\n\ntraining\n\n====================\n')
        early_stopping = EarlyStopping(
                        monitor='val_loss',
                        min_delta=0.0,
                        patience=5
                )
        reduce_lr = ReduceLROnPlateau(
                                monitor='val_loss',
                                factor=0.5,
                                patience=2,
                                min_lr=0.0001
                        )
        tensor_board = TensorBoard(
            log_dir=f'tensorboard_{self.model_name}',
            histogram_freq=1,
            write_images=True
        )
        check_point = ModelCheckpoint(
            f'best_{self.model_savefile}',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=False
        )
        self.history = self.model.fit(self.X_train, self.y_train,
            batch_size=self.batchsize, epochs=self.epochs, validation_split=self.validation_rate,
            callbacks=[early_stopping, reduce_lr, tensor_board, check_point])
        self.model.save(f'last_{self.model_savefile}')

    def drawlossgraph(self):
        """損失関数の描画
        """
        print('\n====================\n\ndrawlossgraph\n\n====================\n')
        loss     = self.history.history['loss']
        nb_epoch = len(loss)
        fig = plt.figure()
        plt.plot(range(nb_epoch), loss,     marker='.', label='loss')
        try:
            val_loss = self.history.history['val_loss']
        except KeyError:
            print('The exception below was ignored')
            print(traceback.format_exc())
            print('No validation data exists')
            with open('trainlog.csv', 'w') as f:
                f.write('epoch,loss,acc\n')
                for ep, (ls, ac) in enumerate(zip(loss, self.history.history['accuracy'])):
                    f.write(f'{ep},{ls},{ac}\n')
        else:
            plt.plot(range(nb_epoch), val_loss, marker='.', label='val_loss')
            with open('trainlog.csv', 'w') as f:
                f.write('epoch,loss,acc,val_loss,val_acc\n')
                for ep, (ls, ac, vls, vac) in enumerate(zip(loss, self.history.history['accuracy'],
                    val_loss, self.history.history['val_accuracy'])):
                    f.write(f'{ep},{ls},{ac},{vls},{vac}\n')
        finally:
            plt.legend(loc='best', fontsize=10)
            plt.grid()
            plt.xlabel('epoch')
            plt.ylabel('loss')
            print("the graph of losses has been stored as \"Loss.png\"")
            fig.savefig("Loss.png")

    def testevaluate(self):
        """テストデータによる評価
        """
        print('\n====================\n\ntestevaluate\n\n====================\n')
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print('Test loss:', test_loss)
        print('Test accuracy:', test_acc)

    def prediction(self):
        """全テストデータで予測、予測結果と間違えたテストデータのインデックスpickle化して保存
        """
        print('\n====================\n\nprediction\n\n====================\n')
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
        print('\n********************\n\ndeeplearning\n\n********************\n')
        sign = cls()
        assert sign.train_test_rate != 1, 'No train data exists. It is no use you having NN learn.'
        sign.loaddataset()
        sign.makecnnmodel()
        sign.training()
        sign.drawlossgraph()
        if len(sign.y_test) == 0:
            print('No test data exists')
        else:
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
        print('\n********************\n\nreconstructmodel\n\n********************\n')
        sign = cls()
        sign.loaddataset()
        try:
            sign.model = tf.keras.models.load_model(f'best_{sign.model_savefile}')
        except OSError as e:
            print("No model exists")
            raise e
        else:
            sign.model.summary()
            if len(sign.y_test) == 0:
                print('No test data exists')
            else:
                sign.testevaluate()
                sign.prediction()
        return sign

    def array2img(self, test_no: int, save_path: Path, extension='png', target_dpi=None, inverse=False, overwrite=True):
        """テストデータを画像ファイルとして保存,
        save_pathがdirなら'{test_no}.{extension}'という名前でそのdirに保存
        save_dirがfileなら'{savedir}.{extension}'という名前でそのパスに保存(呼び出し側で拡張子をつけると2重になるので注意)

        Args:
            test_no (int): テストデータの何番目か
            save_dir (Path): 保存先ディレクトリorパス
            extension (str, optional): ファイル拡張子. Defaults to 'png'.
            target_dpi (np.ndarray, optional): リサイズするdpi. Defaults to None.
            inverse (bool, optional): 白黒反転. Defaults to False.
            overwrite (bool, optional): 同名ファイルに上書きするか. Defaults to True.
        """
        if save_path.is_dir():
            output_filename = f'{test_no}.{extension}'
            output_filepath = Path.cwd() / save_path / output_filename
        else:
            output_filepath = Path(f'{save_path}.{extension}')
        dpi = (self.input_shape[1], self.input_shape[0])            # numpyは(row, column)　opencvは(x, y)で逆(指定の仕方)
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
    print('\n####################\n\nwriting to pickle...\n')
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, protocol=4)
    print('\nfinish!\n\n####################\n')

def serialize_read(filepath: Path):
    """非pickle化

    Args:
        filepath (Path): 読みだすpickleファイル

    Returns:
        Any: 非pickle化されたオブジェクト
    """
    print('\n####################\n\nreading from pickle...\n')
    with open(filepath, 'rb') as f:
        pic = pickle.load(f)
    print('\nfinish!\n\n####################\n')
    return pic


def main():
    # ディープラーニング実行
    #sign = SignClassifier.deeplearning()

    # 保存済みモデル再構築
    sign = SignClassifier.reconstructmodel()

    if len(sign.y_test) == 0:
        pass
    else:
        # failureとクラスごとのディレクトリ作成
        result_dir = Path.cwd() / 'test_results'
        try:
            shutil.rmtree(str(result_dir))
        except:
            pass
        result_dir.mkdir(exist_ok=True)
        (result_dir / 'failure').mkdir(exist_ok=True)
        for i in range(sign.outputs):
            (result_dir / f'{i}').mkdir(exist_ok=True)

        # 間違えたテストデータをpng, csvに出力
        with open('failure.csv', 'w', encoding='utf-8') as f:
            f.write('index,prediction,answer\n')
            for i in sign.index_failure:
                sign.array2img(i, result_dir / f'failure/i{i}_p{sign.predict[i]}_a{sign.y_test[i]}')
                f.write(f'{i},{sign.predict[i]},{sign.y_test[i]}\n')

        # 全テストデータをpngに出力
        for i, y_test in enumerate(tqdm(sign.y_test)):
            #Path(result_dir / f'{y_test}/{i}').mkdir(exist_ok=True)
            sign.array2img(i, result_dir / f'{y_test}/i{i}_p{sign.predict[i]}_a{sign.y_test[i]}')

if __name__ == '__main__':
    main()
