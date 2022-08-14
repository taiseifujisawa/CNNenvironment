import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import traceback
import shutil
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from grad_cam import GradCam
from cnn import sign_classifier_cnn, cifar10_cnn
from cnnconfig import CnnConfig, Cifar10Config2, Cifar10Config3

from keras.datasets import cifar10



class SignClassifier:
    def __init__(self, cnf):
        """cnnの諸元の設定
        """
        self.cnf = cnf

    def loaddataset(self, dataset=None):    # 呼び出しはself.cnf.load_modeで分ける
        """データの読み込み、成形、data augmentation
        """
        print('\n====================\n\nloaddataset\n\n====================\n')

        ## データの読み込み
        # ディレクトリ構造から読み込む場合
        if self.cnf.load_mode == 'directory':
            # ディレクトリ初期化
            shutil.rmtree(self.cnf.splitted_datasetdir, ignore_errors=True)
            # シャッフルして分割
            for d in self.cnf.datasetdir.iterdir():
                # シャッフル
                imgs_shuffled = random.sample(list(d.iterdir()), len(list(d.iterdir())))
                # 分割
                imgs_test = imgs_shuffled[:int(len(imgs_shuffled) * self.cnf.train_test_rate)]
                imgs_rest = imgs_shuffled[int(len(imgs_shuffled) * self.cnf.train_test_rate):]
                imgs_validation = imgs_rest[:int(len(imgs_rest) * self.cnf.train_test_rate)]
                imgs_train = imgs_rest[int(len(imgs_rest) * self.cnf.train_test_rate):]

                # 分割したファイルを置くディレクトリ作成
                (self.cnf.splitted_datasetdir /'train' / d.name).mkdir(parents=True, exist_ok=True)
                (self.cnf.splitted_datasetdir / 'validation' / d.name).mkdir(parents=True, exist_ok=True)
                (self.cnf.splitted_datasetdir / 'test' / d.name).mkdir(parents=True, exist_ok=True)
                # 元データセットからそれぞれの分割先へコピー
                for i in imgs_train:
                    shutil.copy(i, self.cnf.splitted_datasetdir / 'train' / d.name)
                for i in imgs_validation:
                    shutil.copy(i, self.cnf.splitted_datasetdir / 'validation' / d.name)
                for i in imgs_test:
                    shutil.copy(i, self.cnf.splitted_datasetdir / 'test' / d.name)
        # 既にあるデータセットから読み込む場合
        elif self.cnf.load_mode == 'database':
            assert dataset != None, 'set dataset in an argument'
            self.cnf.lossfunc = 'categorical_crossentropy'
            (x_train, y_train), (x_test, y_test) = dataset
            # 出力ノード数をデータセットから検出し書き換え
            self.cnf.outputs = y_test.max() + 1
            self.len_y_train = len(y_train)
            self.len_y_test = len(y_test)
            y_train = to_categorical(y_train)
            y_test = to_categorical(y_test)
        else:
            raise NameError('invalid value was set to "self.cnf.load_mode". check cnnconfig.py.')

        ## data augmentationの設定(正規化含む)
        # ディレクトリ構造から読み込む場合
        if self.cnf.load_mode == 'directory':
            idg = ImageDataGenerator(rescale=1.0/self.cnf.max_pixelvalue, **self.cnf.da_cnf)
        # 既にあるデータセットから読み込む場合
        elif self.cnf.load_mode == 'database':
            idg = ImageDataGenerator(
                rescale=1.0/self.cnf.max_pixelvalue,
                validation_split=self.cnf.validation_rate,
                **self.cnf.da_cnf
            )
        idg_test = ImageDataGenerator(rescale=1.0/self.cnf.max_pixelvalue)

        ## ジェネレータの作成
        # ディレクトリ構造から読み込む場合
        if self.cnf.load_mode == 'directory':
            # 変換
            class_mode = {
                'categorical_crossentropy': 'categorical',
                'binary_crossentropy': 'binary',
                'sparse_categorical_crossentropy': 'sparse',
            }
            # train dataのジェネレータ
            print('train data: ', end='')
            self.train_generator = idg.flow_from_directory(
                self.cnf.splitted_datasetdir / 'train',
                target_size=self.cnf.input_shape,
                color_mode=self.cnf.color,
                shuffle=True,
                class_mode=class_mode[self.cnf.lossfunc],
                batch_size=self.cnf.batchsize
                )
            # validation dataのジェネレータ
            print('validation data: ', end='')
            self.validation_generator = idg.flow_from_directory(
                self.cnf.splitted_datasetdir / 'validation',
                target_size=self.cnf.input_shape,
                color_mode=self.cnf.color,
                shuffle=True,
                class_mode=class_mode[self.cnf.lossfunc],
                batch_size=self.cnf.batchsize
                )
            # test dataのジェネレータ
            print('test data: ', end='')
            self.test_generator = idg_test.flow_from_directory(
                self.cnf.splitted_datasetdir / 'test',
                target_size=self.cnf.input_shape,
                color_mode=self.cnf.color,
                shuffle=False,
                class_mode=class_mode[self.cnf.lossfunc],
                batch_size=self.cnf.batchsize
                )
            # 出力ノード数をデータセットから検出し書き換え
            self.cnf.outputs = len(self.train_generator.class_indices)
            # ディレクトリ名とラベルの対応を表示
            print(
                'labels of train data: {}\nlabels of validation data: {}\nlabels of test data: {}'
                .format(
                    self.train_generator.class_indices,
                    self.validation_generator.class_indices,
                    self.test_generator.class_indices
                    )
                )
        # 既にあるデータセットから読み込む場合
        elif self.cnf.load_mode == 'database':
            # train dataのジェネレータ
            print(f'train data: Found {math.ceil(len(y_train) * (1 - self.cnf.validation_rate))} images belonging to {self.cnf.outputs} classes.')
            self.train_generator = idg.flow(
            x_train, y_train, self.cnf.batchsize, True, subset="training")
            # validation dataのジェネレータ
            print(f'validation data: Found {len(y_train) - math.ceil(len(y_train) * (1 - self.cnf.validation_rate))} images belonging to {self.cnf.outputs} classes.')
            self.validation_generator = idg.flow(
            x_train, y_train, self.cnf.batchsize, True, subset="validation")
            # test dataのジェネレータ
            print(f'test data: Found {len(y_test)} images belonging to {self.cnf.outputs} classes.')
            self.test_generator = idg_test.flow(
            x_test, y_test, self.cnf.batchsize, False)

        ## data augmentationチェック
        def plotImages(images_arr):
            fig, axes = plt.subplots(2, 5, figsize=(10,10))
            axes = axes.flatten()
            for img, ax in zip( images_arr, axes):
                ax.imshow(img)
                ax.axis('off')
            plt.tight_layout()
            fig.suptitle('Examples of train data with data augmentation')
            plt.show()
        def check():
            training_images, training_labels = next(self.train_generator)
            print(training_labels[:10])
            plotImages(training_images[:10])
        # チェックするとき呼ぶ
        check()
        pass

    def makecnnmodel(self, cnn=None):
        """モデルの作成
        """
        print('\n====================\n\nmakecnnmodel\n\n====================\n')
        self.model = cnn

    def training(self):
        """訓練
        """
        print('\n====================\n\ntraining\n\n====================\n')

        ## delete tensorboard logdir
        log_dir = self.cnf.wd / f'tensorboard_{self.cnf.model_name}'
        try:
            shutil.rmtree(str(log_dir))
        except:
            pass

        ## callbacks
        early_stopping = EarlyStopping(
                        monitor='val_loss',
                        patience=self.cnf.earlystopping_patience
                )
        reduce_lr = ReduceLROnPlateau(
                                monitor='val_loss',
                                factor=self.cnf.reducelearningrate_factor,
                                patience=self.cnf.reducelearningrate_patience,
                                min_lr=self.cnf.minimum_learningrate
                        )
        tensor_board = TensorBoard(
            log_dir=self.cnf.wd / f'tensorboard_{self.cnf.model_name}',
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        check_point = ModelCheckpoint(
            self.cnf.wd / f'best_{self.cnf.model_savefile}',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=False
        )

        ## 学習
        # ディレクトリ構造から読み込む場合
        if self.cnf.load_mode == 'directory':
            self.history = self.model.fit(
                self.train_generator,
                steps_per_epoch=math.ceil(self.train_generator.samples / self.cnf.batchsize),
                validation_data=self.validation_generator,
                validation_steps=math.ceil(self.validation_generator.samples / self.cnf.batchsize),
                epochs=self.cnf.epochs,
                verbose=1,
                callbacks=[early_stopping, reduce_lr, tensor_board, check_point])
            self.model.save(self.cnf.wd / f'last_{self.cnf.model_savefile}')
            self.model = tf.keras.models.load_model(self.cnf.wd / f'best_{self.cnf.model_savefile}'
            )
        # 既にあるデータセットから読み込む場合
        elif self.cnf.load_mode == 'database':
            self.history = self.model.fit(
                self.train_generator,
                steps_per_epoch=math.ceil(self.len_y_train * (1 - self.cnf.validation_rate) / self.cnf.batchsize),
                validation_data=self.validation_generator,
                validation_steps=math.ceil(self.len_y_train * self.cnf.validation_rate / self.cnf.batchsize),
                epochs=self.cnf.epochs,
                verbose=1,
                callbacks=[early_stopping, reduce_lr, tensor_board, check_point])
            self.model.save(self.cnf.wd / f'last_{self.cnf.model_savefile}')
            self.model = tf.keras.models.load_model(self.cnf.wd / f'best_{self.cnf.model_savefile}'
            )


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
            with open(self.cnf.wd / 'trainlog.csv', 'w') as f:
                f.write('epoch,loss,acc\n')
                for ep, (ls, ac) in enumerate(zip(loss, self.history.history['accuracy'])):
                    f.write(f'{ep},{ls},{ac}\n')
        else:
            plt.plot(range(nb_epoch), val_loss, marker='.', label='val_loss')
            with open(self.cnf.wd / 'trainlog.csv', 'w') as f:
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
            fig.savefig(self.cnf.wd / "Loss.png")

    def testevaluate(self):
        """テストデータによる評価
        """
        print('\n====================\n\ntestevaluate\n\n====================\n')
        test_loss, test_acc = self.model.evaluate(self.test_generator, verbose=1, batch_size=self.cnf.batchsize)
        print('Test loss:', test_loss)
        print('Test accuracy:', test_acc)

    def prediction(self):
        """全テストデータで予測
        """
        print('\n====================\n\nprediction\n\n====================\n')
        predictions = self.model.predict(self.test_generator, verbose=1, batch_size=self.cnf.batchsize)
        predictions = [np.argmax(pred) for pred in predictions]
        answers = self.test_generator.classes if self.cnf.load_mode == 'directory'\
                                            else np.where(self.test_generator.y == 1)[1]
        print('Test result:')
        print(tf.math.confusion_matrix(answers, predictions).numpy())

    def deeplearning(self):
        """ディープラーニングを実行
        """
        print('\n********************\n\ndeeplearning\n\n********************\n')
        assert self.cnf.train_test_rate != 1, 'No train data exists. It is no use you having NN learn.'
        dataset = cifar10.load_data()
        self.loaddataset(dataset)
        #self.loaddataset()
        #cnn = sign_classifier_cnn(self.cnf)
        cnn = cifar10_cnn(self.cnf)
        self.makecnnmodel(cnn)
        self.training()
        self.drawlossgraph()
        if self.cnf.train_test_rate == 0:
            print('No test data exists')
        else:
            self.testevaluate()
            self.prediction()

    def reconstructmodel(self):
        """保存済みモデルの再構築

        Raises:
            e: 指定した名前の保存済みモデルがない場合

        Returns:
            sign: SignClassifierオブジェクト
        """
        print('\n********************\n\nreconstructmodel\n\n********************\n')
        self.loaddataset()
        try:
            self.model = tf.keras.models.load_model(self.cnf.wd / f'best_{self.cnf.model_savefile}')
        except OSError as e:
            print("No model exists")
            raise e
        else:
            self.model.summary()
            if self.cnf.train_test_rate == 0:
                print('No test data exists')
            else:
                self.testevaluate()
                self.prediction()


def main():
    cnf = CnnConfig()
    cnf = Cifar10Config3()
    sign = SignClassifier(cnf)

    # ディープラーニング実行
    sign.deeplearning()

    # 保存済みモデル再構築
    #sign.reconstructmodel()

    # gradcam起動
    cam = GradCam(sign)
    if sign.cnf.train_test_rate == 0:
        pass
    else:
        cam.batch_singularize()


if __name__ == '__main__':
    main()
