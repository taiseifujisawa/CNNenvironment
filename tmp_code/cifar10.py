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
from cnn import cifar10_cnn
from cnnconfig import Cifar10Config, Cifar10Config2

from keras.datasets import cifar10
#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context

class Sifar10Classifier:
    def __init__(self):
        """cnnの諸元の設定
        """
        self.cnf = Cifar10Config()
        self.cnf = Cifar10Config2()

    def loaddataset(self):
        """データの読み込み、成形、data augmentation
        """
        print('\n====================\n\nloaddataset\n\n====================\n')

        # ディレクトリ初期化
        #shutil.rmtree(self.cnf.splitted_datasetdir, ignore_errors=True)
        # シャッフルして分割

        # 全画像を[0, 1]へ正規化(data augmentationを行える)
        idg = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=self.cnf.validation_rate,
            horizontal_flip=True, vertical_flip=True,
            zoom_range=0.2,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2
            )
        idg_test = ImageDataGenerator(
            rescale=1.0 / 255,
            )


        (x_train,y_train),(x_test,y_test)=cifar10.load_data()
        (self.x_train,self.y_train),(self.x_test,self.y_test)=(x_train / 255,y_train),(x_test / 255,y_test)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        # train dataのジェネレータ
        #print('train data: ', end='')
        self.train_generator = idg.flow(
        x_train, y_train, self.cnf.batchsize, True, subset="training")

        # validation dataのジェネレータ
        #print('train data: ', end='')
        self.validation_generator = idg.flow(
        x_train, y_train, self.cnf.batchsize, True, subset="validation")

        # test dataのジェネレータ
        #print('test data: ', end='')
        self.test_generator = idg_test.flow(
        x_test, y_test, self.cnf.batchsize, False)

        # data augmentationチェック用
        def plotImages(images_arr):
            fig, axes = plt.subplots(2, 5, figsize=(10,10))
            axes = axes.flatten()
            for img, ax in zip(images_arr, axes):
                ax.imshow(img)
                ax.axis('off')
            plt.tight_layout()
            fig.suptitle('Examples of train data with data augmentation')
            plt.show()
        def check():
            training_images, training_labels = next(self.train_generator)
            #print(training_labels[:5])
            plotImages(training_images[:10])
        # チェックするとき呼ぶ
        check()
        pass

    def makecnnmodel(self):
        """モデルの作成
        """
        print('\n====================\n\nmakecnnmodel\n\n====================\n')
        self.model = cifar10_cnn(self.cnf)

    def training(self):
        """訓練
        """
        print('\n====================\n\ntraining\n\n====================\n')

        # delete tensorboard logdir
        log_dir = self.cnf.wd / f'tensorboard_{self.cnf.model_name}'
        try:
            shutil.rmtree(str(log_dir))
        except:
            pass

        # callbacks
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
            log_dir=str(self.cnf.wd / f'tensorboard_{self.cnf.model_name}'),
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        check_point = ModelCheckpoint(
            str(self.cnf.wd / f'best_{self.cnf.model_savefile}'),
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=False
        )

        # 学習
        #self.history = self.model.fit(
        #    self.x_train,
        #    self.y_train,
        #    batch_size=self.cnf.batchsize,
        #    epochs=self.cnf.epochs,
        #    validation_split=self.cnf.validation_rate,
        #    verbose=1,
        #    callbacks=[early_stopping, reduce_lr, tensor_board, check_point])
        #self.model.save(self.cnf.wd / f'last_{self.cnf.model_savefile}')
        #self.model = tf.keras.models.load_model(self.cnf.wd / f'best_{self.cnf.model_savefile}'
        #)

        # 学習
        self.history = self.model.fit(
            self.train_generator,
            steps_per_epoch=math.ceil(len(self.y_train)*0.8 / self.cnf.batchsize),
            validation_data=self.validation_generator,
            validation_steps=math.ceil(len(self.y_train)*0.2 / self.cnf.batchsize),
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
        #test_loss, test_acc = self.model.evaluate(self.x_test, to_categorical(self.y_test), verbose=1, batch_size=self.cnf.batchsize)
        test_loss, test_acc = self.model.evaluate(
            self.test_generator,
            steps=math.ceil(len(self.y_test) / self.cnf.batchsize),
            verbose=1)
        print('Test loss:', test_loss)
        print('Test accuracy:', test_acc)

    def prediction(self):
        """全テストデータで予測
        """
        print('\n====================\n\nprediction\n\n====================\n')
        #predictions = self.model.predict(self.x_test, verbose=1, batch_size=self.cnf.batchsize)
        predictions = self.model.predict(
            self.test_generator,
            steps=math.ceil(len(self.y_test) / self.cnf.batchsize),
            verbose=1)
        predictions = [np.argmax(pred) for pred in predictions]
        answers = self.y_test
        print('Test result:')
        print(tf.math.confusion_matrix(answers, predictions).numpy())

    def deeplearning(self):
        """ディープラーニングを実行
        """
        print('\n********************\n\ndeeplearning\n\n********************\n')
        assert self.cnf.train_test_rate != 1, 'No train data exists. It is no use you having NN learn.'
        self.loaddataset()
        self.makecnnmodel()
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
    sign = Sifar10Classifier()

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
