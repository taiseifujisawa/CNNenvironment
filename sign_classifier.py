import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import math
import random
import cv2
from PIL import Image
import pickle
from tqdm import tqdm
import traceback
import shutil
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.image_utils import array_to_img, img_to_array, load_img, save_img
from grad_cam import GradCam
from cnn import sign_classifier_cnn
from cnnconfig import CnnConfig


class SignClassifier:
    def __init__(self):
        """cnnの諸元の設定
        """
        self.cnf = CnnConfig()

    def loaddataset(self):
        """データの読み込み、成形、data augmentation
        """
        print('\n====================\n\nloaddataset\n\n====================\n')

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

        # 全画像を[0, 1]へ正規化(data augmentationを行える)
        idg = ImageDataGenerator(rescale=1.0 / 255)

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
        self.test_generator = idg.flow_from_directory(
            self.cnf.splitted_datasetdir / 'test',
            target_size=self.cnf.input_shape,
            color_mode=self.cnf.color,
            shuffle=False,
            class_mode=class_mode[self.cnf.lossfunc],
            batch_size=self.cnf.batchsize
            )

        # ディレクトリ名とラベルの対応を表示
        print(
            'labels of train data: {}\nlabels of validation data: {}\nlabels of test data: {}'
            .format(
                self.train_generator.class_indices,
                self.validation_generator.class_indices,
                self.test_generator.class_indices
                )
            )

        # data augmentationチェック用
        def plotImages(images_arr):
            fig, axes = plt.subplots(1, 5, figsize=(10,10))
            axes = axes.flatten()
            for img, ax in zip( images_arr, axes):
                ax.imshow(img)
                ax.axis('off')
            plt.tight_layout()
            plt.show()
        def check():
            training_images, training_labels = next(self.train_generator)
            print(training_labels[:5])
            plotImages(training_images[:5])
        # チェックするとき呼ぶ
        #check()
        pass

    def makecnnmodel(self):
        """モデルの作成
        """
        print('\n====================\n\nmakecnnmodel\n\n====================\n')
        self.model = sign_classifier_cnn(self.cnf)

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

        # 学習
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
        answers = self.test_generator.classes
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
    sign = SignClassifier()

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
