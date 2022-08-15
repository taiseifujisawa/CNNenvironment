import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import traceback
import shutil
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


class DeepLearningCnnClassifier:
    def __init__(self, cnf):
        """cnnの諸元の設定

        Args:
            cnf (configclass): configuration
        """
        self.cnf = cnf
        self.skip_training = False
        cnf.wd.mkdir(exist_ok=True)

    def loaddataset(self, dataset=None, whether_da=False):
        """データの読み込み、成形、data augmentation

        Args:
            dataset (np.ndarray, optional): ((x_train, y_train), (x_test, y_test))型のデータセット. Defaults to None.

        Raises:
            NameError: _description_
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
            if x_train.ndim == 3:
                x_train = np.expand_dims(x_train, axis=-1)
            if x_test.ndim == 3:
                x_test = np.expand_dims(x_test, axis=-1)
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
            fig, axes = plt.subplots(2, 5, figsize=(10, 5))
            axes = axes.flatten()
            for img, ax in zip( images_arr, axes):
                ax.imshow(img)
                ax.axis('off')
            plt.tight_layout()
            fig.suptitle('Examples of train data with data augmentation')
            plt.pause(5)
            plt.clf()
            plt.close()
        def check():
            training_images, training_labels = next(self.train_generator)
            print(training_labels[:10])
            plotImages(training_images[:10])
        if whether_da:
            # チェックするとき呼ぶ
            check()
        else:
            pass

    def makecnnmodel(self, cnn_func=None, whether_train=True):
        """モデルの作成

        Args:
            cnn_func (function, optional): CNNモデルを生成する関数(カッコなし). Defaults to None.
            whether_train (bool, optional): trainするかどうか. Defaults to True.

        Raises:
            e: 訓練しない場合、既存のモデルを読み込む。既存のモデルがない場合raise
        """
        print('\n====================\n\nmakecnnmodel\n\n====================\n')

        if cnn_func != None and whether_train:
            self.model = cnn_func(self.cnf)
        else:
            try:
                self.model = tf.keras.models.load_model(self.cnf.wd / f'best_{self.cnf.model_savefile}')
                self.skip_training = True
            except OSError as e:
                print("No model exists")
                raise e

        self.model.summary()

    def training(self):
        """訓練
        """
        if self.skip_training:  # 既存のモデルを読み込む場合
            pass
        else:
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
        if self.skip_training:  # 既存のモデルを読み込む場合
            pass
        else:
            print('\n====================\n\ndrawlossgraph\n\n====================\n')
            loss     = self.history.history['loss']
            acc = self.history.history['accuracy']
            nb_epoch = len(loss)
            fig = plt.figure()
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.plot(range(nb_epoch), loss,     marker='.', label='train_loss', color='#984ea3')
            ax2.plot(range(nb_epoch), acc,     marker='.', label='train_acc.', color='#377eb8')
            try:
                val_loss = self.history.history['val_loss']
                val_acc = self.history.history['val_accuracy']
            except KeyError:
                print('The exception below was ignored')
                print(traceback.format_exc())
                print('No validation data exists')
                with open(self.cnf.wd / 'trainlog.csv', 'w') as f:
                    f.write('epoch,loss,acc\n')
                    for ep, (ls, ac) in enumerate(zip(loss, self.history.history['accuracy'])):
                        f.write(f'{ep},{ls},{ac}\n')
            else:
                ax1.plot(range(nb_epoch), val_loss, marker='.', label='val_loss', color='#ff7f00')
                ax2.plot(range(nb_epoch), val_acc, marker='.', label='val_acc.', color='r')
                with open(self.cnf.wd / 'trainlog.csv', 'w') as f:
                    f.write('epoch,loss,acc,val_loss,val_acc\n')
                    for ep, (ls, ac, vls, vac) in enumerate(zip(loss, self.history.history['accuracy'],
                        val_loss, self.history.history['val_accuracy'])):
                        f.write(f'{ep},{ls},{ac},{vls},{vac}\n')
            finally:
                h1, l1 = ax1.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax1.legend(h1+h2, l1+l2, loc='center right', fontsize=10)
                plt.grid()
                plt.xlabel('epoch')
                ax1.set_ylabel('loss')
                ax2.set_ylabel('accuracy')
                ax2.set_ylim(0.0, 1.0)
                print("the graph of losses has been stored as \"Loss.png\"")
                fig.savefig(self.cnf.wd / "Loss.png")

    def testevaluate(self):
        """テストデータによる評価
        """

        if self.cnf.train_test_rate == 0:
            pass
        else:
            print('\n====================\n\ntestevaluate\n\n====================\n')
            test_loss, test_acc = self.model.evaluate(self.test_generator, verbose=1, batch_size=self.cnf.batchsize)
            print('Test loss:', test_loss)
            print('Test accuracy:', test_acc)

    def prediction(self):
        """全テストデータで予測
        """
        if self.cnf.train_test_rate == 0:
            pass
        else:
            print('\n====================\n\nprediction\n\n====================\n')
            predictions = self.model.predict(self.test_generator, verbose=1, batch_size=self.cnf.batchsize)
            predictions = [np.argmax(pred) for pred in predictions]
            answers = self.test_generator.classes if self.cnf.load_mode == 'directory'\
                                                else np.where(self.test_generator.y == 1)[1]
            print('Test result:')
            print(tf.math.confusion_matrix(answers, predictions).numpy())
