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
from signtest_GradCAM import GradCam


class SignClassifier:
    def __init__(self, wd: str):
        """cnnの諸元の設定
        """
        RANDOM_SEED = 1
        tf.random.set_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        self.wd = Path(wd)
        self.datasetdir = self.wd / 'dataset'
        self.splitted_datasetdir = self.wd / 'dataset_splitted'
        self.model_name = 'my_model'
        self.model_savefile = 'my_model.h5'
        self.last_layername = "last_conv"
        self.train_test_rate = 0.2          # self.loadfrompickle = Falseなら保存済みのpicklefileのtestdataの割合に書き変わる
        self.validation_rate = 0.2          # self.loadfrompickleによらず一定
        self.input_shape = (250, 1000)      # numpy, (row, column)
        self.outputs = 2
        self.optimizer = Adam(learning_rate=0.001)
        self.lossfunc = 'sparse_categorical_crossentropy'
        self.epochs = 100
        self.batchsize = 16

    def loaddataset(self):
        """データの読み込み、成形、data augmentation
        """
        print('\n====================\n\nloaddataset\n\n====================\n')

        shutil.rmtree(self.splitted_datasetdir)
        for d in self.datasetdir.iterdir():
            imgs_shuffled = random.sample(list(d.iterdir()), len(list(d.iterdir())))
            imgs_test = imgs_shuffled[:int(len(imgs_shuffled) * self.train_test_rate)]
            imgs_rest = imgs_shuffled[int(len(imgs_shuffled) * self.train_test_rate):]
            imgs_validation = imgs_rest[:int(len(imgs_rest) * self.train_test_rate)]
            imgs_train = imgs_rest[int(len(imgs_rest) * self.train_test_rate):]

            (self.splitted_datasetdir /'train' / d.name).mkdir(parents=True, exist_ok=True)
            (self.splitted_datasetdir / 'validation' / d.name).mkdir(parents=True, exist_ok=True)
            (self.splitted_datasetdir / 'test' / d.name).mkdir(parents=True, exist_ok=True)
            for i in imgs_train:
                shutil.copy(i, self.splitted_datasetdir / 'train' / d.name)
            for i in imgs_validation:
                shutil.copy(i, self.splitted_datasetdir / 'validation' / d.name)
            for i in imgs_test:
                shutil.copy(i, self.splitted_datasetdir / 'test' / d.name)

        idg = ImageDataGenerator(rescale=1.0 / 255)
        print('train data: ', end='')
        self.train_generator = idg.flow_from_directory(
            self.splitted_datasetdir / 'train',
            target_size=self.input_shape,
            color_mode='grayscale',
            shuffle=True,
            class_mode="sparse",
            batch_size=self.batchsize
            )
        print('validation data: ', end='')
        self.validation_generator = idg.flow_from_directory(
            self.splitted_datasetdir / 'validation',
            target_size=self.input_shape,
            color_mode='grayscale',
            shuffle=True,
            class_mode="sparse",
            batch_size=self.batchsize
            )
        print('test data: ', end='')
        self.test_generator = idg.flow_from_directory(
            self.splitted_datasetdir / 'test',
            target_size=self.input_shape,
            color_mode='grayscale',
            shuffle=False,
            class_mode="sparse",
            batch_size=self.batchsize
            )

        print(
            'labels of train data: {}\nlabels of validation data: {}\nlabels of test data: {}'.
            format(self.train_generator.class_indices,self.validation_generator.class_indices,self.test_generator.class_indices)
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
        #check()
        pass

    def makecnnmodel(self):
        """モデルの作成(Sequential API)
        """
        print('\n====================\n\nmakecnnmodel\n\n====================\n')
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=self.input_shape+(1,)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', name=self.last_layername),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            #tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(16, activation='relu'),
            #tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.outputs, activation='softmax')
        ], name=self.model_name)
        self.model.summary()
        self.model.compile(optimizer=self.optimizer, loss=self.lossfunc, metrics=['accuracy'])

    def training(self):
        """訓練
        """
        print('\n====================\n\ntraining\n\n====================\n')

        # delete tensorboard logdir
        log_dir = self.wd / f'tensorboard_{self.model_name}'
        try:
            shutil.rmtree(str(log_dir))
        except:
            pass

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
            log_dir=self.wd / f'tensorboard_{self.model_name}',
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        check_point = ModelCheckpoint(
            self.wd / f'best_{self.model_savefile}',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=False
        )
        self.history = self.model.fit(
            self.train_generator,
            steps_per_epoch=math.ceil(self.train_generator.samples / self.batchsize),
            validation_data=self.validation_generator,
            validation_steps=math.ceil(self.validation_generator.samples / self.batchsize),
            epochs=self.epochs,
            verbose=1,
            callbacks=[early_stopping, reduce_lr, tensor_board, check_point])
        self.model.save(self.wd / f'last_{self.model_savefile}')
        self.model = tf.keras.models.load_model(self.wd / f'best_{self.model_savefile}'
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
            with open(self.wd / 'trainlog.csv', 'w') as f:
                f.write('epoch,loss,acc\n')
                for ep, (ls, ac) in enumerate(zip(loss, self.history.history['accuracy'])):
                    f.write(f'{ep},{ls},{ac}\n')
        else:
            plt.plot(range(nb_epoch), val_loss, marker='.', label='val_loss')
            with open(self.wd / 'trainlog.csv', 'w') as f:
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
            fig.savefig(self.wd / "Loss.png")

    def testevaluate(self):
        """テストデータによる評価
        """
        print('\n====================\n\ntestevaluate\n\n====================\n')
        test_loss, test_acc = self.model.evaluate(self.test_generator, verbose=1, batch_size=self.batchsize)
        print('Test loss:', test_loss)
        print('Test accuracy:', test_acc)

    def prediction(self):
        """全テストデータで予測、予測結果と間違えたテストデータのインデックスpickle化して保存
        """
        print('\n====================\n\nprediction\n\n====================\n')
        predictions = self.model.predict(self.test_generator, verbose=1, batch_size=self.batchsize)
        predictions = [np.argmax(pred) for pred in predictions]
        answers = self.test_generator.classes
        index_failure = [i for i, (p, t) in enumerate(zip(predictions, answers)) if p != t]
        serialize_write((predictions, index_failure), self.wd / 'predictions.pkl')
        self.predict = predictions
        self.index_failure = index_failure
        print('Test result:')
        print(tf.math.confusion_matrix(answers, predictions).numpy())

    @classmethod
    def deeplearning(cls, wd=Path.cwd()):
        """ディープラーニングを実行

        Returns:
            sign: SignClassifierオブジェクト
        """
        print('\n********************\n\ndeeplearning\n\n********************\n')
        #sign = cls(r'C:\Users\nktlab\Desktop\Experiment\fujisawa\03_sign_classifier_individual_20220711\dataset_individual\0')
        sign = cls(wd)
        assert sign.train_test_rate != 1, 'No train data exists. It is no use you having NN learn.'
        sign.loaddataset()
        sign.makecnnmodel()
        sign.training()
        sign.drawlossgraph()
        if sign.train_test_rate == 0:
            print('No test data exists')
        else:
            sign.testevaluate()
            sign.prediction()
        return sign

    @classmethod
    def reconstructmodel(cls, wd=Path.cwd()):
        """保存済みモデルの再構築

        Raises:
            e: 指定した名前の保存済みモデルがない場合

        Returns:
            sign: SignClassifierオブジェクト
        """
        print('\n********************\n\nreconstructmodel\n\n********************\n')
        sign = cls(wd)
        sign.loaddataset()
        try:
            sign.model = tf.keras.models.load_model(sign.wd / f'best_{sign.model_savefile}')
        except OSError as e:
            print("No model exists")
            raise e
        else:
            sign.model.summary()
            if sign.train_test_rate == 0:
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
            output_filepath = self.wd / save_path / output_filename
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


def run(subj: int):
    # ディープラーニング実行
    sign = SignClassifier.deeplearning(f'{str(Path.cwd())}')

    # 保存済みモデル再構築
    #sign = SignClassifier.reconstructmodel(f'{str(Path.cwd())}/dataset_individual/{subj}')

    # gradcam起動
    cam = GradCam(sign)

    if sign.train_test_rate == 0:
        pass
    else:
        # failureとクラスごとのディレクトリ作成
        result_dir = sign.wd / 'test_results'
        try:
            shutil.rmtree(str(result_dir))
        except:
            pass
        result_dir.mkdir(exist_ok=True)
        (result_dir / 'failure').mkdir(exist_ok=True)
        for i in range(2):
            (result_dir / f'{i}').mkdir(exist_ok=True)

        # 間違えたテストデータをpng, csvに出力
        with open(sign.wd / 'failure.csv', 'w', encoding='utf-8') as f:
            f.write('index,prediction,answer\n')
            for i in sign.index_failure:
                sign.array2img(i, result_dir / f'failure/i{i}_p{sign.predict[i]}_a{sign.y_test[i]}')
                f.write(f'{i},{sign.predict[i]},{sign.y_test[i]}\n')
                # gradcam
                img = cam.get_cam(i)
                cam.save_img(img, i, result_dir / f'failure/i{i}_p{sign.predict[i]}_a{sign.y_test[i]}_cam')

        # 全テストデータをpngに出力
        for i, y_test in enumerate(tqdm(sign.y_test)):
            #Path(result_dir / f'{y_test}/{i}').mkdir(exist_ok=True)
            sign.array2img(i, result_dir / f'{y_test}/i{i}_p{sign.predict[i]}_a{sign.y_test[i]}')
            # gradcam
            img = cam.get_cam(i)
            cam.save_img(img, i, result_dir / f'{y_test}/i{i}_p{sign.predict[i]}_a{sign.y_test[i]}_cam')

def main():
    for subj in tqdm(range(len([i for i in (Path.cwd() / 'dataset_individual').iterdir() if i.is_dir() and i.name != 'all']))):
        run(subj)



if __name__ == '__main__':
    #main()
    #l = [1,2,4,6,7,8,9,10,11,13,14]
    #for i in tqdm(l):
    #    run(i)
    run(0)
