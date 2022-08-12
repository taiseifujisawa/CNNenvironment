import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import shutil
import math
from tqdm import tqdm
from keras.utils.image_utils import array_to_img, img_to_array, load_img, save_img


class GradCam:
    def __init__(self, trained_model):
        self.trained_model = trained_model
        self.save_dir = self.trained_model.wd / 'test_result'
        #shutil.rmtree(self.save_dir, ignore_errors=True)
        self.save_dir.mkdir(exist_ok=True)
        (self.save_dir / 'failure').mkdir(exist_ok=True)

    def get_cam(self, img: np.ndarray) -> np.ndarray:
        # 対象画像、次元追加(読み込む画像が1枚なため、次元を増やしておかないとmodel.predictが出来ない)
        img = img
        img_wide = np.expand_dims(img, axis=0)

        # GradCAM用に出力を最終CNNマップ(self.last_conv)とsoftmaxとしたモデルを作成(Functional API)
        grad_model = tf.keras.Model([self.trained_model.model.inputs],\
            [self.trained_model.model.get_layer(self.trained_model.last_layername).output,
            self.trained_model.model.output])

        # tapeにself.last_convの出力からout(prediction結果)までの計算を保存
        with tf.GradientTape() as tape:
            # shape: (layercol, layerrow, layerchannel), (outputs,)
            conv_outputs, predictions = grad_model(img_wide)
            if self.trained_model.lossfunc == 'binary_crossentropy':
                class_idx = 0 if predictions[0] < 0.5 else 1
                loss = predictions[0][0]
            else:
                class_idx = np.argmax(predictions[0])       # shape: (1,)
                loss = predictions[0][class_idx]            # shape: (1,)
        # backpropを取得
        grads = tape.gradient(loss, conv_outputs)       # shape: (layercol, layerrow, layerchannel)
        #if grads.numpy().max() <= 0:
            #print()

        # cast <class 'tensorflow.python.framework.ops.EagerTensor'>
        # to <class 'numpy.ndarray'>
        conv_outputs = conv_outputs.numpy()[0]     # shape: (layercol, layerrow, layerchannel)
        grads = grads.numpy()[0]                   # shape: (layercol, layerrow, layerchannel)
        # values smaller than zero in the two numpy arrays above are all casted to zero
        conv_outputs[conv_outputs < 0] = 0     # shape: (layercol, layerrow, layerchannel)
        grads[grads < 0] = 0                   # shape: (layercol, layerrow, layerchannel)

        # global average pooling
        layer_weights = np.mean(grads, axis=(0, 1))     # shape: (layerchannel,)

        # apply weights
        cam = np.sum(
                        np.array([conv_outputs[:, :, i] * layer_weights[i]
                        for i in range(len(layer_weights))]), axis=0
                    )          # shape: (layercol, layerrow)
        # 1枚の場合こちらでも可
        #cam = np.dot(conv_outputs, layer_weights)          # shape: (layercol, layerrow)

        # reluを通す
        cam_relu = np.maximum(cam, 0)
        # camをリサイズ(OpenCVの指定は横×縦になることに注意)
        cam_resized = cv2.resize(
            cam_relu, (self.trained_model.input_shape[1], self.trained_model.input_shape[0])
            , cv2.INTER_LINEAR
            ) # shape: (layercol, layerrow)

        #if cam_resized.max() == 0:
        #    with open('grad_vanish.csv', 'a') as f:
        #        f.write('gradient vanished\n')
        # make heatmap
        heatmap = cam_resized / cam_resized.max() * 255       # shape: (layercol, layerrow)
        # apply color
        hm_colored = cv2.applyColorMap(np.uint8(heatmap), cv2.COLORMAP_JET)    # shape: (layercol, layerrow)

        # 元の画像をカラー化
        org_img = cv2.cvtColor(np.uint8(img * 255), cv2.COLOR_GRAY2BGR)     # shape: (layercol, layerrow)

        # 合成
        out = cv2.addWeighted(src1=org_img, alpha=0.4, src2=hm_colored, beta=0.6, gamma=0)     # shape: (layercol, layerrow)

        return out, class_idx              # shape: (1,)

    def save_img(self, img: np.ndarray, save_dir: Path, save_name: Path, extension='.png'):
        output_filename = f'{save_name}{extension}'
        output_filepath = save_dir / output_filename
        cv2.imwrite(str(output_filepath), img)

    def batch_singularize(self):
        for i, batch in zip(tqdm(range(math.ceil(self.trained_model.test_generator.samples / self.trained_model.batchsize))), self.trained_model.test_generator):
            for j, img_and_label in enumerate(zip(tqdm(batch[0]), batch[1])):
                img, label = img_and_label[0], img_and_label[1]
                test_no = i * self.trained_model.batchsize + j
                cam, pred = self.get_cam(img)
                self.save_img(img * 255, self.save_dir, f'{test_no}_a{label}_p{pred}')
                self.save_img(cam, self.save_dir, f'{test_no}_a{label}_p{pred}_cam')

                if label != pred:
                    self.save_img(img * 255, self.save_dir / 'failure', f'{test_no}_a{label}_p{pred}')
                    self.save_img(cam, self.save_dir / 'failure', f'{test_no}_a{label}_p{pred}_cam')
