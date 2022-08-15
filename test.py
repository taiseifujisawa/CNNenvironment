from deep_learning_cnn_class import DeepLearningCnnClassifier, GradCam, set_seed
from cnn import sign_classifier_cnn, cifar10_cnn, mnist_cnn
from cnnconfig import SignConfig, Cifar10Config, MnistConfig

from keras.datasets import cifar10, mnist, fashion_mnist


def sign_classifier():
  cnf = SignConfig()
  set_seed(cnf.random_seed)
  dl = DeepLearningCnnClassifier(cnf)
  model_gen = sign_classifier_cnn
  dl.loaddataset()
  dl.makecnnmodel(model_gen)
  #dl.makecnnmodel(model_gen, False)
  #dl.makecnnmodel()
  dl.training()
  dl.drawlossgraph()
  #dl.testevaluate()
  dl.prediction()

  cam = GradCam(dl)
  cam.gradcam_batch()

def mnist_classifier():
  cnf = MnistConfig()
  set_seed(cnf.random_seed)
  dl = DeepLearningCnnClassifier(cnf)
  model_gen = mnist_cnn
  dataset = mnist.load_data()
  #dataset = fashion_mnist.load_data()
  dl.loaddataset(dataset)
  dl.makecnnmodel(model_gen)
  #dl.makecnnmodel(model_gen, False)
  dl.training()
  dl.drawlossgraph()
  #dl.testevaluate()
  dl.prediction()

  cam = GradCam(dl)
  cam.gradcam_batch()

def cifar10_classifier():
  cnf = Cifar10Config()
  set_seed(cnf.random_seed)
  dl = DeepLearningCnnClassifier(cnf)
  model_gen = cifar10_cnn
  dataset = cifar10.load_data()
  dl.loaddataset(dataset)
  dl.makecnnmodel(model_gen)
  #dl.makecnnmodel(model_gen, False)
  dl.training()
  dl.drawlossgraph()
  #dl.testevaluate()
  dl.prediction()

  cam = GradCam(dl)
  cam.gradcam_batch()

def cifar10_classifier_vgg16transfer():
  cnf = Cifar10Config()
  set_seed(cnf.random_seed)
  dl = DeepLearningCnnClassifier(cnf)
  from cnn import cifar10_cnn_vgg16transfer
  model_gen = cifar10_cnn_vgg16transfer
  dataset = cifar10.load_data()
  dl.loaddataset(dataset)
  dl.makecnnmodel(model_gen)
  #dl.makecnnmodel(model_gen, False)
  dl.training()
  dl.drawlossgraph()
  dl.testevaluate()
  dl.prediction()

  cam = GradCam(dl)
  cam.gradcam_batch()


if __name__ == '__main__':
  #sign_classifier()
  #mnist_classifier()
  #cifar10_classifier()
  cifar10_classifier_vgg16transfer()
