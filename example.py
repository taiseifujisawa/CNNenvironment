from deep_learning_cnn_class import DeepLearningCnnClassifier, GradCam, set_seed
from cnn import cifar10_cnn
from cnnconfig import Cifar10Config

from keras.datasets import cifar10


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
