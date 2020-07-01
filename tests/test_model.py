import os
import unittest
import shutil
import ax
from sparsemod.model import Trainer
from load_data import prepare_cifar10
from cnn import Net, search_space

class TrainingTestCase(unittest.TestCase):
    """ Test cases to see if we can build a model,
    train it, evaluate, quantize it and prune it"""

    def setUp(self):
        dataset, classes = prepare_cifar10()
        self.models_path = "./models"
        self.trainer = Trainer(pruning=True, datasets=dataset, models_path=self.models_path)
        experiment = ax.Experiment(
                        name="model_test",
                        search_space=search_space(),
                        )
        sobol = ax.Models.SOBOL(search_space=experiment.search_space)
        self.param = sobol.gen(1).arms[0].parameters
        self.net = Net(self.param, classes=classes, input_shape=(3, 32, 32))

        if not os.path.exists(self.models_path):
            os.mkdir(self.models_path)

    def tearDown(self):
        shutil.rmtree(self.models_path)
        shutil.rmtree("./data")
    
    def test_training(self):
        self.trainer.load_dataloaders(batch_size=4, collate_fn=None)
        self.trainer.train(self.net, self.param, name="0", epochs=1, reload=False, old_net=None)
        self.trainer.evaluate(self.net.to('cpu'), quant_mode=False)

