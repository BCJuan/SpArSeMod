import os
import unittest
from sparsemod.model import Trainer
from load_data import prepare_cifar10
from cnn import Net, search_space

class SparseModModelTestCase(unittest.TestCase):
    """ Test cases to see if we can build a model,
    train it, evaluate, quantize it and prune it"""

    def setUp(self):
        dataset, classes = prepare_cifar10()
        self.trainer = Trainer(pruning=True, datasets=dataset, models_path="./models")
        self.net = Net(self.param, classes=classes, input_shape=(3, 32, 32))
        os.mkdir("./models")

    def tearDown(self):
        os.rmdir("./models")
    
    def test_training(self):
        self.trainer.load_dataloaders(batch_size=4, collate_fn=None)
        self.trainer.train(self.net, self.param, name="0", epochs=1, reload=False, old_net=None)

