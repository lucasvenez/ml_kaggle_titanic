from titanic.service import TitanicDatasetLoader

import unittest


class LoadTrainDatasetTest(unittest.TestCase):

    def setUp(self):
        pass

    def given_a_valid_loader_l1(self):
        self.l1: TitanicDatasetLoader = TitanicDatasetLoader()

    def when_train_dataset_d1_is_read_using_l1(self):
        self.d1 = self.l1.read_train_dataset()

    def then_d1_is_not_none(self):
        self.assertIsNotNone(self.d1)

    def test(self):
        self.given_a_valid_loader_l1()
        self.when_train_dataset_d1_is_read_using_l1()
        self.then_d1_is_not_none()
