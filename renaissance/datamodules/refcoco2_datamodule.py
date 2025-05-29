from ..datasets.refcoco2_dataset import Refcoco2Dataset
from .datamodule_base import BaseDataModule



class Refcoco2DataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.refcoco2 = True

    @property
    def dataset_cls(self):
        return Refcoco2Dataset

    @property
    def dataset_cls_no_false(self):
        return Refcoco2Dataset

    @property
    def dataset_name(self):
        return "refcoco2"