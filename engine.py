class TextPredictor:

    def __init__(self):
        self.model = None
        self.dataset = None

    def set_dataset(self, dataset):
        self.dataset = dataset
        