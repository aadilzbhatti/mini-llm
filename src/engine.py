from model import WikiCompleteModel
from data import WikipediaDataset
from trainer import Trainer
from text_completer import TextCompleter

class TextPredictor:

    def __init__(self, device, tokenizer, max_len, vocab_size, n_embd, n_head, n_layer, block_size, dropout, regenerate_dataset=False, num_samples=1000):
        self.model = None

        self.device = device
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.dropout = dropout

        self.dataset = WikipediaDataset(tokenizer, max_len, block_size,  regenerate_dataset, num_samples)

        self.model = WikiCompleteModel(
            self.vocab_size,
            self.n_embd,
            self.n_head,
            self.n_layer,
            self.block_size,
            self.dropout
        ).to(self.device)

        self.trainer = Trainer(self.model, self.dataset, self.tokenizer, self.optimizer, self.checkpoint_dir, self.batch_size, self.max_epochs, self.max_iters, self.eval_interval, self.learning_rate)

        self.text_completer = TextCompleter(self.model, self.tokenizer, self.device)

    def train(self):
        self.trainer.train()

    def get_text_completions(self, context, num_tokens=10):
        return self.text_completer.get_text_completions(context, num_tokens)