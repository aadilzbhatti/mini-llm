class TextCompleter:

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def get_text_completions(self, max_tokens, context=""):
        self.model.eval()
        context = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
        out = self.model.generate(context, max_tokens)
        decoded_text = self.tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
        return decoded_text