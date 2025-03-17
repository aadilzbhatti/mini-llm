class TextCompleter:

    def __init__(self, model, tokenizer, device, block_size):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = block_size

    def get_text_completions(self, max_tokens, context="\n"):
        self.model.eval()
        context = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
        out = self.model.generate(context, max_tokens, self.block_size)
        decoded_text = self.tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
        return decoded_text