#not used, implemented for testing purposes

import torch
from transformers import DataCollatorForLanguageModeling

class RareClassMLMDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, rare_mlm_probabilities=[0.65], rare_token_ids=[4], **kwargs):
        super().__init__(**kwargs)
        
        self.rare_probs = torch.tensor(rare_mlm_probabilities)
        self.rare_tokens = torch.tensor(rare_token_ids)

    def torch_mask_tokens(self, inputs, special_tokens_mask=None, offset_mapping=None):
        labels = inputs.clone()
        
        if special_tokens_mask is None:
            special_ids = torch.tensor(self.tokenizer.all_special_ids, device=inputs.device)
            special_tokens_mask = torch.isin(inputs, special_ids)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix = torch.full(labels.shape, self.mlm_probability, device=inputs.device)
        
        for rare_token, prob in zip(self.rare_tokens, self.rare_probs):
            mask = inputs == rare_token.to(inputs.device)
            probability_matrix[mask] = prob

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replace = torch.bernoulli(torch.full(labels.shape, 0.8, device=inputs.device)).bool() & masked_indices
        inputs[indices_replace] = self.tokenizer.mask_token_id

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=inputs.device)).bool() & masked_indices & ~indices_replace
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, device=inputs.device)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels