import logging
import os

from transformers import AutoTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class TokenizerWrapper:
    def __init__(self, tokenizer_path="NbAiLab/nb-bert-base", blank_token="<blank>"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            logger.info("Tokenizer successfully loaded")
        except OSError:
            logger.warning(
                f"Could not load tokenixer using AutoTokenizer. Attempting to load with PreTrainedTokenizerFast..."
            )
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
            logger.info("Tokenizer successfully loaded")
        except Exception as e:
            logger.error(f"Tokenizer could not be loaded")
            raise e

        self.blank_token = blank_token
        self.add_blank_token()

    def add_blank_token(self):
        self.tokenizer.add_tokens(self.blank_token)
        self.blank_token_id = self.tokenizer.convert_tokens_to_ids(self.blank_token)

    def get_vocab_size(self):
        vocab_size = len(self.tokenizer.vocab)
        return vocab_size
