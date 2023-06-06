from typing import List, Optional, Union

import os

import gdown
import numpy as np
import sentencepiece as spm

from .encoder import SentenceEncoder


__all__ = ['MuSR']


class MuSR:
    r"""
    End-to-end MuSR embedding.

    Args:
        spm_model (str, optional): the path to MuSR's sentencepiece model (``code.model``).
        vocab_file (str, optional): the path to MuSR's vocabulary file (``dict.txt``).
        model_path (str, optional): the path to MuSR's checkpoint file (``checkpoint_768.pt``).
        max_sentences (int, optional): maximum number of sentences in one batch.
        max_tokens (int, optional): maximum number of tokens in one batch.
        stable (bool, optional): if True, mergesort sorting algorithm will be used, otherwise quicksort will be used.
        cpu (bool, optional): if True, forces the use of the CPU even a GPU is available.
    """

    DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    DEFAULT_SPM_MODEL_FILE = os.path.join(DATA_DIR, 'code.model')
    DEFAULT_VOCAB_FILE = os.path.join(DATA_DIR, 'dict.txt')
    DEFAULT_MODEL_FILE = os.path.join(DATA_DIR, 'checkpoint_768.pt')

    SPM_MODEL_PATH = 'https://drive.google.com/file/d/1sTbYsdzQXBOpiN4DGwDnJzmdwqNRRtuy/view?usp=sharing'
    VOCAB_PATH = 'https://drive.google.com/file/d/1ZVngDxEPSag1BnnzSMMSrLsYU1lf1C3R/view?usp=sharing'
    MODEL_PATH = 'https://drive.google.com/file/d/1ICcyN2Q23ZUBJJ0J79wqkEZcZMOmaBkr/view?usp=sharing'

    def __init__(self,
                 spm_model: Optional[str] = None,
                 vocab_file: Optional[str] = None,
                 model_path: Optional[str] = None,
                 max_sentences: Optional[int] = None,
                 max_tokens: Optional[int] = None,
                 stable: bool = False,
                 cpu: bool = False):

        if spm_model is None:
            if not os.path.isfile(self.DEFAULT_SPM_MODEL_FILE):
                print('The sentencepiece model is missing!')
                gdown.download(self.SPM_MODEL_PATH, self.DEFAULT_SPM_MODEL_FILE, quiet=False, fuzzy=True)
            spm_model = self.DEFAULT_SPM_MODEL_FILE
        if vocab_file is None:
            if not os.path.isfile(self.DEFAULT_VOCAB_FILE):
                print('The vocabulary file is missing!')
                gdown.download(self.VOCAB_PATH, self.DEFAULT_VOCAB_FILE, quiet=False, fuzzy=True)
            vocab_file = self.DEFAULT_VOCAB_FILE
        if model_path is None:
            if not os.path.isfile(self.DEFAULT_MODEL_FILE):
                print('The model checkpoint is missing!')
                gdown.download(self.MODEL_PATH, self.DEFAULT_MODEL_FILE, quiet=False, fuzzy=True)
            model_path = self.DEFAULT_MODEL_FILE

        self.spm_model = spm.SentencePieceProcessor(model_file=spm_model)
        self.encoder = SentenceEncoder(
            vocab_file=vocab_file,
            model_path=model_path,
            max_sentences=max_sentences,
            max_tokens=max_tokens,
            sort_kind='mergesort' if stable else 'quicksort',
            cpu=cpu)

    def embed_sentences(self,
                        sentences: Union[List[str], str]) -> np.ndarray:
        r"""
        Computes the MuSR embeddings of the sentences using sentencepiece tokenizer.

        Args:
            sentences (str or List[str]): the sentences to compute the embeddings from.

        Returns:
            np.ndarray: A NumPy array containing the embeddings.
        """

        sentences = [sentences] if isinstance(sentences, str) else sentences
        spm_encoded = [
            ' '.join(self.spm_model.encode_as_pieces(sentence)) for sentence in sentences
        ]

        return self.encoder.encode_sentences(spm_encoded)
