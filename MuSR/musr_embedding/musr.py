from typing import List, Optional, Union

import os

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

    def __init__(self,
                 spm_model: Optional[str] = None,
                 vocab_file: Optional[str] = None,
                 model_path: Optional[str] = None,
                 max_sentences: Optional[int] = None,
                 max_tokens: Optional[int] = None,
                 stable: bool = False,
                 cpu: bool = False):

        if spm_model is None:
            pass
        if vocab_file is None:
            pass
        if model_path is None:
            pass

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
