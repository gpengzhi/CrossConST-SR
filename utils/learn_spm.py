import argparse
import sentencepiece as spm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--model-prefix', required=True)
    parser.add_argument('--vocab-size', type=int, required=True)
    return parser


def main():

    parser = get_parser()
    args = parser.parse_args()

    spm.SentencePieceTrainer.train(
        input=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type='bpe')


if __name__ == '__main__':
    main()
