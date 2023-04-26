import argparse
import os
import torch
import numpy as np

from tqdm import tqdm
from user_src import LaserTransformerModel


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--lang', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    return parser


def main():

    parser = get_parser()
    args = parser.parse_args()

    CKPT = './{}'.format(args.checkpoint)
    input_folder = './bucc2018/{}-en/'.format(args.task)
    output_folder = './bucc2018/{}-en/'.format(args.task)


    model = LaserTransformerModel.from_pretrained(
        '.', checkpoint_file=CKPT, data_name_or_path='./data-bin/europarl.tokenized')
    sentence_encoder = model.models[0].encoder
    sentence_encoder.eval()
    sentence_encoder.cuda()
    dictionary = model.src_dict

    file = '{}-en.training.spm.{}'.format(args.task, args.lang)
    print('Preprocessing {}'.format(file))
    results = []
    lines = open(input_folder + file, 'r').readlines()
    for i in tqdm(range(len(lines)), desc='Embedding...'):
        id = dictionary.encode_line(lines[i].rstrip('\n'), add_if_not_exist=False).cuda()
        length = torch.tensor([id.size(0)]).cuda()
        sentence_embedding = sentence_encoder(id.unsqueeze(0), length, '')['sentemb'][0].tolist()[0]
        results.append(sentence_embedding)

    np.save(output_folder + file.replace('spm', 'emb'), np.array(results, dtype=np.float32))


if __name__ == '__main__':
    main()
