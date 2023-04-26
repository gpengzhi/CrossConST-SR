#!/usr/bin/env bash


prep=europarl.tokenized
tmp=$prep/tmp
org=org

mkdir -p $org $prep $tmp


urlpref=http://opus.nlpl.eu/download.php?f=Europarl/v8/moses
for f in de-en.txt.zip en-fr.txt.zip; do
  if [ ! -f $org/$f ] ; then
    wget $urlpref/$f -O $org/$f
    rm $org/{README,LICENSE}
    unzip $org/$f -d $org
  fi
done


cat $org/Europarl.de-en.de $org/Europarl.en-fr.fr > $tmp/train.src
cat $org/Europarl.de-en.en $org/Europarl.en-fr.en > $tmp/train.tgt


TRAIN=$tmp/train.all
BPE_CODE=$prep/code
BPE_TOKENS=32000
rm -f $TRAIN
cat $tmp/train.* > $TRAIN

# learn sentencepiece model
python3 utils/learn_spm.py --input $TRAIN --model-prefix $BPE_CODE --vocab-size $BPE_TOKENS

# apply sentencepiece model
python3 utils/spm_encode.py --model $BPE_CODE.model --output_format piece --inputs $tmp/train.src --outputs $tmp/train.tok.src
python3 utils/spm_encode.py --model $BPE_CODE.model --output_format piece --inputs $tmp/train.tgt --outputs $tmp/train.tok.tgt


echo "creating train, valid..."
for l in src tgt; do
    awk '{if (NR%10000 == 0)  print $0; }' $tmp/train.tok.$l > $prep/valid.$l
    awk '{if (NR%10000 != 0)  print $0; }' $tmp/train.tok.$l > $prep/train.$l
done
