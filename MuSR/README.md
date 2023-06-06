# MuSR: A One-for-all Multilingual Sentence Representation Model for 223 Languages

## Supported languages

The MuSR model is trained on the following languages:

Acehnese (Arabic script), Acehnese (Latin script), Afrikaans, Akan, Algerian Arabic, Amharic, Armenian, Assamese, Asturian, Awadhi, Ayacucho Quechua, Balinese, Bambara, Banjar (Arabic script), Banjar (Latin script), Bashkir, Basque, Belarusian, Bemba, Bengali, Berber languages, Bhojpuri, Bosnian, Breton, Buginese, Bulgarian, Burmese, Catalan, Cebuano, Central Atlas Tamazight, Central Aymara, Central Kanuri (Arabic script), Central Kanuri (Latin script), Central Kurdish, Chamorro, Chhattisgarhi, Chinese (Simplified), Chinese (Traditional), Chokwe, Chuvash, Cornish, Crimean Tatar, Croatian, Czech, Danish, Dari, Divehi, Dutch, Dyula, Dzongkha, Eastern Panjabi, Eastern Yiddish, Egyptian Arabic, English, Esperanto, Estonian, Ewe, Faroese, Fijian, Filipino, Finnish, Fon, French, Friulian, Galician, Ganda, Georgian, German, Greek, Guarani, Gujarati, Haitian Creole, Halh Mongolian, Hausa, Hebrew, Hindi, Hungarian, Icelandic, Ido, Igbo, Ilocano / Iloko, Indonesian, Interlingua, Interlingue, Irish, Italian, Japanese, Javanese, Jingpho, Kabiyè, Kabuverdianu, Kabyle, Kamba, Kannada, Kashmiri (Arabic script), Kashmiri (Devanagari script), Kashubian, Kazakh, Khmer, Kikongo, Kikuyu, Kimbundu, Kinyarwanda, Korean, Kyrgyz, Lao, Latgalian, Latin, Ligurian, Limburgish, Lingala, Lingua Franca Nova, Lithuanian, Lojban, Lombard, Low German, Luba-Kasai, Luo, Luxembourgish, Macedonian, Magahi, Maithili, Malayalam, Maltese, Maori, Marathi, Meitei (Bengali script), Mesopotamian Arabic, Minangkabau (Latin script), Mizo, Modern Standard Arabic, Moroccan Arabic, Mossi, Najdi Arabic, Nepali, Nigerian Fulfulde, North Azerbaijani, North Levantine Arabic, Northern Kurdish, Northern Sotho, Northern Uzbek, Norwegian Bokmål, Norwegian Nynorsk, Nuer, Nyanja, Occitan, Odia, Pangasinan, Papiamento, Plateau Malagasy, Polish, Portuguese, Romanian, Rundi, Russian, Samoan, Sango, Sanskrit, Santali, Sardinian, Scottish Gaelic, Serbian, Serbo-Croatian, Shan, Shanghainese, Shona, Sicilian, Silesian, Sindhi, Sinhala, Slovak, Slovenian, Somali, South Azerbaijani, South Levantine Arabic, Southern Pashto, Southern Sotho, Southwestern Dinka, Spanish, Standard Latvian, Standard Malay, Standard Tibetan, Sundanese, Swahili, Swati, Swedish, Tagalog, Tajik, Tamasheq (Latin script), Tamasheq (Tifinagh script), Tamil, Tatar, Ta’izzi-Adeni Arabic, Telugu, Thai, Tigrinya, Tok Pisin, Tosk Albanian, Tsonga, Tswana, Tumbuka, Tunisian Arabic, Turkish, Turkmen, Twi, Ukrainian, Umbundu, Upper Sorbian, Urdu, Uyghur, Venetian, Vietnamese, Walloon, Waray, Welsh, West Central Oromo, Western Frisian, Western Persian, Wolof, Xhosa, Yoruba, Yue Chinese, and Zulu.

## Dependencies

* Python 
* PyTorch
* NumPy
* Fairseq
* Sentencepiece
* Gdown

## Usage

```python
import musr_embedding


model = musr_embedding.MuSR()
# spm model, vocab file, and model checkpoint are downloaded automatically.

embeddings = model.embed_sentences(['The weather is good today.', '今天天气很好。'])
# embeddings is a N*Dim Numpy array, where N = number of sentences, and Dim = sentence embedding dimension.
```

If you download the models into a specific directory:

```python
import musr_embedding


path_to_spm_model = ...
path_to_vocab_file = ...
path_to_checkpoint = ...

model = musr_embedding.MuSR(
    spm_model=path_to_spm_model, 
    vocab_file=path_to_vocab_file, 
    model_path=path_to_checkpoint)
```

## Credits


