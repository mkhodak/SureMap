import json
import pdb
import pickle
import shutil
from operator import itemgetter
from urllib import request
import numpy as np
import pyarrow as pa
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import login
from mpi4py import MPI
from tqdm import tqdm
from transformers import pipeline
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers.pipelines.base import KeyDataset


CV_DATA = 'mozilla-foundation/common_voice_13_0'
CV_INFO = 'https://raw.githubusercontent.com/common-voice/cv-dataset/main/datasets/cv-corpus-13.0-2023-03-09.json'
DEVICES = [0, 1] # assumes 48GB memory
MODELS = {'whisper': 'openai/whisper-tiny',
          'wav2vec': 'facebook/wav2vec2-base-100h'}
WAV2VEC = False # if True also runs wav2vec
ENGLISH = True # if False runs on other languages
MINROWS = 10000
SUBSETS = ['test', 'validation', 'train']
HFLOGIN = None # your login here


def main():

    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size
    if not rank:
        login(HFLOGIN)
    with request.urlopen(CV_INFO) as f:
        info = json.load(f)
    if ENGLISH:
        langs = ['en']
    else:
        langs = [lang for lang, val in info['locales'].items() if val['buckets']['test'] >= MINROWS]
        langs[0], langs[1] = langs[1], langs[0]
    normalizer = BasicTextNormalizer()

    for model in (['wav2vec'] if WAV2VEC else []) + ['whisper']:

        pipe = pipeline('automatic-speech-recognition',
                        model=MODELS[model],
                        torch_dtype=torch.float16,
                        device=f'cuda:{rank%len(DEVICES)}')
        if model == 'whisper':
            forced_decoder_ids = pipe.model.config.forced_decoder_ids

        for lang in langs[langs.index('pt'):] if model == 'whisper' else ['en']:

            if model == 'whisper':
                try:
                    pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language=lang.split('-')[0],
                                                                                                 task='transcribe')
                except ValueError:
                    pipe.model.config.forced_decoder_ids = forced_decoder_ids
            if rank:
                cv = comm.bcast(None)
            else:
                cache = f'./data/{lang}'
                cv = load_dataset(CV_DATA, lang, cache_dir=cache)
                comm.bcast(cv)

            for subset in SUBSETS:

                num_rows = cv[subset].num_rows
                arange = np.arange(num_rows)
                dataset = Dataset(cv[subset].data.filter(pa.compute.and_(arange >= (rank*num_rows)//size,
                                                                         arange < ((rank+1)*num_rows)//size)),
                                  cv[subset].info, 
                                  cv[subset].split)
                output = {key+'s': dataset[key] 
                          for key in ['client_id', 'sentence', 'age', 'gender', 'accent']}

                predictions = []
                prog = lambda it: it if rank else tqdm(it, total=dataset.num_rows)
                for prediction in prog(pipe(KeyDataset(dataset, 'audio'),
                                            max_new_tokens=128,
                                            generate_kwargs={'task': 'transcribe'},
                                            batch_size=32 if model == 'whisper' else 1)):
                    predictions.append(prediction['text'])
                output['predictions'] = predictions

                if lang == 'en':
                    for key in ['sentences', 'predictions']:
                        output['normalized_'+key] = [normalizer(s) for s in output[key]]
                if rank:
                    for _, value in sorted(output.items(), key=itemgetter(0)):
                        comm.gather(value)
                else:
                    for key, value in sorted(output.items(), key=itemgetter(0)):
                        output[key] = sum(comm.gather(value), [])
                    with open(f'./data/common_voice/{model}-{lang}-{subset}.pkl', 'wb') as f:
                        pickle.dump(output, f)

            if lang != 'en' and not rank:
                shutil.rmtree(cache)


if __name__ == '__main__':

    main()
