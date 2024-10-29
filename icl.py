import argparse
import json
import os
import pdb
import random
import pandas as pd
from tqdm import tqdm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import LlamaCpp
from mpi4py import MPI
from data import write


DATA_DIR = './data/Adult-Llama' # download from https://archive.ics.uci.edu/dataset/2/adult, unzip, and rename
MODEL_PATH = './ggml-model-Q4_K_M.gguf' # download from https://huggingface.co/JaaackXD/Llama-3-70B-GGUF
HEADER = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
FEATURES = ['workclass', 'hours-per-week', 'sex', 'age', 'occupation', 'capital-loss', 'education', 'capital-gain', 'marital-status', 'relationship']
NAMES = ['id', 'age', 'race', 'sex', 'label', 'pred']
SHOTS = 8
SEED = 0
OUTPUT_DIR = os.path.join(DATA_DIR, f'seed{SEED}')
PRINTERVAL = 100


if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size

    adult = pd.concat([pd.read_csv(os.path.join(DATA_DIR, 'adult.data'), names=HEADER), 
                       pd.read_csv(os.path.join(DATA_DIR, 'adult.test'), header=0, names=HEADER)])
    texts = ['\n'.join(': '.join([attr.replace('-', ' '), str(entry)])
                       for attr, entry in zip(FEATURES, x))
             for _, x in adult[FEATURES].iterrows()]
    answers = [('greater than' if '>' in y else 'less than or equal to') + ' 50K' for y in adult['income']]

    random.seed(SEED)
    while True:
        idx = random.sample(range(len(answers)), SHOTS)
        if SHOTS < 2 or len({answers[j] for j in idx}) == 2:
            break
    template = 'Predict if income exceeds $50K/yr. Respond with only one of the following: greater than 50K | less than or equal to 50K.' + \
               ''.join(f'\nExample {i+1} -\n{texts[j]}\nAnswer: {answers[j]}' for i, j in enumerate(idx)) + '\nTest -\n{text}\nAnswer:'
    prompt = PromptTemplate(input_variables=["text"], template=template)
    llm = LlamaCpp(model_path=MODEL_PATH, temperature=0, n_ctx=1024, n_gpu_layers=-1, verbose=False)
    chain = LLMChain(llm=llm, prompt=prompt, verbose=False)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start, ncorrect, nguess, ntotal = 0, 0, 0, 0
    for i in range(len(texts)):
        fname = os.path.join(OUTPUT_DIR, f'~dump_{i+1}_{rank}.csv')
        if os.path.isfile(fname):
            start = i+1
            df = pd.read_csv(fname, names=NAMES)
            ncorrect += sum(df['pred'] == df['label'])
            nguess += sum('less than or equal to 50K' == df['label'])
            ntotal += len(df['label'])
    with open(os.path.join(OUTPUT_DIR, f'~prompt_{start}_{rank}.txt'), 'w') as f:
        f.write(template)
    if start:
        write(f'{start}\tAccuracy: {ncorrect / ntotal}\tGuessing: {nguess / ntotal}\n')

    entries = [','.join([str(i), str(age), race, sex, answer])
               for i, (age, race, sex, answer) in enumerate(zip(adult['age'], adult['race'], adult['sex'], answers))]
    for data in [texts, answers, entries]:
        random.seed(0)
        random.shuffle(data)
    preds, output = [], []
    for i, text in enumerate(tqdm(texts) if size == 1 else texts):
        if i < start:
            continue
        if i % size == rank:
            pred = chain.run(text).strip()
            preds.append(pred)
            output.append(','.join([entries[i], pred]))
            ncorrect += pred == answers[i]
            nguess += 'less than or equal to 50K' == answers[i]
            ntotal += 1
            acc = ncorrect / ntotal
            guess = nguess / ntotal
        if size == 1:
            print(pred, '\tAccuracy:', acc, '\tGuessing:', guess, '\t', SHOTS, len(FEATURES))
        if not (i+1) % PRINTERVAL:
            with open(os.path.join(OUTPUT_DIR, f'~dump_{i+1}_{rank}.csv'), 'w') as f:
                f.write('\n'.join(output))
            output = []
            write(f'{i+1}\tAccuracy: {acc}\tGuessing: {guess}\n')
    if output:
        with open(os.path.join(OUTPUT_DIR, f'~dump_{i+1}_{rank}.csv'), 'w') as f:
            f.write('\n'.join(output))
        write(f'{i+1}\tAccuracy: {acc}\tGuessing: {guess}\n')

    if not rank:
        dfs = []
        for i in range(len(texts)):
            for j in range(size):
                fname = os.path.join(OUTPUT_DIR, f'~dump_{i+1}_{j}.csv')
                if os.path.isfile(fname):
                    dfs.append(pd.read_csv(fname, names=NAMES))
        pd.concat(dfs).to_csv(os.path.join(OUTPUT_DIR, 'concat.csv'), index=False)
