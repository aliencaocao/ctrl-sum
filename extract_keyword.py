import os
import shutil
import subprocess

model_dir = 'cnn_bert_tagger'
assert os.path.isdir(model_dir), f'model directory ({model_dir}) not found!'
source = input("Enter the source text: ")
max_keywords = int(input("Enter the max number of keywords: "))  # default for CNNDM is 30
conf_threshold = 0.25  # default for CNNDM
summary_size = 10  # default for CNNDM
dataset_name = 'temp'

with open(os.path.join('datasets', dataset_name, 'test.source'), 'w') as s, open(os.path.join('datasets', dataset_name, 'test.target'), 'w') as t:
    s.write(source)
    t.write(source[0])  # target won't be used so just use first word as placeholder

subprocess.call(f'python scripts/preprocess.py {dataset_name} --mode pipeline --split test --num-workers 1')
subprocess.call(f'bash scripts/gpt2_encode.sh {dataset_name}')
subprocess.call(f'bash scripts/binarize_dataset.sh {dataset_name}')
subprocess.call(f'bash scripts/train_seqlabel.sh -g 0 -d {dataset_name} -p {model_dir}')
subprocess.call(f'python scripts/preprocess.py {dataset_name} --split test --mode process_tagger_prediction --tag-pred {model_dir}/test_predictions.txt --threshold {conf_threshold} --maximum-word {max_keywords} --summary-size {summary_size}')

with open(os.path.join('datasets', dataset_name, f'test.ts{conf_threshold}.mw{max_keywords},sumlen{summary_size}.default.predword'), 'r') as f:
    print(f.read())

shutil.rmtree(os.path.join('datasets', dataset_name), ignore_errors=True)  # clean up temp files for next prediction
