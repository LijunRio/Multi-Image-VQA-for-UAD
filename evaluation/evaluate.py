import logging
import os

import pandas as pd
from nlgeval import NLGEval
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score, accuracy_score

logging.getLogger('edu.stanford.nlp').setLevel(logging.ERROR)
NLGEval = NLGEval(no_overlap=False, no_skipthoughts=True, no_glove=True, metrics_to_omit=['METEOR', 'SPICE'])

def tokenize(sentences):
    return [word_tokenize(sentence) for sentence in sentences]

def string_to_label1(res):
    res = res.replace(" ", "").lower()
    if res == 'yes.':
        return 1
    elif res == 'no.':
        return 0
    else:
        print(res)
        raise ValueError('Wrong Input')


def string_to_label2(res):
    res = res.replace(" ", "").lower()
    if res == 'it\'sposttreatmentchange.':
        return 0
    elif res == 'it\'sanormalbrain':
        return 1
    elif res == 'it\'smasslesion.':
        return 2
    elif res == 'it\'slesions.':
        return 3
    elif res == 'it\'senlargedventricles.':
        return 4
    elif res == 'it\'sedema.':
        return 5
    elif res == 'it\'sartefacts.':
        return 6
    elif res == 'it\'scraniotomy.':
        return 7
    elif res == 'it\'sresection.':
        return 8
    else:
        print(res)
        raise ValueError('Wrong Input')

def string_to_label3(res):
    res = res.replace(" ", "").lower()
    if res == 'clinicallyrelevant.':
        return 0
    elif res == 'potentiallyclinicallyrelevant.' or res=='potentiallyclinicallyrelevant':
        return 1
    elif res == 'notapplicable.':
        return 2
    elif res == 'clinicallyirrelevant.':
        return 3
    else:
        print(res)
        raise ValueError('Wrong Input')

def caculate_close_accuracy(data, func,num):
    data = data.copy()
    data['GT_Label'] = data['ground_truth'].apply(func)
    data['Predict_Label'] = data['predict'].apply(func)
    accuracy = accuracy_score(data['GT_Label'], data['Predict_Label'])
    if num !=1:
        f1 = f1_score(data['GT_Label'], data['Predict_Label'], average='micro')
    else: f1 = f1_score(data['GT_Label'], data['Predict_Label'])
    metrics = {'ACC{}'.format(num): accuracy, 'F1{}'.format(num): f1}
    return metrics

def caculate_each_questions(data):
    data.columns = ['filename', 'Question', 'ground_truth', 'predict']
    close_questions = [
        "Is the case normal?",
        "Please describe the condition of the brain.",
        "Can you comment on the severity of the pathology?"
    ]

    open_questions = [
        "Are there areas in the anomaly maps that highlight a normal variation of the healthy, rather than pathological areas (false positives)?",
        "Is the pseudo-healthy reconstruction a plausible restoration of the input to a healthy state?",
        "Do the anomaly maps accurately reflect the selected disease?",
        "Can you describe the differences highlighted between anomaly maps and origin image and why it is the healthy region?"
    ]

    _close_questions = [question.strip() for question in close_questions]
    _open_questions = [question.strip() for question in open_questions]
    data['_Question'] = data['Question'].str.strip()
    # # 计算 close questions 指标
    data1 = data[data['_Question'] == _close_questions[0]]
    data2 = data[data['_Question'] == _close_questions[1]]
    data3 = data[data['_Question'] == _close_questions[2]]
    print(len(data1), len(data2), len(data3))

    F11 = caculate_close_accuracy(data1, string_to_label1,1)['F11']
    F12 = caculate_close_accuracy(data2, string_to_label2,2)['F12']
    F13 = caculate_close_accuracy(data3, string_to_label3,3)['F13']
    F1 = (F11+F12+F13)/3*100
    close_res = data[data['_Question'].isin(_close_questions)]
    close_res['predict'] = close_res['predict'].str.strip().str.lower()
    close_res['ground_truth'] = close_res['ground_truth'].str.strip().str.lower()
    total_correct = (close_res['predict'] == close_res['ground_truth']).sum()
    total_samples = len(close_res)
    accuracy_avg = total_correct / total_samples*100

# open res
    open_res = data[data['_Question'].isin(_open_questions)]
    reference_list = open_res['ground_truth'].tolist()
    candidate_list = open_res['predict'].tolist()
    metrics_nlg = NLGEval.compute_metrics(ref_list=[reference_list], hyp_list=candidate_list)
    final_metric = {'ACC_avg':accuracy_avg, 'F1_avg':F1, **metrics_nlg}
    return final_metric


# caculate forlder
res_folder = '/home/june/Code/VQA-UAD/evaluation/res'
files = [os.path.join(res_folder, f) for f in os.listdir(res_folder) if os.path.isfile(os.path.join(res_folder, f)) and f.endswith('.txt')]
sorted_files = sorted(files, key=os.path.getmtime)
sorted_files = sorted(files, key=os.path.getmtime) #sort by time

all_metrics = []
for file in sorted_files:
    print(file)
    data = pd.read_csv(file, sep='\t', header=None)
    metric = caculate_each_questions(data)
    metric['filename'] = file
    all_metrics.append(metric)
df_all_metrics = pd.DataFrame(all_metrics)
res_file = os.path.join(res_folder, 'res.csv')
df_all_metrics.to_csv(res_file, index=False)

