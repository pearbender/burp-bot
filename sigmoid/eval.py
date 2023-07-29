import torch
import torchaudio
from torchaudio import transforms
import os
import shutil
import argparse
import itertools

from tqdm import tqdm

from model_loader import BurpEvaluator


parser = argparse.ArgumentParser("model-evaluator")
parser.add_argument("-a", "--action", help="Copy or move the false-positives and false-negatives.", choices=['copy', 'move'])
parser.add_argument("-M", "--models", help="Model file[s] to use", type=str, default=['./model.pt'], nargs='+')
parser.add_argument("-p", "--permutate", help="Test all permutation of model combinations", action="store_true")
parser.add_argument("-s", "--single", help="Use model ensemble mode if many model files provided", action="store_true")
parser.add_argument("-o", "--output", help="Directory to store the outputs, ignored without -c or -m", type=str, default='./eval-temp')
parser.add_argument("-T", "--true-data", help="Directory with burp clips", type=str, default='../burps-audio')
parser.add_argument("-F", "--false-data", help="Directory with non-burp clips", type=str, default='../not-burps-audio')
args = parser.parse_args()


burps_folder_path = args.true_data
burps_files = [os.path.join(burps_folder_path, file) for file in os.listdir(
    burps_folder_path) if file.lower().endswith(".wav")]

burps_folder_path = args.false_data
not_burps_files = [os.path.join(burps_folder_path, file) for file in os.listdir(
    burps_folder_path) if file.lower().endswith(".wav")]


def prepare_file(audio_file):
    sig, sr = torchaudio.load(audio_file)
    spec = transforms.MelSpectrogram(
        sr, n_fft=1024, hop_length=None, n_mels=64)(sig)
    spec = transforms.AmplitudeToDB(top_db=80)(spec)

    return spec.unsqueeze(0)


def print_stat(message: str, value, total, length=45):
    print(f"{message: <{length}} {value / total: >9.4%} ( {(total - value) / total: >9.4%} wrong) {value: >5} / {total: <5} {total - value} wrong")


def single_model_mode(model: BurpEvaluator):
    if args.action is not None:
        if os.path.exists(args.output):
            shutil.rmtree(args.output)

        false_positives_dir = os.path.join(args.output, 'false-positives')
        false_negatives_dir = os.path.join(args.output, 'false-negatives')

        os.makedirs(false_positives_dir)
        os.makedirs(false_negatives_dir)


    true_positives = 0
    true_negatives = 0

    false_positives = 0
    false_negatives = 0


    print('finding false-negatives...')
    for burp in tqdm(burps_files, dynamic_ncols=True, leave=False, unit='files'):
        if not model.evaluate_file(burp) >= 0.5:
            false_negatives += 1
            tqdm.write(burp)
            if args.action is not None and args.action == 'move':
                shutil.move(burp, false_negatives_dir)
            elif args.action is not None and args.action == 'copy':
                shutil.copy(burp, false_negatives_dir)
        else:
            true_positives += 1


    print('finding false-positives...')
    for burp in tqdm(not_burps_files, dynamic_ncols=True, leave=False, unit='files'):
        if model.evaluate_file(burp) >= 0.5:
            false_positives += 1
            tqdm.write(burp)
            if args.action is not None and args.action == 'move':
                shutil.move(burp, false_positives_dir)
            elif args.action is not None and args.action == 'copy':
                shutil.copy(burp, false_positives_dir)
        else:
            true_negatives += 1


    total_positives = true_positives + false_negatives
    total_negatives = false_positives + true_negatives

    total_true = true_positives + true_negatives
    total_false = false_positives + false_negatives

    print(f"\n\nTotal data {total_positives + total_negatives} clips")
    print(f"Total burps {total_positives}")
    print(f"Total non-burps {total_negatives}")

    print_stat('Total accuracy', total_true, total_true + total_false)
    print_stat('Percent of burps detected', true_positives, total_positives)
    print_stat('Percent of non-burps detected', true_negatives, total_negatives)
    print_stat('Percent of detections being actually burps', true_positives, true_positives + false_positives)


def print_matrix(matrix, model_len, total):
    top = '|#####|'
    for i in range(model_len):
        top += f' {i: >5} |'
    print(top)

    for i in range(model_len):
        text = f'| {i: >3} |'
        for j in range(model_len):
            text += f'{matrix[i][j] / total: >6.2%} |'
        print(text)


def evaluate_combination(comb, burp_results, not_burp_results, model_filenames):
    true_positives = 0
    false_negatives = 0
    for burp_r in burp_results:
        trues = 0
        for m in comb:
            trues += burp_r[m]
        if trues / len(comb) >= 0.2:
            true_positives += 1
        else:
            false_negatives += 1

    true_negatives = 0
    false_positives = 0
    for not_burp_r in not_burp_results:
        trues = 0
        for m in comb:
            trues += not_burp_r[m]
        if (len(comb) - trues) / len(comb) < 0.2:
            true_negatives += 1
        else:
            false_positives += 1

    total_positives = true_positives + false_negatives
    total_negatives = false_positives + true_negatives

    total_true = true_positives + true_negatives
    total_false = false_positives + false_negatives

    print(f'\n\n==================\nCombo of {comb}:')
    for m in comb:
        print(model_filenames[m])
    print_stat('\nTotal accuracy', total_true, total_true + total_false)
    print_stat('\nPercent of burps detected', true_positives, total_positives)
    print_stat('Percent of non-burps detected', true_negatives, total_negatives)
    print_stat('Percent of detections being actually burps', true_positives, true_positives + false_positives)

    acc = total_true / (total_true + total_false)
    true_acc = true_positives / total_positives
    false_acc = true_negatives / total_negatives
    det_acc = true_positives / (true_positives + false_positives)

    return acc, true_acc, false_acc, det_acc


def multi_model_mode(model_filenames):
    models = []

    for model_file in model_filenames:
        model = AudioClassifier()
        model.load_state_dict(torch.load(model_file))
        model.eval()
        models += [model]

    if os.path.exists(args.output):
        shutil.rmtree(args.output)

    
    burp_results = []

    print('evaluating burps')
    for burp in tqdm(burps_files, dynamic_ncols=True, leave=False, unit='files'):
        burp_stats = []
        tensor = prepare_file(burp)
        for m in models:
            burp_stats += [1 if is_burp_tensor(m, tensor) else 0]
        burp_results += [burp_stats]


    not_burp_results = []
    print('evaluating non-burps')
    for not_burp in tqdm(not_burps_files, dynamic_ncols=True, leave=False, unit='files'):
        burp_stats = []
        tensor = prepare_file(not_burp)
        for m in models:
            burp_stats += [1 if not is_burp_tensor(m, tensor) else 0]
        not_burp_results += [burp_stats]


    print('calculating stat matrices')

    both_true_matrix = [[0 for _ in models] for _ in models]
    both_false_matrix = [[0 for _ in models] for _ in models]
    first_true_matrix = [[0 for _ in models] for _ in models]
    first_false_matrix = [[0 for _ in models] for _ in models]
    both_different_matrix = [[0 for _ in models] for _ in models]

    for b in burp_results:
        for i in range(len(models)):
            for j in range(len(models)):
                if b[i] == b[j] == 1:
                    both_true_matrix[i][j] += 1
                if b[i] == b[j] == 0:
                    both_false_matrix[i][j] += 1
                if b[i] == 1 and b[j] == 0:
                    first_true_matrix[i][j] += 1
                    both_different_matrix[i][j] += 1
                if b[i] == 0 and b[j] == 1:
                    first_false_matrix[i][j] += 1
                    both_different_matrix[i][j] += 1
    
    for i in range(len(models)):
        print(f'[ {i: >3} ]: {model_filenames[i]}')

    print('\nBurps')
    print('Both true:')
    print_matrix(both_true_matrix, len(models), len(burp_results))
    print('Both false:')
    print_matrix(both_false_matrix, len(models), len(burp_results))
    print('First true:')
    print_matrix(first_true_matrix, len(models), len(burp_results))
    print('First false:')
    print_matrix(first_false_matrix, len(models), len(burp_results))
    print('Both different:')
    print_matrix(both_different_matrix, len(models), len(burp_results))

    both_true_matrix = [[0 for _ in models] for _ in models]
    both_false_matrix = [[0 for _ in models] for _ in models]
    first_true_matrix = [[0 for _ in models] for _ in models]
    first_false_matrix = [[0 for _ in models] for _ in models]
    both_different_matrix = [[0 for _ in models] for _ in models]

    for b in not_burp_results:
        for i in range(len(models)):
            for j in range(len(models)):
                if b[i] == b[j] == 1:
                    both_true_matrix[i][j] += 1
                if b[i] == b[j] == 0:
                    both_false_matrix[i][j] += 1
                if b[i] == 1 and b[j] == 0:
                    first_true_matrix[i][j] += 1
                    both_different_matrix[i][j] += 1
                if b[i] == 0 and b[j] == 1:
                    first_false_matrix[i][j] += 1
                    both_different_matrix[i][j] += 1
    
    for i in range(len(models)):
        print(f'[ {i: >3} ]: {model_filenames[i]}')

    print('\nNot Burps')
    print('Both true:')
    print_matrix(both_true_matrix, len(models), len(not_burp_results))
    print('Both false:')
    print_matrix(both_false_matrix, len(models), len(not_burp_results))
    print('First true:')
    print_matrix(first_true_matrix, len(models), len(not_burp_results))
    print('First false:')
    print_matrix(first_false_matrix, len(models), len(not_burp_results))
    print('Both different:')
    print_matrix(both_different_matrix, len(models), len(not_burp_results))

    combs = []

    for i in range(1, len(models) + 1) if args.permutate else [1]:
        for comb in itertools.combinations(range(len(models)), i):
            acc, true_acc, false_acc, det_acc = evaluate_combination(comb, burp_results, not_burp_results, model_filenames)
            
            combs += [(comb, acc, true_acc, false_acc, det_acc)] 

    print("\n\n\n==================\nBest accuracy combination")
    evaluate_combination(max(combs, key=lambda x: x[1])[0], burp_results, not_burp_results, model_filenames)

    print("\n\n\n==================\nLeast false negatives combination")
    evaluate_combination(max(combs, key=lambda x: (x[2], x[1]))[0], burp_results, not_burp_results, model_filenames)

    print("\n\n\n==================\nLeast false positives combination")
    evaluate_combination(max(combs, key=lambda x: (x[3], x[1]))[0], burp_results, not_burp_results, model_filenames)

    print("\n\n\n==================\nBest detections accuracy combination")
    evaluate_combination(max(combs, key=lambda x: (x[4], x[1]))[0], burp_results, not_burp_results, model_filenames)


def main():
    if len(args.models) == 1 or args.single:
        single_model_mode(BurpEvaluator(args.models))
    else:
        #multi_model_mode(args.models)
        pass


if __name__ == "__main__":
    main()
