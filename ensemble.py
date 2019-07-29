import numpy as np
import pandas as pd
from glob import glob
import datetime
import argparse
import sys
import pdb

def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='person classification test code')
    parser.add_argument('--csv_folder', default='./csvs', type=str, help='directory containing candidate csvs')

    return parser.parse_args(argv)

def csv_ensemble(csv_folder):

    csv_files = glob(csv_folder + '/*.csv')

    # filename, 0 negative confidence, 1 positive confidence

    candidates = []
    ID = []
    for csv_file in csv_files:

        df = pd.read_csv(csv_file, header=None)
        data = df.values
        ID = data[:,0]
        prediction = data[:,1:3]
        candidates.append(prediction)

    candidates = np.asarray(candidates)

    averaged = np.mean(candidates, axis=0)
    predicted_classes = np.argmax(averaged,axis=1)

    submit_date = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    submit_file_path = 'submission-final-ensembled-' + submit_date + '.csv'

    print(submit_file_path)

    with open(submit_file_path,'w') as w:
        for i in range(len(ID)):
            line = str(ID[i]) + ',' + str(predicted_classes[i])
            w.write(line + '\n')

    return predicted_classes, submit_file_path


def main(args):
    
    predicted_classes, submit_file_path = csv_ensemble(args.csv_folder)
    print('ensemble done. saved in %s' % (submit_file_path))

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
