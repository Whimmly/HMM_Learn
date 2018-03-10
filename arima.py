#!/usr/bin/env python
import sys, json, codecs, pickle, argparse, datetime
import numpy as np

from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.arima_model import ARIMA

def warn(msg):
    print(msg, file=sys.stderr)

print(datetime.datetime.now())
np.random.seed(seed=None)

args = argparse.ArgumentParser(
        description="Train ARIMA based on given input text")
args.add_argument("-o", required=True, metavar="FILENAME",
        help="file basename for output files")
args.add_argument("-w", default=1000000000, type=int, help="Word count limit")
args.add_argument("input", nargs="?", type=argparse.FileType("r"),
        default=sys.stdin,
        help="input text that will act as training set for the model (can be passed on stdin)")
args = args.parse_args()

lines = [list(map(lambda x: x.lower(), line.split())) for line in args.input] 
words = [word for line in lines for word in line][:args.w]

print("Read files")

alphabet = set(words)
print("Total words:", len(words), "\nUnique words:", len(alphabet),
      "\nLines:", len(lines))
le = LabelEncoder()
le.fit(list(alphabet))

print("Fitted alphabet")

seq = []
max_np_len = 200000
for i in range(0, len(words), max_np_len):
    print(i)
    seq.extend(list(le.transform(words[i:i + max_np_len])))
for i in range(len(seq)):
    seq[i] = float(seq[i])

model = ARIMA(seq, order=(2,1,0))
model_fit = model.fit(disp=0)
output = model_fit.forecast()
prediction = le.inverse_transform([int(symbol.flatten()[0]) for symbol in output])
print("Single forecast:", prediction)
print("Fit features")
model.dates, model.freq, model.missing = None, None, None

with open(args.o + "arima.model_fit", "wb") as f:
    pickle.dump(model_fit, f)
with open(args.o + "arima.model", "wb") as f:
    pickle.dump(model, f)

warn("Output written to arima.model and arima.model_fit")
print(datetime.datetime.now())
