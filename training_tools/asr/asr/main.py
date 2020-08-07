# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import dynet_config

dynet_config.set(mem=2000, random_seed=9, autobatch=False)
# dynet_config.set_gpu()

from config import Config
from dataset import Dataset
import dynet as dy
from network import Network
import sys
from vocoder import MLSAVocoder
import time
import gc


def display_help():
    print ("Neural TTS Model Trainer version 0.9 beta.")
    print ("Usage:")
    print ("\t--train_ctc   <train folder> <dev folder> <model output base> <num itt no improve>")
    print ("\t--resume_ctc   <train folder> <dev folder> <model output base> <num itt no improve>")
    print ("\t--train_rnn   <train folder> <dev folder> <model output base> <num itt no improve>")
    print ("\t--run_ctc     <model output base> <input file> <output file>")
    print ("\t--run_kws     <model output base> <input file> <keyword file> <output file>")


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def evaluate(ds, network):
    cf = 0
    total = 0
    errors = 0
    for fname in ds.dev_list:
        dy.renew_cg()
        cf += 1
        [disc, cont] = ds.read_wave(fname)
        labels = ds.read_lab(fname)
        rez, rez_ts = network.predict(cont, len(labels))
        lbl = []
        for zz in range(len(labels)):
            lbl.append(labels[zz].phoneme)
        sys.stdout.write("GS=")
        for zz in range(len(lbl)):
            sys.stdout.write(" " + lbl[zz])
        sys.stdout.write("\n")
        sys.stdout.write("PR=")
        for zz in range(len(rez)):
            sys.stdout.write(" " + rez[zz])
        sys.stdout.write("\n")
        sys.stdout.flush()
        # print lbl
        # print rez
        print "-------"
        dist = levenshtein(lbl, rez)
        total += max(len(rez), len(lbl))
        errors += dist
    if total == 0:
        total += 1
    return 1.0 - float(errors) / total


def evaluate_rnn(ds, network):
    cf = 0
    total = 0
    errors = 0
    for fname in ds.dev_list:
        dy.renew_cg()
        cf += 1
        [disc, cont] = ds.read_wave(fname)
        labels = ds.read_lab(fname, True)
        rez = network.predict_rnn(cont, len(labels))
        lbl = []
        for zz in range(len(labels)):
            lbl.append(labels[zz].phoneme)
        sys.stdout.write("GS=")
        for zz in range(len(lbl)):
            sys.stdout.write(lbl[zz])
        sys.stdout.write("\n")
        sys.stdout.write("PR=")
        for zz in range(len(rez)):
            sys.stdout.write(rez[zz])
        sys.stdout.write("\n")
        sys.stdout.flush()
        # print lbl
        # print rez
        print "-------"
        dist = levenshtein(lbl, rez)
        total += max(len(rez), len(lbl))
        errors += dist
    if total == 0:
        total += 1
    return 1.0 - float(errors) / total


def run_ctc(model, input_file, output_file):
    ds = Dataset(None, None)
    ds.load_lookups(model + ".ctc-bestACC.encoding")
    config = Config(model + ".ctc-bestACC.config")
    print "===Initializing MLSA Coder==="
    vocoder = MLSAVocoder(ds.sample_rate, config.vocoder_params, config.frame_len_ms)
    vocoder.load_normalization(model + ".ctc-bestACC.norm")
    print "===Initializing ASR Network==="
    network = Network(config, ds, vocoder)
    network.load_model_ctc(model + ".ctc-bestACC.network")

    dy.renew_cg()
    print "===Reading input file==="
    [disc, cont] = ds.read_wave(input_file[:-4])
    print "===Running CTC==="
    start = time.time()
    chars, timestamps = network.predict(cont, num_preds=None)
    stop = time.time()
    print "Execution time: " + str(stop - start)
    with open(output_file, "w") as f:
        for char, ts in zip(chars, timestamps):
            f.write(char.encode('utf-8') + "\t" + str(ts) + "\n")
        f.close()


def partial_dtw(s1, s2):
    import numpy as np
    a = np.zeros((len(s1) + 1, len(s2) + 1))
    for i in xrange(len(s1) + 1):
        a[i, 0] = i
    for i in xrange(len(s2) + 1):
        a[0, i] = i

    for i in xrange(1, len(s1) + 1):
        for j in xrange(1, len(s2) + 1):
            cost = 0
            if s1[i - 1] != s2[j - 1]:
                cost = 1
            m = a[i - 1, j - 1]
            m = min(m, a[i - 1, j])
            m = min(m, a[i, j - 1])
            a[i, j] = m + cost

    m = a[len(s1), len(s2)]
    for i in range(len(s2) + 1):
        if m > a[len(s1), i]:
            m = a[len(s1), i]
    return m


def find_keyword(word, chars, timestamps):
    start = []
    stop = []
    word = unicode(word, 'utf-8')
    l = len(word) + 3
    for i in xrange(len(chars) - l):
        c_start = i
        c_stop = i + l
        if c_stop > len(chars):
            c_stop = len(chars)
        s1 = word
        s2 = chars[c_start:c_stop]
        dtw = partial_dtw(s1, s2)
        if dtw / min(len(s1), len(s2)) < 0.2:
            a_start = timestamps[c_start] - 500
            a_stop = timestamps[c_stop - 1] + 500
            if a_start < 0:
                a_start = 0
            if a_stop > timestamps[-1]:
                a_stop = timestamps[-1]

            # remove duplicates
            if len(start) > 0:
                if a_start >= stop[-1]:
                    start.append(a_start)
                    stop.append(a_stop)

            else:
                start.append(a_start)
                stop.append(a_stop)
    return start, stop


def run_kws(model, kws_file, input_file, output_file):
    ds = Dataset(None, None)
    ds.load_lookups(model + ".ctc-bestACC.encoding")
    config = Config(model + ".ctc-bestACC.config")
    print "===Initializing MLSA Coder==="
    vocoder = MLSAVocoder(ds.sample_rate, config.vocoder_params, config.frame_len_ms)
    vocoder.load_normalization(model + ".ctc-bestACC.norm")
    print "===Initializing ASR Network==="
    network = Network(config, ds, vocoder)
    network.load_model_ctc(model + ".ctc-bestACC.network")

    print "===Reading KWS file==="
    with open(kws_file) as f:
        content = f.readlines()
    words = [x.strip() for x in content]

    dy.renew_cg()
    print "===Reading input file==="
    [disc, cont] = ds.read_wave(input_file[:-4])
    print "===Running CTC==="
    start = time.time()
    chars, timestamps = network.predict(cont, num_preds=None)
    stop = time.time()

    print "===Running KWS==="
    rez = []
    for word in words:
        start_ts, stop_ts = find_keyword(word, chars, timestamps)
        for start, stop in zip(start_ts, stop_ts):
            rez.append({'start': float(start) / 1000, 'stop': float(stop) / 1000, 'cuvant': word})
    import json
    with open(output_file, "w") as f:
        f.write(json.dumps(rez))
        f.close()
    print "Execution time: " + str(stop - start)


def train_ctc(train, dev, output_base, num_itt_no_improve, resume=False):
    if not resume:
        print "===Initializing dataset==="
        ds = Dataset(train, dev)
        config = Config(None)
        print "===Initializing MLSA Coder==="
        vocoder = MLSAVocoder(ds.sample_rate, config.vocoder_params, config.frame_len_ms)
        print "===Initializing ASR Network==="
        network = Network(config, ds, vocoder)
    else:
        print "===Resuming training==="
        ds = Dataset(train, dev)
        ds.load_lookups(output_base + ".ctc-bestACC.encoding")
        config = Config(output_base + ".ctc-bestACC.config")
        print "===Initializing MLSA Coder==="
        vocoder = MLSAVocoder(ds.sample_rate, config.vocoder_params, config.frame_len_ms)
        vocoder.load_normalization(output_base + ".ctc-bestACC.norm")
        print "===Initializing ASR Network==="
        network = Network(config, ds, vocoder)
        network.load_model_ctc(output_base + ".ctc-bestACC.network")
        acc = evaluate(ds, network)
        print "Devset acc at the end of epoch", acc

    itt = num_itt_no_improve
    epoch = 0
    best_acc = acc
    while itt > 0:
        epoch += 1
        sys.stdout.write("Starting epoch " + str(epoch) + "\n")
        nf = len(ds.train_list)
        cf = 0
        for fname in ds.train_list:
            try:
                dy.renew_cg()
                cf += 1
                sys.stdout.write(str(cf) + "/" + str(nf) + " " + fname + "...")
                sys.stdout.flush()
                [disc, cont] = ds.read_wave(fname)
                labels = ds.read_lab(fname)
                start = time.time()
                loss, llforward = network.learn(cont, labels)
                stop = time.time()
                sys.stdout.write("ctc_cost=" + str(llforward) + " exec " + str(stop - start) + "\n")
                if cf % 10000 == 0:
                    acc = evaluate(ds, network)
                    network.save_model(output_base + ".ctc-last")
                    if acc > best_acc:
                        best_acc = acc
                        network.save_model(output_base + ".ctc-bestACC")
                        itt = num_itt_no_improve
                    print "Devset acc", acc
            except:
                print "Something went wrong. Skipping"
                pass
        acc = evaluate(ds, network)
        print "Devset acc at the end of epoch " + str(epoch), acc
        if acc > best_acc:
            best_acc = acc
            network.save_model(output_base + ".ctc-bestACC")
            itt = num_itt_no_improve
        itt -= 1
        network.save_model(output_base + ".ctc-last")


def train_rnn(train, dev, output_base, num_itt_no_improve):
    print "===Initializing dataset==="
    ds = Dataset(train, dev)
    ds.load_lookups(output_base + ".ctc-bestACC.encoding")
    config = Config(output_base + ".ctc-bestACC.config")
    print "===Initializing MLSA Coder==="
    vocoder = MLSAVocoder(ds.sample_rate, config.vocoder_params, config.frame_len_ms)
    vocoder.load_normalization(output_base + ".ctc-bestACC.norm")
    print "===Initializing ASR Network==="
    network = Network(config, ds, vocoder)
    network.load_model_ctc(output_base + ".ctc-bestACC.network")
    itt = num_itt_no_improve
    epoch = 0
    best_acc = 0
    # evaluate_rnn(ds, network)
    while itt > 0:
        epoch += 1
        sys.stdout.write("Starting epoch " + str(epoch) + "\n")
        nf = len(ds.train_list)
        cf = 0
        for fname in ds.train_list:
            dy.renew_cg()
            cf += 1
            sys.stdout.write(str(cf) + "/" + str(nf) + " " + fname + "...")
            sys.stdout.flush()
            [disc, cont] = ds.read_wave(fname)
            labels = ds.read_lab(fname, True)
            start = time.time()
            loss = network.learn_rnn(cont, labels)
            stop = time.time()
            sys.stdout.write("rnn_cost=" + str(loss / len(labels)) + " exec " + str(stop - start) + "\n")
            if cf % 1000 == 0:
                acc = evaluate_rnn(ds, network)
                print "Devset acc", acc
            gc.collect()
        acc = evaluate_rnn(ds, network)
        print "Devset acc at the end of epoch " + str(epoch), acc
        if acc > best_acc:
            network.save_model(output_base + ".rnn-bestACC")
        network.save_model(output_base + ".rnn-last")


if len(sys.argv) <= 1:
    display_help()
elif (sys.argv[1] == "--train_ctc" and len(sys.argv) == 6):
    train_ctc(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
elif (sys.argv[1] == "--resume_ctc" and len(sys.argv) == 6):
    train_ctc(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), True)
elif (sys.argv[1] == "--train_rnn" and len(sys.argv) == 6):
    train_rnn(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
elif (sys.argv[1] == "--run_ctc" and len(sys.argv) == 5):
    run_ctc(sys.argv[2], sys.argv[3], sys.argv[4])
elif (sys.argv[1] == "--run_kws" and len(sys.argv) == 6):
    run_kws(sys.argv[2], sys.argv[4], sys.argv[3], sys.argv[5])
else:
    display_help()
