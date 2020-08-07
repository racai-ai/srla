# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import dynet_config

dynet_config.set(mem=2048, autobatch=False)
#dynet_config.set_gpu()

import dynet as dy
from lts_io import Dataset
from lts_network import Network

def eval_network(network, dataset, examples):
    errs = 0
    for example in examples:
        dy.renew_cg()
        l, p = dataset.process_example(example)
        pp = network.predict(l)
        has_err = False
        if len(pp) != len(p):
            has_err = True
        else:
            for p1, p2 in zip (p, pp):
                if p1 != p2:
                    has_err = True
                    break
        if has_err:
            errs += 1
    return 1.0-float(errs) / float(len(examples))

def train(train_file, dev_file, output_base, num_itt_no_improve):
    dataset = Dataset(train_file, dev_file)
    network = Network(dataset)
    itt = num_itt_no_improve
    epoch = 1
    while itt > 0:
        print "Starting epoch " + str(epoch)
        epoch += 1
        total_loss = 0
        ex_index = 0
        for example in dataset.train_examples:
            ex_index += 1
            dy.renew_cg()
            l, p = dataset.process_example(example)
            loss = network.learn(l, p) / len(p)
            total_loss += loss
            if ex_index % 500 == 0:
                print ex_index
        acc_train = eval_network(network, dataset, dataset.train_examples)
        acc_dev = eval_network(network, dataset, dataset.dev_examples)
        print "train_acc=" + str(acc_train) + " dev_acc=" + str(acc_dev) + " total_loss=" + str(total_loss) + "\n"
        itt -= 1
    
train("../../../corpus/lts/lts-lex.train", "../../../corpus/lts/lts-lex.test", "", 100)