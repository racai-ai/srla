# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

if __name__ == "__main__":
    print "Hello World"

import numpy as np
from os import listdir
import os.path
from os.path import isfile
from os.path import isdir
from os.path import join
import scipy
from scipy.cluster.vq import kmeans
import scipy.io
import scipy.io.wavfile
import sys
import subprocess
import re

from random import shuffle

class Dataset:
    def _get_all_files(self, path):
        rez=[]
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        for f in onlyfiles:
            fn = join (path, f)
            if fn.endswith(".wav") and os.path.isfile(fn.replace(".wav", ".lab")):
                rez.append(fn.replace(".wav", ""))
        
        #the rest of the files
        onlydirs=[f for f in listdir(path) if isdir(join(path, f))]
        for f in onlydirs:
            fn = join (path, f)
            if not fn.endswith("."):
                tmp=self._get_all_files(fn)
                for xx in tmp:
                    rez.append(xx)
        return rez
    
    def __init__(self, train_path, dev_path):
        self.train_list = []
        self.dev_list = []
        if train_path is not None:
            sys.stdout.write("scanning " + train_path + "...")
            self.train_list=self._get_all_files(train_path)
        shuffle(self.train_list)
        
        #sr, wav = scipy.io.wavfile.read(self.train_list[0] + ".wav")
        self.sample_rate = 16000
        sys.stdout.write(" found " + str(len(self.train_list)) + " training files\n")

        if dev_path is not None:
            sys.stdout.write("scanning " + dev_path + "...")
            self.dev_list=self._get_all_files(dev_path)
        sys.stdout.write(" found " + str(len(self.dev_list)) + " dev files\n")
        
        sys.stdout.write("Building feature maps...")
        self.phoneme2int = {}
        self.phoneme2int["_"] = 0
        self.phoneme2int["<EOS>"] = 1
        self.phonemeList = []
        self.phonemeList.append("_")
        self.phonemeList.append("<EOS>")
        self.duration2int = {}
        self.context2int = []
        self.total_states = 0
        for fname in self.train_list:
            pi_list = self.read_lab(fname)
            for pi in pi_list:
                self.total_states += (pi.stop-pi.start) / 5
                phoneme = pi.phoneme
                if phoneme not in self.phoneme2int:
                    self.phoneme2int[phoneme] = len(self.phoneme2int)
                    self.phonemeList.append(phoneme)
                duration = pi.duration
                if duration not in self.duration2int:
                    self.duration2int[duration] = len(self.duration2int)
                if len(self.context2int) == 0:
                    print "Creating context of size " + str(len(pi.context))
                    for zz in range (len(pi.context)):
                        self.context2int.append({})
                for zz in range(len(pi.context)):
                    cont = pi.context[zz]
                    if cont not in self.context2int[zz]:
                        self.context2int[zz][cont] = len(self.context2int[zz])
                
        sys.stdout.write("done\n")
        sys.stdout.write("Statistics:\n")
        
        sys.stdout.write(" \t" + str(len(self.phoneme2int)) + " unique phonemes\n")
        sys.stdout.write(" \t" + str(len(self.duration2int)) + " unique durations\n")
        sys.stdout.write(" \t" + str (len(self.context2int)) + " feature maps\n")
        sys.stdout.write(" \t" + str(self.total_states) + " total states\n")
        self.state2int = None
        print self.phoneme2int

        
    def cluster(self, num_clusters, mlsaVocoder):
        sys.stdout.write("Starting state clustering to " + str(num_clusters) + " number of clusters\n")
        self.state2int = []
        self.num_clusters = num_clusters
        mb = self.total_states * mlsaVocoder.num_params * 8
        mb = mb / 1024
        mb = int(mb / 1024)
        sys.stdout.write("Reading entire dataset into memory (shoud take " + str(mb) + "MB of memory)...")
        sys.stdout.flush()
        last_proc = 0
        index = 0
        all_data = []
        tmp = self.train_list
        for fname in tmp:
            #print fname
            [disc, cont] = self.read_wave(fname)
            mfc = mlsaVocoder.extract_spectrum(cont)
            #print len(mfc)
            for zz in range(len(mfc)):
                all_data.append(mfc[zz])
            index += 1
            proc = index * 100 / len(tmp)
            if proc % 5 == 0 and proc != last_proc:
                sys.stdout.write(" " + str(proc))
                sys.stdout.flush()
                last_proc = proc
        sys.stdout.write(" done\n")
        sys.stdout.write("Starting clustering... ")
        self.state2int, dist = kmeans(all_data, self.num_clusters)
        sys.stdout.write(" distortion=" + str(dist) + "\n")
        
    def ulaw_encode(self, data):
        out_discreete = []
        out_continous = []
        for zz in range(len(data)):
            f = float(data[zz]) / 32768
            sign = np.sign(f)
            encoded = sign * np.log(1.0 + 255.0 * np.abs(f)) / np.log(1.0 + 255.0)
            encoded_d = int((encoded + 1) * 127)
            out_discreete.append(encoded_d)
            out_continous.append(f)
            
        return [out_discreete, out_continous]
    def ulaw_decode(self, data):
        out = []
        for zz in range (len(data)):
            f = float(data[zz]) / 128-1.0
            sign = np.sign(f)
            decoded = sign * (1.0 / 255.0) * (pow(1.0 + 255, abs(f))-1.0)
            decoded = int(decoded * 32768)
            out.append(decoded)
        return out
    
    def read_wave(self, filename):
        sr, wav = scipy.io.wavfile.read(filename + ".wav")
        if sr!=self.sample_rate:
            x=subprocess.check_output(['sox',filename + ".wav", '-r', '16000', '-c', '1', 'tmp.wav'])
            sr, wav = scipy.io.wavfile.read("tmp.wav")
        out = self.ulaw_encode(wav)
        #rec=self.ulaw_decode(out)
        #data2 = np.asarray(rec, dtype=np.int16)
        #scipy.io.wavfile.write("test.wav", 16000, data2)
        return out
    
    def read_lab(self, filename, rnn=False):
        out = []
        with open (filename + ".lab") as f:
            if not rnn:
                out.append(PhoneInfo("<S>", "", 0, 0))
            lines = f.readlines()
            #print lines
            line = lines[0].replace("\n", "")
            line=re.sub(' +',' ',line)
            words = line.split(" ")
            for word in words:
                uniword = unicode(word, 'utf-8').lower()
                for i in range(len(uniword)):
                    char = uniword[i].encode("utf-8")
                    pi = PhoneInfo(char, "", 0, 0)
                    out.append(pi)
                if rnn:
                    out.append(PhoneInfo("_", "", 0, 0))
            if rnn:
                out.append(PhoneInfo("<EOS>", "", 0, 0))
            else:
                out.append(PhoneInfo("<S>", "", 0, 0))
        return out
    
    def make_dataframes(self, audio, frame_len_ms):
        frames = []
        frame_len_samp = self.sample_rate / 1000 * frame_len_ms
        num_frames = len(audio) / frame_len_samp
        if len(audio) % frame_len_samp != 0:
            num_frames += 1
        index = 0
       
        for frame in range (num_frames):
            frm = []
            for zz in range (frame_len_samp):
                if index < len(audio):
                    frm += [audio[index]]
                else:
                    frm += [0]
                index += 1
            frames.append(frm)
            
        return frames
    
    def squared_distance(self, p1, p2):
        dist = 0
        for zz in range(len(p1)):
            dist += (p1[zz]-p2[zz]) ** 2
        return np.sqrt(dist)
    
    def spectrum2int(self, spec):
        
        best_index = 0
        best_score = self.squared_distance(spec, self.state2int[0])
        for zz in range(1, self.num_clusters):
            score = self.squared_distance(spec, self.state2int[zz])
            if score < best_score:
                best_score = score
                best_index = zz
        return best_index
    
    def save_lookups(self, output):
        f = open(output, "w")
        f.write("NUM CHARACTERS " + str(len(self.phoneme2int)) + "\n")
        for phone in self.phoneme2int:
            f.write(phone + " " + str(self.phoneme2int[phone]) + "\n")
        f.close()
        
    def load_lookups(self, output):
        f = open(output, "r")
        lines = f.readlines()
        self.phoneme2int = {}
        self.phonemeList = [""] * (len(lines)-1)
        for iLine in range (1, len(lines)):
            line = lines[iLine].replace("\n", "")
            parts = line.split(" ")
            phoneme = parts[0]
            pIndex = int(parts[1])
            self.phoneme2int[phoneme] = pIndex
            self.phonemeList[pIndex] = unicode(phoneme, 'utf-8')
            
        f.close()
            
class PhoneInfo:
    context2int = {}
    def __init__(self, phoneme, context, start, stop):
        self.phoneme = phoneme
        self.context = context.split("/")
        self.start = start
        self.stop = stop
        self.duration = (stop-start)
            
