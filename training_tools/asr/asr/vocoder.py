# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

from IPython.display import Audio
import librosa
import librosa.display
import numpy as np
from pysptk import blackman
from pysptk import sptk
from pysptk.synthesis import MLSADF
from pysptk.synthesis import Synthesizer
class MLSAVocoder:
    def __init__(self, sample_rate, num_params, frame_len):
        self.num_params = num_params
        self.sample_rate = sample_rate
        self.frame_len = frame_len
        self.alpha = 0.25
        self.frame_length = self.compute_frame_length(sample_rate, frame_len)
        self.mean = None
        self.stdev = None
        print "Setting frame length to", self.frame_length
        
    def compute_frame_length(self, sample_rate, frame_duration):
        s_frame_len = sample_rate / 1000 * frame_duration
        overlapped_frame_len = s_frame_len * 3
        return 2 ** (overlapped_frame_len-1).bit_length()

    
    def extract_pitch(self, audio_signal):
        audio = np.asarray(audio_signal, dtype=np.float64)
        pitch = sptk.swipe(audio, self.sample_rate, self.frame_len * self.sample_rate / 1000, otype=0, min=60.0, max=350.0)
        return pitch
       
    
    def extract_spectrum(self, x, normalize=False):
        
        frames = librosa.util.frame(np.asarray(x, dtype=np.float32), frame_length=self.frame_length, hop_length=self.frame_len * self.sample_rate / 1000).astype(np.float64).T
        frames *= blackman(self.frame_length)
        mc = sptk.mcep(frames, order=self.num_params, alpha=self.alpha)
        if self.mean == None:
            vec = [0] * (self.num_params + 1)
            for tt in range(len(frames)):
                for ii in range (self.num_params + 1):
                    vec[ii] += mc[tt][ii]
            for ii in range(self.num_params + 1):
                vec[ii] /= len(frames)
            stdev = [0] * (self.num_params + 1)
            for tt in range(len(frames)):
                for ii in range (self.num_params + 1):
                    stdev[ii] += (vec[ii]-mc[tt][ii]) ** 2
            for ii in range(self.num_params + 1):
                stdev[ii] = np.sqrt(stdev[ii] / len(frames))
            self.mean = vec
            self.stdev = stdev
        if normalize:
            for tt in range(len(frames)):
                for ii in range(self.num_params + 1):
                    mc[tt][ii] = (mc[tt][ii]-self.mean[ii]) / self.stdev[ii]
        return mc
    
    def synthesize(self, pitch, mc):
        b = sptk.mc2b(mc, self.alpha)
        synthesizer = Synthesizer(MLSADF(order=self.num_params, alpha=self.alpha), self.frame_len * self.sample_rate / 1000)
        source_excitation = sptk.excite(pitch, self.frame_len * self.sample_rate / 1000)
        x_synthesized = synthesizer.synthesis(source_excitation, b)
        return x_synthesized
    
    def save_normalization(self, output):
        f = open(output, "w")
        f.write("NUM_PARAMS " + str(self.num_params) + "\n")
        for val in self.mean:
            f.write(str(val) + " ")
        f.write("\n")
        for val in self.stdev:
            f.write(str(val) + " ")
        f.write("\n")
        
        f.close()
    
    def load_normalization(self, output):
        f = open(output, "r")
        lines = f.readlines()
        self.mean = []
        self.stdev = []
        parts1 = lines[1].replace("\n", "").split(" ")
        parts2 = lines[2].replace("\n", "").split(" ")
        for i in range (self.num_params + 1):
            self.mean.append(float(parts1[i]))
            self.stdev.append(float(parts2[i]))
        f.close()