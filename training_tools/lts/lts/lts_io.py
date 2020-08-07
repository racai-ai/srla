
class Dataset:
    
    def __init__ (self, train, dev):
        self.train_file=train
        self.dev_file=dev
        
        self.train_examples=open(self.train_file, "r").readlines()
        self.dev_examples=open(self.dev_file, "r").readlines()
        
        self.phoneme2int={}
        self.phoneme2int["<EOS>"]=0
        self.grapheme2int={}
        self.phonemelist=[]
        self.phonemelist.append("<EOS>")
        for example in self.train_examples:
            letters, phonemes=self.process_example(example)
            for l in letters:
                if l not in self.grapheme2int:
                    self.grapheme2int[l]=len(self.grapheme2int)
            for p in phonemes:
                if p not in self.phoneme2int:
                    self.phoneme2int[p]=len(self.phoneme2int)
                    self.phonemelist.append(p)
                    
        print "Found "+str(len(self.grapheme2int))+" unique letters and "+str(len(self.phoneme2int))+" unique phonemes"
        print self.grapheme2int
        print self.phoneme2int
        
    
    def process_example(self, example):
        phonemes=[]
        l=example.replace("\n","")
        parts=l.split("\t")
        word=parts[0]
        if parts[1]=="":
            print example
        chars=self._get_characters(word)
        pp=parts[1].split(" ")
        for p in pp:
            phonemes.append(p)
        return chars, phonemes
    
    def _get_characters(self, word):
        chars = []
        uniword = unicode(word, 'utf-8')
        for i in range(len(uniword)):
            char = uniword[i].encode("utf-8")
            chars.append(char)
            
        return chars
            
        
