# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

class Config:
    def __init__(self, config_file):
        self.encoder_size=400
        self.encoder_layers=3
        self.decoder_size=200
        self.decoder_layers=2
        self.vocoder_params=40
        self.frame_len_ms=10
        
        self.rnn_encoder_size=150
        self.rnn_encoder_layers=2
        self.rnn_decoder_size=400
        self.rnn_decoder_layers=2
        self.rnn_subsample=10 # that means 100 ms
        
        if config_file!=None:
            f=open(config_file, "r")
            lines=f.readlines()
            for line in lines:
                line=line.replace("\n","")
                parts=line.split(" ")
                if parts[0]=="ENCODER_SIZE":
                    self.encoder_size=int(parts[1])
                elif parts[0]=="ENCODER_LAYERS":
                    self.encoder_layers=int(parts[1])
                elif parts[0]=="DECODER_SIZE":
                    self.decoder_size=int(parts[1])
                elif parts[0]=="DECODER_LAYERS":
                    self.decoder_layers=int(parts[1])
                elif parts[0]=="VOCODER_PARAMS":
                    self.vocoder_params=int(parts[1])
                elif parts[0]=="FRAME_LEN_MS":
                    self.frame_len_ms=int(parts[1])
            f.close()
                
        
    def save_config(self, output):
        f=open(output, "w")
        f.write("ENCODER_SIZE "+str(self.encoder_size)+"\n")
        f.write("ENCODER_LAYERS "+str(self.encoder_layers)+"\n")
        f.write("DECODER_SIZE "+str(self.decoder_size)+"\n")
        f.write("DECODER_LAYERS "+str(self.decoder_layers)+"\n")
        f.write("VOCODER_PARAMS "+str(self.vocoder_params)+"\n")
        f.write("FRAME_LEN_MS "+str(self.frame_len_ms)+"\n")
        f.close()