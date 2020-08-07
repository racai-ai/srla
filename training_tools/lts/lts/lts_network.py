# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.


import dynet as dy


class Network:
    def __init__(self, dataset):
        self.dataset = dataset
        
        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model)
        
        self.enc_layers = 2
        self.dec_layers = 1
        self.enc_size = 80
        self.dec_size = 80
        self.char_lookup = self.model.add_lookup_parameters((len(self.dataset.grapheme2int), 32))
        self.encoder_fw = []
        self.encoder_bw = []
        self.encoder_fw.append(dy.CoupledLSTMBuilder(1, 32, self.enc_size, self.model))
        self.encoder_bw.append(dy.CoupledLSTMBuilder(1, 32, self.enc_size, self.model)) 
        
        for _ in range (self.enc_layers-1):
            self.encoder_fw.append(dy.CoupledLSTMBuilder(1, self.enc_size * 2, self.enc_size, self.model))
            self.encoder_bw.append(dy.CoupledLSTMBuilder(1, self.enc_size * 2, self.enc_size, self.model)) 
        
        
        self.decoder = dy.CoupledLSTMBuilder(self.dec_layers, self.enc_size * 2, self.dec_size, self.model)
        
        self.att_w1 = self.model.add_parameters((self.enc_size * 2, self.enc_size * 2))
        self.att_w2 = self.model.add_parameters((self.enc_size * 2, self.dec_size))
        self.att_v = self.model.add_parameters((1, self.enc_size * 2))
        
        self.softmaxW = self.model.add_parameters((len(self.dataset.phoneme2int), self.dec_size))
        self.softmaxB = self.model.add_parameters((len(self.dataset.phoneme2int)))
        
    def _attend(self, input_vectors, state):
        w1 = self.att_w1.expr()
        w2 = self.att_w2.expr()
        v = self.att_v.expr()
        
        attention_weights = []
        w2dt = w2 * state.h()[-1]
        for input_vector in input_vectors:
            attention_weight = v * dy.tanh(w1 * input_vector + w2dt)
            attention_weights.append(attention_weight)
            
            
        attention_weights = dy.softmax(dy.concatenate(attention_weights))
        output_vectors = dy.esum(
                                 [vector * attention_weight for vector, attention_weight in zip(input_vectors, attention_weights)])
        return output_vectors, attention_weights
    

    def learn(self, letters, phonemes):
        encoded = self._encode(letters)
        out, _ = self._decode(encoded, len(phonemes) + 1)
        losses = []
        index = 0
        for p in phonemes:
            
            if p in self.dataset.phoneme2int:
                loss = -dy.log(dy.pick(out[index], self.dataset.phoneme2int[p]))
                losses.append(loss)
            index += 1
        loss = -dy.log(dy.pick(out[index], 0))#end of sentence
        losses.append(loss)
        
        loss = dy.esum(losses)
        l = loss.value()
        loss.backward()
        self.trainer.update()
        return l
    
    
    def predict(self, letters):
        encoded = self._encode(letters)
        out, _ = self._decode(encoded)
        phonemes = []
        for o in out:
            p_index = self.argmax(o.value())
            if p_index != 0:
                phonemes.append(self.dataset.phonemelist[p_index])
        return phonemes
    
    def _encode(self, letters):
        x_list = []
        for l in letters:
            if l in self.dataset.grapheme2int:
                x_list.append(self.char_lookup[self.dataset.grapheme2int[l]])
        for i in range (self.enc_layers):
            fw_out = self.encoder_fw[i].initial_state().transduce(x_list)
            bw_out = list(reversed(self.encoder_bw[i].initial_state().transduce(reversed(x_list))))
            x_list = [dy.concatenate([f, b]) for f, b in zip (fw_out, bw_out)]
            
        return x_list
    
    def _decode(self, encoder_output, num_preds=None):
        output = []
        attention = []
        decoder_s = self.decoder.initial_state().add_input(dy.vecInput(self.enc_size*2))
        preds = 0
        while True:
            inp, att = self._attend(encoder_output, decoder_s)
            attention.append(att)
            decoder_s = decoder_s.add_input(inp)
            out = dy.softmax(self.softmaxW.expr() * decoder_s.output() + self.softmaxB.expr())
            output.append(out)
            
            preds += 1
            if num_preds != None and preds == num_preds:
                break
            if num_preds == None:
                if self.argmax(output[-1].value()) == 0 or preds > 100:
                    break
                
        return output, attention
    
    def argmax(self, data):
        max = 0
        for zz in range(1, len(data)):
            if data[zz] > data[max]:
                max = zz
                
        return max    