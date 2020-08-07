# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

from ctc import CTC
import dynet as dy
import numpy as np
import matplotlib.pyplot as plt
class Network:
    def __init__(self, config, dataset, vocoder):
        self.config = config
        self.dataset = dataset
        self.vocoder = vocoder
        self.ctc = CTC()
        
        self.model = dy.Model()
        self.trainer = dy.SimpleSGDTrainer(self.model)
        
        #encoder
        self.encoder_fw = []
        self.encoder_bw = []
        self.encoder_fw.append(dy.LSTMBuilder(1, 3 * (self.config.vocoder_params + 1), self.config.encoder_size, self.model))
        self.encoder_bw.append(dy.LSTMBuilder(1, 3 * (self.config.vocoder_params + 1), self.config.encoder_size, self.model))
        
        for iLayer in range(1, config.encoder_layers):
            self.encoder_fw.append(dy.LSTMBuilder(1, self.config.encoder_size * 2, self.config.encoder_size, self.model))
            self.encoder_bw.append(dy.LSTMBuilder(1, self.config.encoder_size * 2, self.config.encoder_size, self.model))
        #decoder
        self.decoder = dy.LSTMBuilder(self.config.decoder_layers, self.config.encoder_size * 2, self.config.decoder_size, self.model)
        
        #attention
        self.att_w1 = self.model.add_parameters((config.encoder_size * 2, config.encoder_size * 2))
        self.att_w2 = self.model.add_parameters((config.encoder_size * 2, config.decoder_size))
        self.att_v = self.model.add_parameters((1, config.encoder_size * 2))
        
        #softmax
        self.softmaxW = self.model.add_parameters((len(self.dataset.phoneme2int), config.encoder_size * 2))
        self.softmaxB = self.model.add_parameters((len(self.dataset.phoneme2int)))
        
    def _plot_attention(self, matrix, max_weight=None, ax=None):
        """Draw Hinton diagram for visualizing a weight matrix."""
        m_val=[]
        for tmp in matrix:
            m_val.append(tmp.value())
        matrix=m_val
        ax = ax if ax is not None else plt.gca()

        if not max_weight:
            max_weight = 2**np.ceil(np.log(np.abs(matrix).max())/np.log(2))

        ax.patch.set_facecolor('gray')
        ax.set_aspect('equal', 'box')
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        for (x, y), w in np.ndenumerate(matrix):
            color = 'white' if w > 0 else 'black'
            size = np.sqrt(np.abs(w))
            rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                                 facecolor=color, edgecolor=color)
            ax.add_patch(rect)

        ax.autoscale_view()
        ax.invert_yaxis()
        plt.show()
        
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
    
    def argmax(self, data):
        max = 0
        for zz in range(1, len(data)):
            if data[zz] > data[max]:
                max = zz
                
        return max
    
    def _encode(self, input_list):
        x_list = []
        for iX in range (len(input_list)):
            x = dy.inputVector(input_list[iX])#dy.vecInput(3 * (self.config.vocoder_params + 1))
            #x.set(input_list[iX])
            x_list.append(x)
        
        for iLayer in range(self.config.encoder_layers):
            encoder_fw_out = self.encoder_fw[iLayer].initial_state().transduce(x_list)
            encoder_bw_out = list(reversed(self.encoder_bw[iLayer].initial_state().transduce(reversed(x_list))))
            
            x_list = [dy.concatenate([enc_fw, enc_bw]) for enc_fw, enc_bw in zip(encoder_fw_out, encoder_bw_out)]
            
        return x_list
    def _run_rnn(self, init_state, input_vecs):
        s = init_state

        states = s.add_inputs(input_vecs)
        rnn_outputs = [s.output() for s in states]
        return rnn_outputs
    
    def _encode_rnn(self, pre_ctc_output_list):
        x_lst = []
        for iX in range (len(pre_ctc_output_list)):
            #print pre_ctc_output_list[iX].value()
            #print ""
            #x = dy.vecInput(len(pre_ctc_output_list[iX].value()))
            #x=x.set(pre_ctc_output_list[iX].value())#dy.inputVector(pre_ctc_output_list[iX].value())#dy.nobackprop(pre_ctc_output_list[iX])
            x=pre_ctc_output_list[iX].value()
            x_lst.append(x)
        #trick - should work
        dy.renew_cg()
        x_list=[]
        for x in x_lst:
            x_list.append(dy.inputVector(x))
            
        for iLayer in range(self.config.rnn_encoder_layers):
            encoder_fw=self.rnn_encoder_fw[iLayer].initial_state()
            encoder_bw=self.rnn_encoder_bw[iLayer].initial_state()
            encoder_fw_out=self._run_rnn(encoder_fw, x_list)
            encoder_bw_out=self._run_rnn(encoder_bw, x_list[::-1])[::-1]
            #encoder_fw_out = self.rnn_encoder_fw[iLayer].initial_state().transduce(x_list)
            #encoder_bw_out = list(reversed(self.rnn_encoder_bw[iLayer].initial_state().transduce(reversed(x_list))))
            x_list = [dy.concatenate([enc_fw, enc_bw]) for enc_fw, enc_bw in zip(encoder_fw_out, encoder_bw_out)]        
        
        #subsampling
        new_x_list = []
        for iX in range (len(x_list)):
            if iX % self.config.rnn_subsample == 0:
                new_x_list.append(x_list[iX])
        return new_x_list
    
    
    def _decode_rnn(self, encoder_output, num_preds=None):
        output = []
        attention = []
        decoder_s = self.rnn_decoder.initial_state().add_input(dy.vecInput(self.config.rnn_encoder_size * 2))
        preds = 0
        while True:
            inp, att = self._attend(encoder_output, decoder_s)
            attention.append(att)
            decoder_s = decoder_s.add_input(inp)
            
            out = dy.softmax(self.rnnSoftmaxW.expr() * decoder_s.output() + self.rnnSoftmaxB.expr())
            output.append(out)
            
            preds += 1
            if num_preds != None and preds == num_preds:
                break
            if num_preds == None:
                if self.argmax(output[-1].value()) == 1:
                    break
        return output, attention
    
    def _decode(self, encoder_output, num_preds=None, use_softmax=True):
        output = []
        for zz in range (len(encoder_output)):
            out = self.softmaxW.expr() * encoder_output[zz] + self.softmaxB.expr()
            if use_softmax:
                out = dy.softmax(out)
            output.append(out)
        return output
    
    def _compute_ctc_loss(self, embeddings, labels, is_prob):
        cost, losses = self.ctc.compute_cost_np(embeddings, labels, is_prob)
        #print grad.shape
        
        return cost, losses
    
    def _add_dyn_coeff(self, spec):
        tmp = []
        for zz in range (len(spec)):
            frame = [0] * (len(spec[zz]) * 3)
            if zz > 0:
                last_frame = tmp[zz-1]
            else:
                last_frame = [0] * len(spec[zz]) * 2
            ofs = len(spec[zz])
            for tt in range (ofs):
                frame[tt] = spec[zz][tt]
                frame[tt + ofs] = spec[zz][tt]-last_frame[tt]
                frame[tt + ofs + ofs] = frame[tt + ofs]-last_frame[tt + ofs]
            tmp.append(frame)
        return tmp
    
    def learn(self, wave, label_list):
        spec = self.vocoder.extract_spectrum(wave, True)
        spec = self._add_dyn_coeff(spec)
        #print spec
        encoded = self._encode(spec)
        output = self._decode(encoded, len(encoded), True)#len(label_list))
        seq = []
        for iOut in range(len(label_list)):
            p_index = self.dataset.phoneme2int[label_list[iOut].phoneme]
            seq.append(p_index)
            #loss=-dy.log(dy.pick(output[iOut], p_index))
            #losses.append(loss)
        ctc_loss, losses = self._compute_ctc_loss(output, seq, True)
        
        #total_loss=dy.esum(losses)
        total_loss_val = ctc_loss.value()
        if not np.isnan(total_loss_val):
            ctc_loss.backward()
        self.trainer.update()
        return total_loss_val, ctc_loss.value() / len(output)
    
    def predict(self, wave, num_preds=None):
        spec = self.vocoder.extract_spectrum(wave, True)
        spec = self._add_dyn_coeff(spec)
        encoded = self._encode(spec)
        output = self._decode(encoded, num_preds)
        out, out_ts = self.ctc.decode(output, self.config.frame_len_ms)
        rez = []
        
        for zz in range (len(out)):
            index = out[zz]
            rez.append(self.dataset.phonemeList[index])
        return rez, out_ts
    
    def _dtw(self, ctc_out, labels):
        s1 = ctc_out
        s2 = [0] * len(labels)
        for zz in range (len(labels)):
            s2[zz] = self.dataset.phoneme2int[labels[zz].phoneme]
        
        a = np.zeros((len(ctc_out) + 1, len(labels) + 1))
        for i in range(a.shape[0]):
            a[i][0] = i
        for i in range (a.shape[1]):
            a[0][i] = i
        
        for i in range (1, a.shape[0]):
            for j in range (1, a.shape[1]):
                cost = 0
                if s1[i-1] != s2[j-1]:
                    cost = 1
                m = a[i-1][j-1]
                if a[i-1][j] < m:
                    m = a[i-1][j]
                if a[i][j-1] < m:
                    m = a[i][j-1]
                a[i][j] = m + cost
        align = [-1] * len(labels)
        i = a.shape[0]-1
        j = a.shape[1]-1
        #print "--------"
        #print s1
        #print s2
        #print len(s1), len(s2)
        while i > 1 or j > 1:
            #print i,j
            if s1[i-1] == s2[j-1]:
                align[j-1] = i-1
            if i == 1:
                j -= 1
            elif j == 1:
                i -= 1
            else:
                if a[i-1][j-1] <= a[i-1][j] and a[i-1][j-1] <= a[i][j-1]:
                    i -= 1
                    j -= 1
                elif a[i-1][j] <= a[i-1][j-1] and a[i-1][j] <= a[i][j-1]:
                    i -= 1
                else:
                    j -= 1
        if s1[i-1] == s2[j-1]:
            align[j-1] = i-1
        return align
        
    
    def learn_rnn(self, wave, label_list):
        spec = self.vocoder.extract_spectrum(wave, True)
        spec = self._add_dyn_coeff(spec)
        #print spec
        encoded_ctc = self._encode(spec)
        output_ctc = self._decode(encoded_ctc, len(encoded_ctc), True)
        #ctc_out, ctc_out_ts = self.ctc.decode(output_ctc, self.config.frame_len_ms) #for enforcing ctc alignments
        encoded_rnn = self._encode_rnn(encoded_ctc)
        #pairs = self._dtw(ctc_out, label_list) #align ctc output with actual labels
        output_rnn, attentions = self._decode_rnn(encoded_rnn, len(label_list))#len(label_list))
        
        #print ctc_out_ts
        #print pairs
        losses = []
        for iOut in range(len(label_list)):
            p_index = self.dataset.phoneme2int[label_list[iOut].phoneme]
            loss = -dy.log(dy.pick(output_rnn[iOut], p_index))
            losses.append(loss)
            #if pairs[iOut] != -1: #enforce alignments
            #    loss = -dy.log(dy.pick(attentions[iOut], ctc_out_ts[pairs[iOut]] / (self.config.frame_len_ms*self.config.rnn_subsample)))
            #    losses.append(loss*0.2)
        
        total_loss = dy.esum(losses)
        total_loss_val = total_loss.value()
        if not np.isnan(total_loss_val):
            total_loss.backward()
        self.trainer_rnn.update()
        return total_loss_val
    
    def predict_rnn(self, wave, num_preds=None):
        spec = self.vocoder.extract_spectrum(wave, True)
        spec = self._add_dyn_coeff(spec)
        encoded = self._encode(spec)
        #output = self._decode(encoded, len(encoded), True)
        encoded = self._encode_rnn(encoded)
        
        output, att_mat = self._decode_rnn(encoded, num_preds)
        rez = []
        
        #self._plot_attention(att_mat)
        
        for zz in range (len(output)):
            index = self.argmax(output[zz].value())
            rez.append(self.dataset.phonemeList[index])
        return rez
    
    def save_model(self, output_base):
        print "Storing", output_base
        self.config.save_config(output_base + ".config")
        self.vocoder.save_normalization(output_base + ".norm")
        self.dataset.save_lookups(output_base + ".encoding")
        self.model.save(output_base + ".network")
        
    def load_model_ctc(self, model):
        self.model.populate(model)
        #adding rnn decoding network
        self.model_rnn=dy.Model()
        self.trainer_rnn=dy.SimpleSGDTrainer(self.model_rnn)
        self.rnn_encoder_fw = []
        self.rnn_encoder_bw = []
        self.rnn_encoder_fw.append(dy.LSTMBuilder(1, self.config.encoder_size * 2, self.config.rnn_encoder_size, self.model_rnn))
        self.rnn_encoder_bw.append(dy.LSTMBuilder(1, self.config.encoder_size * 2, self.config.rnn_encoder_size, self.model_rnn))
        for zz in range (1, self.config.rnn_encoder_layers):
            self.rnn_encoder_fw.append(dy.LSTMBuilder(1, self.config.rnn_encoder_size * 2, self.config.rnn_encoder_size, self.model_rnn))
            self.rnn_encoder_bw.append(dy.LSTMBuilder(1, self.config.rnn_encoder_size * 2, self.config.rnn_encoder_size, self.model_rnn))
            
        
        self.att_w1 = self.model_rnn.add_parameters((self.config.rnn_encoder_size * 2, self.config.rnn_encoder_size * 2))
        self.att_w2 = self.model_rnn.add_parameters((self.config.rnn_encoder_size * 2, self.config.rnn_decoder_size))
        self.att_v = self.model_rnn.add_parameters((1, self.config.rnn_encoder_size * 2))
        #rnn decoder
        self.rnn_decoder = dy.LSTMBuilder(self.config.rnn_decoder_layers, self.config.rnn_encoder_size * 2, self.config.rnn_decoder_size, self.model_rnn)
        #rnn decoder softmax
        self.rnnSoftmaxW = self.model_rnn.add_parameters((len(self.dataset.phoneme2int), self.config.rnn_decoder_size))
        self.rnnSoftmaxB = self.model_rnn.add_parameters((len(self.dataset.phoneme2int)))
