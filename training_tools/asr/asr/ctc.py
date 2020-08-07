# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import numpy as np
np.seterr(divide='raise', invalid='raise')
import dynet as dy
import sys

class CTC:
    def compute_cost_np(self, params, seq, is_prob=False):
        """
        CTC loss function.
        params - n x m matrix of n-D probability distributions over m frames.
        seq - sequence of phone id's for given example.
        is_prob - whether params have already passed through a softmax
        Returns objective and gradient.
        """
        orig_params = params
        blank = 0
        seq = np.vstack(seq)
        np_params = np.zeros((len(params[0].value()), len(params)))
        for s in range(np_params.shape[0]):
            for t in range (np_params.shape[1]):
                np_params[s, t] = params[t].value()[s]
        
        params = np_params
        
        seqLen = seq.shape[0] # Length of label sequence (# phones)
        numphones = params.shape[0] # Number of labels
        L = 2 * seqLen + 1 # Length of label sequence with blanks
        T = params.shape[1] # Length of utterance (time)

        alphas = np.zeros((L, T))
        betas = np.zeros((L, T))

        # Keep for gradcheck move this, assume NN outputs probs
        if not is_prob:
            params = params - np.max(params, axis=0)
            params = np.exp(params)
            params = params / np.sum(params, axis=0)

        # Initialize alphas and forward pass 
        alphas[0, 0] = params[blank, 0]
        alphas[1, 0] = params[seq[0], 0]
        c = np.sum(alphas[:, 0])
        if c == 0:
            return dy.scalarInput(0), []

        alphas[:, 0] = alphas[:, 0] / c
        llForward = np.log(c)
        #print llForward
        for t in xrange(1, T):
            start = max(0, L-2 * (T-t)) 
            end = min(2 * t + 2, L)
            for s in xrange(start, L):
                l = (s-1) / 2
                # blank
                if s % 2 == 0:
                    if s == 0:
                        alphas[s, t] = alphas[s, t-1] * params[blank, t]
                    else:
                        alphas[s, t] = (alphas[s, t-1] + alphas[s-1, t-1]) * params[blank, t]
                # same label twice
                elif s == 1 or seq[l] == seq[l-1]:
                    alphas[s, t] = (alphas[s, t-1] + alphas[s-1, t-1]) * params[seq[l], t]
                else:
                    alphas[s, t] = (alphas[s, t-1] + alphas[s-1, t-1] + alphas[s-2, t-1]) \
                        * params[seq[l], t]

            # normalize at current time (prevent underflow)
            c = np.sum(alphas[start:end, t])
            alphas[start:end, t] = alphas[start:end, t] / c
            if c == 0:
                return dy.scalarInput(0), []
            llForward += np.log(c)
            #print llForward

        # Initialize betas and backwards pass
        betas[-1, -1] = params[blank, -1]
        betas[-2, -1] = params[seq[-1], -1]
        c = np.sum(betas[:, -1])
        if c == 0:
            return dy.scalarInput(0), []
        betas[:, -1] = betas[:, -1] / c
        llBackward = np.log(c)
        for t in xrange(T-2, -1, -1):
            start = max(0, L-2 * (T-t)) 
            end = min(2 * t + 2, L)
            for s in xrange(end-1, -1, -1):
                l = (s-1) / 2
                # blank
                if s % 2 == 0:
                    if s == L-1:
                        betas[s, t] = betas[s, t + 1] * params[blank, t]
                    else:
                        betas[s, t] = (betas[s, t + 1] + betas[s + 1, t + 1]) * params[blank, t]
                # same label twice
                elif s == L-2 or seq[l] == seq[l + 1]:
                    betas[s, t] = (betas[s, t + 1] + betas[s + 1, t + 1]) * params[seq[l], t]
                else:
                    betas[s, t] = (betas[s, t + 1] + betas[s + 1, t + 1] + betas[s + 2, t + 1]) \
                        * params[seq[l], t]

            c = np.sum(betas[start:end, t])
            betas[start:end, t] = betas[start:end, t] / c

            if c == 0:
                return dy.scalarInput(0), []
            llBackward += np.log(c)
        
        print "Debug:"
        print "\tFORWARD LOSS=", llForward
        print "\tBACKWARD LOSS=", llBackward
        # Compute gradient with respect to unnormalized input parameters
        grad = np.zeros(params.shape)
        ab = alphas * betas
        for s in xrange(L):
            # blank
            if s % 2 == 0:
                grad[blank, :] += ab[s, :]
                ab[s, :] = ab[s, :] / params[blank, :]
            else:
                grad[seq[(s-1) / 2], :] += ab[s, :]
                ab[s, :] = ab[s, :] / (params[seq[(s-1) / 2], :]) 
        absum = np.sum(ab, axis=0)

        # Check for underflow or zeros in denominator of gradient
        llDiff = np.abs(llForward-llBackward)
        if llDiff > 1e-5 or np.sum(absum == 0) > 0:
            print "Diff in forward/backward LL : %f" % llDiff
            print "Zeros found : (%d/%d)" % (np.sum(absum == 0), absum.shape[0])
            return dy.scalarInput(0), []

        division = (params * absum)
        for s in range (division.shape[0]):
            for t in range (division.shape[1]):
                if np.isnan(division[s, t]) or np.isinf(division[s, t]) or division[s, t] == 0:
                    print "Division error"
                    return dy.scalarInput(0), []
        grad = params - grad / division
        
        losses = []
        last_proc = 0
        numLabels = params.shape[0]
        for t in range (T):
            proc = t * 100 / T
            if proc % 5 == 0 and proc != last_proc:
                last_proc = proc
                sys.stdout.write(" " + str(proc))
                sys.stdout.flush()
                
            #print "P=", params[t].value()
            dp = [0] * numLabels
            for l in range (numLabels):
                dp[l] = orig_params[t].value()[l] - grad[l, t]
            
            #print "OP:", orig_params[t].value()
            #print "DP:", dp
            target = dy.inputVector(dp)
            
            losses.append(self._compute_loss(orig_params[t], target, numLabels))
            #target=dy.vecInput(numLabels)
            #target.set(dp)
            #targets.append(target)
            #losses.append(dy.pairwise_rank_loss(orig_params[t], target))
            #losses.append(-dy.log(dy.pick(orig_params[t], self.argmax(dp))))
            #losses.append(-dy.esum(grad_v[t]))
            #print "DP=", dp
        ctc_loss = dy.esum(losses)
        print "\tCTC=", ctc_loss.value()

        return ctc_loss, losses
    
    def _compute_loss(self, p, t, m):
        loss = dy.scalarInput(0)
        for zz in range (m):
            loss += t[zz] * dy.log(p[zz]) + (1.0-t[zz]) * dy.log(1.0-p[zz])
            #loss += dy.pow(t[zz]-p[zz], dy.scalarInput(2))
        return -loss / m
    
    def compute_cost(self, params, seq, is_prob=False):
        """
        CTC loss function.
        params - n x m matrix of n-D probability distributions over m frames.
        seq - sequence of phone id's for given example.
        is_prob - whether params have already passed through a softmax
        Returns objective and gradient.
        """
        blank = 0
        seqLen = len(seq) # Length of label sequence (# phones)
        L = 2 * seqLen + 1 # Length of label sequence with blanks
        T = len(params) # Length of utterance (time)
        numLabels = len(params[0].value())
        #print "T=", T
        
        if not is_prob:
            for t in range (T):
                max_elem = dy.max_dim(params[t], 0)
                #print "max_elem=", max_elem.value()
                pmax = max_elem
                params[t] = dy.exp(params[t]-pmax)
                params[t] = params[t] * dy.pow(dy.sum_elems(params[t]), dy.scalarInput(-1))
            #params = params - dy.max_dim(params,d=0)
            #params = dy.exp(params)
            #params = params / dy.sum(params,axis=0)
        
       
        alphas = [[dy.scalarInput(0)] * L for xx in range(T)]
        betas = [[dy.scalarInput(0)] * L for xx in range(T)]
        #alphas = np.zeros((L,T))
        #betas = np.zeros((L,T))

        # Initialize alphas and forward pass 
        alphas[0][0] = dy.pick(params[0], blank)#params[blank,0]
        alphas[0][1] = dy.pick(params[0], seq[0])#params[seq[0],0]
        c = alphas[0][0] + alphas[0][1]#np.sum(alphas[:,0])
        c_n = dy.pow(c, dy.scalarInput(-1))
        alphas[0][0] = alphas[0][0] * c_n
        alphas[0][1] = alphas[0][1] * c_n
        
        llForward = dy.log(c)#np.log(c)
        #print llForward.value()
        for t in xrange(1, T):
            start = 2 * (T-t)
            if L <= start:
                start = 0
            else:
                start = L-start
            end = min(2 * t + 2, L)
            for s in xrange(start, L):
                l = (s-1) / 2
                # blank
                if s % 2 == 0:
                    if s == 0:
                        alphas[t][s] = alphas[t-1][s] * dy.pick(params[t], blank)
                    else:
                        alphas[t][s] = (alphas[t-1][s] + alphas[t-1][s-1]) * dy.pick(params[t], blank)
                # same label twice
                elif s == 1 or seq[l] == seq[l-1]:
                    alphas[t][s] = (alphas[t-1][s] + alphas[t-1][s-1]) * dy.pick(params[t], seq[l])
                else:
                    alphas[t][s] = (alphas[t-1][s] + alphas[t-1][s-1] + alphas[t-1][s-2]) * dy.pick(params[t], seq[l])

            c = dy.esum(alphas[t])
            c_n = dy.pow(c, dy.scalarInput(-1))
            for tt in range(start, end):
                alphas[t][tt] = alphas[t][tt] * c_n
            llForward += dy.log(c)
            #print llForward.value()
            #print "t=", t, "llForward=", llForward.value()
        print "Debug:"
        print "\tFORWARD LOSS=", llForward.value()
        # Initialize betas and backwards pass
        betas[T-1][L-1] = dy.pick(params[T-1], blank)
        betas[T-1][L-2] = dy.pick(params[T-1], seq[seqLen-1])
        c = betas[T-1][L-1] + betas[T-1][L-2]
        c_n = dy.pow(c, dy.scalarInput(-1))
        betas[T-1][L-1] *= c_n
        betas[T-1][L-2] *= c_n
        llBackward = dy.log(c)
        #print "BACKWARD pass:"
        for t in xrange(T-1, 0, -1):
            t = t-1
            start = 2 * (T-t)
            if L <= start:
                start = 0
            else:
                start = L-start
            end = min(2 * t + 2, L)
            for s in xrange(end, 0, -1):
                s = s-1
                l = (s-1) / 2
                if s % 2 == 0:
                    if s == L-1:
                        betas[t][s] = betas[t + 1][s] * dy.pick(params[t], blank)
                    else:
                        betas[t][s] = (betas[t + 1][s] + betas[t + 1][s + 1]) * dy.pick(params[t], blank)
                # same label twice
                elif s == L-2 or seq[l] == seq[l + 1]:
                    betas[t][s] = (betas[t + 1][s] + betas[t + 1][s + 1]) * dy.pick(params[t], seq[l])
                else:
                    betas[t][s] = (betas[t + 1][s] + betas[t + 1][s + 1] + betas[t + 1][s + 2]) * dy.pick(params[t], seq[l])

            #c = np.sum(betas[start:end, t])
            c = dy.esum(betas[t])
            c_n = dy.pow(c, dy.scalarInput(-1))
            for tt in range (L):
                betas[t][tt] = betas[t][tt] * c_n
            #betas[start:end, t] = betas[start:end, t] / c
            llBackward += dy.log(c)
            #print "t=", t, "llBackward=", llBackward.value()
        #alpha-beta
        print "\tBACKWARD LOSS=", llBackward.value()
        ab = []
        for tt in range (T):
            ab_row = []
            for ll in range(L):
                ab_row.append(alphas[tt][ll] * betas[tt][ll])#*dy.pow(params[tt][ll],dy.scalarInput(-1)))
            ab.append(ab_row)
        ##PHASE 1
        grad_v = [[dy.scalarInput(0)] * numLabels for xx in range (T)]
        for s in xrange(L):
            # blank
            if s % 2 == 0:
                for t in xrange(T):
                    grad_v[t][blank] += ab[t][s]
                    if ab[t][s] != 0:
                        ab[t][s] = ab[t][s] * dy.pow(params[t][blank], dy.scalarInput(-1))
            else:
                for t in xrange(T):
                    grad_v[t][seq[(s-1) / 2]] += ab[t][s]
                    if ab[t][s] != 0:
                        ab[t][s] = ab[t][s] * dy.pow((params[t][seq[(s-1) / 2]]), dy.scalarInput(-1))
        #PHASE 2
        absum = [dy.scalarInput(0)] * T
        for t in xrange(T):
            for s in xrange(L):
                absum[t] += ab[t][s]
        #phase 3
        eps = dy.scalarInput(0.00001)
        for t in xrange(T):
            for s in xrange(numLabels):
                tmp = params[t][s] * absum[t]
                #print tmp.value()
                if tmp > 0:
                    grad_v[t][s] = params[t][s] - grad_v[t][s] * dy.pow(tmp + eps, dy.scalarInput(-1))
                else:
                    grad_v[t][s] = params[t][s]
        #for dynet backprop
        losses = []
        last_proc = 0
        for t in range (T):
            proc = t * 100 / T
            if proc % 5 == 0 and proc != last_proc:
                last_proc = proc
                sys.stdout.write(" " + str(proc))
                sys.stdout.flush()
                
            #print "P=", params[t].value()
            dp = [0] * numLabels
            for l in range (numLabels):
                dp[l] = params[t].value()[l] - grad[t][l].value()
            
            target = dy.inputVector(dp)
            #target.set(dp)
            #targets.append(target)
            losses.append(dy.squared_distance(params[t], target))
            #losses.append(-dy.esum(grad_v[t]))
            #print "DP=", dp
        ctc_loss = dy.esum(losses)
        print "\tCTC=", ctc_loss.value()
        return ctc_loss 
    
    def argmax(self, data):
        max = 0
        for zz in range(1, len(data)):
            if data[zz] > data[max]:
                max = zz
                
        return max
    
    def decode (self, output, frame_len, return_sequences=False):
        out = []
        out_ts = []
        blank = 0
        last_out = -1
        seq = []
        for zz in range (len(output)):
            p_index = self.argmax(output[zz].value())
            #print output[zz].value()
            if p_index != last_out:
                if p_index != blank:
                    out.append(p_index)
                    out_ts.append(zz * frame_len)
                    seq.append(output[zz])
            last_out = p_index
        if return_sequences:
            return out, out_ts, seq
        else:
            return out, out_ts
    
    
