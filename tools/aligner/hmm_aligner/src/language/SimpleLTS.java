/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package language;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import language.ml.ID3;

/**
 *
 * @author tibi
 */
public class SimpleLTS extends LTS {

    ID3 id3 = null;

    public SimpleLTS(String model) throws IOException {
        id3 = ID3.createFromFile(model);
    }

    @Override
    public String getTranscription(String word) {
        List<String> feats = new ArrayList<>();
        String[] phonemes = new String[word.length()];
        String plab = "_";
        for (int k = 0; k < phonemes.length; k++) {
            int ndxP1 = k - 1;
            while (ndxP1 >= 0 && word.charAt(ndxP1) == '-') {
                ndxP1--;
            }

            int ndxP2 = ndxP1 - 1;
            while (ndxP2 >= 0 && word.charAt(ndxP2) == '-') {
                ndxP2--;
            }

            int ndxN1 = k + 1;
            while (ndxN1 < word.length() && word.charAt(ndxN1) == '-') {
                ndxN1++;
            }

            int ndxN2 = ndxN1 + 1;
            while (ndxN2 < word.length() && word.charAt(ndxN2) == '-') {
                ndxN2++;
            }

            String ppl = "_";
            String pl = "_";
            String nl = "_";
            String nnl = "_";

            if (ndxP1 >= 0) {
                pl = word.substring(ndxP1, ndxP1 + 1);
            }
            if (ndxP2 >= 0) {
                ppl = word.substring(ndxP2, ndxP2 + 1);
            }

            String cl = word.substring(k, k + 1);

            if (ndxN1 < word.length()) {
                nl = word.substring(ndxN1, ndxN1 + 1);
            }
            if (ndxN2 < word.length()) {
                nnl = word.substring(ndxN2, ndxN2 + 1);
            }

            feats.clear();
            if (!cl.equals("-")) {
                feats.add("ppl:" + ppl.toLowerCase());
                feats.add("pl:" + pl.toLowerCase());
            }
            feats.add("cl:" + cl.toLowerCase());
            if (!cl.equals("-")) {
                feats.add("nl:" + nl.toLowerCase());
                feats.add("nnl:" + nnl.toLowerCase());
                feats.add("plab:" + plab);
            }
            String phon = id3.classify(feats).split(" ")[0];
            plab = phon;
            if (cl.equals("-")) {
                phonemes[k] = "-";
            } else {
                phonemes[k] = phon;
            }
        }
        StringBuilder trans = new StringBuilder();
        for (int i = 0; i < phonemes.length; i++) {
            trans.append(phonemes[i].replace("-", "").replace(".", " "));
            trans.append(" ");
        }
        return trans.toString().trim().replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("@", "AI");
    }
}
