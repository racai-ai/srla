/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package sphinx;

import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.OutputStreamWriter;
import java.util.List;

import edu.cmu.sphinx.api.SpeechAligner;
import edu.cmu.sphinx.util.TimeFrame;
import edu.cmu.sphinx.result.WordResult;
import edu.cmu.sphinx.linguist.acoustic.Unit;
import io.WavFile;
import java.io.BufferedWriter;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Set;
import java.util.TreeSet;
import language.LTS;
import language.SimpleLTS;

public final class Aligner {

    public Aligner(String model_base, String wav_file, String transcript_filename, String output_base) throws Exception {
        // audio and transcript file paths
        File file = new File(wav_file);
        File transcript_file = new File(transcript_filename);

        // read transcript from file
        BufferedReader transcript_file_reader = new BufferedReader(new FileReader(transcript_file));
        StringBuffer transcript = new StringBuffer();
        List<String> origList = new ArrayList<>();
        List<String> labList = new ArrayList<>();

        String line = null;
        while ((line = transcript_file_reader.readLine()) != null) {
            String lab = makeLab(line.replace("\r", ""));
            if (!lab.trim().isEmpty()) {
                transcript.append(lab).append("\n");
                origList.add(line);
                labList.add(lab);
            }
        }
        double[] wav_data = WavFile.readWave(wav_file);
        //WavFile.save2Wave(wav_data, 16000, new DataOutputStream(new FileOutputStream("test.wav")));
        //System.exit(0);

        // perform alignment
        System.out.println("Performing initial HMM alignment. This will take a while...");
        String tmp_dict = createTemporaryDictionary(transcript_filename, model_base);
        SpeechAligner aligner = new SpeechAligner(model_base, tmp_dict, null);

        List<WordResult> results = aligner.align(file.toURI().toURL(), transcript.toString());
        System.out.println("Performing DTW word-level alignments");
        int[] alignments = dtw(results, labList);
        // write out results
        //CSVWriter writer = new CSVWriter(new OutputStreamWriter(System.out), ',');
        String lab = "";
        String txt = "";
        long start = 0;
        long stop = 0;
        System.out.println("Writing alignments");
        int ofs = 0;
        for (int i = 0; i < labList.size(); i++) {
            String out_base = output_base + "_" + String.format("%04d", i);
            String[] parts = labList.get(i).trim().split(" ");
            writeAlign(out_base, wav_data, results, ofs, parts, origList.get(i), labList.get(i), alignments);
            ofs += parts.length;
            //ofs += writeAlign(out_base, wav_data, results, ofs, parts, origList.get(i), labList.get(i));
        }
    }

    String makeLab(String text) {
        String newstr = text.replaceAll("[^A-Za-z ăîșțâĂÎȘȚÂ]+", "");
        return newstr.toUpperCase();
    }

    private String createTemporaryDictionary(String transcript_filename, String model_base) throws IOException {
        System.out.println("Creating dictionary for OOV words");
        System.out.print("\tLoading LTS model...");
        LTS lts = new SimpleLTS(model_base + "/lts.id3");
        System.out.println("done");
        System.out.print("\tCounting unique words...");
        BufferedReader br = new BufferedReader(new FileReader(transcript_filename));
        String line = null;
        Set<String> wordList = new TreeSet<>();
        while ((line = br.readLine()) != null) {
            String lab = makeLab(line);

            String[] parts = lab.split(" ");
            for (int i = 0; i < parts.length; i++) {
                if (!wordList.contains(parts[i])) {
                    wordList.add(parts[i]);
                }
            }
        }
        br.close();
        System.out.println(" found " + wordList.size() + " unique words");
        System.out.print("\tCreating transcription lexicon...");
        BufferedWriter bw = new BufferedWriter(new FileWriter("tmp.dict"));
        for (String word : wordList) {
            String pt = lts.getTranscription(word);
            bw.write(word.toLowerCase() + "\t" + pt.toUpperCase() + "\n");
        }
        bw.close();
        System.out.println("done");
        return "tmp.dict";
    }

    private int writeAlign(String out_base, double[] wav_data, List<WordResult> results, int ofs, String[] parts, String txt, String lab, int[] align) throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter(out_base + ".txt"));
        bw.write(txt + "\n");
        bw.close();
        bw = new BufferedWriter(new FileWriter(out_base + ".lab"));
        bw.write(lab + "\n");
        bw.close();
        int skip = 0;
        long start = results.get(align[ofs]).getTimeFrame().getStart();

        long stop = results.get(align[ofs + parts.length - 1]).getTimeFrame().getEnd();
        long start_sample = start * 16;
        long stop_sample = stop * 16;
        double[] data = new double[(int) (stop_sample - start_sample + 1)];
        for (int zz = 0; zz < data.length; zz++) {
            data[zz] = wav_data[zz + (int) start_sample];
        }
        WavFile.save2Wave(data, 16000, new DataOutputStream(new FileOutputStream(out_base + ".wav")));
        return 0;
    }

    private int[] dtw(List<WordResult> results, List<String> labList) {
        String[] rWords = new String[results.size()];
        List<String> tmp = new ArrayList<>();
        for (int i = 0; i < labList.size(); i++) {
            String[] parts = labList.get(i).split(" ");
            for (int j = 0; j < parts.length; j++) {
                tmp.add(parts[j].toLowerCase());
            }
        }
        String[] pWords = tmp.toArray(new String[0]);
        for (int i = 0; i < results.size(); i++) {
            rWords[i] = results.get(i).getWord().toString().toLowerCase();
        }

        int a[][] = new int[rWords.length + 1][pWords.length + 1];
        for (int i = 0; i < a.length; i++) {
            a[i][0] = i;
        }
        for (int i = 0; i < a[0].length; i++) {
            a[0][i] = i;
        }

        for (int i = 1; i < a.length; i++) {
            for (int j = 1; j < a[0].length; j++) {
                int cost = 0;
                if (!pWords[j - 1].equals(rWords[i - 1])) {
                    cost = 1;
                }
                int min = a[i - 1][j - 1];
                if (min > a[i - 1][j]) {
                    min = a[i - 1][j];
                }
                if (min > a[i][j - 1]) {
                    min = a[i][j - 1];
                }
                a[i][j] = cost + min;
            }
        }

        int[] rez = new int[pWords.length];
        int i = a.length - 1;
        int j = a[0].length - 1;
        while (i > 1 || j > 1) {
            rez[j - 1] = i - 1;
            if (i == 1) {
                j--;
            } else if (j == 1) {
                i--;
            } else {
                if (a[i - 1][j - 1] < a[i - 1][j] && a[i - 1][j - 1] < a[i][j - 1]) {
                    i--;
                    j--;
                } else if (a[i - 1][j] < a[i - 1][j - 1] && a[i - 1][j] < a[i][j - 1]) {
                    i--;
                } else {
                    j--;
                }
            }
        }
        rez[j - 1] = i - 1;
        return rez;
    }
}
