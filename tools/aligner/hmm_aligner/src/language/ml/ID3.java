/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package language.ml;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;


/**
 *
 * @author adrian
 */
public class ID3 {

    //<editor-fold defaultstate="collapsed" desc="Utilities">
    /**
     *
     */
    public static boolean verbose = false;
    private static int cutOff = 0;
    private static long countTotal = -1;
    private static long countDone = 0;
    private static long oldPercentShow = 0;

    /**
     * Reseteaza contorul pentru antrenare
     */
    private static void reset() {
        countTotal = -1;
        countDone = 0;
        oldPercentShow = -1;
    }

    /**
     * Afiseaza progresul
     *
     * @param cnt
     */
    private static void reportProgress(long cnt) {
        if (!verbose) {
            return;
        }
        if (countDone == 0) {
            System.out.print("%0");
        }
        long p1 = 100 * countDone / countTotal;
        countDone += cnt;
        long p2 = 100 * countDone / countTotal;
        for (long i = p1 + 1; i <= p2; i++) {
            if (i % 10 == 0) {
                System.out.print(i);
                if (i % 50 == 0) {
                    //System.out.println();
                }
            } else if (i % 5 == 0) {
                System.out.print("|");
            } else {
                System.out.print(".");
            }
        }
        if (countDone == countTotal) {
            System.out.println("%");
            reset();
        }
    }

    //</editor-fold>
    String field, value;
    ID3 yes;
    ID3 no;
    double count;
    double entropy;

    private double getEntropy(TrainData dataSet) {
        double count = 0;
        for (FeatureSets fSets : dataSet.values()) {
            count += fSets.count;
        }

        double h = 0;
        for (FeatureSets fSets : dataSet.values()) {
            double px = fSets.count / count;
            h -= px * Math.log(px) / Math.log(2);
        }
        return h;
    }

    /**
     *
     * @param train
     * @param cutOff
     */
    public ID3(TrainData train, int cutOff) {
        this.cutOff = cutOff;
        Split(train);
    }

    /**
     *
     * @param train
     */
    public ID3(TrainData train) {
        Split(train);
    }

    private void buildLeaf(TrainData train) {
        field = null;
        yes = null;
        no = null;

        String[] result = new String[train.keySet().size()];
        int pos = 0;
        for (String key : train.keySet()) {
            result[pos] = key;
            pos++;
        }
        boolean ok = false;
        while (!ok) {
            ok = true;
            for (int i = 0; i < result.length - 1; i++) {
                if (train.get(result[i]).count < train.get(result[i + 1]).count) {
                    ok = false;
                    String tmp = result[i];
                    result[i] = result[i + 1];
                    result[i + 1] = tmp;
                }
            }
        }

        value = "";
        for (int i = 0; i < result.length; i++) {
            if (i != 0) {
                value += " ";
            }
            value += result[i] + " " + (train.get(result[i]).count / count);
        }
    }

    private void Split(TrainData train) {
        // Calcul count
        count = 0;
        for (FeatureSets fSets : train.values()) {
            count += fSets.count;
        }

        // Extragere feature-uri distincte
        List<String> features = new ArrayList<String>();
        for (String key : train.keySet()) {
            for (FeatureSet fSet : train.get(key).getSets()) {
                for (String feat : fSet.getSet()) {
                    if (!features.contains(feat)) {
                        features.add(feat);
                    }
                }
            }
        }

        if (countTotal == -1) {
            countTotal = (long) count;
            if (verbose) {
                System.out.println(countTotal + " entries");
                System.out.println(features.size() + " features");
            }
        }

        if (count < cutOff) {
            buildLeaf(train);
            reportProgress((long) count);
            return;
        }

        if (train.keySet().size() == 1) {
            buildLeaf(train);
            reportProgress((long) count);
            return;
        }

        if (features.size() == 0) {
            buildLeaf(train);
            reportProgress((long) count);
            return;
        }

        entropy = getEntropy(train);

        // Determinare feature cu gain maxim
        TrainData bestYesSet = new TrainData();
        TrainData bestNoSet = new TrainData();
        String bestFeature = "";
        double bestGain = -1, bestCountYes = 0, bestCountNo = 0;

        Map<String, List<List<String>>> tmp;
        for (String feature : features) {

            // Impartire in yes si no
            TrainData featYesSet = new TrainData();
            TrainData featNoSet = new TrainData();
            for (String key : train.keySet()) {
                for (FeatureSet set : train.get(key).getSets()) {
                    FeatureSet newSet = new FeatureSet(set.getCount());
                    for (String feat : set.getSet()) {
                        if (!feat.equals(feature)) {
                            newSet.add(feat);
                        }
                    }

                    if (set.contains(feature)) {
                        if (!featYesSet.containsKey(key)) {
                            featYesSet.put(key, new FeatureSets());
                        }
                        featYesSet.get(key).add(newSet);
                    } else {
                        if (!featNoSet.containsKey(key)) {
                            featNoSet.put(key, new FeatureSets());
                        }
                        featNoSet.get(key).add(newSet);
                    }
                }
            }
            // Calcul count
            double countYes = 0;
            for (FeatureSets fSets : featYesSet.values()) {
                countYes += fSets.count;
            }
            double countNo = 0;
            for (FeatureSets fSets : featNoSet.values()) {
                countNo += fSets.count;
            }

            double gain = entropy - countYes / count * getEntropy(featYesSet) - countNo / count * getEntropy(featNoSet);

            if ((countYes > cutOff || countYes == 0) && (countNo > cutOff || countNo == 0)) {
                if (bestFeature == null || gain > bestGain) {
                    bestGain = gain;
                    bestFeature = feature;
                    bestYesSet = featYesSet;
                    bestNoSet = featNoSet;
                    bestCountYes = countYes;
                    bestCountNo = countNo;
                }
            }
        }

        if (bestGain == -1) {
            buildLeaf(train);
            reportProgress((long) count);
            return;
        }

        if (bestCountYes * bestCountNo == 0) {
            buildLeaf(train);
            reportProgress((long) count);
            return;
        }

        field = bestFeature;
        value = null;
        yes = new ID3(bestYesSet);
        no = new ID3(bestNoSet);
    }

    /**
     *
     * @param feats
     * @return
     */
    public String classify(List<String> feats) {
        if (field == null) {
            if (value == "") {
                System.err.println("Naspa");
            }
            return value;
        }
        if (feats.contains(field)) {
            return yes.classify(feats);
        } else {
            return no.classify(feats);
        }
    }

    //  <editor-fold defaultstate="collapsed" desc="Save and Load ID3">
    /**
     *
     * @param fileName
     * @return
     * @throws FileNotFoundException
     * @throws IOException
     */
    public static ID3 createFromFile(String fileName) throws FileNotFoundException, IOException {
        BufferedReader br = new BufferedReader(new FileReader(fileName));
        ID3 id3 = new ID3(br);
        br.close();
        return id3;
    }

    public static void shrink(String fileName) throws FileNotFoundException, IOException{
        List<String[]> model = new ArrayList<String[]>();
        BufferedReader br = new  BufferedReader(new FileReader(fileName));
        for(String line=br.readLine();line!=null;line=br.readLine()){
            String[] toks=line.split("[ \t]");
            if(toks.length>1){
                model.add(new String[]{toks[0], toks[1]});
            }
        }
        br.close(); 
        
        BufferedWriter bw = new BufferedWriter(new FileWriter(fileName));
        for(String[] toks: model){
            bw.write(toks[0]+"\t"+toks[1]+"\n");
        }
        bw.close();         
    }
    /**
     *
     * @param fileName
     * @param cutOff
     * @param verbose
     * @return
     */
    public static ID3 trainFromFile(String fileName, int cutOff, boolean verbose) {
        TrainData train = new TrainData();
        try {
            BufferedReader br = new BufferedReader(new FileReader(fileName));

            String line;
            long rows = 0;
            boolean end = false, endPack = false;
            do {
                line = br.readLine();
                if (line == null || line.length() == 0) {
                    if (endPack) {
                        if (verbose) {
                            System.out.println(rows + " rows");
                        }
                        end = true;
                        train.buildFeatureSets();

                    } else {
                        endPack = true;
                    }
                } else {
                    if (endPack) {
                        endPack = false;
                    }
                    if (rows != 0 && rows % 100000 == 0) {
                        if (verbose) {
                            System.out.println(rows + " rows ...");
                        }
                    }
                    rows++;
                    train.addInput(line);

                }

            } while (!end);
            if (verbose) {
                System.out.println(train.size() + " outputs");
            }

        } catch (FileNotFoundException ex) {
            //Logger.getLogger(Classifiers.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            //Logger.getLogger(Classifiers.class.getName()).log(Level.SEVERE, null, ex);
        }

        ID3.verbose = verbose;
        ID3 id3 = new ID3(train, cutOff);
//        try {
//            id3.viterbi = new Viterbi(fileName);
//        } catch (IOException ex) {
//            Logger.getLogger(ID3O.class.getName()).log(Level.SEVERE, null, ex);
//        }
        return id3;
    }

    private ID3(BufferedReader br) throws IOException {
        String line = br.readLine();
        String[] tokens = line.split("\t");
        if (tokens[0].equals("N")) {
            field = tokens[1];
            value = null;
        } else {
            field = null;
            value = tokens[1];
        }
        if (tokens.length > 2) {
            count = Double.parseDouble(tokens[2]);
            entropy = Double.parseDouble(tokens[3]);
        }
        if (tokens[0].equals("N")) {
            yes = new ID3(br);
            no = new ID3(br);
        }
    }
    
    /**
     *
     * @param out
     * @throws IOException
     */
    public void saveModel(PrintStream out) throws IOException {
        if (field != null) {
            out.print("N\t" + field);
        } else {
            out.print("L\t" + value);
        }
        out.print("\t" + count);
        out.println("\t" + entropy);
        if (yes != null) {
            yes.saveModel(out);
            no.saveModel(out);
        }
    }

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Save to dot file">
    private String name;
    private static int next = 0;

    /**
     *
     * @param stream
     * @throws IOException
     */
    public void saveGraph(PrintStream stream) throws IOException {
        next = 0;
        List<String> nodes = new ArrayList<String>();
        List<String> edges = new ArrayList<String>();
        getDotItems(nodes, edges);
        stream.println("digraph id3 {\n");
        stream.println("\tnode [shape=record];");
        for (String node : nodes) {
            stream.println("\t" + node + ";");
        }
        for (String edge : edges) {
            stream.println("\t" + edge + ";");
        }
        stream.print("}");
    }

    private void getDotItems(List<String> nodes, List<String> edges) {
        next++;
        name = "n" + next;
        long cnt = (long) count;
        String entr = String.format("%.3f", entropy);
        if (field == null) { // frunza
            String[] tokens = value.split(" ");
            String rez = "";
            String color = "green";
            if (tokens.length == 2) {
                rez = tokens[0] + " 100%";
            } else {
                for (int i = 0; i < tokens.length - 1; i += 2) {
                    if (i != 0) {
                        rez = rez + "|";
                    }
                    rez = rez + "{ " + tokens[i] + " | " + String.format("%d", Math.round(100 * Double.parseDouble(tokens[i + 1]))) + "% }";
                }
                color = "yellow";
            }

            nodes.add(name + " [label=\"{ { #" + cnt + " | " + entr + " } | { " + rez + " } }\", style=\"filled\" fillcolor=\"" + color + "\"]");
        } else {
            nodes.add(name + " [label=\"{ { #" + cnt + " | " + entr + " } | " + field.replace("\"", "\\\"") + " }\"]");
            yes.getDotItems(nodes, edges);
            no.getDotItems(nodes, edges);
            edges.add(name + " -> " + yes.name + " [label=\"Yes\"]");
            edges.add(name + " -> " + no.name + " [label=\"No\"]");
        }
    }
    //</editor-fold>
}
