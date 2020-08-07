/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package language.ml;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 *
 * @author adrian
 */
public class TrainData extends HashMap<String, FeatureSets> {

    /**
     * <b>Arbore de caracteristici (cu greutati in noduri)</b>
     */
    public class Tree extends HashMap<String, Tree> {

        /**
         * <b>Numar de caracteristici din pachet</b>
         */
        public int count = 0;

        /**
         * <b>Adauga un nou set de caracteristici in arbore </b>
         *
         * @param fSet <b>Un set de caracteristici</b>
         */
        public void add(FeatureSet fSet) {
            if(fSet.size()==0){
                count+=fSet.count;
            } else {
            
                fSet.sort();
                String key = fSet.getFirst();
                fSet.removeFirst();

                if (!containsKey(key)) {
                    put(key, new Tree());
                }
                get(key).add(fSet);
            }
        }

        /**
         * Parcurge arborele si intoarce listele de caracteristici</b>
         *
         * @return
         */
        FeatureSets listFSets() {
            FeatureSets result = new FeatureSets();
            if (count != 0) {
                FeatureSet newfSet = new FeatureSet(count);
                result.add(newfSet);
            } 
            for (String key : keySet()) {
                for (FeatureSet entry : get(key).listFSets().getSets()) {
                    entry.addFirst(key);
                    result.add(entry);
                }
            }
            return result;
        }
    }

    /**
     * Arbori cu iesire(key) si caracteristici (tree)
     */
    private class TreeMap extends HashMap<String, Tree> {

        // Sparge 
        public boolean add(String entry) {
            String[] toks = entry.split("[\t ]");
            if (toks.length == 0) {
                return false;
            }

            FeatureSet fset = new FeatureSet(1);
            for (int i = 0; i < toks.length; i++) {
                if (toks[i].length() > 0) {
                    fset.add(toks[i]);
                }
            }
            String key = fset.getFirst();
            fset.removeFirst();
            fset.sort();

            if (!containsKey(key)) {
                put(key, new Tree());
            }
            get(key).add(fset);
            return true;
        }
    }

    /**
     * Arborele in care sunt stocate datele initiale
     */
    TreeMap treeMap = new TreeMap();

    /**
     * Chei numerice prntru fiecare iesire
     */
    private Map<String, Integer> dict = new HashMap<String, Integer>();

    /**
     * Adauga o noua intrare in arborele temporar
     *
     * @param entry
     */
    public void addInput(String entry) {
        treeMap.add(entry);
    }

    /**
     * Construire date de antrenare ponderate
     */
    public void buildFeatureSets() {
        List<String> keys = new ArrayList<String>();
        for (String key : treeMap.keySet()) {
            keys.add(key);
        }
        for (String key : keys) {
            put(key, treeMap.get(key).listFSets());
            treeMap.remove(key);
        }
    }
}
