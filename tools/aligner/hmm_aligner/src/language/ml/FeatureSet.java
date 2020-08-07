/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package language.ml;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.SortedSet;

/**
 * <b>Structura de date:</b><br/>
 * Lista cu caracteristici<br/>
 * Numarul de repetitii (count)<br/>
 *
 * @author adrian
 */
public class FeatureSet {

    List<String> data;
    int count = 0;
    /**
     * <b>Numar de aparitii</b>
     * @return 
     */
    public int getCount(){
        return count;
    }

    /**
     *
     * @param count
     */
    public FeatureSet(int count) {
        data=new ArrayList<String>();
        this.count = count;
    }
    
    @Override
    public String toString() {
        return count + "=" + super.toString();
    }

    /**
     *
     * @return
     */
    public Iterable<String> getSet() {
        return data;
    }

    /**
     *
     * @param feat
     */
    public void add(String feat) {
        data.add(feat);        
    }

    /**
     *
     * @param feat
     * @return
     */
    public boolean contains(String feat) {
        return data.contains(feat);
    }

    int size() {
        return data.size();
    }

    void sort() {
        Collections.sort(data);
    }

    String getFirst() {
        return data.get(0);
    }

    void removeFirst() {
        data.remove(0);
    }

    void addFirst(String feat) {
        data.add(0,feat);
    }
}
