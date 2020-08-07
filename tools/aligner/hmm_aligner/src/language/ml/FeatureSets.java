/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package language.ml;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author adrian
 */
public class FeatureSets {

    /**
     *
     */
    public int count;
    private List<FeatureSet> data;

    /**
     *
     */
    public FeatureSets() {
        data = new ArrayList<FeatureSet>();
        count=0;
    }
    
    /**
     *
     * @return
     */
    public int length(){
        return data.size();
    }

    /**
     *
     * @return
     */
    public List<FeatureSet> getSets() {
        return data;
    }

    /**
     *
     * @param set
     */
    public void add(FeatureSet set) {
        data.add(set);
        count+=set.getCount();                
    }
    
    

}
