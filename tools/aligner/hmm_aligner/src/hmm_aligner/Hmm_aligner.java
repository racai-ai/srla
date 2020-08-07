/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package hmm_aligner;

import sphinx.Aligner;

/**
 *
 * @author tibi
 */
public class Hmm_aligner {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        if (args.length!=4){
            displayHelp();
        }else{
            Aligner aligner=new Aligner(args[0], args[1], args[2], args[3]);
        }
    }

    private static void displayHelp() {
        System.out.println("SRLA voice aligner v0.9");
        System.out.println("Usage:");
        System.out.println("hmm_aligner.jar <model base> <wav file> <text file> <output base>");
        
    }
    
}
