package ch.idsia.blip.core.io.bn;


import ch.idsia.blip.core.utils.BayesianNetwork;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.logExp;


/**
 * Read a Bayesian Network from a solver output
 */
public class BnResReader {

    private static final Logger log = Logger.getLogger(
            BnResReader.class.getName());

    private Float scoreTot;

    public ArrayList<Double> scores;

    private ArrayList<int[]> p;

    public static BayesianNetwork ex(String s) throws FileNotFoundException {
        return new BnResReader().go(s);
    }

    /**
     * Construct a Bayesian network from a res output (note that the CPTs are void).
     */
    public static BayesianNetwork ex(BufferedReader rd) {
        return new BnResReader().go(rd);
    }

    public BayesianNetwork go(String s) throws FileNotFoundException {
        File f_bn_original = new File(s);

        return go(new BufferedReader(new FileReader(f_bn_original)));
    }

    private BayesianNetwork go(BufferedReader rd) {

        p = new ArrayList<int[]>();

        scores = new ArrayList<Double>();

        try {
            String line;

            while ((line = rd.readLine()) != null) {
                String[] a = line.replace("(", "").replace(")", "").replace(":", "").split(
                        "\\s+");

                if (a.length <= 1) {
                    break;
                }

                scores.add(Double.valueOf(a[1]));

                if (a.length == 2) {
                    p.add(new int[0]);
                    continue;
                }

                String[] b = a[2].replace("(", "").replace(")", "").replace("\\s+", "").split(
                        ",");
                int[] c = new int[b.length];

                for (int i = 0; i < b.length; i++) {
                    c[i] = Integer.valueOf(b[i]);
                }

                Arrays.sort(c);

                p.add(c);
            }

            line = rd.readLine();
            while (line.trim().equals("")) {
                line = rd.readLine();
            }

            String[] a = line.split(":");

            scoreTot = Float.valueOf(a[1]);

        } catch (Exception e) {
            logExp(log, e);
            return null;
        }

        BayesianNetwork bn = new BayesianNetwork(p.size());

        for (int i = 0; i < p.size(); i++) {
            bn.l_parent_var[i] = p.get(i);
        }

        return bn;
    }

}
