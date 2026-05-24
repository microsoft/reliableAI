package ch.idsia.blip.core.io.bn;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;

import java.io.*;
import java.util.Arrays;
import java.util.logging.Logger;

import static ch.idsia.blip.core.io.bn.BnUaiReader.readPotent;
import static ch.idsia.blip.core.utils.RandomStuff.logExp;


public class BnErgReader {

    private static final Logger log = Logger.getLogger(
            BnErgReader.class.getName());

    public static BayesianNetwork ex(BufferedReader rd_net) {
        try {
            return new BnErgReader().go(rd_net);
        } catch (IOException e) {
            logExp(log, e);
        }
        return null;
    }

    public static BayesianNetwork ex(String s) throws FileNotFoundException {
        File f_bn_original = new File(s);

        return ex(new BufferedReader(new FileReader(f_bn_original)));
    }

    private BayesianNetwork go(BufferedReader rd) throws IOException {

        int n = Integer.valueOf(rd.readLine().trim());
        BayesianNetwork bn = new BayesianNetwork(n);
        String[] ar = splitLine(rd);

        for (int i = 0; i < n; i++) {
            bn.l_ar_var[i] = Integer.valueOf(ar[i]);
        }

        int[][] orig_parents = new int[n][];

        // Read parents
        for (int i = 0; i < n; i++) {
            ar = splitLine(rd);
            int p_ar = Integer.valueOf(ar[0]);
            TIntArrayList pars = new TIntArrayList();

            for (int j = 0; j < p_ar; j++) {
                pars.add(Integer.valueOf(ar[j + 1]));
            }

            int[] parents = pars.toArray();

            orig_parents[i] = parents;
            Arrays.sort(parents);
            bn.l_parent_var[i] = parents;
        }

        String s = "";

        while (!"/* Probabilities */".equals(s)) {
            s = rd.readLine();
        }

        readPotent(rd, n, bn, orig_parents);

        return bn;
    }

    private String[] splitLine(BufferedReader rd) throws IOException {
        return rd.readLine().trim().split("[ \\t]+");
    }

    private int nextInt(BufferedReader br) throws IOException {
        String s = "";

        while (s.trim().length() == 0) {
            s = br.readLine();
        }
        return Integer.valueOf(s.trim());
    }
}
