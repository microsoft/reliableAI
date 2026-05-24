package ch.idsia.blip.core.io.bn;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;

import java.io.*;
import java.util.Arrays;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class BnUaiReader {

    private static final Logger log = Logger.getLogger(
            BnUaiReader.class.getName());

    public static BayesianNetwork ex(BufferedReader rd_net) {
        try {
            return new BnUaiReader().go(rd_net);
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

        String s = rd.readLine().trim();

        if (!s.equals("BAYES")) {
            p("NOAAOAOAOAOAO");
            return null;
        }

        int n = Integer.valueOf(rd.readLine().trim());
        BayesianNetwork bn = new BayesianNetwork(n);
        String[] ar = splitLine(rd);

        for (int i = 0; i < n; i++) {
            bn.l_ar_var[i] = Integer.valueOf(ar[i]);
        }

        int n2 = Integer.valueOf(rd.readLine().trim());

        if (!(n2 == n)) {
            p("NOAAOAOAOAOAOsafasfasfas");
            return null;
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
            pars.remove(i);
            int[] parents = pars.toArray();

            orig_parents[i] = parents;
            Arrays.sort(parents);
            bn.l_parent_var[i] = parents;
        }

        // Read potentials
        readPotent(rd, n, bn, orig_parents);

        return bn;
    }

    public static void readPotent(BufferedReader rd, int n, BayesianNetwork bn, int[][] orig_parents) throws IOException {
        String[] ar;
        double p;

        for (int i = 0; i < n; i++) {
            int n_pt = nextInt(rd);
            int n_par = n_pt / bn.l_ar_var[i];
            double[] pt = new double[n_pt];
            int t = 0;

            for (int j = 0; j < n_par; j++) {
                ar = splitLine(rd);
                double cnt = 0;

                for (String a : ar) {
                    p = Double.valueOf(a);
                    pt[t++] = p;
                    cnt += p;
                }
                if (!doubleEquals(cnt, 1.0)) {
                    pf("SUM NOT 1.0: %s %.2f \n", Arrays.toString(ar), cnt);
                }
            }

            bn.l_potential_var[i] = pt;

        }
    }

    public static String[] splitLine(BufferedReader rd) throws IOException {
        return rd.readLine().trim().split("[ \\t]+");
    }

    public static int nextInt(BufferedReader br) throws IOException {
        String s = "";

        while (s.trim().length() == 0) {
            s = br.readLine();
        }
        return Integer.valueOf(s.trim());
    }
}
