package ch.idsia.blip.core.io;


import ch.idsia.blip.core.utils.MarkovNetwork;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;

import java.io.*;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.logExp;
import static ch.idsia.blip.core.utils.RandomStuff.p;


public class MarkovUaiReader {

    private static final Logger log = Logger.getLogger(
            MarkovUaiReader.class.getName());

    public static MarkovNetwork ex(BufferedReader rd_net) {
        try {
            return new MarkovUaiReader().go(rd_net);
        } catch (IOException e) {
            logExp(log, e);
        }
        return null;
    }

    public static MarkovNetwork ex(String s) throws FileNotFoundException {
        File f_bn_original = new File(s);

        return ex(new BufferedReader(new FileReader(f_bn_original)));
    }

    private MarkovNetwork go(BufferedReader rd) throws IOException {

        String s = rd.readLine().trim();

        if (!s.equals("MARKOV")) {
            p("NOAAOAOAOAOAO");
            return null;
        }

        int n_vars = Integer.valueOf(rd.readLine().trim());
        MarkovNetwork mk = new MarkovNetwork(n_vars);
        String[] ar = splitLine(rd);

        for (int i = 0; i < n_vars; i++) {
            mk.l_ar_var[i] = Integer.valueOf(ar[i]);
        }

        mk.n_cliques = Integer.valueOf(rd.readLine().trim());

        mk.l_vars_cliques = new int[mk.n_cliques][];
        mk.l_pot_cliques = new double[mk.n_cliques][];

        // Read cliques
        for (int i = 0; i < mk.n_cliques; i++) {
            ar = splitLine(rd);
            int p_ar = Integer.valueOf(ar[0]);
            TIntArrayList pars = new TIntArrayList();

            for (int j = 0; j < p_ar; j++) {
                pars.add(Integer.valueOf(ar[j + 1]));
            }
            mk.l_vars_cliques[i] = pars.toArray();
        }

        mk.updateCliqueAssignments();

        // Read potentials
        readPotent(rd, n_vars, mk);

        return mk;
    }

    public static void readPotent(BufferedReader rd, int n, MarkovNetwork mn) throws IOException {
        String[] ar;

        for (int i = 0; i < mn.n_cliques; i++) {
            int n_pt = nextInt(rd);
            double[] pt = new double[n_pt];
            int t = 0;

            ar = splitLine(rd);
            for (String a : ar) {
                pt[t++] = Double.valueOf(a);
            }

            mn.l_pot_cliques[i] = pt;
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
