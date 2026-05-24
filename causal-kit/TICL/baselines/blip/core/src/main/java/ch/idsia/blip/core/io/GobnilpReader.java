package ch.idsia.blip.core.io;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.ParentSet;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static ch.idsia.blip.core.utils.RandomStuff.closeIt;
import static ch.idsia.blip.core.utils.RandomStuff.logExp;


/**
 * Reads gobnilp
 */
public class GobnilpReader {

    private static final Logger log = Logger.getLogger(
            GobnilpReader.class.getName());

    private static final Pattern ptrn_sk = Pattern.compile("^BN score is (.+)");
    private static final Pattern ptrn_str = Pattern.compile("^(\\d+)<-");

    public ParentSet[] new_str;
    public double sk;
    private int n_var;

    public void go(String s) {
        BufferedReader rd = null;

        try {
            rd = new BufferedReader(new FileReader(s));
            go(rd);
        } catch (Exception e) {
            logExp(log, e);
        } finally {
            closeIt(log, rd);
        }
    }

    public BayesianNetwork getBn() {
        BayesianNetwork bn = new BayesianNetwork(n_var);

        for (int i = 0; i < n_var; i++) {
            bn.setParents(i, new_str[i].parents);
        }

        return bn;
    }

    private void go(BufferedReader br) throws IOException {
        String l2;

        sk = -Double.MAX_VALUE;

        HashMap<Integer, ParentSet> temp = new HashMap<Integer, ParentSet>();

        while ((l2 = br.readLine()) != null) {

            Matcher mtch = ptrn_str.matcher(l2);

            if (mtch.find()) {
                // p(l2);
                String[] aux = l2.split("<-");
                int v = Integer.valueOf(aux[0]);
                String[] aux2 = aux[1].split(" ");
                int[] ps;

                if (!aux2[0].isEmpty()) {
                    String[] aux3 = aux2[0].split(",");

                    ps = new int[aux3.length];
                    for (int j = 0; j < ps.length; j++) {
                        ps[j] = Integer.valueOf(aux3[j]);
                    }
                } else {
                    ps = new int[0];
                }
                double p_sk = Double.valueOf(aux2[1]);

                Arrays.sort(ps);
                temp.put(v, new ParentSet(p_sk, ps));
                // pf("%d - %s %.3f\n", v, Arrays.toString(ps), p_sk);
            }

            mtch = ptrn_sk.matcher(l2);
            if (mtch.find()) {
                sk = Double.valueOf(mtch.group(1).trim());
            }

        }

        n_var = temp.size();
        new_str = new ParentSet[n_var];
        for (int i = 0; i < n_var; i++) {
            new_str[i] = temp.get(i);
        }
    }

    public static BayesianNetwork ex(String f) {
        GobnilpReader g = new GobnilpReader();

        g.go(f);
        return g.getBn();
    }
}
