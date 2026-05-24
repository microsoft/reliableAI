package ch.idsia.blip.core.io.bn;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.RandomStuff;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


/**
 * Read a Bayesian Network from a string
 */
public class BnMdlReader {

    /**
     * Construct a Bayesian network from a mdl description
     *
     * @param path path of file
     * @return the associated Bayesian network (note that the CPTs are void).
     */
    public static BayesianNetwork go(String path) throws IOException {

        String c = RandomStuff.readFile(path);

        String lineSeparator = System.getProperty("line.separator");

        c = c.replace(lineSeparator, "").replace(" ", "").replace("][", "] [");

        String[] aux = c.split(" ");

        BayesianNetwork bn = new BayesianNetwork(aux.length);

        List<String> aux2 = new ArrayList<String>();

        int i = 0;

        // Get names
        for (String a : aux) {
            // System.out.println(a);
            a = a.replaceAll("\\[", "").replaceAll("\\]", "");
            if (a.contains("|")) {
                String[] aux3 = a.split("\\|");

                bn.l_nm_var[i] = aux3[0].trim();
                aux2.add(aux3[1].trim());
            } else {
                bn.l_nm_var[i] = a.trim();
                aux2.add("");
            }
            i++;
        }

        i = 0;
        // Get parent sets
        for (String a : aux2) {
            if (a.trim().isEmpty()) {
                bn.l_parent_var[i++] = new int[0];
                continue;
            }

            String[] aux3 = a.split(":");
            int[] par = new int[aux3.length];

            for (int j = 0; j < aux3.length; j++) {
                par[j] = find(bn.l_nm_var, aux3[j].trim());
            }
            bn.l_parent_var[i++] = par;
        }

        return bn;
    }

    private static int find(String[] l, String f) {
        for (int i = 0; i < l.length; i++) {
            if (t(f).equals(t(l[i]))) {
                return i;
            }
        }
        return -1;
    }

    private static String t(String f) {
        return f.toLowerCase().trim();
    }
}
