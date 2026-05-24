package ch.idsia.blip.core.io.bn;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.other.BnBuilder;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static ch.idsia.blip.core.utils.RandomStuff.logExp;


/**
 * Read a Bayesian Network from a .net format file.
 */
public class BnNetReader {

    private static final Logger log = Logger.getLogger(
            BnNetReader.class.getName());

    /**
     * Patter to capture the nodes
     */
    private static final Pattern ptrn_node = Pattern.compile(
            "node ([\\w-.]+) \\{([^\\}]+)\\}");

    /**
     * Patter to capture the states
     */
    private static final Pattern ptrn_states = Pattern.compile(
            "states = \\(([^\\)]+)\\);");

    /**
     * Patter to capture the parents
     */
    private static final Pattern ptrn_parents = Pattern.compile(
            "potential \\(([^\\\\)]+)\\) \\{([^\\}]+)\\}");

    /**
     * Patter to capture the potentials
     */
    private static final Pattern ptrn_potentials = Pattern.compile(
            "data = \\((.+)\\);");

    /**
     * Built Bayesian Network
     */
    private BnBuilder bn;

    public static BayesianNetwork ex(BufferedReader rd_net) {
        return new BnNetReader().go(rd_net);
    }

    public static BayesianNetwork ex(String s) {
        File f_bn_original = new File(s);

        try {
            return ex(new BufferedReader(new FileReader(f_bn_original)));
        } catch (FileNotFoundException e) {
            logExp(log, e);
        }

        return null;
    }

    /**
     * Check if the given potentials represent a correct distribution
     *
     * @param bn     Bayesian network we are working with
     * @param par    parents of variable
     * @param ix_var variable of interest
     * @param probs  list of potentials
     */
    private static void checkPotentials(BnBuilder bn, int[] par, int ix_var, double[] probs) {
        double tot_probs = 0.0;

        for (int i = 0; i < probs.length; i++) {
            if (probs[i] < 0.0) {
                log.info(
                        String.format(
                                "Invalid (< 0.0) potential! ix_var: %d, %s, ix_prob: %d, getPotential: %.2f",
                                ix_var, bn.l_nm_var.get(ix_var), i, probs[i]));
            }
            if (probs[i] > 1.0) {
                log.severe(
                        String.format(
                                "Invalid (> 1.0) potential! ix_var: %d, %s, ix_prob: %d, getPotential: %.2f",
                                ix_var, bn.l_nm_var.get(ix_var), i, probs[i]));
            }
            tot_probs += probs[i];
        }
        double check_tot = 1.0;

        for (int aPar : par) {
            check_tot *= bn.l_ar_var.get(aPar);
        }
        if (Math.abs(check_tot - tot_probs) > Math.pow(2, -15)) {
            log.severe(
                    String.format(
                            "Different potential sum! ix_var: %d, %s, tot: %.10f, check:%.10f",
                            ix_var, bn.l_nm_var.get(ix_var), tot_probs,
                            check_tot));
        }

    }

    /**
     * @param content portion of the file
     * @return potentials
     */
    private static double[] extractPotentials(String content) {
        Matcher mtch_data = ptrn_potentials.matcher(content);

        if (!mtch_data.find()) {
            return new double[0];
        }

        String aux3 = mtch_data.group(1).replace("(", " ").replace(")", " ");
        String[] aux4 = aux3.trim().split("\\s+");
        double[] probs = new double[aux4.length];

        for (int i = 0; i < aux4.length; i++) {
            probs[i] = Double.valueOf(aux4[i]);
        }
        return probs;
    }

    private BayesianNetwork go(BufferedReader rd_net) {
        bn = new BnBuilder();

        String content = "";

        try {
            while (rd_net.ready()) {
                content += rd_net.readLine();
                content = content.replace("  ", " ").replace('\n', ' ').replace("{", " { ").replace("}", " } ").replace(";", "; ").replace(" ;", ";").replaceAll(
                        " +", " ");

                Matcher mtch = ptrn_node.matcher(content);

                if (mtch.find()) {
                    int s = content.length();

                    content = content.substring(0, mtch.start())
                            + content.substring(mtch.end(), s);
                    extractNode(mtch);
                    continue;
                }

                mtch = ptrn_parents.matcher(content);
                if (mtch.find()) {
                    int s = content.length();

                    content = content.substring(0, mtch.start())
                            + content.substring(mtch.end(), s);
                    extractParents(mtch);
                }

            }
        } catch (IOException e) {
            log.severe(String.format("Error reading bn: %s", e.getMessage()));
        }

        BayesianNetwork b = bn.toBn();

        return b;
    }

    /**
     * Reorder the Bn in the alphabetical order of its variables.
     */
    
    /*
     private void reorderBn() {

     BnBuilder new_bn = new BnBuilder(bn.n_var);

     new_bn.l_nm_var.clear();
     new_bn.l_nm_var.addAll(bn.l_nm_var);
     Collections.sort(new_bn.l_nm_var);
     int[] ord = new int[bn.n_var];

     for (int i = 0; i < bn.n_var; i++) {
     ord[i] = Collections.binarySearch(new_bn.l_nm_var,
     bn.name( i));
     }

     for (int i = 0; i < bn.n_var; i++) {
     int o = ord[i];

     new_bn.l_ar_var.set(o, bn.arity(i));
     new_bn.l_values_var.set(o, bn.values(i));
     new_bn.l_potential_var.set(o, bn.potentials(i));

     reorderParents(new_bn, ord, i, o);
     }
     for (int i = 0; i < bn.n_var; i++) {
     int o = ord[i];

     reorderProbs(new_bn, ord, i, o);
     }

     bn = new_bn;
     } */

    /*
     private void reorderProbs(BayesianNetwork new_bn, int[] ord, int i, int o) {
     // Reorder the potentials for each variable
     int[] old_p = ArrayUtils.expandArray(bn.parents(i), i);
     int[] new_p = new int[old_p.length];
     int j = 0;

     for (int p : old_p) {
     new_p[j++] = ord[p];

     }
     double[] probs = BnNetUtils.reorganize(new_bn, bn.potentials(i),
     new_p, ArrayUtils.expandArray(new_bn.parents(o), o));

     // System.out.println(o + " " + Arrays.toString(new_p) + " " + Arrays.toString(expandArray(new_bn.parents(o), o)));
     new_bn.l_potential_var.set(o, probs);
     } */

    /*
     private void reorderParents(BayesianNetwork new_bn, int[] ord, int i, int o) {
     // Reorder parents for each variable
     int[] p_var = bn.parents(i);
     int[] new_p_var = new int[p_var.length];
     int j = 0;

     for (int p : p_var) {
     new_p_var[j++] = ord[p];

     }
     Arrays.sort(new_p_var);
     new_bn.setParents(o, new_p_var);
     } */

    /**
     * Read the parents in the given string.
     *
     * @param mtch_parents portion to analyze
     */
    private void extractParents(Matcher mtch_parents) {

        // Extract parent set
        int[] par;
        String nm_var;
        String aux = mtch_parents.group(1);

        String[] aux2 = aux.split("\\|");

        nm_var = aux2[0].trim();
        if (aux2.length == 1 || "".equals(aux2[1].trim())) {
            par = new int[0];
        } else {
            String[] aux3 = aux2[1].trim().split(" ");

            par = new int[aux3.length];

            for (int i = 0; i < aux3.length; i++) {
                par[i] = bn.l_nm_var.indexOf(aux3[i]);
            }
        }
        int ix_var = bn.l_nm_var.indexOf(nm_var);

        // Extract potential
        double[] probs = extractPotentials(mtch_parents.group(2));

        // Check on potential
        checkPotentials(bn, par, ix_var, probs);

        // Save parent set
        bn.l_parent_var.set(ix_var, par);

        // Save potentials
        bn.l_potential_var.set(ix_var, probs);

    }

    /**
     * Extract nodes from the content of the file
     *
     * @param mtch_nodes matched part to extract
     */
    private void extractNode(Matcher mtch_nodes) {

        TreeMap<String, String> map = new TreeMap<String, String>();

        // Match all the nodes

        Matcher mtch_states;

        // Get names
        String s_var = mtch_nodes.group(1);
        // Get row_values
        String aux = mtch_nodes.group(2);

        mtch_states = ptrn_states.matcher(aux.trim());
        if (!mtch_states.find()) {
            log.severe(aux);
        }
        List<String> val_var = new ArrayList<String>();

        for (String val : mtch_states.group(1).trim().split("\" \"")) {
            val_var.add(val.replaceAll("\"", ""));
        }

        // Save variables and arities to network
        bn.l_nm_var.add(s_var);
        bn.l_ar_var.add(val_var.size());
        bn.l_values_var.add(val_var.toArray(new String[val_var.size()]));

        bn.l_parent_var.add(new int[0]);
        bn.l_potential_var.add(new double[0]);

        bn.n_var = bn.l_nm_var.size();
    }

}
