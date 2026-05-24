package ch.idsia.blip.core.io.bn;


import ch.idsia.blip.core.utils.BayesianNetwork;

import java.io.IOException;
import java.io.Writer;
import java.text.DecimalFormat;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.logExp;
import static ch.idsia.blip.core.utils.RandomStuff.wf;


/**
 * Write a Bayesian Network as a .net format file.
 */
public class BnNetWriter extends BnWriter {

    private static final DecimalFormat fmt = new DecimalFormat("####");

    private static final Logger log = Logger.getLogger(
            BnNetWriter.class.getName());

    /**
     * Writes a Bayesian network in a .net file format.
     *
     * @param wr writer for the file
     * @param bn bayesian network to save
     */
    @Override
    public void go(Writer wr, BayesianNetwork bn) throws IOException {

        try {
            wr.write("net { }\n\n");

            writeNodes(wr, bn);

            writePotentials(wr, bn);

        } catch (IOException e) {
            log.severe(e.getMessage());
        }
    }

    /**
     * Write down the potentials.
     *
     * @param rd_wr writer where to graph
     * @param bn    Bayesian network, from where we get the potentials
     * @throws IOException if there is any problem in the writing
     */
    private void writePotentials(Writer rd_wr, BayesianNetwork bn)
        throws IOException {

        for (int i = 0; i < bn.n_var; i++) {
            wf(rd_wr, "potential ( %s", bn.name(i));
            int[] ps = bn.parents(i);

            if (ps.length == 0) {
                writePotentialsSimple(rd_wr, bn, i);
            } else {
                writePotentialsComplete(rd_wr, bn, i, ps);
            }

            rd_wr.flush();
        }
    }

    /**
     * Write potentials when the variable has no parents.
     *
     * @param rd_wr writer where to graph
     * @param bn    Bayesian network, from where we get the potentials
     * @param i     index of variable
     * @throws IOException if there is a problem in the writing
     */
    private void writePotentialsSimple(Writer rd_wr, BayesianNetwork bn, int i) throws IOException {
        // Write for variable with no parents
        wf(rd_wr, " ) { \n data = ( ");
        for (double p : bn.potentials(i)) {
            writeP(rd_wr, p);
        }
        wf(rd_wr, "); \n}\n\n");
    }

    /**
     * Write potentials when the variable has parents.
     *
     * @param rd_wr writer where to graph
     * @param bn    Bayesian network, from where we get the potentials
     * @param i     index of variable
     * @param ps    parent of the variable (array of indexes)
     * @throws IOException if there is a problem in the writing
     */
    private void writePotentialsComplete(Writer rd_wr, BayesianNetwork bn, int i, int[] ps) throws IOException {
        // Write for variables with parents
        wf(rd_wr, " |");
        for (int p : ps) {
            wf(rd_wr, " %s", bn.name(p));
        }

        // Write potentials
        wf(rd_wr, " ) { \n data = \n");

        // Compute intervals (how do we know when to place parenthesis?)
        int[] interval = new int[ps.length + 1];
        int ar = bn.arity(i);
        int ix = 0;

        interval[ix] = ar;
        for (int j = ps.length - 1; j > -1; j--) {
            ar *= bn.arity(ps[j]);
            interval[j + 1] = ar;
        }

        for (int ignored : interval) {
            wf(rd_wr, "(");
        }

        ix = 0;
        double[] pt = bn.potentials(i);

        for (double p : pt) {
            writeP(rd_wr, p);
            ix += 1;

            String sep = "";

            for (int in : interval) {
                if (((ix % in) == 0) && (ix < pt.length)) {
                    sep = String.format(")\n%s(", sep);
                    rd_wr.flush();
                }
            }
            wf(rd_wr, sep);
        }

        for (int in : interval) {
            wf(rd_wr, ")");
        }
        wf(rd_wr, "; \n}\n\n");
    }

    /**
     * Write the metadata of all variables in the Bayesian network
     *
     * @param rd_wr writer for the file
     * @param bn    Bayesian network of interest
     * @throws IOException if there is a problem in the writing
     */
    private static void writeNodes(Writer rd_wr, BayesianNetwork bn)
        throws IOException {

        for (int i = 0; i < bn.n_var; i++) {
            String nm = bn.name(i);

            wf(rd_wr, "node %s { \n states = ( ", nm);
            for (String v : bn.values(i)) {
                wf(rd_wr, "\"%s\" ", v);
            }
            wf(rd_wr, "); \n");

            wf(rd_wr, " parents = ( ");
            for (int j : bn.parents(i)) {
                wf(rd_wr, "\"%s\" ", bn.name(j));
            }
            wf(rd_wr, ");\n");

            if (bn.positions != null && bn.positions.containsKey(nm)) {
                double[] g = bn.positions.get(nm);

                wf(rd_wr, " position = ( %s %s ); \n", f(g[0]), f(g[1]));
            }

            wf(rd_wr, "}\n\n");
        }

    }

    private static String f(double v) {
        return fmt.format(v).replace(".", ",");
    }

    public static void ex(String s, BayesianNetwork bn) {
        ex(bn, s);
    }

    public static void ex(BayesianNetwork bn, String s) {
        new BnNetWriter().go(s, bn);
    }

    public static void ex(BayesianNetwork bn, Writer w) {

        try {
            new BnNetWriter().go(w, bn);
        } catch (IOException e) {
            logExp(log, e);
        }

    }
}
