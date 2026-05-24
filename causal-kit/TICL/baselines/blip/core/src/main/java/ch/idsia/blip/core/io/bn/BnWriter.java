package ch.idsia.blip.core.io.bn;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.RandomStuff;

import java.io.IOException;
import java.io.Writer;
import java.text.DecimalFormat;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public abstract class BnWriter {

    private static final DecimalFormat fmt = new DecimalFormat("#.######");

    private static final Logger log = Logger.getLogger(
            BnNetWriter.class.getName());

    public void go(String net, BayesianNetwork bn) {
        Writer g = null;

        try {
            g = getWriter(net);
            go(g, bn);
        } catch (Exception e) {
            logExp(log, e);
        } finally {
            closeIt(log, g);
        }
    }

    public abstract void go(Writer wr, BayesianNetwork bn) throws IOException;

    /**
     * Write a single potential
     *
     * @param rd_wr writer for the file
     * @param p     potential
     */
    public void writeP(Writer rd_wr, double p) {
        try {
            wf(rd_wr, " %s", fmt.format(p));
        } catch (Exception e) {
            System.out.printf("p: %.2f", p);
            RandomStuff.logExp(log, e);
        }
    }

    protected void writePotential(Writer wr, BayesianNetwork bn, int i) throws IOException {
        wf(wr, "\n");
        double[] pt = bn.potentials(i);
        int t = 0;

        wf(wr, "%d\n", pt.length);
        int ar = bn.l_ar_var[i];

        if (ar == 0) {
            p("whatttt");
        }
        int n_par = pt.length / ar;

        for (int j = 0; j < n_par; j++) {
            wf(wr, " ");
            double sum = 0;

            for (int z = 0; z < ar - 1; z++) {
                double p = pt[t++];

                p = Math.floor(p * 1000000d) / 1000000d;
                writeP(wr, p);
                sum += p;
            }

            // last number as 1 minus the sum of the others (ensure they will sum to 1)
            writeP(wr, 1 - sum);
            t++;
            wf(wr, "\n");
        }

        wr.flush();
    }
}
