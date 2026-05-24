package ch.idsia.blip.api.common;


import ch.idsia.blip.api.Api;
import ch.idsia.blip.core.utils.BayesianNetwork;
import org.kohsuke.args4j.Option;

import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.getBayesianNetwork;

public class HammingDist extends Api {

    private static final Logger log = Logger.getLogger(
            HammingDist.class.getName());

    @Option(name = "-n1", required = true, usage = "First Bayesian network path")
    private String s_f_bn;

    @Option(name = "-n2", required = true, usage = "Second Bayesian network path")
    private String s_s_bn;

    /**
     * Command line execution
     *
     * @param args parameters provided
     */
    public static void main(String[] args) {
        defaultMain(args, new HammingDist());
    }

    public int computeDist(BayesianNetwork f_bn, BayesianNetwork s_bn) {

        if (f_bn.n_var != s_bn.n_var) {
            System.out.println("Warning! Different network sizes!");
            return -1;
        }

        boolean[] f_ar = f_bn.moralize().arcs;
        boolean[] s_ar = s_bn.moralize().arcs;

        int dist = 0;

        for (int i = 0; i < f_ar.length; i++) {
            if (f_ar[i] != s_ar[i]) {
                dist++;
            }
        }

        return dist;
    }

    @Override
    public void exec() throws Exception {

        BayesianNetwork f_bn = getBayesianNetwork(s_f_bn);
        BayesianNetwork s_bn = getBayesianNetwork(s_s_bn);

        int dist = computeDist(f_bn, s_bn);

        System.out.println(dist);
    }
}

