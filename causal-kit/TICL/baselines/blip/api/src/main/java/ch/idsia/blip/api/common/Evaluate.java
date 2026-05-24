package ch.idsia.blip.api.common;


import ch.idsia.blip.api.Api;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.score.BDeu;
import ch.idsia.blip.core.utils.score.BIC;
import ch.idsia.blip.core.common.LLEval;
import org.kohsuke.args4j.Option;

import java.util.Arrays;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class Evaluate extends Api {

    @Option(name = "-n", required = true, usage = "Bayesian network file path")
    private String bn_p;

    @Option(name = "-d", required = true, usage = "Datafile path")
    private String dat_p;

    @Option(name = "-l", usage = "List of single parent score")
    private boolean list = false;

    public static void main(String[] args) {
        defaultMain(args, new Evaluate());
    }

    @Override
    public void exec() throws Exception {

        BayesianNetwork bn = getBayesianNetwork(bn_p);

       DataSet dat = getDataSet(dat_p);

        int[] inv_index = new int[bn.n_var];
        for (int i = 0; i < bn.n_var; i++) {
            inv_index[i] = ArrayUtils.index(bn.l_nm_var[i], dat.l_nm_var);
        }

        LLEval l = new LLEval();
        l.go(bn, dat_p);

        double bic_v = 0;
        double bic_e = 0;
        double bdeu_v = 0;
        BIC bic = new BIC(dat);
        BDeu bdeu = new BDeu(1, dat);
        for (int a = 0; a < bn.n_var; a++) {
            int inv_a = inv_index[a];
            int[] p = bn.parents(a);
            int[] inv_p = new int[p.length];
            String[] names_p = new String[p.length];
            for (int j = 0; j < p.length; j++) {
                inv_p[j] = inv_index[p[j]];
                names_p[j] = bn.l_nm_var[p[j]];
            }
            bic_e = bic.computeScore(inv_a, inv_p);
            if (list)
                pf("%s - %s: %.2f \n", bn.l_nm_var[a], Arrays.toString(names_p), bic_e);
            bic_v += bic_e;
            // p( bic.computeScore(a, bn.parents(a)));
            bdeu_v += bdeu.computeScore(inv_a, inv_p);
        }

        pf("LL: %3.3f, BIC: %6.3f, BDeu: %6.3f \n", l.ll, bic_v, bdeu_v);
    }
}
