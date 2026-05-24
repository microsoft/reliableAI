package ch.idsia.blip.core.common;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.io.dat.BaseFileLineReader;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.data.array.TDoubleArrayList;
import ch.idsia.blip.core.utils.RandomStuff;

import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.getDataSetReader;


public class LLEval {

    private static final Logger log = Logger.getLogger(LLEval.class.getName());

    public double ll;

    public TDoubleArrayList ls;

    public boolean log10 = false;

    private short[] samp;

    private int[] inv_index;

    public static double ex(BayesianNetwork bn, String s) {
        return new LLEval().go(bn, s);
    }

    public double go(BayesianNetwork bn, String s) {
        return go(bn, getDataSetReader(s));
    }

    /**
     * @param bn
     * @param dat_rd
     * @return computed log-likelihood of the given Bayesian network over the given data
     */
    public double go(BayesianNetwork bn, BaseFileLineReader dat_rd) {

        this.ll = 0.0D;

        this.ls = new TDoubleArrayList();

        int cnt = 0;

        double l;

        try {
            dat_rd.readMetaData();

            inv_index = new int[bn.n_var];
            for (int i = 0; i < bn.n_var; i++) {
                inv_index[i] = ArrayUtils.index(bn.l_nm_var[i], dat_rd.l_s_names);
            }

            samp = new short[bn.n_var];

            while (!dat_rd.concluded) {

                dat_rd.next();
                for (int ix = 0; ix < bn.n_var; ix++) {
                    samp[inv_index[ix]] = dat_rd.samp[ix];
                }

                if (this.log10) {
                    l = bn.getLogLik10(samp);
                } else {
                    l = bn.getLogLik(samp);
                }
                this.ll += l;
                this.ls.add(l);
                cnt++;
            }
        } catch (Exception e) {
            System.out.println("Sample: " + Arrays.toString(samp));
            RandomStuff.logExp(log, e);
        }
        this.ll /= cnt;

        return this.ll;
    }


}
