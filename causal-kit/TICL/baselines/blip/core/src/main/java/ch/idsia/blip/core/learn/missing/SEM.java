package ch.idsia.blip.core.learn.missing;


import ch.idsia.blip.core.App;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;

import java.io.File;

import static ch.idsia.blip.core.utils.RandomStuff.p;
import static ch.idsia.blip.core.utils.RandomStuff.pf;


public class SEM extends App {

    protected int time_for_variable;

    protected int tw;

    protected String path;

    protected int n_var;

    protected DataSet dat;

    protected String dat_path;

    protected BayesianNetwork bn;

    public void init(String path, int time, int tw) {
        super.init();
        this.path = path;
        if (!new File(path).exists()) {
            new File(path).mkdir();
        }
        this.time_for_variable = time;
        p(this.time_for_variable);
        this.tw = tw;
    }

    protected boolean stationary(BayesianNetwork bn, BayesianNetwork newBn) {

        if (bn.n_var != newBn.n_var) {
            return false;
        }

        double diff = 0;
        double tot = 0;
        int cnt = 0;

        for (int i = 0; i < bn.n_var; i++) {
            double[] pot = bn.potentials(i);
            double[] new_pot = newBn.potentials(i);

            if (pot.length != new_pot.length) {
                return false;
            }

            for (int j = 0; j < pot.length; j++) {
                diff = Math.max(diff, pot[j] - new_pot[j]);
                double d = Math.abs(pot[j] - new_pot[j]);

                tot += d;
                cnt++;
            }
        }

        pf("diff %.8f \n", tot / cnt);
        return tot / cnt < 0.000001;
    }

    protected boolean stationaryStr(BayesianNetwork bn, BayesianNetwork newBn) {
        if (bn.n_var != newBn.n_var) {
            return false;
        }

        for (int i = 0; i < bn.n_var; i++) {
            int[] p = bn.parents(i);
            int[] newP = newBn.parents(i);

            if (p.length != newP.length) {
                return false;
            }

            for (int j = 0; j < bn.parents(i).length; j++) {
                if (p[j] != newP[j]) {
                    return false;
                }
            }
        }

        return true;
    }

    protected double elaps() {
        return (System.currentTimeMillis() - start) / 1000.0;
    }
}
