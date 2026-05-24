package ch.idsia.blip.core.learn.solver.samp;


import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.analyze.Entropy;

import java.util.Random;

import static ch.idsia.blip.core.utils.RandomStuff.getDataSet;


public class EntropySampler implements Sampler {

    protected int n;

    protected Random r;

    protected DataSet dat;

    protected double[] weight;

    protected Object lock = new Object();

    @Override
    public int[] sample() {
        return sampleWeighted(n, weight);
    }

    public EntropySampler(String ph_dat, int n, Random r) {
        dat = getDataSet(ph_dat);
        this.n = n;
        this.r = r;
    }

    @Override
    public void init() {

        weight = new double[n];
        Entropy e = new Entropy(dat);

        for (int i = 0; i < n; i++) {
            weight[i] = e.computeH(i);
        }
    }

    public int[] sampleWeighted(int n, double[] weights) {

        int[] new_ord = new int[n];

        synchronized (lock) {

            boolean[] selected = new boolean[n];

            for (int j = 0; j < n; j++) {

                double tot = 0;

                for (int i = 0; i < n; i++) {
                    if (!selected[i]) {
                        tot += weights[i];
                    }
                }
                double v = r.nextDouble() - Math.pow(2, -10);
                int sel = -1;

                for (int i = 0; i < n && sel == -1; i++) {
                    if (!selected[i]) {
                        double s = weights[i] / tot;

                        if (s <= 0 || v <= s) {
                            sel = i;
                        }
                        v -= s;
                    }
                }

                // p(sel);

                selected[sel] = true;
                new_ord[j] = sel;

                // Entropy e = new Entropy(dat);
                // pf("%.4f ", e.computeH(sel));
            }
        }
        // p("");

        return new_ord;
    }
}
