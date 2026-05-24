package ch.idsia.blip.core.learn.solver.samp;


import ch.idsia.blip.core.utils.analyze.Entropy;

import java.util.Random;


public class EntropyRSampler extends EntropySampler {

    public EntropyRSampler(String ph_dat, int n, Random r) {
        super(ph_dat, n, r);
    }

    @Override
    public void init() {
        weight = new double[n];
        Entropy e = new Entropy(dat);

        for (int i = 0; i < n; i++) {
            weight[i] = 1.0 / e.computeH(i);
        }
    }

}
