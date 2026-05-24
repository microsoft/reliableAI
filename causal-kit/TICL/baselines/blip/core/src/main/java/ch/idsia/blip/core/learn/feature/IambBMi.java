package ch.idsia.blip.core.learn.feature;


import ch.idsia.blip.core.utils.analyze.BayesMutualInformation;
import ch.idsia.blip.core.io.dat.DatFileReader;

import java.io.IOException;


/**
 * Uses Bayesian Mutual Information as if it were precise
 */
public class IambBMi extends Iamb {

    final BayesMutualInformation bmi;

    @Override
    public double[] getTresholds() {
        return new double[] { 0.005};
    }

    // Mutual information
    public IambBMi(DatFileReader dr) throws IOException {
        super(dr);
        this.bmi = new BayesMutualInformation(dat);
    }

    @Override
    protected void prepare() {}

    @Override
    protected double computeCMI(int x, int y, int[] z) {
        return bmi.computeCMI(x, y, z);
    }

    @Override
    protected boolean condInd(int x, int y, int[] z, double alpha) {
        if (forward) {
            return bmi.computeCMI(x, y, z, 0.05) < alpha;
        } else {
            return bmi.computeCMI(x, y, z, 0.95) < alpha;
        }

    }

    @Override
    public String getName() {
        return "iambBMi";
    }
}
