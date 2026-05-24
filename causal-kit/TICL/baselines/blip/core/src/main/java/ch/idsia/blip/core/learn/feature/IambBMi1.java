package ch.idsia.blip.core.learn.feature;


import ch.idsia.blip.core.io.dat.DatFileReader;

import java.io.IOException;


/**
 * BMI - alpha adjustment
 */
public class IambBMi1 extends IambBMi {

    public IambBMi1(DatFileReader dat) throws IOException {
        super(dat);
    }

    @Override
    public String getName() {
        return "iambBMi1";
    }

    @Override
    protected boolean condInd(int x, int y, int[] z, double alpha) {
        double est_I = computeCMI(x, y, z);

        return est_I < alpha;
    }

    @Override
    protected double computeCMI(int x, int y, int[] z) {

        fixEss(x, y, z);

        return bmi.computeCMI(x, y, z);

    }

    void fixEss(int x, int y, int[] z) {
        bmi.ess = (dat.l_n_arity[x] - 1) * (dat.l_n_arity[y] - 1);
        for (int p : z) {
            bmi.ess *= dat.l_n_arity[p];
        }
    }
}
