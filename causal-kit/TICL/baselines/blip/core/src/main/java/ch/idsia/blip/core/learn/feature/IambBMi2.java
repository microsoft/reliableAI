package ch.idsia.blip.core.learn.feature;


import ch.idsia.blip.core.io.dat.DatFileReader;

import java.io.IOException;


/**
 * BMI - alpha adjustment, normal test?
 */
public class IambBMi2 extends IambBMi1 {

    public IambBMi2(DatFileReader dat) throws IOException {
        super(dat);
    }

    @Override
    public String getName() {
        return "iambBMi1";
    }

    @Override
    protected boolean condInd(int x, int y, int[] z, double alpha) {
        fixEss(x, y, z);
        bmi.alpha = alpha;
        return bmi.condInd(x, y, z);

    }

    @Override
    protected double computeCMI(int x, int y, int[] z) {

        fixEss(x, y, z);

        return bmi.computeCMI(x, y, z);

    }
}
