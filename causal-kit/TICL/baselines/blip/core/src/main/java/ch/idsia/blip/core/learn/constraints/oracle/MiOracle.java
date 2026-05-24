package ch.idsia.blip.core.learn.constraints.oracle;


import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.analyze.MutualInformation;


public class MiOracle extends Oracle {

    private final double eps;
    private final MutualInformation mutualInf;

    public MiOracle(double eps, DataSet dat) {
        super(dat);
        this.eps = eps;
        mutualInf = new MutualInformation(dat);
    }

    @Override
    public boolean condInd(int x, int y, int[] z) {
        double mi;

        if (z.length == 0) {
            mi = mutualInf.computeMi(x, y);
        } else {
            mi = mutualInf.computeCMI(x, y, z);
        }

        // System.out.printf("%.3f - eps %.3f \n", mi, eps);

        return mi < eps;
    }
}
