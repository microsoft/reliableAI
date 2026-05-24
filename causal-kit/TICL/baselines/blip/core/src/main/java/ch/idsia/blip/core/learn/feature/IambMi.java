package ch.idsia.blip.core.learn.feature;


import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.analyze.MutualInformation;
import ch.idsia.blip.core.io.dat.DatFileReader;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


/**
 * IAMB analyzer - precise Mutual information
 */
public class IambMi extends Iamb {

    @Override
    public double[] getTresholds() {
        return new double[] { 1, 2, 3, 4, 5};
    }

    // Mutual information
    private MutualInformation mi;

    public IambMi(DatFileReader dr) throws IOException {
        super(dr);
        this.mi = new MutualInformation(dr.read());
    }

    @Override
    protected void prepare() {}

    @Override
    protected double computeCMI(int x, int y, int[] z) {
        return mi.computeCMI(x, y, z);
    }

    @Override
    protected boolean condInd(int x, int y, int[] z, double alpha) {
        mi.alpha = alpha;
        return mi.condInd(x, y, z);
    }

    public List<String> fgh(int[] s) {
        return fgh(dat, s);
    }

    List<String> fgh(DataSet dat, int[] s) {
        List<String> g = new ArrayList<String>();

        for (int e : s) {
            g.add(dat.l_nm_var[e]);
        }
        return g;
    }

    @Override
    public String getName() {
        return "iambMi";
    }

}
