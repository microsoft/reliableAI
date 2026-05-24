package ch.idsia.blip.core.utils.score;


import ch.idsia.blip.core.utils.DataSet;


/**
 * Computes the DyBIC.
 * <p/>
 * Remenber: last index in values is for missing data (let's simply ignore them!)
 */
class DyBIC extends BIC {

    private final double threshold;

    public DyBIC(DataSet dat, double threshold) {
        super(dat);
        this.threshold = threshold;
    }

    @Override
    protected double getPenalization(int arity) {
        return Math.log(dat.n_datapoints) * (arity - 1) * threshold;
    }

    @Override
    protected double getPenalization(int arity, int p_arity) {
        return Math.log(dat.n_datapoints) * (arity - 1) * p_arity * threshold;
    }
}
