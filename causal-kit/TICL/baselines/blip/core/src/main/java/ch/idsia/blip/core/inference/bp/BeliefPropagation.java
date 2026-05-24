package ch.idsia.blip.core.inference.bp;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.inference.BaseInference;
import ch.idsia.blip.core.inference.ve.BayesianFactor;
import ch.idsia.blip.core.utils.data.hash.TIntIntHashMap;


public class BeliefPropagation extends BaseInference {

    public BeliefPropagation(BayesianNetwork bn, int verb) {
        super(bn, verb);
    }

    @Override
    public BayesianFactor query(int[] query, TIntIntHashMap evidence) {
        return null;
    }
}
