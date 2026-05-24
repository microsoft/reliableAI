package ch.idsia.blip.core.inference;


import ch.idsia.blip.core.App;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.inference.ve.BayesianFactor;
import ch.idsia.blip.core.utils.data.hash.TIntIntHashMap;


public abstract class BaseInference extends App {
    protected final BayesianNetwork bn;

    public BaseInference(BayesianNetwork bayesNet, int verb) {
        this.bn = bayesNet;
        this.verbose = verb;

        prepare();
    }

    public BaseInference(BayesianNetwork bayesNet, boolean verb) {
        this(bayesNet, verb ? 1 : 0);
    }

    public abstract BayesianFactor query(int[] paramArrayOfInt, TIntIntHashMap paramTIntIntHashMap);
}
