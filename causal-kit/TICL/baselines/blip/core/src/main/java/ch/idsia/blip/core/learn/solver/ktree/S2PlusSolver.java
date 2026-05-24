package ch.idsia.blip.core.learn.solver.ktree;


import ch.idsia.blip.core.utils.tw.AstarKtree;
import ch.idsia.blip.core.utils.tw.KTree;
import ch.idsia.blip.core.utils.data.ArrayUtils;

import java.util.logging.Logger;


/**
 * IS ALIVE! IS AAAAALIVE!
 */
public class S2PlusSolver extends S2Solver {

    private static final Logger log = Logger.getLogger(
            S2PlusSolver.class.getName());

    private int[] vars;

    private AstarKtree sampler;

    public String ph_astar;

    @Override
    protected void almost() {
        vars = new int[n_var];
        for (int j = 0; j < n_var; j++) {
            vars[j] = j + 1;
        }

        sampler = new AstarKtree(n_var, tw, this);
    }

    @Override
    protected String name() {
        return "Gobnilp S2+ Solver!";
    }

    @Override
    protected KTree sampleKtree(int i, int j) {

        // Sample an initial complete clique
        ArrayUtils.shuffleArray(vars, rand);

        int[] R = new int[tw];

        System.arraycopy(vars, 0, R, 0, tw);

        KTree k = sampler.go(R, availableTime(), i, j);

        return k;
    }
}
