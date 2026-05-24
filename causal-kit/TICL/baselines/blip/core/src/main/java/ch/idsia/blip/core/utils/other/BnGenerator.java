package ch.idsia.blip.core.utils.other;

import ch.idsia.blip.core.App;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.data.ArrayUtils;

import java.util.*;
import java.lang.Math;

public class BnGenerator extends App {

    private int n_var;

    private int max_degree = 5;

    private int max_val = 4;

    private DFGenerator df;

    @Override
    public void prepare() {
        super.prepare();

        df = new DFGenerator();
    }

    public BnGenerator(int n_var) {
        this.n_var = n_var;

        prepare();
    }


    public BayesianNetwork create() {
        BayesianNetwork bn = new BayesianNetwork(n_var);

        int[] r = new int[n_var];

        // Arity
        for (int i = 0; i < n_var; i++) {
            int a = randInt(2, max_val);
            bn.l_ar_var[i] = a;
            bn.l_values_var[i] = new String[a];
            for (int j = 0; j < a; j++) {
                bn.l_values_var[i][j] = "v" + j;
            }

            r[i] = i;
        }

        bn.l_ar_var = new int[] {
                4, 2, 2, 3, 3, 3, 2, 3, 4, 2, 4, 3, 3, 3, 3, 2, 3, 2, 4, 2, 2, 2
        };

        // Everything else
        for (int i = 0; i < n_var; i++) {

            // Choose number of parents
            int n_p = randInt(0, Math.min(i, max_degree));

            ArrayUtils.shuffleArray(r, this.rand, i-1);
            int[] p = new int[n_p];
            System.arraycopy(r, 0, p, 0, n_p);
            Arrays.sort(p);
            bn.l_parent_var[i] = p;

            int npp = 1;
            for (int j = 0; j < n_p; j++)
                npp *= bn.l_ar_var[p[j]];

            double[] pot = df.generateDistributionFunction(bn.l_ar_var[i],npp);
            bn.l_potential_var[i] = pot;

        }

        return bn;
    }
}