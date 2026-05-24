package ch.idsia.blip.core.utils.other;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;

import java.util.Random;

import static ch.idsia.blip.core.utils.RandomStuff.getRandom;


public class DFGenerator {

    private final Random random = getRandom();
    private final int seed = (int) (100000 * random.nextFloat());
    private final MersenneTwister rn = new MersenneTwister(seed);

    double[] drawDistribution(int i, BayesianNetwork bn) {

        int nvv = bn.arity(i);

        int npp = 1;

        for (int p : bn.parents(i)) {
            npp *= bn.arity(p);
        }

        return generateDistributionFunction(nvv, npp);
    }

    public double[] generateUniformDistribution(int n) {
        double distribution[] = new double[n];
        double normalization = 0.0;

        for (int i = 0; i < n; i++) {
            distribution[i] = -Math.log(rn.nextDouble());
            normalization += distribution[i];
        }
        for (int i = 0; i < n; i++) {
            distribution[i] /= normalization;
        }
        return (distribution);
    }

    public double[] generateDistributionFunction(int nv, int np) {
        double distribution[] = new double[nv * np];
        int cont = 0;

        for (int i = 0; i < np; i++) {
            double distr[] = generateUniformDistribution(nv);

            // double distr[] = generateDistribution(nv, alphas);
            for (int j = 0; j < nv; j++) {
                distribution[cont] = distr[j];
                cont++;
            }
        }
        return (distribution);
    }

    public int[][] generateUniformDataset(int card, int n_datapoints) {

        TIntArrayList[] r = new TIntArrayList[card];

        for (int v = 0; v < card; v++) {
            r[v] = new TIntArrayList();
        }

        double[] distr = generateUniformDistribution(card);

        for (int j = 0; j < n_datapoints; j++) {
            r[sampleV(distr)].add(j);
        }

        int[][] row = new int[card][];

        for (int v = 0; v < card; v++) {
            row[v] = r[v].toArray();
        }

        return row;

    }

    public int sampleV(double[] distr) {
        double v = rn.nextDouble();
        int i = 0;

        while (i < distr.length) {
            v -= distr[i];
            if (v < 0) {
                return i;
            }
            i++;
        }

        return 0;
    }
}
