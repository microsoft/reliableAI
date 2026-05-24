package ch.idsia.blip.core.inference.ve;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;

import java.util.ArrayList;
import java.util.List;


/**
 * Simulation execution for variable elimination algorithms
 */
public class Simulation {

    private final BayesianNetwork bn;
    private final int verbose;

    public Simulation(BayesianNetwork bn, int verbose) {
        this.bn = bn;
        this.verbose = verbose;
    }

    public Simulation(BayesianNetwork bn, boolean verbose) {
        this(bn, verbose ? 1 : 0);
    }

    /**
     * Simulate the variable elimination algorithm, memorize
     * the maximum amount of space required
     * t
     *
     * @param order an elimination order to follow
     * @return maximum number of potentials in memory at any time
     */
    public double simulateInference(int[] order) {

        List<FakeFactor> factors = new ArrayList<FakeFactor>();

        double max_size = 0;

        for (int v : order) {
            // Create factors from nodes
            FakeFactor psi = new FakeFactor(v, bn);

            factors.add(psi);
        }

        // For each factor in the elimination order
        for (int v : order) {

            simulateElimin(v, factors);

            // Compute current size
            double size = 0;

            for (FakeFactor psi : factors) {
                size += psi.size;
            }

            if (size > max_size) {
                max_size = size;
            }

        }

        return max_size;
    }

    /**
     * Simulate the the elimination of a single variable
     *
     * @param v       variable to eliminate
     * @param factors factors in play
     */
    public void simulateElimin(int v, List<FakeFactor> factors) {
        if (verbose > 2) {
            System.out.printf("\nEliminating: %d # ", v);
        }

        // Search factor to multiply (they contain the variable to eliminate)
        List<FakeFactor> toMult = new ArrayList<FakeFactor>();

        for (FakeFactor psi : factors) {
            if (psi.dom.contains(v)) {
                toMult.add(psi);

                if (verbose > 2) {
                    System.out.printf("\n %s", psi.dom);
                }
            }
        }

        TIntArrayList n_dom = new TIntArrayList();

        // Multiply factors
        for (FakeFactor psi : toMult) {
            n_dom.addAll(psi.dom);
            factors.remove(psi);
        }

        // Marginalize variable
        n_dom.remove(v);

        // Re-add to factors list
        if (!n_dom.isEmpty()) {

            FakeFactor f = new FakeFactor(n_dom, bn);

            factors.add(f);

            if (verbose > 2) {
                System.out.printf(" # final: %s ", n_dom);
            }
        }
    }

    /**
     * Simulation for a real factor
     */
    public static class FakeFactor {

        /**
         * Domain of the factor
         */
        public TIntArrayList dom;

        /**
         * Total number of probabilities
         */
        int size;

        /**
         * Build from the CPT of a the variable of a bayesian network
         *
         * @param v  variable of interest
         * @param bn network of interest
         */
        public FakeFactor(int v, BayesianNetwork bn) {
            setDomain(v, bn);
            setSize(bn);
        }

        /**
         * Build from a custom domain
         *
         * @param n_dom domain to use
         * @param bn    network of interest
         */
        public FakeFactor(TIntArrayList n_dom, BayesianNetwork bn) {
            dom = new TIntArrayList();
            dom.addAll(n_dom);

            setSize(bn);
        }

        /**
         * Update the size of the factor, based on the domain.
         *
         * @param bn network of interest
         */
        private void setSize(BayesianNetwork bn) {
            // Set size
            int s = 1;

            for (int l : dom.toArray()) {
                s *= bn.arity(l);
            }
            size = s;
        }

        /**
         * Set the domain from the map of a network (variable and its parents)
         *
         * @param v  variable of interest
         * @param bn network of interest
         */
        private void setDomain(int v, BayesianNetwork bn) {
            dom = new TIntArrayList();
            for (int p : bn.parents(v)) {
                dom.add(p);
            }
            dom.add(v);
        }
    }

}
