package ch.idsia.blip.core.utils.other;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.inference.ve.BayesianFactor;
import ch.idsia.blip.core.inference.ve.VariableElimination;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;

import static ch.idsia.blip.core.utils.data.ArrayUtils.pos;


/**
 * KL divergence
 */

public class KLDiv {

    /**
     * Minimum value for probabilities (avoid zero)
     */
    private final double eps = Math.pow(2, -20);

    /**
     * Verbose flag
     */
    public boolean verbose;
    private VariableElimination inf_Q;
    private VariableElimination inf_P;

    /**
     * Compute the Kullback-Liebler Divergence between the two bayesian networks.
     *
     * @param bn_P First Bayesian network (P)
     * @param bn_Q Second Bayesian network (Q)
     * @return value of the KL Divergence
     */
    public double getKLDivergence(BayesianNetwork bn_P, BayesianNetwork bn_Q) {

        double kl = 0;
        double v = 0;
        double k;

        inf_Q = new VariableElimination(bn_Q, false);
        inf_P = new VariableElimination(bn_P, false);

        boolean sameStructure = bn_P.equalStructure(bn_Q);

        for (int i = 0; i < bn_P.n_var; i++) {

            if (verbose) {
                System.out.printf("\nVar: %s (%d)\n", bn_P.name(i), i);
            }

            // Get joint distribution over P: P(X_i, pa_i)
            double[] joint_P = getJointDistribution(bn_P, i);
            // Get condition distribution over P: P(X_i | pa_i)
            double[] cond_P = bn_P.potentials(i);
            // Get condition distribution over P: P(X_i | pa_i)
            double[] cond_Q;

            if (false) { // sameStructure) {
                cond_Q = bn_Q.potentials(i);
            } else {
                cond_Q = getCondDistribution(bn_P, i);
            }

            // Sum
            for (int j = 0; j < joint_P.length; j++) {
                double j_p = check(joint_P[j]);
                double c_p = check(cond_P[j]);
                double c_q = check(cond_Q[j]);

                k = j_p * (Math.log(c_p) - Math.log(c_q));
                kl += k;
            }
        }

        return kl;
    }

    /**
     * Get the conditional distribution over bn_P of the variable v and its parents in the bn_Q map.
     *
     * @param bn_P network where to pick the parents for the variable
     * @param v    variable of interest
     * @return joint distribution as factor
     */
    public double[] getCondDistribution(BayesianNetwork bn_P, int v) {

        TIntArrayList query = new TIntArrayList();

        // If the variable has no parents in bn_P
        if (bn_P.parents(v).length == 0) {
            query.add(v);
            BayesianFactor j_inf = inf_Q.query(query.toArray());

            return j_inf.getPotent();
        }

        // Get Q(pa_i) in parents
        for (int p : bn_P.parents(v)) {
            query.add(p);
        }
        BayesianFactor p_inf = inf_Q.query(query.toArray());
        double[] parents = p_inf.getPotent();

        // Get Q(X_i, pa_i) in joint
        query.add(v);
        BayesianFactor j_inf = inf_Q.query(query.toArray());
        double[] joint = j_inf.getPotent();
        double[] cond = new double[joint.length];

        // Get Q(X_i | pa_i) in cond

        int l = 0;
        int j = 0;

        int v_card = j_inf.card[pos(v, j_inf.dom)];
        int v_stride = j_inf.stride[pos(v, j_inf.dom)];

        int p = parents.length;

        for (double parent : parents) {

            int k = j;

            for (int h = 0; h < v_card; h++) {
                cond[k] = joint[k] / parent;
                k += v_stride;
            }

            // Advance to next starting position
            l++;
            if (l == v_stride) {
                l = 0;
                j = k - v_stride + 1;
            } else {
                j++;
            }
        }

        return cond;
    }

    /**
     * @param v potential value
     * @return max(v, epsilon value)
     */
    private double check(double v) {
        return (v > eps) ? v : eps;
    }

    /**
     * Get the joint distribution over bn_P of the variable v and its parents in the bn_Q map.
     *
     * @param bn_Q network where to writeSample the query
     * @param v    variable of interest
     * @return joint distribution as factor
     */
    public double[] getJointDistribution(BayesianNetwork bn_Q, int v) {

        // Do the query
        TIntArrayList q = new TIntArrayList();

        q.add(v);
        for (int p : bn_Q.parents(v)) {
            q.add(p);
        }

        BayesianFactor fact = inf_P.query(q.toArray());

        return fact.getPotent();
    }

}
