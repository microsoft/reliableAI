package ch.idsia.blip.core.inference.ve;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.data.array.TDoubleArrayList;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import ch.idsia.blip.core.utils.data.hash.TIntIntHashMap;

import java.util.*;

import static ch.idsia.blip.core.utils.data.ArrayUtils.*;
import static ch.idsia.blip.core.utils.RandomStuff.p;


/**
 * Inference factor based on Bayesian probabilities
 */
public class BayesianFactor {

    /**
     * Flag if we want to use log computations
     */
    public static boolean logComp = true;

    /**
     * Domain
     */
    public int[] dom;

    /**
     * Cardinality of each variable
     */
    public int[] card;

    /**
     * Stride of each variable
     */
    public int[] stride;

    /**
     * Total size of potentials
     */
    private int size;

    /**
     * Potentials (ordered by variables in numerical order)
     */
    public double[] potent;

    /**
     * Void constructor.
     */
    public BayesianFactor() {
        dom = new int[0];
        card = new int[0];
        updateStride();
        potent = new double[size];
    }

    /**
     * Normal construction: provide domain and arities of variable of interest
     *
     * @param n_dom  new domain
     * @param n_card new cardinality
     */
    public BayesianFactor(TreeSet<Integer> n_dom, Map<Integer, Integer> n_card) {
        // Set domain
        TIntArrayList aux_dom = new TIntArrayList();
        TIntArrayList aux_card = new TIntArrayList();

        for (int v : n_dom) {
            aux_dom.add(v);
            aux_card.add(n_card.get(v));
        }

        dom = aux_dom.toArray();
        card = aux_card.toArray();

        // Set stride
        updateStride();

        // Prepare potentials
        potent = new double[size];
    }

    /**
     * Construction as reduction of the given factor by the given variable
     *
     * @param psi original factor
     * @param v   variable to remove
     */
    public BayesianFactor(BayesianFactor psi, int v) {

        int p_v = pos(v, psi.dom);

        // Set domain
        dom = removeElementAt(psi.dom, p_v);

        // Set cardinality
        card = removeElementAt(psi.card, p_v);

        // Set stride
        updateStride();

        // Prepare potentials
        potent = new double[size];
    }

    /**
     * Construction as union of the given factors
     *
     * @param psi1 first factor
     * @param psi2 second factor
     */
    public BayesianFactor(BayesianFactor psi1, BayesianFactor psi2) {

        // Set domain
        TIntArrayList aux_dom = new TIntArrayList();
        TIntArrayList aux_card = new TIntArrayList();

        int i = 0;
        int j = 0;

        while ((i < psi1.dom.length) && (j < psi2.dom.length)) {
            if (psi1.dom[i] == psi2.dom[j]) {
                aux_dom.add(psi1.dom[i]);
                aux_card.add(psi1.card[i]);
                i++;
                j++;
            } else if (psi1.dom[i] < psi2.dom[j]) {
                aux_dom.add(psi1.dom[i]);
                aux_card.add(psi1.card[i]);
                i++;
            } else {
                aux_dom.add(psi2.dom[j]);
                aux_card.add(psi2.card[j]);
                j++;
            }
        }

        while (i < psi1.dom.length) {
            aux_dom.add(psi1.dom[i]);
            aux_card.add(psi1.card[i]);
            i++;
        }

        while (j < psi2.dom.length) {
            aux_dom.add(psi2.dom[j]);
            aux_card.add(psi2.card[j]);
            j++;
        }

        card = aux_card.toArray();
        dom = aux_dom.toArray();

        // Set stride
        updateStride();

        // Prepare potentials
        potent = new double[size];

    }

    /**
     * Construction from the CPT of a Bayesian Network
     *
     * @param v  variable of interest
     * @param bn bayesian network
     */
    public BayesianFactor(int v, BayesianNetwork bn) {

        // Set domain
        int[] ps = bn.parents(v);

        dom = new int[ps.length + 1];
        for (int i = 0; i < ps.length; i++) {
            dom[i] = ps[i];
        }
        dom[ps.length] = v;
        Arrays.sort(dom);

        // Set cardinality
        card = new int[dom.length];
        for (int i = 0; i < dom.length; i++) {
            card[i] = bn.arity(dom[i]);
        }

        // Set stride
        updateStride();

        /*
         System.out.println(bn.name( v));
         System.out.println(dom + " - " + card + " - " + stride + " - " + size);
         System.out.println(bn.potentials(v).length);
         */

        // Prepare potentials
        potent = new double[size];
        updatePotent(bn.potentials(v));
        reorganizePotent(bn, v);

    }

    private void reorganizePotent(BayesianNetwork bn, int v) {

        if (bn.parents(v).length == 0) {
            return;
        }

        double[] new_pt = new double[potent.length];

        int[] ord = cloneArray(bn.parents(v));

        ArrayUtils.reverse(ord);
        ord = addArray(v, ord);
        int[] new_ord = expandArray(bn.parents(v), v);

        if (Arrays.equals(ord, new_ord)) {
            return;
        }

        for (int i = 0; i < potent.length; i++) {
            TIntIntHashMap t = bn.getAssignmentFromIndex(ord, v, i);
            int n_i = bn.getIndexFromAssignment(new_ord, t);

            // pf("%d - %s - %d \n", i, t, n_i);
            new_pt[n_i] = potent[i];
        }

        potent = new_pt;
    }

    public BayesianFactor(int v, BayesianNetwork bn, double[] new_p) {
        // Set domain
        dom = new int[] { v};

        // Set cardinality
        card = new int[] { bn.arity(v)};

        // Set stride
        updateStride();

        potent = new double[size];
        cloneArray(new_p, potent);
    }

    /**
     * @param nums List of values
     * @return sum of values
     */
    private static double logSum(TDoubleArrayList nums) {

        double max_exp = nums.max();
        double sum = 0.0;

        for (int i = 0; i < nums.size(); i++) {
            sum += Math.pow(10, nums.getQuick(i) - max_exp);
            // printf("nums[thread]: %.5f, max_exp: %.5f, diff: %.5f", nums[thread], max_exp, nums[thread] - max_exp);
        }

        return Math.log10(sum) + max_exp;
    }

    /**
     * @param nums List of values
     * @return sum of values
     */
    private static double listSum(TDoubleArrayList nums) {
        double s = 0;

        for (int i = 0; i < nums.size(); i++) {
            s += nums.getQuick(i);
        }
        return s;
    }

    /**
     * Update value of stride
     */
    private void updateStride() {
        int s = 1;

        stride = new int[dom.length];
        for (int i = 0; i < dom.length; i++) {
            stride[i] = s;
            s *= card[i];
        }
        size = s;
    }

    /**
     * Update potentials from the given
     *
     * @param n_potent potentials to copy
     */
    public void updatePotent(double[] n_potent) {

        // Copy potentials
        int l = 0;

        for (double p : n_potent) {
            potent[l++] = logComp ? Math.log10(p) : p;
        }
    }

    public String toString() {

        return String.format(
                "size: %d && dom: %s && card: %s && stride; %s && potent:%s\n\n",
                size, Arrays.toString(dom), Arrays.toString(card),
                Arrays.toString(stride), printPotent());
    }

    /**
     * Check if potentials sum to 1
     *
     * @return if potentials are ok
     */
    private double diffPotent() {
        double sum = 0.0;

        for (double p : potent) {
            sum += logComp ? Math.pow(10, p) : p;
        }

        return Math.abs(sum - 1.0);
    }

    /**
     * Print potentials
     *
     * @return description of potentials
     */
    public String printPotent() {
        StringBuilder str = new StringBuilder();

        str.append("[ ");
        for (int j = 0; j < potent.length; j++) {
            str.append(String.format("%.6f ", getPotent(j)));
        }
        str.append("]");

        return str.toString();
    }

    /**
     * Normalize to 1 the potentials distribution
     */
    public void normalize() {

        double sum = 0.0;

        for (int i = 0; i < size; i++) {
            sum += logComp ? Math.pow(10, potent[i]) : potent[i];
        }
        for (int i = 0; i < size; i++) {
            potent[i] = logComp ? potent[i] - Math.log10(sum) : potent[i] / sum;
        }

    }

    /**
     * Get potentials from index
     *
     * @param j index
     * @return potential
     */
    public double getPotent(int j) {
        return logComp ? Math.pow(10, potent[j]) : potent[j];
    }

    /**
     * Get all potentials (eventually in normal form)
     *
     * @return potentials of the factor
     */
    public double[] getPotent() {
        double[] d = new double[size];

        for (int i = 0; i < size; i++) {
            d[i] = logComp ? Math.pow(10, potent[i]) : potent[i];
        }
        return d;
    }

    /**
     * Reduce a factor by the given evidence
     *
     * @param e   index of the evidence variable
     * @param val value of the evidence variable
     * @return resulting factor
     */
    public BayesianFactor reduction(int e, int val) {

        BayesianFactor n_psi = new BayesianFactor(this, e);

        int p = n_psi.size;

        int i_v = pos(e, dom);

        int v_card = card[i_v];
        int v_stride = stride[i_v];

        // First position of evidence variable with given variable
        int k = val * v_stride;
        int l = 0;

        for (int i = 0; i < p; i++) {

            // System.out.println(thread+ " " + tw);
            n_psi.potent[i] = logComp ? Math.pow(10, potent[k]) : potent[k];

            // Advance to next position in the potentials
            k++;
            l++;
            if (l == v_stride) {
                l = 0;
                k += v_stride * (v_card - 1);
            }
        }

        if (logComp) {
            // Go back to log
            for (int i = 0; i < p; i++) {
                n_psi.potent[i] = Math.log10(n_psi.potent[i]);
            }
        }

        /*
         // Normalize?
         for (int thread = 0; thread < p; thread++) {
         n_potent[thread] = logComp ? Math.log10 (n_potent[thread] / sum)  : n_potent[thread] / sum;
         }
         */
        return n_psi;
    }

    /**
     * Compute the product of two factors, each with its domain and probabilities distribution.
     *
     * @param psi2 Second factor
     * @return resulting factor
     */
    public BayesianFactor product(BayesianFactor psi2) {

        int j = 0;
        int k = 0;

        // Create new factor
        BayesianFactor n_psi = new BayesianFactor(this, psi2);
        int d = n_psi.dom.length;
        int p = n_psi.potent.length;

        // System.out.println("N_psi: " + n_psi.dom + " - " + RandomStuff.printOrdMap(n_psi.stride));

        // Auxilary structures
        int[] assignment = new int[d];
        int[] card = new int[d];

        int[] stride1 = new int[d];
        int[] shift1 = new int[d];

        int[] stride2 = new int[d];
        int[] shift2 = new int[d];

        for (int m = 0; m < n_psi.dom.length; m++) {
            assignment[m] = 0;

            int v = n_psi.dom[m];

            card[m] = n_psi.card[m];

            stride1[m] = (find(v, dom) ? stride[pos(v, dom)] : 0);
            shift1[m] = (card[m] - 1) * stride1[m];

            stride2[m] = (find(v, psi2.dom) ? psi2.stride[pos(v, psi2.dom)] : 0);
            shift2[m] = (card[m] - 1) * stride2[m];
        }

        // System.out.println("p: " + p);
        for (int i = 0; i < p; i++) {
            // n_psi.potent[thread] = potent[j] + psi2.potent[tw];
            n_psi.potent[i] = logComp
                    ? potent[j] + psi2.potent[k]
                    : potent[j] * psi2.potent[k];
            for (int l = 0; l < d; l++) {
                assignment[l] += 1;
                if (assignment[l] == card[l]) {
                    assignment[l] = 0;
                    j -= shift1[l];
                    k -= shift2[l];
                } else {
                    j += stride1[l];
                    k += stride2[l];
                    break;
                }
            }
        }

        return n_psi;
    }

    /**
     * Sums out a variable from a factor
     *
     * @param v variable to sum out
     * @return new factor
     */
    public BayesianFactor marginalization(int v) {

        BayesianFactor n_psi = new BayesianFactor(this, v);

        int p = n_psi.size;

        int l = 0;
        int j = 0;

        int i_v = pos(v, dom);

        int v_card = card[i_v];
        int v_stride = stride[i_v];

        List<TDoubleArrayList> aux = new ArrayList<TDoubleArrayList>();

        for (int i = 0; i < p; i++) {
            aux.add(new TDoubleArrayList());
        }

        for (int i = 0; i < p; i++) {

            // Sum (of logarithms)
            int k = j;

            for (int h = 0; h < v_card; h++) {
                aux.get(i).add(potent[k]);
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

        for (int i = 0; i < p; i++) {
            n_psi.potent[i] += logComp
                    ? logSum(aux.get(i))
                    : listSum(aux.get(i));
        }

        /*
         // Normalize?
         for (int thread = 0; thread < p; thread++) {
         // n_psi.potent[thread] = logComp ? Math.log10 (n_psi.potent[thread] / sum)  : n_psi.potent[thread] / sum;
         n_psi.potent[thread] = logComp ? Math.log10 (n_psi.potent[thread] )  : n_psi.potent[thread] ;
         }*/

        return n_psi;
    }

    /**
     * Combine a factor of the same dimension
     *
     * @param psi2 Second factor
     */
    public void combine(BayesianFactor psi2) {

        // System.out.printf("psi1: %s%n", psi1);
        // System.out.printf("psi2: %s%n", psi2);

        double sum = 0.0;

        for (int i = 0; i < potent.length; i++) {
            potent[i] = logComp
                    ? potent[i] + psi2.potent[i]
                    : potent[i] * psi2.potent[i];
            sum += logComp ? Math.pow(10, potent[i]) : potent[i];
        }
        for (int i = 0; i < potent.length; i++) {
            potent[i] = logComp ? potent[i] - Math.log10(sum) : potent[i] / sum;
        }
    }

    public int mostProbable() {
        Double best;
        int ix = -1;

        if (logComp) {
            best = -Double.MAX_VALUE;
        } else {
            best = 0.0;
        }

        for (int i = 0; i < potent.length; i++) {
            if (potent[i] > best) {
                best = potent[i];
                ix = i;
            }
        }
        return ix;
    }
}
