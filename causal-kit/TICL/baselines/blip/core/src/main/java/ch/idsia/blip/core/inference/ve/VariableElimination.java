package ch.idsia.blip.core.inference.ve;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.arcs.Undirected;
import ch.idsia.blip.core.inference.BaseInference;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import ch.idsia.blip.core.utils.data.hash.TIntIntHashMap;
import ch.idsia.blip.core.utils.data.set.TIntHashSet;
import ch.idsia.blip.core.utils.RandomStuff;

import java.util.*;

import static ch.idsia.blip.core.utils.data.ArrayUtils.sameArray;


public class VariableElimination extends BaseInference {
    public EliminMethod elim = EliminMethod.Heu;

    public VariableElimination(BayesianNetwork bayesNet, boolean verb) {
        super(bayesNet, verb);
    }

    public VariableElimination(BayesianNetwork bayesNet, int verb) {
        super(bayesNet, verb);
    }

    private void shuffleGreedy(int[] greedy) {
        int n_inver = this.rand.nextInt(1 + greedy.length / 4);

        for (int i = 0; i < n_inver; i++) {
            int a = this.rand.nextInt(greedy.length - 1);
            int b = this.rand.nextInt(greedy.length - 1);

            int t = greedy[b];

            greedy[b] = greedy[a];
            greedy[a] = t;
        }
    }

    private static int heuristicEliminationCriteria(int v, BayesianNetwork bn, Undirected arcs, TIntArrayList vars, EliminMethod method) {
        if (method == EliminMethod.MinFill) {
            return arcs.findFillArcs(v, vars).length;
        }
        if (method == EliminMethod.MinWidth) {
            TIntHashSet aux = new TIntHashSet();

            aux.add(v);
            int o;

            for (int i = 0; i < vars.size(); i++) {
                o = vars.getQuick(i);
                if ((o != v) && (arcs.check(v, o))) {
                    aux.add(o);
                }
            }
            for (int i : arcs.findFillArcs(v, vars)) {
                int[] aux2 = arcs.r_index(i);

                aux.add(aux2[0]);
                aux.add(aux2[1]);
            }
            int size = 1;

            for (int a : aux.toArray()) {
                size *= bn.arity(a);
            }
            return size;
        }
        return 0;
    }

    public static TIntArrayList findNuisance(BayesianNetwork bn, TIntArrayList query, TIntIntHashMap ev_keys, boolean verbose) {
        TIntArrayList nuisance = new TIntArrayList();

        TIntArrayList relevant = new TIntArrayList();

        relevant.addAll(query);
        relevant.addAll(ev_keys.keys());
        for (int v = 0; v < bn.n_var; v++) {
            if (!relevant.contains(v)) {
                if ((!containsOne(bn.getAncestors(v), relevant))
                        || (!containsOne(bn.getDescendents(v), relevant))) {
                    nuisance.add(v);
                }
            }
        }
        nuisance.sort();
        if (verbose) {
            System.out.printf("Nuisance: %s%n", nuisance);
        }
        return nuisance;
    }

    private static boolean containsOne(TIntHashSet v1, TIntArrayList v2) {
        for (int i = 0; i < v2.size(); i++) {
            if (v1.contains(v2.getQuick(i))) {
                return true;
            }
        }
        return false;
    }

    public BayesianFactor query(int[] query, TIntIntHashMap evidence) {
        if (this.verbose > 1) {
            logf("New query. Query: %s, evidence: %s\n", Arrays.toString(query),
                    evidence);
        }
        if ((query == null) || (query.length == 0)) {
            return new BayesianFactor();
        }
        Arrays.sort(query);
        if (evidence == null) {
            evidence = new TIntIntHashMap();
        }
        int[] ev_keys = evidence.keys();

        TIntArrayList barren = findBarren(query, ev_keys);

        TIntArrayList rel2 = new TIntArrayList();

        for (int v = 0; v < this.bn.n_var; v++) {
            if (!barren.contains(v)) {
                rel2.add(v);
            }
        }
        int[] rel = rel2.toArray();

        TIntArrayList init = new TIntArrayList();

        for (int r : rel) {
            if ((!find(r, query)) && (!evidence.containsKey(r))) {
                init.add(r);
            }
        }
        int[] order = new int[0];

        long start = System.currentTimeMillis();

        if (!init.isEmpty()) {
            order = findEliminationOrder(init.toArray());
        }
        if (this.verbose > 1) {
            System.out.printf("Elimination order: %s\n", Arrays.toString(order));
            RandomStuff.pf("Required: %d \n", System.currentTimeMillis() - start);
        }
        List<BayesianFactor> factors = new ArrayList<BayesianFactor>();
        int i;

        for (int v : rel) {
            BayesianFactor psi = new BayesianFactor(v, this.bn);

            for (int e : ev_keys) {
                if (find(e, psi.dom)) {
                    psi = psi.reduction(e, evidence.get(e));
                }
            }
            for (i = 0; i < barren.size(); i++) {
                int b = barren.get(i);

                if (find(b, psi.dom)) {
                    psi = psi.marginalization(b);
                }
            }
            if (psi.dom.length > 0) {
                factors.add(psi);
            }
        }
        for (int o : order) {
            if (this.verbose > 1) {
                System.out.println(this.verbose);
                System.out.printf("\nEliminating: %d # ", o);
            }
            List<BayesianFactor> toMult = new ArrayList<BayesianFactor>();

            for (BayesianFactor psi : factors) {

                if (find(o, (psi.dom))) {
                    toMult.add(psi);
                    if (this.verbose > 2) {
                        System.out.printf("\n %s", Arrays.toString((psi).dom));
                    }
                }
            }
            BayesianFactor n_psi = toMult.get(0);

            toMult.remove(0);
            factors.remove(n_psi);
            for (BayesianFactor psi : toMult) {
                try {
                    n_psi = n_psi.product(psi);
                } catch (Exception ex) {
                    System.out.println(ex.getMessage());
                }
                factors.remove(psi);
            }
            n_psi.normalize();

            n_psi = n_psi.marginalization(o);
            if (n_psi.dom.length > 0) {
                factors.add(n_psi);
                if (this.verbose > 2) {
                    System.out.printf(" # final: %s ",
                            Arrays.toString(n_psi.dom));
                }
            }
        }
        if (this.verbose > 1) {
            System.out.printf("Final factors: %d%n", factors.size());
        }
        BayesianFactor res = factors.get(0);

        factors.remove(0);
        if (!factors.isEmpty()) {
            for (BayesianFactor f : factors) {
                res = res.product(f);
            }
        }
        res.normalize();
        if ((!sameArray(res.dom, query)) && (this.verbose > 0)) {
            logf(
                    "Resulting factors has a domain (%s) different than query (%s)!",
                    Arrays.toString(res.dom), Arrays.toString(query));
        }
        if (this.verbose > 1) {
            logf(
                    String.format("Result: %s - %d%n", Arrays.toString(res.dom),
                    res.potent.length));
        }
        return res;
    }

    public BayesianFactor query(int q, TIntIntHashMap e) {
        return query(new int[] { q}, e);
    }

    public BayesianFactor query(int[] q) {
        return query(q, new TIntIntHashMap());
    }

    public BayesianFactor query(int q, HashMap<Integer, double[]> e) {
        return query(new int[] { q}, e);
    }

    private BayesianFactor query(int[] query, HashMap<Integer, double[]> evidence) {
        if (this.verbose > 1) {
            logf("New query. Query: %s, evidence: %s\n", Arrays.toString(query),
                    evidence);
        }
        if ((query == null) || (query.length == 0)) {
            return new BayesianFactor();
        }
        Arrays.sort(query);
        if (evidence == null) {
            evidence = new HashMap<Integer, double[]>();
        }
        Set<Integer> s = evidence.keySet();
        int[] ev_keys = new int[s.size()];
        int j = 0;

        for (int c : s) {
            ev_keys[(j++)] = c;
        }
        Arrays.sort(ev_keys);

        TIntArrayList barren = findBarren(query, ev_keys);

        TIntArrayList rel2 = new TIntArrayList();

        for (int v = 0; v < this.bn.n_var; v++) {
            if (!barren.contains(v)) {
                rel2.add(v);
            }
        }
        int[] rel = rel2.toArray();

        TIntArrayList init = new TIntArrayList();

        for (int r : rel) {
            if (!find(r, query)) {
                init.add(r);
            }
        }
        int[] order = new int[0];

        if (!init.isEmpty()) {
            order = findEliminationOrder(init.toArray());
        }
        if (this.verbose > 1) {
            System.out.printf("Elimination order: %s\n", Arrays.toString(order));
        }
        List<BayesianFactor> factors = new ArrayList<BayesianFactor>();

        for (int e : ev_keys) {
            BayesianFactor psi = new BayesianFactor(e, this.bn, evidence.get(e));

            factors.add(psi);
        }
        for (int v : rel) {
            BayesianFactor psi = new BayesianFactor(v, this.bn);

            factors.add(psi);
        }

        for (BayesianFactor psi : factors) {
            for (int i = 0; i < barren.size(); i++) {
                int b = barren.get(i);

                if (find(b, psi.dom)) {
                    psi = psi.marginalization(b);
                }
            }
        }

        for (int o : order) {
            if (this.verbose > 1) {
                System.out.println(this.verbose);
                System.out.printf("\nEliminating: %d # ", o);
            }
            List<BayesianFactor> toMult = new ArrayList<BayesianFactor>();

            for (BayesianFactor psi : toMult) {
                if (find(o, psi.dom)) {
                    toMult.add(psi);
                    if (this.verbose > 2) {
                        System.out.printf("\n %s", Arrays.toString(psi.dom));
                    }
                }
            }

            BayesianFactor n_psi = toMult.get(0);

            toMult.remove(0);
            factors.remove(n_psi);
            for (BayesianFactor psi : toMult) {
                n_psi = n_psi.product(psi);
                factors.remove(psi);
            }
            n_psi = n_psi.marginalization(o);
            if (n_psi.dom.length > 0) {
                factors.add(n_psi);
                if (this.verbose > 2) {
                    System.out.printf(" # final: %s ",
                            Arrays.toString(n_psi.dom));
                }
            }
        }
        if (this.verbose > 1) {
            System.out.printf("Final factors: %d%n", factors.size());
        }
        BayesianFactor res = (factors).get(0);

        factors.remove(0);
        if (!factors.isEmpty()) {
            for (BayesianFactor f : factors) {
                res = res.product(f);
            }
        }
        res.normalize();
        if ((sameArray(res.dom, query)) && (this.verbose > 0)) {
            logf(
                    "Resulting factors has a domain (%s) different than query (%s)!",
                    Arrays.toString(res.dom), Arrays.toString(query));
        }
        if (this.verbose > 1) {
            logf(
                    String.format("Result: %s - %d%n", Arrays.toString(res.dom),
                    res.potent.length));
        }
        return res;
    }

    public int[] findEliminationOrder(int[] vars) {
        return findNewEliminatonOrder(vars);
    }

    private int[] findNewEliminatonOrder(int[] vars) {
        if ((this.elim == EliminMethod.MinFill)
                || (this.elim == EliminMethod.MinWidth)) {
            return tryEliminationOrder(vars, this.elim);
        }
        if (this.elim == EliminMethod.Greedy) {
            Simulation sim = new Simulation(this.bn, this.verbose);

            return greedyEliminationOrder(vars, sim);
        }
        Simulation sim = new Simulation(this.bn, this.verbose);

        int[] ord = vars.clone();
        int[] bestOrd = vars.clone();
        double bestEval = Double.MAX_VALUE;

        for (int i = 0; i < vars.length * 2; i++) {
            shuffleGreedy(ord);

            double eval = sim.simulateInference(ord);

            if (eval < bestEval) {
                bestOrd = ord.clone();
                bestEval = eval;
            }
        }
        return bestOrd;
    }

    private int[] greedyEliminationOrder(int[] n_vars, Simulation sim) {
        TIntArrayList vars = new TIntArrayList(n_vars);

        int[] order = new int[n_vars.length];

        List<Simulation.FakeFactor> factors = new ArrayList<Simulation.FakeFactor>();

        for (int i = 0; i < vars.size(); i++) {
            Simulation.FakeFactor psi = new Simulation.FakeFactor(vars.get(i),
                    this.bn);

            factors.add(psi);
        }
        for (int i = 0; i < n_vars.length; i++) {
            int best_size = Integer.MAX_VALUE;
            int best_v = -1;

            for (int v : vars.toArray()) {
                List<Simulation.FakeFactor> n_factors = new ArrayList<Simulation.FakeFactor>();

                n_factors.addAll(factors);
                sim.simulateElimin(v, n_factors);

                int size = 0;

                for (Simulation.FakeFactor psi : n_factors) {
                    size += psi.size;
                }
                if (size < best_size) {
                    best_size = size;
                    best_v = v;
                }
            }
            sim.simulateElimin(best_v, factors);
            order[i] = best_v;
            vars.remove(best_v);
        }
        return order;
    }

    public int[] tryEliminationOrder(int[] n_vars, EliminMethod method) {
        int[] order = new int[n_vars.length];
        TIntArrayList vars = new TIntArrayList();

        vars.addAll(n_vars);

        Undirected arcs = new Undirected(this.bn);

        int k = 0;

        while (!vars.isEmpty()) {
            int bestV = -1;
            int bestHeur = Integer.MAX_VALUE;
            int v;
            int heur;

            for (int i = 0; i < vars.size(); i++) {
                v = vars.getQuick(i);
                heur = heuristicEliminationCriteria(v, this.bn, arcs, vars,
                        method);
                if (heur < bestHeur) {
                    bestHeur = heur;
                    bestV = v;
                }
            }
            order[(k++)] = bestV;
            vars.remove(bestV);
            for (int i : arcs.findFillArcs(bestV, vars)) {
                arcs.mark(i);
            }
        }
        return order;
    }

    private TIntArrayList findBarren(int[] query, int[] ev_keys) {
        TIntArrayList barren = new TIntArrayList();

        Arrays.sort(ev_keys);

        TIntArrayList rev_topol = new TIntArrayList();

        rev_topol.addAll(this.bn.getTopologicalOrder());

        rev_topol.reverse();
        for (int i = 0; i < rev_topol.size(); i++) {
            int v = rev_topol.get(i);

            if (isBarren(query, ev_keys, barren, v)) {
                barren.add(v);
            }
        }
        barren.sort();
        if (this.verbose > 2) {
            System.out.printf("Barren: %s\n", barren);
        }
        return barren;
    }

    private boolean isBarren(int[] query, int[] evidence, TIntArrayList barren, int v) {
        if (find(v, query)) {
            return false;
        }
        if (find(v, evidence)) {
            return false;
        }
        for (int c = 0; c < this.bn.n_var; c++) {
            if (Arrays.binarySearch(this.bn.parents(c), v) >= 0) {
                if (!barren.contains(c)) {
                    return false;
                }
            }
        }
        return true;
    }

    public BayesianFactor query(int i) {
        return query(new int[] { i});
    }

    public int mpe(int i, TIntIntHashMap m) {
        BayesianFactor v = query(i, m);
        int max = -1;
        double max_p = -1.7976931348623157E308D;

        for (int j = 0; j < v.potent.length; j++) {
            if (v.potent[j] > max_p) {
                max_p = v.potent[j];
                max = j;
            }
        }
        return max;
    }

    public static enum EliminMethod {
        MinFill, Heu, Greedy, MinWidth;

        private EliminMethod() {}
    }
}
