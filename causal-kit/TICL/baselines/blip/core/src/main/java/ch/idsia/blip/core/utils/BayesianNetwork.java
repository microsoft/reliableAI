package ch.idsia.blip.core.utils;


import ch.idsia.blip.core.Base;
import ch.idsia.blip.core.utils.arcs.NamedDirected;
import ch.idsia.blip.core.utils.arcs.Undirected;
import ch.idsia.blip.core.utils.other.TopologicalOrder;
import ch.idsia.blip.core.utils.tw.TreeWidth;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import ch.idsia.blip.core.utils.data.hash.TIntIntHashMap;
import ch.idsia.blip.core.utils.data.set.TIntHashSet;
import ch.idsia.blip.core.utils.exp.CyclicGraphException;
import ch.idsia.blip.core.utils.exp.TreeWidthExceededException;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.*;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.*;


/**
 * A Bayesian network, and the main operations to do on them.
 */
public class BayesianNetwork extends Base implements Serializable {

    /**
     * Logger.
     */
    private static final Logger log = Logger.getLogger(
            BayesianNetwork.class.getName());

    /**
     * Lower value to consider in probabilities computation
     */
    private final double eps = Math.pow(2, -10);

    /**
     * Number of variable in the network.
     */
    public int n_var;

    /**
     * Name of each variable.
     */
    public String[] l_nm_var;

    /**
     * Arity of each variable.
     */
    public int[] l_ar_var;

    /**
     * List of values of each variable.
     */
    public String[][] l_values_var;

    /**
     * Parents of each variable.
     */
    public int[][] l_parent_var;

    /**
     * CPT of each variable (for the order of the potentials see the FAQ).
     */
    public double[][] l_potential_var;

    /**
     * Topological order on bn
     */
    private int[] topOrder;

    public HashMap<String, double[]> positions;

    /**
     * Construct void network
     *
     * @param n size of network
     */
    public BayesianNetwork(int n) {

        l_nm_var = new String[n];
        l_ar_var = new int[n]; // List of variables arity
        l_values_var = new String[n][]; // List of variables row_values
        l_parent_var = new int[n][]; // List of parent set for variable
        l_potential_var = new double[n][]; // List of probabilities

        for (int i = 0; i < n; i++) {
            l_nm_var[i] = "N" + String.valueOf(i);
            l_ar_var[i] = 0;
            l_values_var[i] = new String[0];
            l_parent_var[i] = new int[0];
            l_potential_var[i] = new double[0];
        }
        n_var = n;
    }

    public BayesianNetwork(ParentSet[] best_str) {

        this(best_str.length);

        for (int i = 0; i < best_str.length; i++) {
            l_parent_var[i] = best_str[i].parents;
        }
    }

    public double getLogLik10(short[] sample) {
        assert (sample.length == this.n_var);

        double logLik = 0.0D;
        double l = 0.0D;

        for (int i = 0; i < this.n_var; i++) {
            double p = getPotential(i, sample);

            l = Math.log10(p);

            logLik += l;
        }
        return logLik;
    }

    /**
     * Get the log likelihood for a complete sample
     *
     * @param sample complete assignment of all variables
     * @return computed log likelihood
     */
    public double getLogLik(short[] sample) {
        assert (sample.length == n_var);

        double logLik = 0.0;
        double l = 0;

        // System.out.println(Arrays.toString(sample));

        for (int i = 0; i < n_var; i++) {
            double p = getPotential(i, sample);

            // pf("%d | %.5f \n", i, p);
            l = Math.log(p);

            logLik += l;
        }

        return logLik;
    }

    /**
     * Get the probability of the assignment for a variable
     *
     * @param n      index of the variable
     * @param sample complete assignment of all variables
     * @return probability of assignment
     */
    public double getPotential(int n, short[] sample) {

        int ix = potentialIndex(n, sample);

        ix *= arity(n);
        ix += sample[n];
        double[] p = potential(n);
        double ps;

        if (ix >= p.length) {
            ps = 0;
        } else {
            ps = p[ix];
        }

        if (ps < eps) {
            ps = eps;
        }

        return ps;
    }

    /**
     * Returns a CPT index from a sample, given an arbitrary order.
     *
     * @param sample List of values for each variable
     * @return index associated with the given sample and the order of variables in the CPT.
     */
    public int potentialIndex(int n, short[] sample) {
        int ix = 0;
        // Add parents index value
        int ix_ml = 1;

        int[] ps = parents(n);

        // for (int n : order) {
        for (int i = ps.length - 1; i >= 0; i--) {
            int p = ps[i];

            ix += sample[p] * ix_ml; // Shift index
            ix_ml *= arity(p); // Compute cumulative shifter
            // log.severe("P:" + par + " - v: " + sample[par] + " - " + arity(par) + " - " + ix );
        }
        return ix;
    }

    private double[] potential(int var) {
        return l_potential_var[var];
    }

    /**
     * Returns a sample from a CPT index, given an arbitrary order.
     *
     * @param order List of variable indexes (order in the CPT)
     * @param n_var total number of variables
     * @param index index of the CPT of the variable
     * @return List of values for each variable, associated with the given index.
     */
    public TIntIntHashMap getAssignmentFromIndex(int[] order, int n_var, int index) {
        TIntIntHashMap t = new TIntIntHashMap();

        // for (int i = order.length-1; i >= 0; i--) {

        // int n = order[i];
        for (int n : order) {
            int ar = arity(n);

            t.put(n, (index % ar));
            index /= ar;
        }
        return t;
    }

    /**
     * @return a cached topological order.
     */
    public int[] getTopologicalOrder() {
        if (topOrder == null) {
            topOrder = TopologicalOrder.find(n_var, l_parent_var);
        }
        return topOrder;
    }

    public String toString() {

        StringBuilder str = new StringBuilder(String.format("Num: %d\n", n_var));

        str.append("# Variables\n");
        for (int i = 0; i < n_var; i++) {
            if (l_nm_var != null) {
                str.append("   ").append(name(i));
            }
            if (l_ar_var != null) {
                str.append(" (").append(arity(i)).append(") ");
            }
            if (l_values_var != null) {
                str.append(Arrays.toString(values(i)));
            }

            str.append("\n    ");

            if (l_parent_var != null) {
                printStructure(str, i);
            }

            if (l_potential_var != null) {
                str.append(" - (").append(potential(i).length).append(") ").append(
                        " [ ");
                for (double p : potential(i)) {
                    str.append(String.format("%.3f ", p));
                }
                str.append("]");
            }

            str.append("\n\n");
        }

        /*
         new_str.append("\n# Structure\n\n");
         for (int thread = 0; thread < n_var; thread++) {
         if (l_parent_var != null) {
         new_str.append("  ").append(name( thread)).append(" - ");
         printStructure(new_str, thread);
         new_str.append("\n");
         }
         }*/

        return str.toString();
    }

    public String[] values(int i) {
        return l_values_var[i];
    }

    public String name(int i) {
        return l_nm_var[i];
    }

    public void toGraph(PrintWriter w) {

        try {

            wf(w, "digraph Base {\n");
            wf(w, "labelloc=\"t\"\n");
            wf(w, "label=\"Nodes: %d, arcs: %s\"\n", n_var, numEdges());
            ;
            for (int v1 = 0; v1 < n_var; v1++) {
                wf(w, "\"%s\" \n", name(v1));
                for (int v2 : parents(v1)) {
                    wf(w, "\"%s\" -> \"%s\" \n", name(v2), name(v1));
                }
            }
            wf(w, "}\n");
        } catch (IOException ex) {
            logExp(log, ex);
        }
    }

    private void printStructure(StringBuilder str, int i) {
        int[] p = parents(i);

        str.append(" [");
        if (p.length > 0) {

            str.append(name(p[0]));
            for (int j = 1; j < p.length; j++) {
                str.append(", ").append(name(p[j]));
            }
        }
        str.append("]");
    }

    /**
     * @param v variable of interest
     * @return all the descendants of the variable (indexes)
     */
    public TIntHashSet getDescendents(int v) {
        TIntHashSet aux = new TIntHashSet();

        PriorityQueue<Integer> ev = new PriorityQueue<Integer>();

        for (int p : parents(v)) {
            ev.add(p);
        }

        while (!ev.isEmpty()) {
            int a = ev.poll();

            for (int p : parents(a)) {
                ev.add(p);
            }
            aux.add(a);
        }

        return aux;
    }

    /**
     * @param v variable of interest
     * @return all the ancestors of the variable (indexes)
     */
    public TIntHashSet getAncestors(int v) {
        TIntHashSet aux = new TIntHashSet();
        int[] ord = getTopologicalOrder();

        int i = 0;

        while (ord[i] != v) {
            i++;
        }

        while (i < n_var) {
            int d = ord[i];

            for (int p : parents(d)) {
                if (aux.contains(p) || (p == v)) {
                    aux.add(d);
                }
            }
            i++;
        }

        return aux;
    }

    /**
     * Find out if the map is equivalent of the dand network.
     *
     * @param bn_q the dand network
     * @return if the map of the network (number of nodes, parents for each node) is equivalent.
     */
    public boolean equalStructure(BayesianNetwork bn_q) {
        if (n_var != bn_q.n_var) {
            return false;
        }

        for (int i = 0; i < n_var; i++) {
            int[] my = parents(i);

            Arrays.sort(my);
            int[] oth = bn_q.parents(i);

            Arrays.sort(oth);
            if (!(Arrays.equals(my, oth))) {
                return false;
            }
        }

        return true;
    }

    /**
     * Check if the network is acyclic.
     *
     * @return whether the graph is acyclic.
     */
    public boolean isAcyclic() {
        int[] res = getTopologicalOrder();

        return ((res != null) && (res.length == n_var));
    }

    public Undirected moralize() {

        Undirected ar = new Undirected(n_var);

        for (int n = 0; n < n_var; n++) {
            int[] ps = parents(n);

            for (int j = 0; j < ps.length; j++) {
                ar.mark(n, ps[j]);
                for (int j2 = j + 1; j2 < ps.length; j2++) {
                    ar.mark(ps[j], ps[j2]);
                }
            }
        }

        return ar;
    }

    public List<TIntArrayList> findPaths(Undirected ar) {

        List<TIntArrayList> candidates = new ArrayList<TIntArrayList>();

        boolean[] visited = new boolean[n_var];

        for (int n = 0; n < n_var; n++) {
            visited[n] = false;
        }

        for (int n = 0; n < n_var; n++) {
            if (visited[n]) {
                continue;
            }
            TIntArrayList path = new TIntArrayList();

            visit(-1, n, ar, path, candidates, visited);
        }

        List<TIntArrayList> paths = select(ar, candidates);
        List<TIntArrayList> paths2 = new ArrayList<TIntArrayList>();

        for (TIntArrayList pt : paths) {
            if (!paths2.contains(pt)) {
                paths2.add(pt);
            }
        }

        return paths2;
    }

    private List<TIntArrayList> select(Undirected ar, List<TIntArrayList> candidates) {
        List<TIntArrayList> paths = new ArrayList<TIntArrayList>();

        while (!candidates.isEmpty()) {
            TIntArrayList path = candidates.get(0);

            candidates.remove(0);
            if (cordless(path, ar, candidates)) {
                paths.add(path);
            }
        }

        return paths;
    }

    private void visit(int p, int n, Undirected ar, TIntArrayList path, List<TIntArrayList> candidate, boolean[] visited) {

        if (visited[n]) {
            if (path.contains(n)) {
                // int g = path.lastIndexOf(n);

                // path.add(n);
                path.remove(0, path.lastIndexOf(n));

                // Keep only path of 4 or more nodes,
                // with no cords
                if (path.size() > 3) {
                    candidate.add(path);
                }
            }
            return;
        }

        visited[n] = true;
        path.add(n);
        for (int n2 = 0; n2 < n_var; n2++) {
            if (n != n2 && p != n2 && ar.check(n, n2)) {
                TIntArrayList path2 = new TIntArrayList();

                path2.addAll(path);
                visit(n, n2, ar, path2, candidate, visited);
            }
        }
    }

    private boolean cordless(TIntArrayList path, Undirected ar, List<TIntArrayList> candidates) {

        boolean cordless = true;

        int[] p = path.toArray();

        for (int j = 0; j < p.length; j++) {
            for (int j1 = j + 2; j1 < p.length; j1++) {

                if (j == 0 && j1 == p.length - 1) {
                    continue;
                }

                if (ar.check(p[j], p[j1])) {
                    // System.out.printf("%d - %d, %b", p[j], p[j1], ar.check(p[j], p[j1]));
                    TIntArrayList new_path = new TIntArrayList();

                    new_path.addAll(path);
                    new_path.remove(j + 1, j1 - (j + 1));

                    if (new_path.size() > 3) {
                        candidates.add(new_path);
                    }
                    cordless = false;
                }
            }
        }

        // System.out.println("@@@@ " + cordless + " " + path );

        return cordless;
    }

    public NamedDirected directed() {
        NamedDirected D = new NamedDirected(n_var);

        for (int n = 0; n < n_var; n++) {
            for (int p : parents(n)) {
                D.mark(p, n);
            }
        }
        D.setNames(l_nm_var);
        return D;
    }

    public int arity(int j) {
        return l_ar_var[j];
    }

    public int[] parents(int j) {
        return l_parent_var[j];
    }

    public void checkTreeWidth(int tw) throws TreeWidthExceededException {
        int k = TreeWidth.go(this);

        if (k > tw) {
            throw new TreeWidthExceededException(k, tw);
        }
    }

    public void checkAcyclic() throws CyclicGraphException {
        if (!isAcyclic()) {
            throw new CyclicGraphException(this);
        }
    }

    public int[] blanket(int i) {
        TIntHashSet b = new TIntHashSet();

        b.addAll(parents(i));
        for (int j = 0; j < n_var; j++) {
            if (!find(i, parents(j))) {
                continue;
            }
            b.add(j);
            b.addAll(parents(j));
        }
        b.remove(i);
        int[] s = b.toArray();

        Arrays.sort(s);
        return s;
    }

    public int numEdges() {
        int t = 0;

        for (int[] p : l_parent_var) {
            t += p.length;
        }
        return t;
    }

    public int treeWidth() {
        return TreeWidth.go(this);
    }

    public int[] childrens(int t) {
        TIntArrayList g = new TIntArrayList();

        for (int i = 0; i < n_var; i++) {
            if (find(t, parents(i))) {
                g.add(i);
            }
        }
        return g.toArray();
    }

    public double[] potentials(int i) {
        return l_potential_var[i];
    }

    public void setParents(int i, int[] parents) {
        l_parent_var[i] = parents;
    }

    public void toGraph(PrintWriter w, TreeSet<String> highligth) {
        try {

            wf(w, "digraph Base {\n");
            wf(w, "labelloc=\"t\"\n");
            wf(w, "label=\"Nodes: %d, arcs: %s\"\n", n_var, numEdges());
            ;
            for (int v1 = 0; v1 < n_var; v1++) {
                String v = name(v1);

                if (highligth != null && highligth.contains(v)) {
                    wf(w, "\"%s\" [style=filled, fillcolor=red]  \n", v);
                } else {
                    wf(w, "\"%s\" \n", v);
                }
                for (int v2 : parents(v1)) {
                    wf(w, "\"%s\" -> \"%s\" \n", name(v2), v);
                }
            }
            wf(w, "}\n");
        } catch (IOException ex) {
            logExp(log, ex);
        }
    }

    public void toGraph(PrintWriter w, int[] keys) {
        try {
            if (keys != null) {
                Arrays.sort(keys);
            }

            wf(w, "digraph Base {\n");
            wf(w, "labelloc=\"t\"\n");
            wf(w, "label=\"Nodes: %d, arcs: %s\"\n", n_var, numEdges());
            ;
            for (int v1 = 0; v1 < n_var; v1++) {
                String v = name(v1);

                if (keys != null && find(v1, keys)) {
                    wf(w, "\"%s\" [style=filled, fillcolor=red]  \n", v);
                } else {
                    wf(w, "\"%s\" \n", v);
                }
                for (int v2 : parents(v1)) {
                    wf(w, "\"%s\" -> \"%s\" \n", name(v2), v);
                }
            }
            wf(w, "}\n");
        } catch (IOException ex) {
            logExp(log, ex);
        }
    }

    public void writeGraph(String s, int[] keys) {
        try {
            if (!new File(System.getProperty("user.home") + "/Tools/dot").exists()) {
                return;
            }

            PrintWriter w = new PrintWriter(s + ".dot", "UTF-8");

            toGraph(w, keys);
            w.close();

            dot(f("./dot -Tpng %s.dot -o %s.png", s, s));
            dot(f("./dot -Tpdf %s.dot -o %s.pdf", s, s));

        } catch (IOException e) {
            logExp(e);
        } catch (InterruptedException e) {
            logExp(e);
        } catch (Exception e) {
            logExp(e);
        }

    }

    private void dot(String h) throws IOException, InterruptedException {
        Process proc = Runtime.getRuntime().exec(h, new String[0],
                new File(System.getProperty("user.home") + "/Tools"));

        exec(proc);

        // To close them
        proc.getInputStream().close();
        proc.getOutputStream().close();
        proc.getErrorStream().close();
    }

    public void writeGraph(String s) {
        writeGraph(s, null);
    }

    public void setPotential(int v, double[] new_probs) {
        l_potential_var[v] = new_probs;
    }

    public int getIndexFromAssignment(int[] new_ord, TIntIntHashMap t) {
        int ix = 0;
        // Add parents index value
        int ix_ml = 1;

        // for (int n : order) {
        // for (int i = new_ord.length-1; i >= 0; i--) {
        // int p = new_ord[i];
        for (int p : new_ord) {
            ix += t.get(p) * ix_ml; // Shift index
            ix_ml *= arity(p); // Compute cumulative shifter
            // log.severe("P:" + par + " - v: " + sample[par] + " - " + arity(par) + " - " + ix );
        }
        return ix;
    }
}
