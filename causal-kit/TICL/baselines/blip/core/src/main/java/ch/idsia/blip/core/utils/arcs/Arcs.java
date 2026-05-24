package ch.idsia.blip.core.utils.arcs;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;

import java.io.IOException;
import java.io.PrintWriter;

import static ch.idsia.blip.core.utils.cmd.RunTimeout.cmdTimeout;
import static ch.idsia.blip.core.utils.RandomStuff.logExp;


public abstract class Arcs {

    // size of arc matrix
    public int size;

    // arcs matrix
    public boolean[] arcs;

    // number of variables
    public int n;

    public String[] names;

    Arcs(BayesianNetwork bn) {
        this(bn.n_var);

        prepare();

        for (int n = 0; n < bn.n_var; n++) {
            for (int p : bn.parents(n)) {
                mark(n, p);
            }
        }
    }

    protected Arcs() {}

    void prepare() {}

    Arcs(int n_var) {
        this(n_var, false);
    }

    Arcs(int n_var, boolean b) {
        n = n_var;
        size = getSize();

        arcs = new boolean[size];

        // Initialize matrix
        for (int i = 0; i < size; i++) {
            arcs[i] = b;
        }

        names = new String[n];
    }

    protected abstract int getSize();

    /**
     * Mark by row / column
     *
     * @param v1 row
     * @param v2 column
     */
    public void mark(int v1, int v2) {
        mark(index(v1, v2));
    }

    protected abstract int index(int v1, int v2);

    /**
     * Mark by index
     *
     * @param i index
     */
    void mark(int i) {
        arcs[i] = true;
    }

    /**
     * Empty by row / column
     *
     * @param v1 row
     * @param v2 column
     */
    public void empty(int v1, int v2) {
        empty(index(v1, v2));
    }

    /**
     * Empty by index
     *
     * @param i index
     */
    void empty(int i) {
        arcs[i] = false;
    }

    /**
     * Check by row / column
     *
     * @param v1 row
     * @param v2 column
     * @return content of row / column
     */
    public boolean check(int v1, int v2) {
        // System.out.println(v1 + " - " + v2);
        boolean d;

        try {
            d = arcs[index(v1, v2)];
        } catch (ArrayIndexOutOfBoundsException ex) {
            System.out.printf("%s (v1: %d, v2: %d)\n", ex.getMessage(), v1, v2);
            throw (ex);
        }
        return d;
    }

    /**
     * Check by index
     *
     * @param n index
     * @return content of index
     */
    public boolean check(int n) {

        boolean d;

        try {
            d = arcs[n];
        } catch (ArrayIndexOutOfBoundsException ex) {
            System.out.printf("%s (n: %d)\n", ex.getMessage(), n);
            throw (ex);
        }
        return d;
    }

    public int[] neighbours(int v) {
        TIntArrayList aux = new TIntArrayList();

        for (int i = 0; i < n; i++) {
            if (i != v && check(i, v)) {
                aux.add(i);
            }
        }

        return aux.toArray();
    }

    public abstract int[] r_index(int i);

    public void graph(String s) {
        try {
            PrintWriter w = new PrintWriter(s + ".dot", "UTF-8");

            w.print(this);
            w.close();
            String h = String.format("dot -Tpng %s.dot -o %s.png", s, s);

            cmdTimeout(h, false, false, 1000000);
        } catch (IOException e) {
            logExp(e);
        }
    }

    public String name(int v) {
        if (names != null) {
            return names[v];
        } else {
            return "N" + v;
        }
    }

}
