package ch.idsia.blip.core.learn.solver;


import ch.idsia.blip.core.App;
import ch.idsia.blip.core.utils.other.TopologicalOrder;
import ch.idsia.blip.core.utils.arcs.Directed;
import ch.idsia.blip.core.learn.solver.ps.Provider;
import ch.idsia.blip.core.learn.solver.samp.Sampler;
import ch.idsia.blip.core.learn.solver.src.Searcher;
import ch.idsia.blip.core.utils.data.array.TDoubleArrayList;
import ch.idsia.blip.core.utils.ParentSet;
import ch.idsia.blip.core.utils.RandomStuff;

import java.io.IOException;
import java.io.Writer;
import java.util.HashMap;
import java.util.TreeSet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.data.ArrayUtils.sameArray;
import static ch.idsia.blip.core.utils.RandomStuff.*;


public abstract class BaseSolver extends App {

    private final Logger log = Logger.getLogger(BaseSolver.class.getName());

    public ParentSet[][] sc;

    // Best map found yet
    public double best_sk = -Double.MAX_VALUE;

    // Best map found yet
    public ParentSet[] best_str;

    public boolean testAcycility = false;

    public int n_var;

    protected boolean atLeastOne = false;

    public String res_path;

    private TDoubleArrayList sk_proposed;

    protected int numIter;

    public int out_solutions = 1;

    public TreeSet<Solution> open;

    private double worstSolScore;

    private Sampler smp;

    public int delta;

    public TreeSet<Solution> tryFirst;

    private int[][] test_parent;

    public boolean write_solutions = true;

    protected abstract String name();

    @Override
    protected void prepare() {

        super.prepare();

        last_improvement = start;

        if (verbose > 0) {
            logf("Starting %s \n", name());
            logf("max_exec_time: %.2f \n", max_exec_time);
            logf("thread_pool_size: %d \n", thread_pool_size);
            logf("seed: %d \n", seed);
        }

        sk_proposed = new TDoubleArrayList();

        best_sk = -Double.MAX_VALUE;
        best_str = new ParentSet[n_var];

        if (out_solutions > 1) {
            open = new TreeSet<Solution>();
        }

        counter = System.currentTimeMillis();

        test_parent = new int[n_var][];
    }

    public ParentSet[] go(String path) throws Exception {
        this.res_path = path;

        return go();
    }

    protected void writeStructure(String res_path) {
        writeStructure(res_path, this.best_sk, this.best_str);
        int i;

        if ((this.out_solutions > 1) && (this.write_solutions)) {
            i = 0;
            for (Solution s : this.open) {
                writeStructure(RandomStuff.f("%s-%d", res_path, i++), s.sk,
                        s.str);
            }
        }
    }

    public ParentSet[] go() {

        prepare();

        Provider pro = getProvider();

        if (this.verbose > 0) {
            log("Read scores... \n");
        }
        this.sc = pro.getParentSets();

        almost();

        try {
            ExecutorService es = Executors.newCachedThreadPool();

            for (int i = 0; i < thread_pool_size; i++) {
                if (verbose > 0) {
                    logf("Starting %d searcher \n", i);
                }
                es.execute(getNewSearcher(i));
            }

            es.shutdown();
            es.awaitTermination(Integer.MAX_VALUE, TimeUnit.MINUTES);

            conclude();
        } catch (InterruptedException e) {
            logExp(log, e);
        }

        writeStructure(this.res_path);

        return this.best_str;

    }

    protected void almost() {
        if (tryFirst == null) {
            return;
        }

        for (Solution s : tryFirst) {
            tryNow(s);
        }
    }

    protected void tryNow(Solution s) {
        ParentSet[] t = new ParentSet[this.n_var];

        for (int i = 0; i < this.n_var; i++) {
            t[i] = findParSet(s.str[i].parents, this.sc[i]);
            if (t[i] == null) {
                return;
            }
        }
        newStructure(s.str);
    }

    private ParentSet findParSet(int[] pset, ParentSet[] sc) {
        for (ParentSet p : sc) {
            if (sameArray(p.parents, pset)) {
                return p;
            }
        }

        return null;
    }

    public abstract Sampler getSampler();

    protected abstract Searcher getSearcher();

    protected abstract Provider getProvider();

    public boolean still_time = true;

    public double elapsed;

    private double last_improvement;

    protected long counter;

    public void checkTime() {

        elapsed = (System.currentTimeMillis() - start) / 1000.0;

        if (max_exec_time > 0 && elapsed > max_exec_time) {
            still_time = false;
        }

        if (delta > 0
                && (System.currentTimeMillis() - last_improvement) / 1000.0
                        > delta) {
            still_time = false;
        }
    }

    protected void conclude() {

        if (verbose > 0) {
            log("... Ended! \n");
            logf("Num Iterations total: %d \n", numIter);
            logf("Scores statistics: \n%s \n", sk_proposed.stats());
        }
    }

    public void writeGraph(String s, String[] nm) {
        Directed d = new Directed(best_str);

        d.names = nm;
        d.graph(s);
    }

    public void writeGraph(String s) {
        Directed d = new Directed(best_str);

        d.graph(s);
    }

    /**
     * Write the best map found so far.
     */
    public void writeStructure(String s, double sk, ParentSet[] str) {

        if (!atLeastOne) {
            return;
        }

        Writer w;

        try {
            w = getWriter(s);

            // writer.graph("Quick Structure: " + printQuick(best_str));

            // writer.graph("\n\nExpanded Structure: \n\n");
            if (str != null && str.length > 0) {
                for (int i = 0; i < str.length; i++) {

                    ParentSet pSet = str[i];

                    if (pSet != null) {
                        wf(w, "%d: %.2f %s\n", i, pSet.sk,
                                (pSet.parents.length > 0
                                ? " (" + combine(pSet.parents, ",") + ")"
                                : ""));
                    } else {
                        wf(w, "%d: 0 0\n", i);
                    }
                }
            }

            wf(w, "\nScore: %.3f \n", sk);

            w.flush();
            w.close();
        } catch (IOException e) {
            logExp(log, e);
        }

    }

    /**
     * Combine the parent set for output
     *
     * @param s parent set in array form
     * @param d delimiter
     * @return parent set in mdl form
     */
    private String combine(int[] s, String d) {
        int k = s.length;

        if (k == 0) {
            return "";
        }
        StringBuilder out = new StringBuilder();

        out.append(s[0]);
        for (int x = 1; x < k; ++x) {
            out.append(d).append(s[x]);
        }
        return out.toString();
    }

    public String printQuick(ParentSet[] best_str) {
        StringBuilder s = new StringBuilder();

        for (int i = 0; i < n_var; i++) {
            s.append(String.format("[%d", i));
            ParentSet pSet = best_str[i];

            if (pSet.parents.length > 0) {
                s.append("|");
                s.append(combine(pSet.parents, ":"));
            }
            s.append("]");
        }

        return s.toString();
    }

    /**
     * Check that the resulting network is acyclic
     *
     * @return if the network is acyclic
     */
    public boolean testAcyclic(ParentSet[] new_str) {

        synchronized (lock) {
            for (int i = 0; i < n_var; i++) {
                test_parent[i] = new_str[i].parents;
            }
        }
        int[] res = TopologicalOrder.find(n_var, test_parent);

        return ((res != null) && (res.length == n_var));
    }

    public void newStructure(ParentSet[] new_str) {

        if (new_str == null) {
            return;
        }

        synchronized (lock) {

            numIter++;

            double new_sk = getSk(new_str);

            if (testAcycility && !testAcyclic(new_str)) {
                log.severe("Network CYCLIC!");
            }

            if (testAcycility && !testComplete(new_str)) {
                log.severe("Network NOT COMPLETE!");
            }

            if (verbose > 0) {
                sk_proposed.add(new_sk);
            }

            if (new_sk > best_sk) {

                checkTime();

                if (verbose > 0) {
                    logf("New improvement! %.5f (after %.1f s.)\n", new_sk,
                            elapsed);
                }

                best_sk = new_sk;

                cloneStr(new_str, best_str);

                atLeastOne = true;

                if (res_path != null) {
                    writeStructure(res_path, best_sk, new_str);
                }

                // best_order = vars.clone();

                last_improvement = System.currentTimeMillis();
            }

            if (verbose > 2) {
                if ((System.currentTimeMillis() - counter) / 10000.0 > 1) {
                    logf("%.5f %.1f \n", best_sk, elapsed);
                    counter = System.currentTimeMillis();
                }
            }

            // logf("%d \n", numIter);

            if (out_solutions > 1) {
                propose(new_sk, new_str);
            }

            /*
             p(new_sk);
             p(best_sk);
             for (ParentSet p: new_str)
             p(p);
             */
        }
    }

    public ParentSet[] getPSetArray(int[] new_str, ParentSet[][] pps) {
        ParentSet[] ps = new ParentSet[new_str.length];

        for (int v = 0; v < new_str.length; v++) {
            ps[v] = getPSet(new_str, v, pps);
        }
        return ps;
    }

    public ParentSet[] getPSetArray(int[] new_str) {
        return getPSetArray(new_str, sc);
    }

    private boolean testComplete(ParentSet[] new_str) {
        for (int i = 0; i < n_var; i++) {
            ParentSet ps = new_str[i];

            if (ps == null) {
                return false;
            }
            if (ps.sk > sc[i][0].sk) {
                return false;
            }
        }
        return true;
    }

    private void propose(double new_sk, ParentSet[] new_str) {

        boolean toDropWorst = false;

        if (open.size() > (out_solutions + 1)) {

            if (new_sk < worstSolScore) {
                // log.conclude("pruned");
                return;
            }

            toDropWorst = true;
        }

        for (Solution s : open) {
            if (s.same(new_str)) {
                return;
            }
        }

        // Drop worst element in queue, to make room!
        if (toDropWorst) {
            open.pollLast();
            worstSolScore = open.last().sk;
        } else // If we didn't drop any element, check if we have to update the current
        // worst score!
        if (new_sk < worstSolScore) {
            worstSolScore = new_sk;
        }

        open.add(new Solution(new_sk, new_str));
    }

    protected double getSk(ParentSet[] new_str) {
        double new_sk = 0;

        for (ParentSet p : new_str) {
            new_sk += p.sk;
        }
        return new_sk;
    }

    protected double availableTime() {
        return max_exec_time * 1000 - (System.currentTimeMillis() - start);
    }

    public BaseSearcher getNewSearcher(int i) {
        return new BaseSearcher(this, i);
    }

    public ParentSet getPSet(int[] last_str, int var) {
        return getPSet(last_str, var, sc);
    }

    public class BaseSearcher implements Runnable {

        protected final BaseSolver solver;

        protected final int thread;

        public BaseSearcher(BaseSolver solver, int thread) {
            this.solver = solver;
            this.thread = thread;
        }

        private Searcher src;

        @Override
        public void run() {

            src = getSearcher();

            src.init(sc, thread);

            while (still_time) {

                // Obtains a new candidate solution
                ParentSet[] str = src.search();

                // Propose the new solution
                newStructure(str);

                checkTime();
            }

        }
    }


    public static class Solution implements Comparable<Solution> {
        public final ParentSet[] str;
        private final double sk;

        public Solution(double sk, ParentSet[] str) {
            this.sk = sk;
            this.str = str;
        }

        @Override
        public int compareTo(Solution other) {
            if (sk < other.sk) {
                return 1;
            }
            return -1;
        }

        public boolean same(ParentSet[] o_str) {

            if (str.length != o_str.length) {
                return false;
            }

            for (int i = 0; i < str.length; i++) {
                if (!sameArray(str[i].parents, o_str[i].parents)) {
                    return false;
                }
            }

            return true;
        }
    }

    public ParentSet getPSet(int[] str, int var, ParentSet[][] pps) {
        return pps[var][str[var]];
    }

    @Override
    public void init(HashMap<String, String> options) {
        super.init(options);
        delta = gInt("delta", 0);
        out_solutions = gInt("out_solutions", 0);
        res_path = gStr("res_path", null);
    }

}

