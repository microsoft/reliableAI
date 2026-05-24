package ch.idsia.blip.core.learn.solver.src;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.io.GobnilpReader;
import ch.idsia.blip.core.learn.solver.ClOptSolver;
import ch.idsia.blip.core.learn.solver.src.obs.ObsSearcher;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.data.FastList;
import ch.idsia.blip.core.utils.data.common.TIntIterator;
import ch.idsia.blip.core.utils.data.set.TIntHashSet;
import ch.idsia.blip.core.utils.exp.CyclicGraphException;
import ch.idsia.blip.core.utils.ParentSet;
import ch.idsia.blip.core.utils.other.StreamGobbler;

import java.io.File;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.TreeSet;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.data.ArrayUtils.cloneArray;
import static ch.idsia.blip.core.utils.data.ArrayUtils.findAll;
import static ch.idsia.blip.core.utils.RandomStuff.*;


public class ClOptSearcher extends ObsSearcher {

    private static final Logger log = Logger.getLogger(
            ClOptSearcher.class.getName());

    private final int w;

    private final String ph_work;

    private final String ph_gobnilp;

    int iter = 0;

    double[] maxSk;

    double[] minSk;

    private BitSet acceptable;

    private TIntHashSet todo;

    private TreeSet<Result> cand;

    private Result[] bests;

    public ClOptSearcher(ClOptSolver solver) {
        super(solver);
        this.w = solver.w;
        this.ph_work = solver.ph_work;
        this.ph_gobnilp = solver.ph_gobnilp;
    }

    private int[][] availParents;

    @Override
    public void init(ParentSet[][] scores, int thread) {
        super.init(scores, thread);

        availParents = new int[n_var][];

        for (int i = 0; i < n_var; i++) {
            TIntHashSet l = new TIntHashSet();

            for (ParentSet ps : scores[i]) {
                for (int p : ps.parents) {
                    l.add(p);
                }
            }

            availParents[i] = l.toArray();
            Arrays.sort(availParents[i]);
        }

        minSk = new double[n_var];
        maxSk = new double[n_var];

        for (int i = 0; i < n_var; i++) {
            int j = m_scores[i].length - 1;

            minSk[i] = m_scores[i][j].sk;
            maxSk[i] = m_scores[i][0].sk;
        }
    }

    /**
     * Select a variable, try to improve the score moving it on the list
     *
     * @param vars old variable order
     * @return if an improvement was possible
     */
    public boolean greedy(int[] vars, int ix) {

        int theChosen = vars[ix];

        // pf("theChosen: %d \n", theChosen);
        // p(Arrays.toString(vars));

        // Index of the best switch
        ParentSet[] best_str = null;
        int[] best_vars = new int[n_var];
        // Gain from the best switch
        double best_sk = sk;

        ParentSet[] new_str = cloneStr(str);

        int[] new_vars = new int[n_var];

        for (int i = 0; i < n_var; i++) {
            new_vars[i] = vars[i];
        }

        acceptable = new BitSet(n_var);
        for (int i = 0; i < ix; i++) {
            acceptable.set(vars[i]);
        }

        for (int ix2 = ix; ix2 < n_var; ix2++) {
            // Switch in the order ix2 and ix
            varSwitch(ix2, new_str, new_vars);
            double sk = skore(new_str);

            if (sk > best_sk) {
                best_sk = sk;
                best_str = cloneStr(new_str);
                cloneArray(new_vars, best_vars);
            }

            // checkCorrect(new_str, new_vars);

        }

        // If no best gain, return false
        if (best_str == null) {
            return false;
        }

        // Save best
        str = best_str;
        sk = best_sk;
        cloneArray(best_vars, vars);
        return true;

    }

    private void checkCorrect(ParentSet[] new_str, int[] new_vars) {

        BayesianNetwork bn = new BayesianNetwork(new_str);

        try {
            bn.checkAcyclic();
        } catch (CyclicGraphException e) {
            bn.writeGraph(solver.res_path + ".png");
            e.printStackTrace();
        }
    }

    private ParentSet[] cloneStr(ParentSet[] s) {
        ParentSet[] new_s = new ParentSet[s.length];

        for (int i = 0; i < s.length; i++) {
            new_s[i] = s[i];
        }
        return new_s;
    }

    private void varSwitch(int ix, ParentSet[] str, int[] vars) {
        int b = vars[ix];
        int a = vars[ix - 1];

        acceptable.clear(a);

        // pf("Old parent set for %d: %s \n", b, str[b]);

        // If in the parent set of a there is b, we need to assign a new parent set
        if (find(a, str[b].parents)) {
            str[b] = bests(b);
        }

        // pf("New parent set for %d: %s \n", b, str[b]);

        // pf("Old parent set for %d: %s \n", a, str[a]);

        // Find new best parent set for a, now that b is available
        acceptable.set(b);
        // if (find(b, availParents[a]))
        str[a] = bests(a);

        // pf("New parent set for %d: %s \n", a, str[a]);
        // p("");

        ArrayUtils.swapArray(vars, ix, ix - 1);
    }

    private ParentSet bests(int a) {
        for (ParentSet pSet : m_scores[a]) {
            if (acceptable(a, pSet.parents)) {
                return pSet;
            }
        }
        return null;
    }

    protected boolean acceptable(int v, int[] parents) {
        for (int p : parents) {
            if (!acceptable.get(p)) {
                return false;
            }
        }
        return true;
    }

    private double skore(ParentSet[] s) {
        double check = 0.0;

        for (ParentSet p : s) {
            check += p.sk;
        }
        return check;

    }

    @Override
    public ParentSet[] search() {

        vars = smp.sample();

        if (solver.verbose > 2) {
            solver.log("going! \n");
        }

        // Find initial map!
        this.searchF(vars);

        if (solver.verbose > 2) {
            solver.logf("Initial: %.5f (check: %.5f) \n", sk, checkSk());
        }

        vars = new BayesianNetwork(str).getTopologicalOrder();

        solver.checkTime();

        FastList<Integer> todo = new FastList<Integer>(solver.rand);

        initT(todo);

        while (solver.still_time) {

            // Choose a random variable
            int ix = todo.rand();

            if (greedy(vars, ix)) {
                if (todo.size() != n_var - 1) {
                    initT(todo);
                }
            } else {
                todo.delete(ix);
            }

            if (todo.size() == 0) {
                break;
            }

            solver.checkTime();
        }

        checkCorrect(str, vars);

        if (solver.verbose > 2) {
            solver.logf("After greedy! %.5f - %.3f \n", solver.elapsed,
                    skore(str));
        }

        return str;
    }

    private void initT(FastList todo) {
        todo.reset();
        for (int i = 1; i < n_var; i++) {
            todo.add(i);
        }
    }

    private void searchF(int[] vars) {

        // Get the first w variables for the Gobnilp optimization
        int[] selec = new int[w];

        System.arraycopy(vars, 0, selec, 0, w);
        Arrays.sort(selec);

        new File(ph_work).mkdir();
        String jkl = f("%s/%d-%d.jkl", ph_work, thread, iter);
        String out = f("%s/%d-%d.gob", ph_work, thread, iter);
        String err = f("%s/%d-%d.err", ph_work, thread, iter);

        // Write .jkl simplified problem
        try {
            writeScores(jkl, selec);
        } catch (Exception e) {
            logExp(log, e);
        }

        // Solve the problem with gobnilp
        try {
            solveGobnilp(jkl, out, err);
        } catch (IOException e) {
            logExp(log, e);
        } catch (InterruptedException e) {
            logExp(log, e);
        }

        // Read gobnilp solution, translate into an initial solution
        readGobnilp(selec, out);

        // Find the best solution by minimize loss
        initBests(vars);

        // First update of available best
        updateBests(-1);

        // for each remaining ones
        while (!todo.isEmpty()) {

            // Get the best one yet
            Result res = cand.pollFirst();

            // update the lists
            done(res.v);
            // update map
            str[res.v] = res.p;
        }

        iter++;
    }

    private void done(int v) {
        todo.remove(v);
        cand.remove(bests[v]);
        bests[v] = null;

        // update available parents
        acceptable.set(v);

        // Update tryFirst
        updateBests(v);
    }

    private void updateBests(int newAvailable) {

        // Update best handlers
        TIntIterator it = todo.iterator();

        while (it.hasNext()) {
            int v = it.next();

            if (newAvailable != -1 && !find(newAvailable, availParents[v])) {
                continue;
            }

            ParentSet bestPset = bests(v);

            if (bestPset == bests[v].p) {
                continue;
            }

            cand.remove(bests[v]);
            double sk = (bestPset.sk - maxSk[v]) / (minSk[v] - maxSk[v]);
            Result c = new Result(sk, v, bestPset);

            cand.add(c);
            bests[v] = c;
        }
    }

    private void initBests(int[] vars) {
        // Init list of variables to evaluate
        todo = new TIntHashSet();
        cand = new TreeSet<Result>();
        bests = new Result[n_var];

        for (int i = w; i < n_var; i++) {
            int v = vars[i];

            todo.add(v);
            Result r = new Result(1, v, m_scores[v][m_scores[v].length - 1]);

            cand.add(r);
            bests[v] = r;
        }

    }

    private void readGobnilp(int[] selec, String out) {
        GobnilpReader g = new GobnilpReader();

        g.go(out);

        str = new ParentSet[n_var];
        acceptable = new BitSet(n_var);

        for (int i = 0; i < w; i++) {
            int v = selec[i];

            ParentSet oldP = g.new_str[i];
            int[] newAr = new int[oldP.parents.length];

            for (int j = 0; j < newAr.length; j++) {
                newAr[j] = selec[oldP.parents[j]];
            }

            str[v] = new ParentSet(oldP.sk, newAr);
            acceptable.set(v);
        }
    }

    private void solveGobnilp(String jkl, String out, String err) throws IOException, InterruptedException {
        ProcessBuilder builder = new ProcessBuilder(ph_gobnilp, "-f=jkl", jkl);

        Process p = builder.start();

        StreamGobbler e = new StreamGobbler(p.getErrorStream(), err);

        e.start();
        StreamGobbler o = new StreamGobbler(p.getInputStream(), out);

        o.start();

        p.waitFor();
    }

    private void writeScores(String jkl, int[] selec) throws Exception {
        Writer wr = getWriter(jkl);

        wr.write(String.format("%d\n", w));

        for (int i = 0; i < w; i++) {

            ArrayList<ParentSet> newPs = new ArrayList<ParentSet>();

            for (ParentSet p : m_scores[selec[i]]) {
                // Check if the parent set contains only value in selec
                if (!findAll(p.parents, selec)) {
                    continue;
                }

                // put the parents with the new index
                int[] newAr = new int[p.parents.length];

                for (int j = 0; j < newAr.length; j++) {
                    newAr[j] = pos(p.parents[j], selec);
                }

                newPs.add(new ParentSet(p.sk, newAr));
            }

            wr.write(String.format("%d %d\n", i, newPs.size()));
            for (ParentSet p : newPs) {
                wr.write(p.prettyPrint() + "\n");
            }
            wr.flush();
        }

        wr.close();
    }

    private class Result implements Comparable<Result> {

        public final ParentSet p;

        public final double sk;

        private final int v;

        public Result(double sk, int v, ParentSet p) {
            this.p = p;
            this.sk = sk;
            this.v = v;
        }

        @Override
        public int compareTo(Result o) {
            if (sk < o.sk) {
                return -1;
            }
            if (sk > o.sk) {
                return 1;
            }

            if (v < o.v) {
                return 1;
            }
            if (v > o.v) {
                return -1;
            }

            if (equals(o)) {
                return 0;
            } else {
                return -1;
            }
        }
    }
}
