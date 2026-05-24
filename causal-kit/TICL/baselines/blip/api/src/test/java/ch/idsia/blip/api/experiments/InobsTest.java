package ch.idsia.blip.api.experiments;


import ch.idsia.blip.core.learn.solver.*;
import ch.idsia.blip.core.utils.ParentSet;
import ch.idsia.blip.core.utils.RandomStuff;
import ch.idsia.blip.core.utils.data.FastList;
import org.junit.Test;

import java.io.File;
import java.util.HashMap;
import java.util.Random;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class InobsTest {

    String path = "../experiments/inobs/";

    @Test
    public void test() throws Exception {
        WinAsobsSolver o = new WinAsobsSolver();

        o.max_windows = (5);
        // String set = "accidents.test";
        // String set = "plants.ts";
        String s = "accidents.test";

        go(s, o, null, 1000, 1, 1);
    }

    @Test
    public void compare() throws Exception {

        for (String prob : new String[] { "plants.ts", "accidents.test"}) { // {"simple",  "accidents", , "test", "simple"

            HashMap<String, ScoreSolver> solvers = new HashMap<String, ScoreSolver>();

            // addAsobs(solvers, "inasobs2");
            // addAsobs(solvers, "inasobs4");
            // addAsobs(solvers, "inasobs2");

            // addAsobs(solvers, "inasobs5");
            // addAsobs(solvers, "inasobs6");
            // addAsobs(solvers, "inasobs7");

            // for (String r: new String[] {"null", "ent", "r_ent"}) { //
            // addAsobs(solvers, "inasobs2", r);
            // // addAsobs(solvers, "inasobs3", r);
            // // addObs(solvers, "inobs4", r);
            // }

            addWinAsobs(solvers, 1);
            addWinAsobs(solvers, 2);
            addWinAsobs(solvers, 3);
            addWinAsobs(solvers, 4);
            addWinAsobs(solvers, 5);

            /* addAsobs(solvers, "winasobs2");
             addAsobs(solvers, "winasobs");
             addAsobs(solvers, "inasobs2");
             addAsobs(solvers, "inasobs");*/


            for (String s : solvers.keySet()) { //
                go(prob, solvers.get(s), s, 10, 7, 1);
            }

            p("");
        }
    }

    private void addWinAsobs(HashMap<String, ScoreSolver> solvers, int i) {
        WinAsobsSolver o = new WinAsobsSolver();

        o.max_windows = (i);
        solvers.put("winasobs-" + i, o);
    }

    private void go(String prob, ScoreSolver o, String s, int i, int k, int v) throws Exception {
        o.verbose = v;
        ParentSet[][] sc = RandomStuff.getScoreReader(f("%s.jkl", path + prob),
                1);

        o.dat_path = f("%s.dat", path + prob);
        o.init(sc, i);
        new File(path + prob).mkdir();
        String res;

        if (s != null) {
            o.logWr = getWriter(f("%s/%s.log", path + prob, s));
            res = f("%s/%s.res", path + prob, s);
        } else {
            res = f("%s/%s.res", path + prob, "test");
        }

        o.go(res);
        pf("%10s %.3f \n", s, o.best_sk);
    }

    private void addObs(HashMap<String, ScoreSolver> solvers, String s) {
        addObs(solvers, s, null);
    }

    private void addObs(HashMap<String, ScoreSolver> solvers, String s, String e) {
        ObsSolver obs = new ObsSolver();

        obs.searcher = s;
        obs.sampler = e;
        solvers.put(s + "-" + e, obs);
    }

    private void addClopt(HashMap<String, ScoreSolver> solvers, String s) {
        ClOptSolver clopt = new ClOptSolver();

        clopt.initAdv(s);
        String p = System.getProperty("user.dir");

        p += "/../experiments/";
        clopt.setup(p + "temp/", p + "gobnilp");
        solvers.put(s, clopt);
    }

    private void addAsobs(HashMap<String, ScoreSolver> solvers, String s) {
        addAsobs(solvers, s, null);
    }

    private void addAsobs(HashMap<String, ScoreSolver> solvers, String s, String e) {
        AsobsSolver asobs = new AsobsSolver();

        asobs.searcher = s;
        asobs.sampler = e;
        solvers.put(s + "-" + e, asobs);

    }

    @Test
    public void testFastList() {
        int n_var = 1000;
        FastList<Integer> todo = new FastList<Integer>(getRandom());

        for (int i = 1; i < n_var; i++) {
            todo.add(i);
        }

        while (!(todo.size() == 0)) {
            int ix = todo.rand();

            p(ix);
            todo.delete(ix);

        }
    }

    @Test
    public void times() {

        int n_var = 100;

        long start = System.nanoTime();
        Random rand = getRandom();
        int[] todo = new int[n_var];

        for (int i = 0; i < 100; i++) {
            todo[i] = i;
        }

        for (int i = 0; i < 1000; i++) {
            int index = rand.nextInt(n_var - 1);
            // Simple swapArray
            int a = todo[index];

            todo[index] = todo[0];
            todo[0] = a;
        }

        pf("elapsed: %d \n", System.nanoTime() - start);

        start = System.nanoTime();

        FastList<Integer> td = new FastList<Integer>(getRandom());

        for (int i = 0; i < 1000; i++) {
            td.add(i);
        }

        for (int i = 0; i < 1000; i++) {
            td.delete(i);
            td.add(i);
        }

        pf("elapsed: %d \n", System.nanoTime() - start);
    }

    private void initGreedy(int n_var, java.util.Random rand, int[] todo) {

        // randomize based on how much we explored
        int rnd = n_var - 1;

        // randomize first part of array (already seen)
        for (int i = n_var - 1; i > 1; i--) {}
    }

    private void initT(FastList todo, int n_var) {
        todo.reset();
        for (int i = 1; i < n_var; i++) {
            todo.add(i);
        }
    }
}

