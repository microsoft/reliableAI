package ch.idsia.blip.api.experiments;


import ch.idsia.blip.core.learn.solver.brtl.BrutalSolver;
import ch.idsia.blip.core.utils.ParentSet;
import ch.idsia.blip.core.utils.RandomStuff;
import org.junit.Test;

import java.io.File;
import java.util.HashMap;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class NewBrutalTest {

    String path = "../experiments/newBrutal/";

    @Test
    public void test() throws Exception {
        BrutalSolver o = new BrutalSolver();

        o.searcher = "max";
        // o.sampler = "r_ent";
        // go("plants.ts", o, null, 1000, 1, 3);
        go("random2000-0", o, null, 1000, 1, 3);
    }

    @Test
    public void compare() throws Exception {

        for (String prob : new String[] {
            "plants.ts", "accidents.valid", "test",
            "what"}) { // "accidents", , "test", "simple" , "what", "accidents.test"

            HashMap<String, BrutalSolver> solvers = new HashMap<String, BrutalSolver>();

            /*
             for (String r: new String[] {"null", "ent", "r_ent"}) { //
             addBrutal(solvers, "def", r);
             } */

            addBrutal(solvers, "max2");
            addBrutal(solvers, "max");
            addBrutal(solvers, "def");

            /*
             for (String r: new String[] {"null", "ent", "r_ent"}) { //
             addAsobs(solvers, "inasobs2", r);
             }             */

            for (String s : solvers.keySet()) { //
                go(prob, solvers.get(s), s, 10, 7, 1);
            }

            p("");
        }
    }

    private void addBrutal(HashMap<String, BrutalSolver> solvers, String s) {
        addBrutal(solvers, s, "def");
    }

    private void go(String prob, BrutalSolver o, String s, int i, int k, int v) throws Exception {
        o.verbose = v;
        ParentSet[][] sc = RandomStuff.getScoreReader(f("%s.jkl", path + prob),
                1);

        o.dat_path = f("%s.dat", path + prob);
        o.init(sc, i, k);
        new File(path + prob).mkdir();
        String res;

        if (s != null) {
            o.logWr = getWriter(f("%s/%s.log", path + prob, s));
            res = f("%s/%s.res", path + prob, s);
        } else {
            res = f("%s/%s.res", path + prob, "test");
        }
        o.tw = 3;
        o.go(res);
        pf("%8s \t %.3f \n", s, o.best_sk);
    }

    private void addBrutal(HashMap<String, BrutalSolver> solvers, String s, String e) {
        BrutalSolver b = new BrutalSolver();

        b.searcher = s;
        b.sampler = e;
        solvers.put(s + "-" + e, b);
    }

}

