package ch.idsia.blip.api.exp;


import ch.idsia.blip.api.learn.scorer.GreedyScorerApi;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.io.bn.BnUaiWriter;
import ch.idsia.blip.core.utils.score.BDeu;
import ch.idsia.blip.core.utils.score.K2;
import ch.idsia.blip.core.utils.score.Score;
import ch.idsia.blip.core.learn.param.ParLeBayes;
import ch.idsia.blip.core.learn.solver.WinAsobsSolver;
import ch.idsia.blip.core.learn.solver.brtl.BrutalMaxSolver;
import ch.idsia.blip.core.learn.solver.brtl.BrutalSolver;
import ch.idsia.blip.core.utils.other.IncorrectCallException;
import ch.idsia.blip.core.common.LLEval;
import ch.idsia.blip.core.utils.ParentSet;
import ch.idsia.blip.core.utils.data.ArrayUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class ExpWinMine {

    String path = "/home/loskana/Documents/bazaar/2017/winmine";

    // String score = "bdeu";
    String score = "k2";

    public static void main(String[] args) {
        try {
            new ExpWinMine().gogo();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    private void gogo() throws IOException, IncorrectCallException {

        // "nltcs" , "plants", "kdd" ,  "baudio", "jester", "bnetflix",  "accidents","tretail",    "pumsb_star",   "dna",   "kosarek", "msweb", "book", "tmovie"
        // "tmovie"
        // for (String s : new String[] { "nltcs", "plants", "kdd", "baudio",  "jester", "bnetflix", "accidents", "tretail", "pumsb_star", "dna", "kosarek", "msweb"}) { //
        for (String s : new String[] {"ad"}) { //  "tmovie", "book"""c20ng", "cr52", "cwebkb"
            // score = "k2";
            // go2(s);
            score = "bdeu";
            go2(s);
        }

    }

    private void go(String s) throws IOException, IncorrectCallException {

        String train = f("%s/data1/%s.ts.dat", path, s);
        DataSet d_train = getDataSet(train);
        String test = f("%s/data1/%s.test.dat", path, s);

        pf("%s \t %d \t ", s, d_train.n_var);

        // WINMINE

        String win = f("%s/final/%s/%s.win.uai", path, score, s);

        if (!new File(win).exists()) {

            BufferedReader br = getReader(
                    f("%s/model11/%s/%s.ts.win", path, score, s));

            String l;
            int ix = -1;
            ParentSet[] ps = new ParentSet[d_train.n_var];
            int[] p = new int[0];

            while ((l = br.readLine()) != null) {
                l = l.trim().toLowerCase();

                if ("".equals(l)) {
                    if (ix != -1) {
                        ps[ix] = new ParentSet(0, p);
                        p = new int[0];
                        ix = -1;
                    }
                    continue;
                }

                int v = Integer.valueOf(l.replace("n", ""));

                if (ix == -1) {
                    ix = v;
                } else {
                    p = ArrayUtils.expandArray(p, v);
                }
            }

            for (int i = 0; i < ps.length; i++) {
                if (ps[i] == null) {
                    ps[i] = new ParentSet(0, new int[0]);
                }
            }

            BayesianNetwork bn = new BayesianNetwork(ps);

            bn = ParLeBayes.ex(bn, d_train);
            BnUaiWriter.ex(bn, win);
        }

        eval(d_train, test, getBayesianNetwork(win));

        // JKL
        String jkl = f("%s/model2/%s.ts.%s.jkl", path, s, score);

        if (!new File(jkl).exists()) {
            GreedyScorerApi.main(
                    new String[] {
                "", "-d", train, "-j", jkl, "-b", "0", "-c",
                score, "-t", String.valueOf(d_train.n_var), "-n", "15"
            });
        }

        // ASOBS

        String aso = f("%s/final/%s/%s.aso.uai", path, score, s);

        if (!new File(aso).exists()) {
            String res = f("%s/model2/%s.ts.%s.res", path, s, score);

            WinAsobsSolver gs = new WinAsobsSolver();

            gs.init();
            gs.res_path = res;
            gs.max_exec_time = 120;
            gs.init(getScoreReader(jkl));
            gs.thread_pool_size = 3;

            gs.go();

            BayesianNetwork bn = ParLeBayes.ex(getBayesianNetwork(res), d_train);

            BnUaiWriter.ex(bn, aso);
        }

        eval(d_train, test, getBayesianNetwork(aso));

        // GOB
        /*
         String gob = f("%s/final/%s.gob.uai", path, set);
         if (!new File(gob).exists()) {
         String jkl = f("%s/model2/%s.ts.bdeu.jkl", path, set);
         String res = f("%s/model2/%s.ts.gob", path, set);

         WinAsobsSolver gs = new WinAsobsSolver();
         gs.init();
         gs.res_path = res;
         gs.max_exec_time = 1;
         gs.max_windows = 1;
         gs.init(getScoreReader(jkl));
         gs.thread_pool_size = 3;
         // gs.verbose = 2;

         gs.go();

         BayesianNetwork bn = ParLeBayes.ex(getBayesianNetwork(res), d_train);
         BnUaiWriter.ex(bn, gob);
         }

         eval(d_train, test, getBayesianNetwork(gob));
         */

        p("");

    }

    private void eval(DataSet d_train, String test, BayesianNetwork bn_win) {

        Score sc;

        if (score.equals("bdeu")) {
            sc = new BDeu(1, d_train);
        } else {
            sc = new K2(d_train);
        }

        double sc_win = 0;

        for (int n = 0; n < bn_win.n_var; n++) {
            int[] p = bn_win.parents(n);

            if (p.length > 0) {
                sc_win += sc.computeScore(n, p);
            } else {
                sc_win += sc.computeScore(n);
            }
        }
        double ll_win = LLEval.ex(bn_win, test);

        pf("\t %.2f \t %.2f", sc_win, ll_win);

    }

    private void go2(String s) throws IOException, IncorrectCallException {

        String train = f("%s/data1/%s.ts.dat", path, s);
        DataSet d_train = getDataSet(train);
        String test = f("%s/data1/%s.test.dat", path, s);

        pf("%s \t %d \t ", s, d_train.n_var);

        // JKL
        String jkl = f("%s/model2/%s.ts.%s.jkl", path, s, score);

        if (!new File(jkl).exists()) {
            GreedyScorerApi.main(
                    new String[] {
                "", "-d", train, "-j", jkl, "-b", "0", "-c",
                score, "-t", String.valueOf(d_train.n_var), "-n", "15"
            });
        }

        // Brutal MAX

        String max = f("%s/brutal/%s/%s.max.res", path, score, s);
        if (!new File(max).exists()) {
            BrutalMaxSolver gs = new BrutalMaxSolver();
            gs.init();
            gs.tw = 7;
            gs.res_path = max;
            gs.max_exec_time = 1200;
            gs.init(getScoreReader(jkl));
            gs.thread_pool_size = 3;

            gs.go();
        }

        String max_bn = max.replace(".res", ".uai");
        if (!new File(max_bn).exists()) {
            BayesianNetwork bn = ParLeBayes.ex(getBayesianNetwork(max), d_train);
            BnUaiWriter.ex(bn, max_bn);
        }

        eval(d_train, test, getBayesianNetwork(max_bn));

        // Brutal k-g

        String kg = f("%s/brutal/%s/%s.kg.res", path, score, s);
        if (!new File(kg).exists()) {
            BrutalSolver gs = new BrutalSolver();
            gs.init();
            gs.tw = 7;
            gs.res_path = kg;
            gs.max_exec_time = 1200;
            gs.init(getScoreReader(jkl));
            gs.thread_pool_size = 3;

            gs.go();
        }

        String kg_bn = kg.replace(".res", ".uai");
        if (!new File(kg_bn).exists()) {
            BayesianNetwork bn = ParLeBayes.ex(getBayesianNetwork(kg), d_train);
            BnUaiWriter.ex(bn, kg_bn);
        }

        eval(d_train, test, getBayesianNetwork(kg_bn));

        // GOB
        /*
         String gob = f("%s/final/%s.gob.uai", path, set);
         if (!new File(gob).exists()) {
         String jkl = f("%s/model2/%s.ts.bdeu.jkl", path, set);
         String res = f("%s/model2/%s.ts.gob", path, set);

         WinAsobsSolver gs = new WinAsobsSolver();
         gs.init();
         gs.res_path = res;
         gs.max_exec_time = 1;
         gs.max_windows = 1;
         gs.init(getScoreReader(jkl));
         gs.thread_pool_size = 3;
         // gs.verbose = 2;

         gs.go();

         BayesianNetwork bn = ParLeBayes.ex(getBayesianNetwork(res), d_train);
         BnUaiWriter.ex(bn, gob);
         }

         eval(d_train, test, getBayesianNetwork(gob));
         */

        p("");

    }

}
