package ch.idsia.blip.api.exp;

import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.score.BDeu;
import ch.idsia.blip.core.utils.score.BIC;
import ch.idsia.blip.core.utils.score.K2;
import ch.idsia.blip.core.utils.score.Score;
import ch.idsia.blip.core.learn.param.ParLeOpt;
import ch.idsia.blip.core.learn.scorer.IndependenceScorer;
import ch.idsia.blip.core.learn.solver.WinAsobsPertSolver;

import java.io.File;
import java.io.Writer;

import static ch.idsia.blip.core.utils.RandomStuff.*;

public class ExpBDeuAlpha {

    // double[] alphas = new double[] {-2, -1, 0, 1, 2, 3, 4, 5, 6, 7};

    int[] alphas1 = new int[] {-5, -3,-1, 0, 1, 3, 5, 7};
    // int[] alphas2 = new int[] {-7, -6, -5, -4, -3, -2 ,-1, 0, 1, 2, 3, 4, 5, 6, 7};

    String locPath = System.getProperty("user.home") + "/Documents/bazaar/2017/alpha/";
    
    private int thread = 1;

    private int tw = 5;

    private double time = 10;

    private String lowest_cmb;

    private double lowest;

    public static void main(String[] args) {
        try {
            if (args.length > 1) {
                    new ExpBDeuAlpha().go(args[0], args[1]);
            } else {
                new ExpBDeuAlpha().test();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void go(String bn_name, String path) throws Exception {
        String orig = f("%s/data/%s.dat", path, bn_name);
        DataSet dat = getDataSet(orig);

        String test = f("%s/data/orig/%s.test.dat", path, bn_name.substring(0, bn_name.lastIndexOf('.')));

        lowest = -Double.MAX_VALUE;
        lowest_cmb = "";

        String path_w = f("%s/res/", path);
        if (!new File(path_w).exists())
            new File(path_w).mkdirs();
        Writer w = getWriter(path_w + bn_name);

        wf (w, "%6s # ", "");
        // for (int a2: alphas2) {
        //     wf (w, "%12.4f", Math.pow(2, a2));
        // }
        wf(w, "\n");

        for (int a1: alphas1) {

            double alpha1 = Math.pow(2, a1);
            String work = f("%s/work/%s/%d/", path, bn_name, a1);

            go(work, dat, test, w, f("%.2f", alpha1), new BDeu(alpha1, dat));

            w.flush();
        }

        wf (w, "\n");

        String work = f("%s/work/%s/bic/", path, bn_name);
        go(work, dat, test, w, "bic", new BIC(dat));
        wf (w, "\n");

        work = f("%s/work/%s/k2/", path, bn_name);
        go(work, dat, test, w, "k2", new K2(dat));
        wf (w, "\n");

        wf (w, "lowest: %12.4f, combination: %20s", lowest, lowest_cmb);
        wf (w, "\n");

        w.flush();

        w.close();
    }

    private void go(String work, DataSet dat, String test, Writer w, String m, Score score) throws Exception {

        if (!new File(work).exists())
            new File(work).mkdirs();

        String jkl = work + "jkl";
        if (!new File(jkl).exists()) {
            IndependenceScorer is = new IndependenceScorer();
            is.init();
            is.ph_scores = jkl;
            is.max_pset_size = tw;
            is.max_exec_time = dat.n_var * time;
            is.score = score;
            is.thread_pool_size = this.thread;

            is.go(dat);
        }

        String res = work + "res";
        if (!new File(res).exists()) {
            WinAsobsPertSolver gs = new WinAsobsPertSolver();
            gs.init();
            gs.res_path = res;
            gs.max_exec_time = dat.n_var / 10.0 * time ;
            gs.thread_pool_size = thread;
            gs.sc = getScoreReader(jkl);
            gs.n_var = dat.n_var;

            gs.go();
        }

        BayesianNetwork bn = getBayesianNetwork(res);

        wf (w, "%6s", m);
        ParLeOpt p = new ParLeOpt();
        BayesianNetwork bn_p = p.go(bn,dat, test);
        writeBayesianNetwork(bn_p,work+"bn.uai" );

        double ll = p.highestLL;
        double alpha = p.highestA;

         wf (w, "%12.4f %5.2f", ll, alpha);

        if (ll > lowest) {
            lowest = ll;
            lowest_cmb = f("%s %.2f", m, alpha);
        }


//        for (int a2: alphas2) {
//            double alpha2 = Math.pow(2, a2);
//            String bn_p = work + "bn" + a2 + ".uai";
//            if (!new File(bn_p).exists()) {
//                BayesianNetwork bn2 = ParLeBayes.ex(bn, dat, alpha2);
//                writeBayesianNetwork(bn2,bn_p );
//            }
//
//            double ll = LLEval.ex(getBayesianNetwork(bn_p), test);
//            wf (w, "%12.4f", ll);
//
//            if (ll > lowest) {
//                lowest = ll;
//                lowest_cmb = f("%s %.2f", m, alpha2);
//            }
//
//        }

        wf (w, "\n");
    }

    private void test() throws Exception {

        // "nltcs" , "plants", "kdd" ,  "baudio", "jester", "bnetflix",  "accidents","tretail",    "pumsb_star",   "dna",   "kosarek", "msweb", "book", "tmovie"
        // "tmovie"
        // for (String s : new String[] { "nltcs", "plants", "kdd", "baudio",  "jester", "bnetflix", "accidents", "tretail", "pumsb_star", "dna", "kosarek", "msweb"}) { //
        this.thread = 3;

        go("nltcs.1", locPath);

        for (String s : new String[]{"nltcs", "plants", "kdd", "baudio",  "jester", "bnetflix", "accidents", "tretail", "pumsb_star", "dna"}) {
        go(s, locPath);
    }

    }

}
