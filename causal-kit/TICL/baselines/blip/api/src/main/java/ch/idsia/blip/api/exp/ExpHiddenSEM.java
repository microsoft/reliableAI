package ch.idsia.blip.api.exp;


import ch.idsia.blip.api.learn.scorer.IndependenceScorerApi;
import ch.idsia.blip.api.learn.solver.tw.BrutalGreedyAdvSolverApi;
import ch.idsia.blip.api.learn.solver.win.WinAsobsSolverApi;
import ch.idsia.blip.api.learn.param.ParLeApi;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.io.bn.BnUaiReader;
import ch.idsia.blip.core.io.bn.BnUaiWriter;
import ch.idsia.blip.core.io.dat.DatFileLineReader;
import ch.idsia.blip.core.learn.missing.HiddenSEM;
import ch.idsia.blip.core.learn.missing.LLEvalHidden;
import ch.idsia.blip.core.learn.missing.SEM2;
import ch.idsia.blip.core.learn.param.ParLe;
import ch.idsia.blip.core.learn.param.ParLeBayes;
import ch.idsia.blip.core.learn.scorer.IndependenceScorer;
import ch.idsia.blip.core.learn.solver.brtl.BrutalMaxSolver;
import ch.idsia.blip.core.common.LLEval;
import ch.idsia.blip.core.utils.RandomStuff;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.Writer;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;

import static java.lang.Integer.*;


public class ExpHiddenSEM extends ExpSemImputation {
    private static final Logger log = Logger.getLogger(
            ExpHiddenSEM.class.getName());
    String path = System.getProperty("user.home") + "/Desktop/SEM/hidden/";
    private int time = 60;

    protected String suffix = "-sem/";

    public static void main(String[] args) {
        try {
            if (args.length == 4) {
                new ExpHiddenSEM().go(args[0], args[1], valueOf(args[2]),
                        valueOf(args[3]), 1);
            } else {
                new ExpHiddenSEM().test();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    protected void test()
        throws Exception {
        String n = "nltcs";

        RandomStuff.p(n);
        def(n);
        for (int p = 1; p < 10; p++) {
            gogo(n, p);
        }
    }

    private void def(String net)
        throws Exception {
        String d1 = RandomStuff.f("%s/data/%s.ts.dat", this.path, net);

        DataSet dat = RandomStuff.getDataSet(d1);

        this.max_time = dat.n_var;

        String newPath = RandomStuff.f("%s/%s/", this.path, net);
        File f = new File(newPath);

        if (!f.exists()) {
            f.mkdirs();
        }
        String jkl = newPath + "jkl";

        f = new File(jkl);
        if (!f.exists()) {
            IndependenceScorer is = new IndependenceScorer();
            HashMap<String, String> o = new HashMap<String, String>();

            o.put("ph_scores", jkl);
            o.put("max_pset_size", "5");
            o.put("time_for_variable", String.valueOf(this.max_time));
            RandomStuff.p(String.valueOf(this.max_time));
            o.put("scoreNm", "bdeu");
            o.put("alpha", "1.0");
            o.put("thread_pool_size", String.valueOf(this.thread));
            is.init(o);
            is.go(dat);
        }
        String res = newPath + "res";

        f = new File(res);
        if (!f.exists()) {
            BrutalMaxSolver gs = new BrutalMaxSolver();

            prp(gs, res, this.max_time, RandomStuff.getScoreReader(jkl, 0),
                    this.thread, this.tw);
            gs.go();
        }
        String uai = newPath + "res.uai";

        f = new File(uai);
        BayesianNetwork bn;

        if (!f.exists()) {
            ParLe pl = new ParLeBayes(2.0D);

            bn = pl.go(RandomStuff.getBayesianNetwork(res), dat);
            BnUaiWriter.ex(bn, uai);
        } else {
            bn = BnUaiReader.ex(uai);
        }
        LLEval l = new LLEval();

        l.go(bn, RandomStuff.f("%s/data/%s.test.dat", this.path, net));

        RandomStuff.pf("def %.4f \n", l.ll);
    }

    private void gogo(String net, int per)
        throws Exception {
        String dat_path = RandomStuff.f("%s/data/%s.ts.dat", this.path, net);

        String newPath = RandomStuff.f("%s/%s/%d/", this.path, net, valueOf(per));
        File f = new File(newPath);

        if (!f.exists()) {
            f.mkdirs();
        }
        String model = newPath + "final.uai";

        f = new File(model);
        BayesianNetwork bn;

        if (!f.exists()) {
            HiddenSEM mSem = new HiddenSEM();

            mSem.thread_pool_size = this.thread;
            mSem.init(newPath, this.max_time, this.tw);
            bn = mSem.go(dat_path, per);
        } else {
            bn = RandomStuff.getBayesianNetwork(model);
        }
        LLEvalHidden l = new LLEvalHidden(this.thread);
        double ll = l.go(RandomStuff.f("%s/data/%s.test.dat", this.path, net),
                newPath + "final.uai");

        RandomStuff.pf("final.uai %d %.4f \n", valueOf(per), ll);
        ll = l.go(RandomStuff.f("%s/data/%s.test.dat", this.path, net),
                newPath + "final.win.uai");
        RandomStuff.pf("final.win.uai %d %.4f \n", valueOf(per), ll);
    }

    private void tesgo()
        throws IOException, InterruptedException {
        String base = this.path + "/data/new/";
        String d = base + "child-test-test.dat";

        for (String s : new String[] {
            "child-train-sem/child-train-2-0.win.uai",
            "child-train-sem/child-train-2-0.uai",
            "child-train-sem/base.kmax.uai" }) {
            double start = System.currentTimeMillis();
            double llIj = llIjgp(d, base + s);
            double tIj = System.currentTimeMillis() - start;

            start = System.currentTimeMillis();
            double llEv = LLEvalHidden.ex(d, base + s);
            double tEv = System.currentTimeMillis() - start;

            RandomStuff.pf("%20s | %.4f (ijgp) %.4f | %.4f (llev) %.4f\n", s,
                    llIj, tIj, llEv, tEv);
        }
    }

    private double llIjgp(String d, String bn_path)
        throws IOException, InterruptedException {
        DatFileLineReader ds = new DatFileLineReader(d);

        ds.readMetaData();
        double ll = 0.0D;

        while (!ds.concluded) {
            short[] sample = ds.next();

            ll += RandomStuff.ijgp(bn_path, sample);
        }
        return ll / ds.n_datapoints;
    }

    private void random()
        throws IOException, InterruptedException {
        String bn_path = this.path + "/data/new/child-train-sem/";
        String d = bn_path + "test2.dat";

        for (String s : new String[] {
            "base.win.uai", "child-train-2-0.uai",
            "child-train-2-0.red.uai", "child-train-1-0.win.uai",
            "child-train-1-0.red.uai" }) {
            double t = LLEvalHidden.ex(d, bn_path + s);

            RandomStuff.pf("%s %.2f \n", s, t);
        }
    }

    private void go(String base, String nm, int i, int j, int threads)
        throws Exception {
        String d = base + nm + ".dat";

        String path = base + nm + suffix;
        File p = new File(path);

        if (!p.exists()) {
            p.mkdir();
        }
        SEM2 sem = new SEM2(0);

        String bn_name = RandomStuff.f("%s-%d-%d", nm, valueOf(j), valueOf(i));

        String out = RandomStuff.f("%s/%s", path, bn_name);

        if (new File(out + ".uai").exists()) {
            return;
        }
        RandomStuff.p(out);

        sem.init(out, this.time, 5, j);
        sem.thread_pool_size = threads;
        sem.go(d, nm);

        double t = LLEvalHidden.ex(d, out);

        Writer w = RandomStuff.getWriter(out + ".ll");

        RandomStuff.wf(w, "%.2f", t);
        w.close();
    }

    public void testEVERYTHING()
        throws Exception {
        String h = "child-train";
        String s = System.getProperty("user.home")
                + "/Desktop/SEM/hidden/data/new/";

        testSEM(s, h);
    }

    private void goSEM(String base, String nm)
        throws Exception {
        for (int i = 0; i < 5; i++) {
            for (int j = 1; j < 10; j++) {
                go(base, nm, i, j, 0);
            }
        }
    }

    private void testSEM(String base, String nm)
        throws IOException, InterruptedException {
        String h = base + nm + "-sem/base";

        String d = base + nm + "-sem/test.dat";

        score(d, h);
        winasobs(h);
        kMax(h);
        parle(d, h);

        HashMap<String, Double> bns = new HashMap<String, Double>();

        for (File file : new File(base + nm + "-sem/").listFiles()) {
            if (!file.isDirectory()) {
                if (file.getName().endsWith(".uai")) {
                    bns.put(file.getName(),
                            LLEvalHidden.ex(d, file.getAbsolutePath()));
                }
            }
        }
        Map<String, Double> s_bns = RandomStuff.sortByValues(bns);

        for (String bn_name: s_bns.keySet()) {
            RandomStuff.pf("%20s %10.5f \n", bn_name, bns.get(bn_name));
        }

        RandomStuff.pf("\n LOG LIKELIHOOD\n");

        String[] gg = new String[] { "kmax", "win" };

        for (String g: gg) {
            LLEval lleval = new LLEval();

            lleval.go(BnUaiReader.ex(RandomStuff.f("%s.%s.uai", h, g)),
                    new DatFileLineReader(d));
            RandomStuff.pf("%10s %10.5f \n", g, lleval.ll);
        }
    }

    private void winasobs(String s)
        throws FileNotFoundException {
        if (new File(s + ".win.res").exists()) {
            return;
        }
        WinAsobsSolverApi.main(
                new String[] {
            "", "-win", "20", "-j", s + ".jkl", "-r",
            s + ".win.res", "-t", RandomStuff.f("%d", valueOf(this.time)), "-pb",
            "7", "-v", "1" });
    }

    private void parle(String h, String s) {
        for (String r : new String[] { "kmax", "win" }) {
            if (!new File(s + "." + r + ".uai").exists()) {
                ParLeApi.main(
                        new String[] {
                    "", "-d", h, "-r", s + "." + r + ".res",
                    "-n", s + "." + r + ".uai" });
            }
        }
    }

    private void kMax(String s)
        throws FileNotFoundException {
        if (new File(s + ".kmax.res").exists()) {
            return;
        }
        BrutalGreedyAdvSolverApi.main(
                new String[] {
            "", "-src", "max", "-w", "5", "-j", s + ".jkl",
            "-r", s + ".kmax.res", "-t", RandomStuff.f("%d", valueOf(this.time)),
            "-pb", "7", "-v", "1" });
    }

    private void score(String s, String h) {
        if (new File(h + ".jkl").exists()) {
            return;
        }
        IndependenceScorerApi.main(
                new String[] {
            "", "-n", "4", "-pc", "bdeu", "-d", s, "-j",
            h + ".jkl", "-t", RandomStuff.f("%d", valueOf(this.time)), "-pb",
            "7", "-v", "2" });
    }
}
