package ch.idsia.blip.api.experiments;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.api.learn.scorer.GreedyScorerApi;
import ch.idsia.blip.api.learn.scorer.IndependenceScorerApi;
import ch.idsia.blip.api.learn.solver.AsobsAdvSolverApi;
import ch.idsia.blip.api.learn.solver.win.WinAsobsSolverApi;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.arcs.Und;
import ch.idsia.blip.core.utils.graph.UndSeparator;
import ch.idsia.blip.core.io.GraphWriter;
import ch.idsia.blip.core.io.bn.BnNetReader;
import ch.idsia.blip.core.io.bn.BnNetWriter;
import ch.idsia.blip.core.io.bn.BnResReader;
import ch.idsia.blip.core.io.dat.CsvToDat;
import ch.idsia.blip.core.inference.ve.BayesianFactor;
import ch.idsia.blip.core.inference.ve.VariableElimination;
import ch.idsia.blip.core.learn.param.ParLe;
import ch.idsia.blip.core.learn.param.ParLeBayes;
import ch.idsia.blip.core.learn.scorer.IndependenceScorer;
import ch.idsia.blip.core.learn.solver.brtl.QuietGreedySolver;
import ch.idsia.blip.core.utils.data.hash.TIntIntHashMap;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.io.Writer;
import java.util.List;

import static ch.idsia.blip.api.exp.ExpUKang.readUnd;
import static ch.idsia.blip.core.utils.RandomStuff.*;


public class Random extends TheTest {

    @Test
    public void new_test5() throws Exception {

        String b = "/home/loskana/Desktop/pigs.dat";
        DataSet g = getDataSet(b);
        writeDataSet(g, b+"2");
    }

    @Test
    public void new_test8() throws Exception {


        String p = "/home/loskana/Desktop/";

        String h = "accidents.test";

        AsobsAdvSolverApi.main(new String[] {"",
                "-j",  p + h + ".jkl",
                "-b", "1",
                "-d", p + h + ".dat",
                "-r", p + h + ".res",
                "-smp", "r_ent",
                "-t", "99999",
                "-v", "1"
        });
    }

    @Test
    public void new_test6() throws Exception {


    String p = "/home/loskana/Desktop/";

        WinAsobsSolverApi.main(new String[] {"",
        "-j",  p + "pigs_new.jkl",
                "-b", "1",
                "-d", p + "pigs_new.res",
                "-t", "99999",
                "-v", "1",
                "-win", "1"
    });
    }

    @Test
    public void new_test7() throws Exception {


        String p = "/home/loskana/Desktop/";

        GreedyScorerApi.main(new String[] {"",
                "-j",  p + "ad.test.jkl",
                "-b", "1",
                "-d", p + "ad.test.dat",
                "-t", "600",
                "-v", "1"
        });
    }

    @Test
    public void new_test4() throws Exception {

        String b = "/home/loskana/Desktop/ukang/edges.txt";
        Und u = readUnd(b);
        List<Und> lu = UndSeparator.go(u);

        p(lu.size());
    }

    @Test
    public void new_test3() throws Exception {

        // String t = "10k-file1";
        String b = "/home/loskana/Desktop/ukang/16223";

        File dir = new File(b);
        File[] directoryListing = dir.listFiles();

        if (directoryListing != null) {
            for (File child : directoryListing) {
                if (child.isFile() && !child.getName().endsWith(".dot")) {
                    work3(child.getAbsolutePath());
                }

            }
        }
    }

    private void work3(String s) throws IOException {
        Und u = readUnd(s);

        new File(s + "-pc").mkdir();
        List<Und> lu = UndSeparator.go(u);

        pf("%s - %d", s, lu.size());
        int i = 0;

        for (Und u2 : lu) {
            u2.graph(f("%s-pc/%04d-%d", s, u2.n, i++));
        }

    }

    @Test
    public void new_test2() throws Exception {

        // String t = "10k-file1";
        String b = "/home/loskana/Documents/structural-heuristic/comparison/result";

        File dir = new File(b);
        File[] directoryListing = dir.listFiles();

        if (directoryListing != null) {
            for (File child : directoryListing) {
                if (child.isFile() && child.getName().endsWith(".net")) {
                    work2(child.getAbsolutePath());
                }

            }
        }
    }

    private void work2(String s) {
        p(s);

        BayesianNetwork b = BnNetReader.ex(s);

        s = s.replace(".net", "");
        // SamGe.ex(pb, set + ".dat", 5000);

        BnNetWriter.ex(s + "/true.net", b);

        // BnErgWriter.ex(set + "/true.erg", pb);
        // BnUaiWriter.ex(set + "/true.uai", pb);
    }

    @Test
    public void new_test() throws Exception {

        // String t = "10k-file1";
        String b = "/home/loskana/Desktop/data/";
        File dir = new File(b);
        File[] directoryListing = dir.listFiles();

        if (directoryListing != null) {
            for (File child : directoryListing) {
                if (child.isFile() && child.getName().contains(".dat")) {
                    work(child.getAbsolutePath());
                }
            }
        }
    }

    private void work(String s) throws Exception {
        p(s);

        String base = s.substring(0, s.lastIndexOf("."));

        String dat = base + ".dat";

        if (!new File(dat).exists()) {
            CsvToDat.go(s, dat);
        }

        File f = new File(base);

        f.mkdir();

        String jkl = base + "/jkl";

        if (!new File(jkl).exists()) {
            IndependenceScorer is = new IndependenceScorer();

            is.max_pset_size = 4;
            is.max_exec_time = 1;
            is.verbose = 1;
            is.ph_scores = jkl;
            is.scoreNm = "bic";
            is.go(getDataSet(dat));
        }

    }

    @Test
    public void newFileReader() throws Exception {
        IndependenceScorer is = new IndependenceScorer();

        is.max_pset_size = 3;
        is.max_exec_time = 1;
        is.verbose = 1;
        is.scoreNm = "bdeu";

        // default
        is.ph_scores = basePath + "/scorer/child-5000-new.jkl";
        is.go(getDataSet(basePath + "/scorer/child-5000.dat"));

        // no variable names
        is.ph_scores = basePath + "/scorer/child-5000-new2.jkl";
        is.go(getDataSet(basePath + "/scorer/child-5000-2.dat"));

        // No cardinalities
        is.ph_scores = basePath + "/scorer/child-5000-new3.jkl";
        is.go(getDataSet(basePath + "/scorer/child-5000-3.dat"));

        // no both
        is.ph_scores = basePath + "/scorer/child-5000-new4.jkl";
        is.go(getDataSet(basePath + "/scorer/child-5000-4.dat"));
    }

    @Test
    public void infer() throws Exception {

        // String t = "10k-file1";

        String t = "asia";

        String b = "/home/loskana/Desktop/new/" + t;

        String dat = b + "-5000.dat";
        String jkl = b + ".jkl";
        String res = b + "-quiet.res";
        String net = b + ".net";

        // if (! new File(net).exists()) {
        BnResReader r = new BnResReader();
        ParLe p = new ParLeBayes(10);

        p.go(r.go(res), getDataSet(dat));
        BnNetWriter.ex(p.bn, getWriter(net));
        // }

        BayesianNetwork f = BnNetReader.ex(net);
        java.util.Random rnd = getRandom();
        int q = rnd.nextInt(f.n_var);
        TIntIntHashMap ev = new TIntIntHashMap();

        for (int j = 0; j < 500; j++) {
            int e = rnd.nextInt(f.n_var);

            if (e == q) {
                continue;
            }
            ev.put(e, rnd.nextInt(f.arity(e)));
        }

        VariableElimination inf = new VariableElimination(f, 1);

        inf.elim = VariableElimination.EliminMethod.Greedy;
        query(q, ev, inf);

        inf.elim = VariableElimination.EliminMethod.Heu;
        query(q, ev, inf);

        /*
         inf.elim = Inference.EliminMethod.MinFill;
         query(q, ev, inf);

         inf.elim = Inference.EliminMethod.MinWidth;
         query(q, ev, inf);
         */

        String input = b + ".inp";
        String output = b + ".out";

        Writer w = getWriter(input);

        wf(w, "<batch> \n");
        wf(w, "<query network=\"%s\">", net);
        wf(w, "<exactmap timelimit=\"60\"><id>A</id> </exactmap>");
        wf(w, "</query></batch>");
        w.close();

        // BatchTool.main(new String[]{input, output});

    }

    private void query(int q, TIntIntHashMap ev, VariableElimination inf) {
        long start = System.currentTimeMillis();
        BayesianFactor fac = inf.query(q, ev);
        long elaps = System.currentTimeMillis() - start;

        pf("Time: %d \n", elaps);
        pf("Query: %d, evidence: %s \n", q, ev.toString());
        pf("Factor: %s \n", fac.toString());
    }

    @Test
    public void output() throws Exception {

        // String t = "10k-file1";

        // String t = "GSE31312";

        // String t = "diabetes";

        String t = "child";

        String b = "/home/loskana/Desktop/new/" + t;
        String dat = b + "-5000.dat";
        String jkl = b + ".jkl";
        String res;

        BnResReader r = new BnResReader();

        if (!new File(jkl).exists()) {
            IndependenceScorer is = new IndependenceScorer();

            is.max_pset_size = 3;
            is.max_exec_time = 1;
            is.verbose = 1;
            is.go(getDataSet(dat));
        }

        String q;
        BayesianNetwork f;

        q = "-quiet";
        res = b + q + ".res";
        new File(b + q + ".png").delete();
        QuietGreedySolver quiet = new QuietGreedySolver();

        quiet.thread_pool_size = 1;
        quiet.verbose = 1;
        quiet.tw = 3;
        quiet.init(getScoreReader(jkl, 0), 100);
        quiet.go(res);

        GraphWriter.go(res, dat, b + q);

        /*
         q = "-nice";
         res = pb + q + ".res";
         AsobsNiceSolver nice = new AsobsNiceSolver();
         nice.max_exec_time = 10;
         nice.verbose = 1;
         nice.ex(getScoreReader(jkl, 0), res);
         */
    
        /*
         q = "-brut";
         res = pb + q + ".res";
         BrutalMCSolver brut = new BrutalMCSolver();
         brut.max_exec_time = 10;
         brut.tw = 3;
         brut.ex(getScoreReader(jkl, 0), res);
         f = r.ex(res);
         f.directed().graph(pb + q);



         */
        // NamedDirected d = f.directed();

    }

    @Test
    public void testest(){
        IndependenceScorerApi.main(new String[] {
                "",
                "-d", "/home/loskana/Desktop/test/hepar2-5000.dat",
                "-b", "0",
                "-j", "/home/loskana/Desktop/test/new.jkl",
                "-t", "6",
                "-u", "0",
                "-v", "2"
        });
    }
}
