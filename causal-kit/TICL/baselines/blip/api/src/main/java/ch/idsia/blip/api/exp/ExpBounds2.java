package ch.idsia.blip.api.exp;


import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.analyze.Entropy;
import ch.idsia.blip.core.utils.score.BIC;
import ch.idsia.blip.core.learn.scorer.BaseScorer;
import ch.idsia.blip.core.learn.scorer.IndependenceScorer;
import ch.idsia.blip.core.learn.scorer.SeqUltScorer;
import ch.idsia.blip.core.utils.data.array.TDoubleArrayList;
import ch.idsia.blip.core.utils.data.hash.TIntIntHashMap;
import ch.idsia.blip.core.utils.math.FastMath;

import java.io.BufferedReader;
import java.io.File;
import java.io.Writer;
import java.util.HashMap;
import java.util.Map;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class ExpBounds2 {

    String path = System.getProperty("user.dir") + "/experiments/uci_data/";

    public static void main(String[] args) {
        try {

            if (args.length == 2) {
                new ExpBounds2().go(args[0], args[1]);
            } else if (args.length == 1) {
                // new ExpBounds2().tryglass();
                new ExpBounds2().testBounds(args[0]);
                // new ExpBounds2().testCoroll();
                // new ExpBounds2().test6();
                // new ExpBounds2().testBounds();
            } else
             new ExpBounds2().test();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void test() throws Exception {
        path = "/home/loskana/Desktop/bounds/data/";
        go(path + "autos.arff", "7");
    }

    private void tryglass() throws Exception {
        go(path + "glass.arff", "4");
    }

    private void go(String s, String p) throws Exception {
        int pmax = Integer.valueOf(p);

        SeqUltScorer sq = new SeqUltScorer();
        DataSet dat = getDataFromArff(s);

        Entropy h = new Entropy(dat);

        // p(h.computeHCond(0, new int[]{1, 2, 3}));
        // p(h.computeHCond(7, new int[]{1, 2, 3}));

        // p(h.computeHCond(1, new int[]{0, 2, 7}));
        // p(h.computeHCond(6, new int[]{0, 2, 7}));

        long start = System.currentTimeMillis();


        sq.max_pset_size = pmax;
        sq.verbose = 2;
        sq.score = new BIC(dat);
        sq.ph_scores = s + ".jkl";
        sq.thread_pool_size = 1;
        sq.max_exec_time = dat.n_var * 20;
        sq.go(dat);


        int elapsed = (int) Math.ceil((System.currentTimeMillis() - start) / 1000.0);

        int t = 0;

        for (int i = 1; i <= pmax; i++) {
            t += dat.n_var * comb(dat.n_var - 1, i);
        }

        pf("& %20s & %20s & %20s & %20s & %20s & %20s & %20s & %20s & %20s & %20s\n",
                "pmax", "tot", "time", "h(X|pi)", "h(Y|pi)", "h(X|pi) + h(Y|pi)", "h(X)",
                "h(Y)", "h(X) + h(Y)", "new");
        pf("& %20d & %20d & %20d & %20d & %20d & %20d & %20d & %20d & %20d & %20d \n",
                pmax, t, elapsed, sq.prune_xpi, sq.prune_ypi, sq.prune_xypi, sq.prune_x,
                sq.prune_y, sq.prune_xy, sq.prune_new);
    }

    private void test6() throws Exception {
        String h = "car";

        ult(h, getDataSet(path + h + ".dat"), new HashMap<String, Integer>());

        h = "diabetes";
        ult(h, getDataFromArff(path + h + ".arff"),
                new HashMap<String, Integer>());
    }

    private void testCoroll() {

        HashMap<String, Integer> hmp = new HashMap<String, Integer>();

        for (File file : new File(path).listFiles()) {
            if (file.isDirectory()) {
                continue;
            }

            if (!(file.getName().endsWith(".arff"))) {
                continue;
            }

            p(file.getName());

            DataSet dat = getDataFromArff(file.getAbsolutePath());

            corollary(file.getName(), dat, hmp);
        }

        Map<String, Integer> hsp = sortByValues(hmp);

        for (String s : hsp.keySet()) {
            p(s);
        }
    }

    private void testBounds(String path) throws Exception {

        int i = 0;

        HashMap<String, Integer> hmp = new HashMap<String, Integer>();

        p(path);

        String data_path = path + "data";

        String h = null;

        for (File file : new File(data_path).listFiles()) {
            if (file.isDirectory()) {
                continue;
            }

            if (!(file.getName().endsWith(".arff"))) {
                continue;
            }

            String s = "";

            DataSet dat = getDataSet(file.getAbsolutePath());

            String nm = file.getName().replace(".arff", "");

            for (int k : new int[] { 3, 4, 5, 6, 7}) {

                BufferedReader rd = getReader(path + "/result/" + nm + "-" + k);

                h = rd.readLine();

                if (!s.equals("")) {
                    s += "\n& & ";
                }

                s += rd.readLine() + "\\\\";
            }

            while (s.contains("  ")) {
                s = s.replace("  ", " ");
            }

            s = f(
                    "\\multirow{3}{*}{%s} & \\multirow{3}{*}{%d} & \\multirow{3}{*}{%d} \n",
                    nm, dat.n_var, dat.n_datapoints)
                            + s;
            hmp.put(s, dat.n_var);
        }

        p(h);
        Map<String, Integer> hsp = sortByValues(hmp);

        for (String s : hsp.keySet()) {
            p(s);
        }
    }

    private void corollary(String l, DataSet dat, HashMap<String, Integer> hmp) {

        double c2 = 1 + log2(dat.n_datapoints) - log2(log2(dat.n_datapoints));
        int s2 = (int) Math.ceil(c2);

        pf("%s - c2 %d \n", l, s2);

        double[] h = new double[dat.n_var];
        Entropy ent = new Entropy(dat);

        for (int i = 0; i < dat.n_var; i++) {
            h[i] = ent.computeH(i);
        }

        if (l.equals("autos.arff")) {
            p("ciao");
        }

        TDoubleArrayList ss = new TDoubleArrayList();
        TIntIntHashMap sm = new TIntIntHashMap();

        for (int i = 0; i < dat.n_var; i++) {
            double c1 = 0;

            for (int j = 0; j < dat.n_var; j++) {
                if (i == j) {
                    continue;
                }

                double a = Math.min(h[i], h[j]);

                a /= ((dat.l_n_arity[i] - 1) * (dat.l_n_arity[j] - 1));

                double c = 1 + log2(a) + log2(dat.n_datapoints)
                        - log2(log2(dat.n_datapoints));

                // pc = Math.max(pc, 1);

                if (c > c1) {
                    c1 = c;
                }
            }

            pf("# %d - c1 %.2f\n", i, c1);

            ss.add(c1);
            int c = (int) Math.ceil(c1);

            if (c == 0) {
                p("cia");
            }
            if (!sm.containsKey(c)) {
                sm.put(c, 0);
            }

            sm.put(c, sm.get(c) + 1);
        }

        String s1 = "";

        for (int k : sm.keys()) {
            if (!s1.equals("")) {
                s1 += ", ";
            }
            s1 += f("%d (%d)", k, sm.get(k));
        }

        String s = f("%15s & %5d & %5d & %40s & %5d \\\\",
                l.replace(".arff", ""), dat.n_var, dat.n_datapoints, s1, s2);

        hmp.put(s, dat.n_var);

    }

    private double log2(double n) {
        return FastMath.log(n) / FastMath.log(2);
    }

    public void testN() throws Exception {
        String l = "adult";
        String s = f("%s/%s.dat", path, l);
        DataSet dat = getDataSet(s);

        IndependenceScorer sq = new IndependenceScorer();

        sq.max_pset_size = 3;
        sq.verbose = 0;
        sq.scoreNm = "bic";
        sq.ph_scores = "g";
        sq.thread_pool_size = 1;
        sq.max_exec_time = dat.n_var * 20;
        p(sq.max_exec_time);
        sq.go(dat);
    }

    private void ult(String l, DataSet dat, HashMap<String, Integer> hmp) throws Exception {

        pf("%s \n", l);
        String sh = "";

        for (int pmax : new int[] { 3, 4, 5}) {

            long start = System.currentTimeMillis();
            SeqUltScorer sq = new SeqUltScorer();

            exe(l, dat, sq, "ult", pmax);

            int t = 0;

            for (int i = 1; i <= pmax; i++) {
                t += dat.n_var * comb(dat.n_var - 1, i);
            }

            String s = f("& %d & %d & %d & %d & %d", pmax, t, sq.prune_xpi,
                    sq.prune_ypi, sq.prune_xypi);

            pf(s);
            sh += s + "\n";
            pf("      (required: %.1f seconds) \n",
                    (System.currentTimeMillis() - start) / 1000.0);

            exe(l, dat, new IndependenceScorer(), "ind", pmax);
        }

        hmp.put(sh, dat.n_var);

        p("\n");
    }

    public static double factorial(int n) {
        double fact = 1; // this  will be the result

        for (int i = 1; i <= n; i++) {
            fact *= i;
        }
        return fact;
    }

    public static int   comb(int n, int k) {
        return (int) (factorial(n) / (factorial(n - k) * factorial(k)));
    }

    private void exe(String l, DataSet dat, BaseScorer sq, String s, int pmax) throws Exception {
        String jkl = f("%s/%s/%s-%d.jkl", path, s, l, pmax);

        // if (new File(jkl).exists())
        // return;

        sq.max_pset_size = pmax;
        sq.verbose = 0;
        sq.scoreNm = "bic";
        sq.ph_scores = jkl;
        // sq.thread_pool_size = 1;
        sq.max_exec_time = dat.n_var * 20;
        sq.go(dat);
    }

    private void see5() throws Exception {
        for (String net : new String[] {
            "car", "nursery", "breast", "adult",
            "zoo", "letter", "mushroom", "wdbc", "lung"}) {
            fff(net);
        }
    }

    private void fff(String l) throws Exception {
        String s = f("%s/%s.dat", path, l);
        DataSet dat = getDataSet(s);

        IndependenceNewScorer is = new IndependenceNewScorer();

        is.max_pset_size = 1;
        is.go(dat);

        Entropy e = new Entropy(dat);

        Writer w = getWriter(f("%s/nnnn/%s.dat", path, l));

        for (int i = 0; i < dat.n_var; i++) {
            double t = 0;

            for (int j = 0; j < dat.n_var; j++) {
                if (i == j) {
                    continue;
                }
                t += is.oneScoressss[j].get(i);
            }

            wf(w, "%.5f %.2f %d \n", e.computeH(i), t, i);
        }

        w.close();

        w = getWriter(f("%s/nnnn/%s.plt", path, l));
        wf(w, "set terminal png size 800,600 \n");
        wf(w, "set output '%s.png' \n", l);
        wf(w, "plot  '%s.dat' \n", l);
        w.close();

        ProcessBuilder pb = new ProcessBuilder("gnuplot", f("%s.plt", l));

        pb.directory(new File(f("%s/nnnn/", path)));

        procOutput(pb.start());

    }

    private void see4() throws Exception {
        // gh("adult", 5);
        // gh("zoo", 5);
        // gh("mushroom", 5);
        gh("wdbc", 5);
    }

    public void see() throws Exception {

        // for (String net : new String[]{"car", "nursery", "breast", "adult", "zoo", "letter", "mushroom", "wdbc", "lung"}) {
        for (String net : new String[] {
            "breast", "adult", "zoo", "letter",
            "mushroom", "wdbc", "lung"}) {

            gh(net, 5);
        }

    }

    private void gh(String net, int k) throws Exception {

        String s = f("%s/%s.dat", path, net);
        DataSet dat = getDataSet(s);

        IndependenceNewScorer seq = new IndependenceNewScorer();

        g(s, k, seq, net);
    }

    public void see3() throws Exception {
        // gh("adult", 3);
        gh("zoo", 3);
        // gh("letter", 3);
        // gh("mushroom", 3);
        // gh("wdbc", 3);
    }

    public void see2() throws Exception {

        for (String net : new String[] { "new5", "new10", "new15", "new20"}) {

            gh(net, 5);
        }

    }

    private void g(String s, int k, IndependenceNewScorer seq, String net) throws Exception {

        String jkl = f("%s/new/%s.jkl", path, net);
        String res = f("%s/new/%s", path, net);

        // if (!new File(res).exists()) {
        seq.max_pset_size = k;
        seq.scoreNm = "bic";
        seq.max_exec_time = 10000000;
        seq.verbose = 1;
        seq.ph_scores = jkl;
        // seq.choice_variables = "4";
        // seq.thread_pool_size = 1;
        // seq.thread_pool_size = 1;
        seq.wr = getWriter(res + ".all");
        seq.wr2 = getWriter(res + ".tryFirst");
        seq.go(s);
        seq.wr.close();
        // }

        for (String str : new String[] { "all", "tryFirst"}) {
            String plt = f("%s/new/%s.%s.plt", path, net, str);

            Writer wr = getWriter(plt);

            wf(wr, "set terminal png size 800,600 \n");
            wf(wr, "set output '%s.%s.png' \n", net, str);
            wf(wr, "set style increment user \n");
            wf(wr, "set style line 1 lc rgb 'black' pt 3 \n");
            wf(wr, "set style line 2 lc rgb 'blue'  pt 3 \n");
            wf(wr, "set style line 3 lc rgb 'red' pt 3 \n");
            wf(wr, "set style line 4 lc rgb 'green'  pt 3 \n");

            wf(wr, "set style data points \n");
            wf(wr, "plot '%s.%s' using 1:2:3 linecolor variable with points \n",
                    net, str);
            wr.close();

            ProcessBuilder pb = new ProcessBuilder("gnuplot",
                    f("%s.%s.plt", net, str));

            pb.directory(new File(f("%s/new/", path)));

            procOutput(pb.start());

        }
    }
}
