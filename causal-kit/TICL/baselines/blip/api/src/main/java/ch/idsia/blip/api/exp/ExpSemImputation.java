package ch.idsia.blip.api.exp;


import ch.idsia.blip.core.App;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.io.bn.BnUaiReader;
import ch.idsia.blip.core.io.bn.BnUaiWriter;
import ch.idsia.blip.core.io.dat.*;
import ch.idsia.blip.core.learn.missing.HardMissingSEM;
import ch.idsia.blip.core.learn.param.ParLe;
import ch.idsia.blip.core.learn.param.ParLeBayes;
import ch.idsia.blip.core.learn.scorer.IndependenceScorer;
import ch.idsia.blip.core.learn.solver.brtl.BrutalMaxSolver;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.ParentSet;
import ch.idsia.blip.core.utils.RandomStuff;

import java.io.File;
import java.io.IOException;
import java.io.Writer;
import java.util.HashMap;
import java.util.Map;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class ExpSemImputation extends App {

    int max_fold = 5;

    String locPath = System.getProperty("user.home")
            + "/Desktop/SEM/imputation/";

    String path;

    protected int max_time = 1;

    protected int thread = 1;

    protected int tw = 6;

    int[] percs = { 1, 2, 3, 4, 5, 8, 10, 12, 15 };

    protected String bnPath;

    protected String sem;

    protected String suffix =  "sem";

    protected String competit = "rfsrc";

    public static void main(String[] args)
        throws IOException {
        try {
            if (args.length > 1) {
                if (args[0].equals("prepare")) {
                    new ExpSemImputation().prepareNow(args);
                } else if (args[0].equals("go")) {
                    new ExpSemImputation().goNow(args);
                } else if (args[0].equals("measure")) {
                    new ExpSemImputation().measureNow(args);
                }
            } else {
                new ExpSemImputation().test();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    protected int[] getStrawImputation(DataSet d) {
        int[] straw = new int[d.n_var];

        for (int i = 0; i < d.n_var; i++) {
            int max = -1;
            int count = 0;

            for (int j = 0; j < d.l_n_arity[i]; j++) {
                int l = d.row_values[i][j].length;

                if (l > count) {
                    count = l;
                    max = j;
                }
            }
            straw[i] = max;
        }
        return straw;
    }


    public ExpSemImputation() {
        init();
        prepare();
    }

    protected void test()
        throws Exception {
        this.thread = 3;
        this.path = locPath;

       // prepareNow(new String[] { "", locPath, "accidents.test" });

        // goNow(new String[] { "", locPath, "ad", "1", "15" });

       //  prepareNow(new String[] { "", locPath, "nltcs.test" });
        goNow(new String[] { "", locPath, "nltcs.test", "1", "5" });
    }

    private void doALL()
        throws Exception {
        HashMap<String, Integer> bns = new HashMap<String, Integer>();

        for (File file : new File(this.path + "/data/").listFiles()) {
            if (!file.isDirectory()) {
                if (file.getName().endsWith(".dat")) {
                    DataSet d = getDataSet(file.getAbsolutePath());

                    RandomStuff.writeDataSet(d, this.path + "/" + file.getName());

                    bns.put(file.getName().replace(".dat", ""), d.n_var);
                }
            }
        }
        Map<String, Integer>  s_bns = RandomStuff.sortByValues(bns);

        for (String bn_name : (s_bns.keySet())) {
            prepareNow(new String[] { this.path, bn_name });
        }
    }

    protected void prepareNow(String[] args)
        throws IOException {
        this.path = args[1];
        String bn_name = args[2];

        String orig = f("%s/../data/%s.dat", this.path, bn_name);

        DataSet dat = null;
        short[][] samples = new short[0][];

        for (int f = 1; f <= this.max_fold; f++) {
            prepare(bn_name, f);
            for (int p = 0; p < this.percs.length; p++) {
                int per = this.percs[p];

                String newPath = f("%s/work/%s/fold%d/%d/",
                        this.path, bn_name, f,
                        per);
                File fi = new File(newPath);

                if (!fi.exists()) {
                    fi.mkdirs();
                }
                String missing = newPath + "missing.dat";

                fi = new File(missing);
                if (!fi.exists()) {
                    if (dat == null) {
                        dat = getDataSet(orig);

                        BaseFileLineReader dr = RandomStuff.getDataSetReader(
                                orig);

                        try {
                            dr.readMetaData();
                            samples = new short[dat.n_datapoints][];
                            int i = 0;

                            while (!dr.concluded) {
                                samples[(i++)] = dr.next();
                            }
                        } catch (IOException e) {
                            RandomStuff.logExp(e);
                        }
                    }
                    DatFileLineWriter dflw = new DatFileLineWriter(dat,
                            RandomStuff.getWriter(missing));

                    dflw.writeMetaData();
                    short[] s = new short[dat.n_var];

                    for (int r = 0; r < samples.length; r++) {
                        ArrayUtils.cloneArray(samples[r], s);
                        for (int n = 0; n < dat.n_var; n++) {
                            int ra = randInt(0, 100);

                            if (ra < per) {
                                s[n] = -1;
                            }
                        }
                        dflw.next(s);
                    }
                }
            }
        }
    }

    protected void goNow(String[] args)
        throws Exception {
        this.path = args[1];
        String bn_name = args[2];

        p(bn_name);

        int fo = Integer.valueOf(args[3]);
        int per = Integer.valueOf(args[4]);

        max_time = Integer.valueOf(args[5]);

        String newPath = f("%s/work/%s/fold%d/%d/", this.path,
                bn_name, fo, per);
        String missing = newPath + "missing.dat";

        sem = newPath + "/" + suffix + "/";

        bnPath = sem + "new.uai";

        File fi = new File(sem);
        if (!fi.exists()) {
            fi.mkdir();
        }
        p(bnPath);

        HardMissingSEM mSem;
        mSem = getHardMissingSEM();

        fi = new File(bnPath);
        if (!fi.exists()) {

            mSem.init(sem, this.max_time, this.tw);
            mSem.thread_pool_size = this.thread;
            BayesianNetwork bn = mSem.go(missing);

            RandomStuff.writeBayesianNetwork(bn, bnPath);
        }

        String compl = newPath + suffix + "-compl.dat";
        fi = new File(compl);
        p(compl);
        if (!fi.exists()) {
            BayesianNetwork bn = getBayesianNetwork(bnPath);
            mSem.writeCompletedDataSet(missing, bn, compl);
        }
    }

    protected HardMissingSEM getHardMissingSEM() {
        return new HardMissingSEM();
    }

    private void prepare(String net, int j)
        throws IOException {
        String s = f("%s/work/%s/fold%d/", this.path, net, j);
        File f = new File(s);

        if (!f.exists()) {
            f.mkdir();
        }
    }

    private void def(String net)
        throws Exception {
        String d1 = f("%s/data/%s.dat", this.path, net);
        DataSet dat = getDataSet(d1);

        String s = f("%s/work/%s/", this.path, net);
        File f = new File(s);

        if (!f.exists()) {
            f.mkdir();
        }
        this.max_time = dat.n_var;

        String jkl = s + "jkl";

        f = new File(jkl);
        if (!f.exists()) {
            IndependenceScorer is = new IndependenceScorer();
            HashMap<String, String> o = new HashMap<String, String>();

            o.put("ph_scores", jkl);
            o.put("max_pset_size", "5");
            o.put("time_for_variable", String.valueOf(this.max_time));
            o.put("scoreNm", "bdeu");
            o.put("alpha", "10.0");
            o.put("thread_pool_size", String.valueOf(this.thread));
            is.init(o);
            is.go(dat);
        }
        String res = s + "res";

        f = new File(res);
        if (!f.exists()) {
            BrutalMaxSolver gs = new BrutalMaxSolver();

            prp(gs, res, this.max_time, RandomStuff.getScoreReader(jkl, 0),
                    this.thread, Math.min(this.tw, dat.n_var - 1));
            gs.go();
        }
        String uai = s + "res.uai";

        f = new File(uai);
        BayesianNetwork bn;

        if (!f.exists()) {
            ParLe pl = new ParLeBayes(2.0D);

            bn = pl.go(RandomStuff.getBayesianNetwork(res), dat);
            BnUaiWriter.ex(bn, uai);
        } else {
            bn = BnUaiReader.ex(uai);
        }
        bn.writeGraph(s + "res");
    }

    protected void measureNow(String[] args)
            throws IOException {
        this.path = args[1];
        String bn_name = args[2];

        RandomStuff.pf("### %s ###\n", bn_name);

        String path = f("%s/../data/%s.dat", this.path, bn_name);

        String res_path = f("%s/res/data/%s/", this.path, bn_name);
        if (!new File(res_path).exists())
            new File(res_path).mkdirs();

        DataSet d = getDataSet(path);
        int[] straw = getStrawImputation(d);
        for (int p = 0; p < this.percs.length; p++) {
            int per = this.percs[p];

            Writer res = getWriter(res_path + "res-" + per);

            Writer sct = getWriter(res_path + "sct-" + per);

            Writer tms = getWriter(res_path + "tms-" + per);

            RandomStuff.pf("@@@ %d \n", per);
            for (int f = 1; f <= this.max_fold; f++) {

                measure(path, d, straw, bn_name, per, f, res, sct, tms);

                res.flush();
                sct.flush();
                tms.flush();
            }

            res.close();
        }
    }


    protected void measure(String s_orig, DataSet d, int[] straw, String bn_name, int per, int fold, Writer res, Writer sct, Writer tms)
        throws IOException {

        //  p(competit);

        String newPath = f("%s/work/%s/fold%d/%d/",
                this.path, bn_name, fold,
                per);

        DatFileLineReader orig = new DatFileLineReader(s_orig);

        orig.readMetaData();

        DatFileLineReader missing = new DatFileLineReader(
                newPath + "missing.dat");

        missing.readMetaData();

        int t_missing = 0;

        int t_straw = 0;

        String s_sem = newPath + "sem-compl.dat";
        if (!new File(s_sem).exists()) {
            pf("%s - NOT EXISTS!!!\n", s_sem);
            return;
            // DataSet s = getDataSet(s_sem.replace(".dat", ".arff"));
            // DatFileWriter.ex(s, s_sem);
        }

        DatFileLineReader sem = new DatFileLineReader(s_sem);
        sem.readMetaData();

        int t_sem = 0;

        String s_comp = newPath + competit + "-compl.dat";
        // String s_comp = s_sem.replace("/imputation5/", "/imputation1/");
        // s_comp = s_comp.replace("/imputation10/", "/imputation1/");

        p(s_comp);

        if (!new File(s_comp).exists()) {
            pf("%s - NOT EXISTS!!!\n", s_comp);
            return;
            // DataSet s = getDataSet(s_comp.replace(".dat", ".arff"));
            // DatFileWriter.ex(s, s_comp);
        }


        DatFileLineReader comp = new DatFileLineReader(s_comp);
        comp.readMetaData();
                int t_com = 0;


        while (!missing.concluded) {
            short[] o = orig.next();
            short[] m = missing.next();

            short[] r = comp.next();

            short[] s = sem.next();

            for (int n = 0; n < m.length; n++) {
                if (m[n] == -1) {
                    t_missing++;

                    if (o[n] == r[n]) {
                        t_com++;
                    }
                    if (o[n] == s[n]) {
                        t_sem++;
                    }

                    if (o[n] == straw[n]) {
                        t_straw++;
                    }
                }
            }
        }

        double v_comp = t_straw * 1.0D / t_com;
        double v_sem = t_straw * 1.0D / t_sem;

                RandomStuff.pf("Missing: %d / %d (%.2f), %s: %.2f, sem: %.2f \n",
                t_missing, d.n_datapoints * d.n_var,
                t_missing * 1.0D / (d.n_datapoints * d.n_var), competit,
                v_comp, v_sem);

                        RandomStuff.pf("(correct)  %s: %d, sem: %d, straw: %d \n", competit, t_com,
                t_sem, t_straw);


        int tm_comp = Integer.valueOf(
                 RandomStuff.getReader(newPath + competit + "-compl.sec").readLine());


        int tm_sem = Integer.valueOf(
                RandomStuff.getReader(newPath + "sem-compl.sec").readLine());

        wf(res, "%.3f\t%.3f\t%.3f\n", tm_sem *1.0 / tm_comp, t_sem * 1.0 / t_com, t_sem*1.0/t_straw);


        wf(tms, "%d\t%04d\t%.4f\n", tm_sem,  d.n_var, tm_sem * 1.0 / d.n_var);

        wf(sct, "%04d\t%d\t%.4f\t%d\t%.4f\t%.4f\n", d.n_var, t_sem, t_sem * 1.0/ t_missing, t_com, t_com * 1.0/ t_missing, t_sem * 1.0/t_com);
    }

    protected double last(String ll, int j) {
        ll = ll.replace("%", "").trim();
        String[] aux = ll.split("\\s+");

        return Double.valueOf(aux[j]);
    }

    private DataSet complete(DataSet d, BayesianNetwork bn) {
        return null;
    }

    private class Result {
        private final double acc;
        private final double mae;

        public Result(double mae, double acc) {
            this.mae = mae;
            this.acc = acc;
        }
    }

    protected void prp(BrutalMaxSolver gs, String res, int max_time, ParentSet[][] scoreReader, int thread, int tw) {
        gs.init();
        gs.res_path = res;
        gs.max_exec_time = max_time;
        gs.init(scoreReader);
        gs.thread_pool_size = thread;
        gs.tw = tw;
    }
}
