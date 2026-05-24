package ch.idsia.blip.api.exp;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.io.dat.ArffFileLineWriter;
import ch.idsia.blip.core.io.dat.BaseFileLineReader;
import ch.idsia.blip.core.io.dat.DatFileLineReader;
import ch.idsia.blip.core.io.dat.DatFileWriter;
import ch.idsia.blip.core.learn.missing.HardMissingSEM;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.RandomStuff;
import ch.idsia.blip.core.utils.other.StreamGobbler;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.Writer;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class ExpSemClassification extends ExpSemImputation {

    String locPath = System.getProperty("user.home")
            + "/Desktop/SEM/classification/";

    private String[] classifiers = {
        "weka.classifiers.bayes.NaiveBayes",
        "weka.classifiers.trees.J48",
            "weka.classifiers.trees.RandomForest",
            "weka.classifiers.trees.RandomTree",
            "weka.classifiers.trees.REPTree"
    };

    private String sem;

    private String compl;

    private String bnPath;

    public static void main(String[] args) throws IOException {
        try {
            if (args.length > 1) {
                if (args[0].equals("prepare")) {
                    new ExpSemClassification().prepareNow(args);
                } else if (args[0].equals("go")) {
                    new ExpSemClassification().goNow(args);
                } else if (args[0].equals("measure")) {
                    new ExpSemClassification().measureNow(args);
                } else if (args[0].equals("measure2")) {
                    new ExpSemClassification().measureNow2(args);
                } else if (args[0].equals("eval")) {
                    new ExpSemClassification().evalNow(args);
                }
            } else {
                new ExpSemClassification().test();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void doALL()
        throws Exception {
        HashMap<String, Integer> bns = new HashMap<String, Integer>();

        for (File file : new File(this.path + "/data/").listFiles()) {
            if (!file.isDirectory()) {
                if (file.getName().endsWith(".arff")) {
                    DataSet d = RandomStuff.getDataSet(file.getAbsolutePath());

                    bns.put(file.getName().replace(".arff", ""), d.n_var);
                }
            }
        }
        Map<String, Integer> s_bns = RandomStuff.sortByValues(bns);

        for (String bn_name : (s_bns.keySet())) {
            prepareNow(new String[] { "", this.path, bn_name });
        }
    }

    protected void prepareNow(String[] args)
        throws IOException {
        this.path = args[1];
        String bn_name = args[2];

        for (int f = 1; f <= this.max_fold; f++) {
            String d = RandomStuff.f("%s/data/%s.arff", this.path, bn_name);

            String s = RandomStuff.f("%s/work/%s/fold%d/", this.path, bn_name, f);
            File fi = new File(s);

            if (!fi.exists()) {
                fi.mkdirs();
            }
            Random r = new Random(System.currentTimeMillis() % 99999L);
            String d_train = s + "train.arff";
            String d_test = s + "test.arff";
            int seed = Math.abs(r.nextInt()) % 99999;

            if (!new File(d_train).exists()) {
                RandomStuff.pf("seed %d \n", seed);
                weka(
                        RandomStuff.f(
                                "java -cp weka.jar weka.filters.supervised.instance.StratifiedRemoveFolds -i %s -o %s -c last -N 10 -F %d -V -S %d",
                                d, d_train, Integer.valueOf(f),
                                Integer.valueOf(seed)));

                weka(
                        RandomStuff.f(
                                "java -cp weka.jar weka.filters.supervised.instance.StratifiedRemoveFolds -i %s -o %s -c last -N 10 -F %d -S %d",
                                d, d_test, Integer.valueOf(f),
                                Integer.valueOf(seed)));
            }
            DataSet dat = null;
            short[][] samples = new short[0][];

            for (int p = 0; p < this.percs.length; p++) {
                int per = this.percs[p];

                String newPath = RandomStuff.f("%s/work/%s/fold%d/%d/",
                        this.path, bn_name, f, per);

                fi = new File(newPath);
                if (!fi.exists()) {
                    fi.mkdirs();
                }
                String missing = newPath + "missing.arff";

                fi = new File(missing);
                if (!fi.exists()) {
                    if (dat == null) {
                        dat = RandomStuff.getDataSet(d_train);

                        BaseFileLineReader dr = RandomStuff.getDataSetReader(
                                d_train);

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
                    ArffFileLineWriter dflw = new ArffFileLineWriter(dat,
                            RandomStuff.getWriter(missing));

                    dflw.writeMetaData();
                    short[] si = new short[dat.n_var];

                    for (int rs = 0; rs < samples.length; rs++) {
                        ArrayUtils.cloneArray(samples[rs], si);
                        for (int n = 0; n < dat.n_var; n++) {
                            int ra = randInt(0, 100);

                            if (ra < per) {
                                si[n] = -1;
                            }
                        }
                        dflw.next(si);
                    }
                    dflw.close();
                }
                DataSet ds = RandomStuff.getDataSet(missing, true);

                RandomStuff.writeDataSet(ds, missing.replace("arff", "dat"));
            }
        }
    }

    protected void goNow(String[] args)
        throws Exception {
        this.path = args[1];
        String bn_name = args[2];

        RandomStuff.p(bn_name);

        int fo = Integer.valueOf(args[3]);
        int per = Integer.valueOf(args[4]);

        String newPath = RandomStuff.f("%s/work/%s/fold%d/%d/", this.path,
                bn_name, fo, per);
        String missing = newPath + "missing.arff";

        sem = newPath + "/sem/";
        File fi = new File(sem);

        if (!fi.exists()) {
            fi.mkdir();
        }

        compl = newPath + "sem-compl.arff";

        bnPath = sem + "new.uai";

        HardMissingSEM mSem = new HardMissingSEM();

        fi = new File(compl);
        if (!fi.exists()) {

            mSem.init(sem, this.max_time, this.tw);
            mSem.thread_pool_size = this.thread;
            BayesianNetwork bn = mSem.go(missing);

            RandomStuff.writeBayesianNetwork(bn, bnPath);
        }

        BayesianNetwork bn = getBayesianNetwork(bnPath);
        mSem.writeCompletedDataSet(missing, bn, compl);
    }

    protected void measureNow(String[] args)
        throws IOException {
        this.path = args[1];
        String bn_name = args[2];

        Writer wr = RandomStuff.getWriter(
                RandomStuff.f("%s/res/%s", this.path, bn_name));

        RandomStuff.wf(wr, "### %s ###\n", bn_name);
        for (int p = 0; p < this.percs.length; p++) {
            int per = this.percs[p];

            RandomStuff.wf(wr, "\n@@@ %d \n", per);
            for (String classi : this.classifiers) {
                RandomStuff.wf(wr, "\n### %s \n", classi);
                for (String m : new String[] {
                    "orig", "sem", "rfsrc", "missing" }) {
                    double mae = 0.0D;
                    double auc = 0.0D;

                    for (int f = 1; f <= this.max_fold; f++) {
                        String out = RandomStuff.f(
                                "%s/work/%s/fold%d/%d/eval/%s/%s-out", this.path,
                                bn_name, f, per, classi, m);

                        BufferedReader br = RandomStuff.getReader(out);

                        boolean ok = false;
                        String ll;

                        while ((ll = br.readLine()) != null) {
                            if (ll.contains("Error on test data")) {
                                ok = true;
                            }
                            if ((ok) && (ll.contains("Mean absolute error"))) {
                                mae += last(ll, 3);
                            }
                            if ((ok) && (ll.contains("Weighted Avg."))) {
                                auc += last(ll, 8);
                            }
                        }
                    }
                    RandomStuff.wf(wr, "%10s, auc: %4.4f, mae: %4.4f \n", m,
                            auc / this.max_fold, mae / this.max_fold);

                    wr.flush();
                }
            }
        }
        wr.close();
    }

    protected void measureNow2(String[] args)
            throws IOException {
        this.path = args[1];
        String bn_name = args[2];

        RandomStuff.pf("### %s ###\n", bn_name);

        for (int p = 0; p < this.percs.length; p++) {
            int per = this.percs[p];

            RandomStuff.pf("@@@ %d \n", per);
            for (int f = 1; f <= this.max_fold; f++) {

                String d_f = f("%s/work/%s/fold%d/train.dat", this.path, bn_name, f);

                DataSet d;
                if (!new  File(d_f).exists()) {
                    d  =  getDataSet(d_f.replace(".dat", ".arff"));
                   DatFileWriter.ex(d, d_f);
                } else
                d = getDataSet(d_f);

                int[] straw = getStrawImputation(d);

                String newPath = f("%s/work/%s/fold%d/%d/",
                        this.path, bn_name, f,
                        per);

                measure(d_f, d, straw, newPath);
            }
        }
    }

    protected void measure(String s_orig, DataSet d, int[] straw, String newPath)
            throws IOException {
        String s_rfsrc = newPath + "rfsrc-compl.dat";
        String s_sem = newPath + "sem-compl.dat";

        if (!new File(s_rfsrc).exists()) {
            DataSet s = getDataSet(s_rfsrc.replace(".dat", ".arff"));
            DatFileWriter.ex(s, s_rfsrc);
        }
        if (!new File(s_sem).exists()) {
            DataSet s = getDataSet(s_sem.replace(".dat", ".arff"));
            DatFileWriter.ex(s, s_sem);
        }

        DatFileLineReader orig = new DatFileLineReader(s_orig);

        orig.readMetaData();

        DatFileLineReader missing = new DatFileLineReader(
                newPath + "missing.dat");

        missing.readMetaData();

        DatFileLineReader rfsrc = new DatFileLineReader(s_rfsrc);

        rfsrc.readMetaData();

        DatFileLineReader sem = new DatFileLineReader(s_sem);

        sem.readMetaData();

        int t_missing = 0;

        int t_rfsrc = 0;
        int t_sem = 0;
        int t_straw = 0;

        while (!missing.concluded) {
            short[] o = orig.next();
            short[] m = missing.next();

            short[] r = rfsrc.next();
            short[] s = sem.next();

            for (int n = 0; n < m.length; n++) {
                if (m[n] == -1) {
                    t_missing++;
                    if (o[n] == r[n]) {
                        t_rfsrc++;
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
        double v_rfsrc = t_straw * 1.0D / t_rfsrc;
        double v_sem = t_straw * 1.0D / t_sem;

        int tm_rfsrc = Integer.valueOf(
                RandomStuff.getReader(newPath + "rfsrc-compl.sec").readLine());
        int tm_sem = Integer.valueOf(
                RandomStuff.getReader(newPath + "sem-compl.sec").readLine());

        RandomStuff.pf("Missing: %d / %d (%.2f), rfsrc: %.2f, sem: %.2f \n",
                t_missing, d.n_datapoints * d.n_var,
                t_missing * 1.0D / (d.n_datapoints * d.n_var),
                Double.valueOf(v_rfsrc), Double.valueOf(v_sem));
        RandomStuff.pf("(correct)  rfsrc: %d, sem: %d, straw: %d \n", t_rfsrc,
                t_sem, t_straw);
        RandomStuff.pf("(time)  rfsrc: %d, sem: %d \n", tm_rfsrc, tm_sem);
    }

    protected void test()
        throws Exception {
        this.thread = 0;
        this.path = locPath;

        doALL();

        goNow(new String[] { "go", locPath, "audiology", "1", "1" });
    }

    protected void evalNow(String[] args)
        throws IOException, InterruptedException {
        this.path = args[1];
        String bn_name = args[2];

        RandomStuff.pf("### %s ###\n", bn_name);
        for (int j = 1; j <= this.max_fold; j++) {
            String foldPath = RandomStuff.f("%s/work/%s/fold%d/", this.path,
                    bn_name, j);

            String train = foldPath + "train.arff";
            String test = foldPath + "test.arff";

            DataSet a = RandomStuff.getDataSet(train);

            for (int p = 0; p < this.percs.length; p++) {
                int per = this.percs[p];

                String newPath = RandomStuff.f("%s/work/%s/fold%d/%d/",
                        this.path, bn_name, j, per);

                for (String classi : this.classifiers) {
                    eval(newPath, classi, train, test, "orig");

                    String missing = newPath + "missing.arff";

                    eval(newPath, classi, missing, test, "missing");

                    String sem = newPath + "sem-compl.arff";

                    translate(a, sem);
                    eval(newPath, classi, sem, test, "sem");

                    String rfsrc = newPath + "rfsrc-compl.arff";

                    translate(a, rfsrc);
                    eval(newPath, classi, rfsrc, test, "rfsrc");
                }
            }
        }
    }

    private void translate(DataSet a, String arff)
        throws IOException {
        if (new File(arff).exists()) {
            return;
        }
        String dat = arff.replace(".arff", ".dat");
        DatFileLineReader dflr = new DatFileLineReader(dat);

        ArffFileLineWriter aflw = new ArffFileLineWriter(a, arff);

        aflw.writeMetaData();

        dflr.readMetaData();
        while (!dflr.concluded) {
            short[] s = dflr.next();

            aflw.next(s);
        }
        dflr.close();
        aflw.close();
    }

    private void eval(String newPath, String classifier, String trainData, String testData, String name)
        throws IOException, InterruptedException {
        ProcessBuilder builder = new ProcessBuilder("java", "-cp", "weka.jar",
                classifier, "-t", trainData, "-T", testData);

        builder.directory(
                new File(System.getProperty("user.home") + "/Tools/weka-3-8-1/"));

        Process proc = builder.start();

        String evalPath = RandomStuff.f("%s/eval/%s", newPath, classifier);

        File f = new File(evalPath);

        if (!f.exists()) {
            f.mkdirs();
        }
        String out = RandomStuff.f("%s/%s-out", evalPath, name);

        StreamGobbler e = new StreamGobbler(proc.getErrorStream(),
                RandomStuff.f("%s/%s-err", evalPath, name));

        e.start();
        StreamGobbler o = new StreamGobbler(proc.getInputStream(), out);

        o.start();

        proc.waitFor();
    }

    protected void weka(String cmd)
        throws IOException {
        ProcessBuilder builder = new ProcessBuilder(cmd.split("\\s+"));

        RandomStuff.cmd(builder,
                System.getProperty("user.home") + "/Tools/weka-3-8-1/");
    }
}
