package ch.idsia.blip.core.learn.missing;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.io.ScoreReader;
import ch.idsia.blip.core.io.bn.BnResReader;
import ch.idsia.blip.core.io.bn.BnUaiWriter;
import ch.idsia.blip.core.io.dat.*;
import ch.idsia.blip.core.utils.score.BDeu;
import ch.idsia.blip.core.inference.ve.BayesianFactor;
import ch.idsia.blip.core.inference.ve.VariableElimination;
import ch.idsia.blip.core.learn.param.ParLe;
import ch.idsia.blip.core.learn.param.ParLeBayes;
import ch.idsia.blip.core.learn.scorer.IndependenceScorer;
import ch.idsia.blip.core.learn.scorer.concurrency.NotifyingThread;
import ch.idsia.blip.core.learn.scorer.concurrency.ThreadCompleteListener;
import ch.idsia.blip.core.learn.solver.BaseSolver;
import ch.idsia.blip.core.learn.solver.ScoreSolver;
import ch.idsia.blip.core.learn.solver.WinAsobsSolver;
import ch.idsia.blip.core.learn.solver.brtl.BrutalMaxSolver;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import ch.idsia.blip.core.utils.data.hash.TIntIntHashMap;
import ch.idsia.blip.core.utils.ParentSet;
import ch.idsia.blip.core.utils.RandomStuff;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.TreeSet;

import static ch.idsia.blip.core.utils.RandomStuff.f;
import static ch.idsia.blip.core.utils.RandomStuff.p;


public class HardMissingSEM extends SEM implements ThreadCompleteListener {

    protected TreeSet<BaseSolver.Solution> bests;

    protected int[] allMissing;

    protected short[][] samples;

    protected int todo;

    protected TIntArrayList[][] completion;

    protected int n_compl;

    protected DataSet cleanDat;

    int max_cnt = 5;

    protected int cnt;

    protected int done;

    protected String bnPath;

    public BayesianNetwork go(String dat_path)
        throws Exception {
        this.dat_path = dat_path;

        prepare();

        bn = learnBase();

        cnt = 1;
        boolean same = false;

        while (!same && cnt < max_cnt) {

            bnPath = f ("%s/em-%d.uai", path, cnt);

            BnUaiWriter.ex(bn, bnPath);

            logf("expectation %d, expectation (%.2f) \n", cnt, elaps());

            expect();

            BayesianNetwork newBn = learn();

            if ((stationaryStr(bn, newBn)) || (stationary(bn, newBn))) {
                same = true;
            }
            bn = newBn;
            cnt += 1;
        }
        BnUaiWriter.ex(bn, path + "final.uai");

        return bn;
    }

    protected void expect()
        throws IOException, InterruptedException {
        for (int t = 0; t
                < Math.min(allMissing.length, thread_pool_size); t++) {
            Runnable r = getEmSearcher(t);
            NotifyingThread tr = new NotifyingThread(r);

            tr.addListener(this);
            tr.start();
        }
        while (done != allMissing.length) {
            synchronized (lock) {
                lock.wait();
            }
        }
    }

    protected Runnable getEmSearcher(int t) {
        return new HardEmSearcher(t);
    }

    private void complete(String bn_path, short[] sample)
        throws IOException, InterruptedException {
        String evid_path = bn_path + ".evid";

        TIntArrayList missing = new TIntArrayList();
        String s = "";

        for (int n = 0; n < n_var; n++) {
            if (sample[n] != -1) {
                s = s + f(" %d %d", n, sample[n]);
            } else {
                missing.add(n);
            }
        }
        PrintWriter w = new PrintWriter(evid_path, "UTF-8");

        RandomStuff.wf(w, "%d%s", n_var - missing.size(), s);
        w.close();

        String r = f("./daoopt -f %s -e %s", bn_path, evid_path);
        Process proc = Runtime.getRuntime().exec(r, new String[0],
                new File(System.getProperty("user.home") + "/Tools"));
        ArrayList<String> out = RandomStuff.exec(proc);
        String res = (String) out.get(out.size() - 1);

        String[] aux = res.split(" ");

        for (int i = 0; i < missing.size(); i++) {
            int n = missing.get(i);

            sample[n] = Short.valueOf(aux[(n + 3)]);
        }
        proc.getInputStream().close();
        proc.getOutputStream().close();
        proc.getErrorStream().close();
    }

    private DataSet cloneDataSet(DataSet dat) {
        DataSet newData = new DataSet();

        newData.n_var = dat.n_var;
        newData.missing_l = new int[n_var][];
        newData.l_nm_var = new String[n_var];
        ArrayUtils.cloneArray(dat.l_nm_var, newData.l_nm_var);
        newData.l_n_arity = new int[n_var];
        ArrayUtils.cloneArray(dat.l_n_arity, newData.l_n_arity);
        newData.row_values = new int[n_var][][];
        for (int i = 0; i < dat.n_var; i++) {
            newData.row_values[i] = new int[newData.l_n_arity[i]][];
        }
        return newData;
    }

    protected void prepare() {
        super.prepare();

        start = System.currentTimeMillis();

        bests = new TreeSet<BaseSolver.Solution> ();

        dat = RandomStuff.getDataSet(dat_path, true);
        n_var = dat.n_var;

        cleanDat = RandomStuff.getDataSet(dat_path, false);

        tw = Math.min(n_var - 1, tw);

        allMissing = new int[0];
        for (int i = 0; i < n_var; i++) {
            if (dat.missing_l[i] != null) {
                allMissing = ArrayUtils.union(allMissing,
                        dat.missing_l[i]);
            }
        }
        BaseFileLineReader dr = RandomStuff.getDataSetReader(dat_path);

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
        completion = new TIntArrayList[n_var][];
        for (int n = 0; n < n_var; n++) {
            completion[n] = new TIntArrayList[dat.l_n_arity[n]];
            for (int v = 0; v < dat.l_n_arity[n]; v++) {
                completion[n][v] = new TIntArrayList(1000);
            }
        }

    }

    private BayesianNetwork learnBase()
        throws Exception {
        String g = "base";

        String base = f("%s/%s", path, g);

        String jkl = base + ".jkl";

        IndependenceScorer is = new IndependenceScorer();

        prp(is, jkl, 1, 5, "bdeu", 1.0D,
                thread_pool_size);
        is.verbose = 0;
        is.go(dat);

        String res = base + ".res";
        ParentSet[][] sc = new ScoreReader(jkl).readScores();

        WinAsobsSolver gs = new WinAsobsSolver();

        prp(gs, res, sc, 5, thread_pool_size);
        gs.write_solutions = false;
        gs.verbose = 1;
        gs.go();

        BayesianNetwork newbn = BnResReader.ex(res);

        newbn.writeGraph(base);
        newbn.l_nm_var = dat.l_nm_var;

        ParLe p = new ParLeBayes(2.0D);

        newbn = p.go(newbn, dat);

        return newbn;
    }

    protected BayesianNetwork learn()
        throws Exception {
        String base = f("%s/%s-%d", path, "em", cnt);

        String jkl = base + ".jkl";

        IndependenceScorer is = new IndependenceScorer();

        int pset_time = Math.min(time_for_variable * dat.n_var, 1200);
        p(pset_time);
        prp(is, jkl, tw, pset_time, "bdeu", 1.0D,
                thread_pool_size);
        is.verbose = 0;
        DataSet d = setScore(is);

        is.go(d);

        String res = base + ".res";
        ParentSet[][] sc = new ScoreReader(jkl).readScores();

        BrutalMaxSolver gs = new BrutalMaxSolver();

        int brt_time = Math.min(time_for_variable * dat.n_var / 10, 600);
        prp(gs, res, sc, brt_time, thread_pool_size);
        gs.tw = tw;
        gs.tryFirst = bests;
        gs.out_solutions = 10;
        gs.write_solutions = false;
        gs.verbose = 1;
        gs.go();

        bests.addAll(gs.open);

        BayesianNetwork newbn = BnResReader.ex(res);

        newbn.writeGraph(base);
        newbn.l_nm_var = dat.l_nm_var;

        ParLe p = new ParLeBayes(2.0D);

        newbn = p.go(newbn, d);

        String bn_path = base + ".uai";

        BnUaiWriter.ex(bn, bn_path);

        return newbn;
    }

    protected DataSet setScore(IndependenceScorer is) {
        int[][][] n_compl = new int[n_var][][];

        for (int n = 0; n < n_var; n++) {
            n_compl[n] = new int[dat.l_n_arity[n]][];
            for (int v = 0; v < dat.l_n_arity[n]; v++) {
                n_compl[n][v] = new int[dat.row_values[n][v].length + completion[n][v].size()];
                ArrayUtils.cloneArray(dat.row_values[n][v], n_compl[n][v]);
                int d = dat.row_values[n][v].length;

                for (int i = 0; i < completion[n][v].size(); i++) {
                    n_compl[n][v][(i + d)] = completion[n][v].getQuick(i);
                }
                Arrays.sort(n_compl[n][v]);
            }
        }
        DataSet d = new DataSet(dat);

        d.row_values = n_compl;

        is.score = new BDeu(1.0D, d);

        return d;
    }

    protected void prp(IndependenceScorer is, String jkl, int i, int max_time, String score, double alpha, int thread_pool_size) {
        is.init();
        is.ph_scores = jkl;
        is.max_pset_size = i;
        is.max_exec_time = max_time;
        is.scoreNm = score;
        is.alpha = alpha;
        is.thread_pool_size = thread_pool_size;
    }

    protected void prp(ScoreSolver gs, String res, ParentSet[][] sc, int max_time, Integer thread_pool_size) {
        gs.init();
        gs.res_path = res;
        gs.max_exec_time = Math.max(max_time, 1);
        gs.init(sc);
        gs.thread_pool_size = thread_pool_size;
    }

    public void notifyOfThreadComplete() {
        synchronized (lock) {
            lock.notify();
        }
    }

    protected void prepExpect() {
        for (int n = 0; n < n_var; n++) {
            for (int v = 0; v < bn.arity(n); v++) {
                completion[n][v].ensureCapacity(
                        dat.row_values[n][v].length);
                ArrayUtils.cloneArray(dat.row_values[n][v],
                        completion[n][v]._d);
            }
        }
        n_compl = 0;

        todo = 0;
        done = 0;
    }

    protected class HardEmSearcher
            implements Runnable {
        private int t;

        public HardEmSearcher(int t) {
            this.t = t;
        }

        public void run() {
            VariableElimination vEl = new VariableElimination(
                   bn, false);

            while (true) {
                int r = getNextMissingRow();

                if (r == -1) {
                    return;
                }
                short[] sample =samples[r];

                TIntArrayList q = new TIntArrayList();
                TIntIntHashMap e = new TIntIntHashMap();

                for (int n = 0; n <n_var; n++) {
                    if ((n >= sample.length) || (sample[n] == -1)) {
                        q.add(n);
                    } else {
                        e.put(n, sample[n]);
                    }
                }

                long st = System.currentTimeMillis();

                expect(vEl, r, q, e);

                // pf("%d - %d - %d \n",r, q.size(), System.currentTimeMillis() - st);
            }
        }

        protected void expect(VariableElimination vEl, int r, TIntArrayList q, TIntIntHashMap e) {
            TIntIntHashMap res = new TIntIntHashMap();
            for (int i = 0; i < q.size(); i++) {
                int var = q.get(i);
                int val = vEl.query(var, e).mostProbable();
                res.put(var, val);
            }
            addResult(r, res);
        }
    }

    protected void addResult(int row, TIntIntHashMap res) {
        synchronized (lock) {
            for (int var : res.keys()) {
                completion[var][res.get(var)].add(row);
            }
            done += 1;
        }
    }

    protected int getNextMissingRow() {

        int r;

        synchronized (lock) {
            if (todo >= allMissing.length) {
                return -1;
            }
            r = todo;
            todo += 1;
        }

        return allMissing[r];
    }

    protected void prp(ScoreSolver gs, ParentSet[][] scoreReader, int max_time, Integer thread_pool_size) {
        prp(gs, null, scoreReader, max_time, thread_pool_size);
    }

    public void writeCompletedDataSet(String missing, BayesianNetwork bn, String out)
            throws IOException {
        BaseFileLineReader rd = RandomStuff.getDataSetReader(missing);

        rd.readMetaData();

        int n_var = rd.n_var;

        BaseFileLineWriter wr = null;

        if (out.endsWith("dat")) {
            wr = new DatFileLineWriter(bn, RandomStuff.getWriter(out));
        } else if (out.endsWith("arff")) {
            ArffFileLineReader d = new ArffFileLineReader(missing);

            d.readMetaData();
            bn.l_values_var = d.l_v_names;
            wr = new ArffFileLineWriter(bn, RandomStuff.getWriter(out), "test");
        }
        if (wr == null) {
            return;
        }
        wr.writeMetaData();

        VariableElimination vEl = new VariableElimination(bn, false);

        while (!rd.concluded) {
            short[] sm = rd.next();

            TIntArrayList q = new TIntArrayList();
            TIntIntHashMap e = new TIntIntHashMap();

            for (int n = 0; n < n_var; n++) {
                if ((n >= sm.length) || (sm[n] == -1)) {
                    q.add(n);
                } else {
                    e.put(n, sm[n]);
                }
            }

            if (q.size() > 0)
                update(vEl, sm, q, e);

            wr.next(sm);
        }
        wr.close();
    }

    protected void update(VariableElimination vEl, short[] sm, TIntArrayList q, TIntIntHashMap e) {
        for (int v : q.toArray()) {
            BayesianFactor f = vEl.query(v, e);

            sm[v] = (short) f.mostProbable();
        }
    }
}
