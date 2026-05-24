package ch.idsia.blip.core.learn.missing;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.analyze.Analyzer;
import ch.idsia.blip.core.io.bn.BnResReader;
import ch.idsia.blip.core.io.bn.BnUaiWriter;
import ch.idsia.blip.core.io.dat.DatFileLineReader;
import ch.idsia.blip.core.utils.score.MissingBIC;
import ch.idsia.blip.core.inference.ve.BayesianFactor;
import ch.idsia.blip.core.inference.ve.VariableElimination;
import ch.idsia.blip.core.learn.param.ParLe;
import ch.idsia.blip.core.learn.param.ParLeMissing;
import ch.idsia.blip.core.learn.scorer.concurrency.NotifyingThread;
import ch.idsia.blip.core.learn.solver.WinAsobsSolver;
import ch.idsia.blip.core.learn.solver.brtl.BrutalMaxSolver;
import ch.idsia.blip.core.utils.data.FastList;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import ch.idsia.blip.core.utils.data.hash.TIntIntHashMap;
import ch.idsia.blip.core.utils.other.DFGenerator;
import ch.idsia.blip.core.utils.ParentSet;
import ch.idsia.blip.core.utils.RandomStuff;

import java.io.File;
import java.io.IOException;
import java.util.*;


public class HiddenSEM extends SoftMissingSEM {
    private int n_hidden;
    private ArrayList<Map<int[], Double>> scores;
    private String base;
    private String jkl;
    private short[][] completed;

    protected void prepare() {
        super.prepare();

        this.start = System.currentTimeMillis();

        this.bests = new TreeSet();

        this.dat = RandomStuff.getDataSet(this.dat_path);

        this.scores = new ArrayList();
        for (int j = 0; j < this.n_var; j++) {
            this.scores.add(new TreeMap());
        }
        this.allMissing = new int[this.dat.n_datapoints];
        for (int k = 0; k < this.dat.n_datapoints; k++) {
            this.allMissing[k] = k;
        }
        File dir = new File(this.path);

        if (!dir.exists()) {
            return;
        }
        for (File file : dir.listFiles()) {
            if (!file.isDirectory()) {
                if ((file.getName().startsWith("em"))
                        || (file.getName().startsWith("sem"))) {
                    file.delete();
                }
            }
        }
        try {
            DatFileLineReader dr = new DatFileLineReader(this.dat_path);

            dr.readMetaData();

            this.samples = new short[this.dat.n_datapoints][];
            this.completed = new short[this.dat.n_datapoints][];
            int i = 0;

            while (!dr.concluded) {
                this.samples[i] = dr.next();
                this.completed[i] = new short[this.n_hidden];
                i++;
            }
        } catch (IOException ex) {
            RandomStuff.logExp(ex);
        }
    }

    public BayesianNetwork go(String dat_path, int n_hidden)
        throws Exception {
        this.n_hidden = n_hidden;
        this.dat_path = dat_path;

        prepare();

        this.start = System.currentTimeMillis();

        logf("First learning (%.2f)\n", new Object[] { Double.valueOf(elaps())});

        addHiddenVariables();

        addHiddenValues(this.dat, n_hidden);

        performEM("em");

        this.bn = performSEM();

        this.bn.writeGraph(this.path + "new-g");
        BnUaiWriter.ex(this.bn, this.path + "final.uai");

        this.bn = winasobs();
        this.bn.writeGraph(this.path + "new-w");
        BnUaiWriter.ex(this.bn, this.path + "final.win.uai");

        return this.bn;
    }

    protected Runnable getEmSearcher(int t) {
        return new HEmSearcher();
    }

    private BayesianNetwork winasobs()
        throws Exception {
        String res = this.base + ".win.res";
        WinAsobsSolver gs = new WinAsobsSolver();

        prp(gs, RandomStuff.getScoreReader(this.jkl, 0), this.time_for_variable,
                this.thread_pool_size);
        gs.go(res);
        BayesianNetwork newBn = BnResReader.ex(res);

        return ParLe.ex(newBn, this.dat);
    }

    private BayesianNetwork maximizSEM(int i)
        throws Exception {
        BayesianNetwork newBn = learn("sem" + i);

        return ParLeMissing.ex(newBn, this.dat, this.completion, this.weights);
    }

    private BayesianNetwork learn(String g)
        throws Exception {
        this.base = RandomStuff.f("%s/%s", this.path, g);

        this.jkl = (this.base + ".jkl");
        SemIndependenceScorer is = new SemIndependenceScorer();

        is.setScores(this.scores, this.n_var);
        prp(is, this.jkl, this.tw, this.time_for_variable, "bdeu", 1.0D,
                this.thread_pool_size);
        is.verbose = 0;
        is.score = new MissingBIC(this.dat, this.completion, this.weights);
        is.go(this.dat);

        String res = this.base + ".res";
        BrutalMaxSolver gs = new BrutalMaxSolver();

        prp(gs, res, RandomStuff.getScoreReader(this.jkl, 0), this.time_for_variable,
                this.thread_pool_size);
        gs.tw = this.tw;
        gs.verbose = 0;
        gs.out_solutions = 10;
        gs.tryFirst = this.bests;
        gs.go();

        this.bests.addAll(gs.open);

        BayesianNetwork newbn = BnResReader.ex(res);

        newbn.writeGraph(this.base);

        return newbn;
    }

    protected void expect()
        throws IOException, InterruptedException {
        this.todo = 0;
        for (int t = 0; t < Math.min(this.samples.length, this.thread_pool_size); t++) {
            Runnable r = getEmSearcher(t);
            NotifyingThread tr = new NotifyingThread(r);

            tr.addListener(this);
            tr.start();
        }
        while (this.todo != this.samples.length) {
            synchronized (this.lock) {
                this.lock.wait();
            }
        }
    }

    private BayesianNetwork performSEM()
        throws Exception {
        this.bn = maximizSEM(0);

        int i = 1;

        for (;;) {
            logf("SEM iteration %d, expectation (%.2f) \n",
                    new Object[] { Integer.valueOf(i), Double.valueOf(elaps())});

            expect();

            BayesianNetwork newBn = maximizSEM(i);

            if ((stationaryStr(this.bn, newBn)) || (i > 15)) {
                return newBn;
            }
            this.bn = newBn;
            i++;
        }
    }

    private void performEM(String s)
        throws Exception {
        this.bn = ParLe.ex(this.bn, this.dat);

        int i = 1;

        for (;;) {
            logf("%s expectation %d, expectation (%.2f) \n",
                    new Object[] {
                s, Integer.valueOf(i),
                Double.valueOf(elaps())});

            expect();

            BayesianNetwork newBn = ParLe.ex(this.bn, this.dat);

            if ((stationary(this.bn, newBn)) || (i > 15)) {
                break;
            }
            this.bn = newBn;
            i++;
        }
    }

    public void addHiddenValues(DataSet dat, int k)
        throws IOException {
        int card = 3;

        String[] aux = new String[dat.n_var + k];
        int[] aux2 = new int[dat.n_var + k];
        int[][][] aux3 = new int[dat.n_var + k][][];

        DFGenerator DFg = new DFGenerator();

        for (int i = 0; i < dat.n_var; i++) {
            aux[i] = dat.l_nm_var[i];
            aux2[i] = dat.l_n_arity[i];
            aux3[i] = dat.row_values[i];
        }
        Analyzer an = new Analyzer(dat);

        for (int i = 0; i < k; i++) {
            int v = dat.n_var + i;

            aux[v] = RandomStuff.f("MS%d", new Object[] { Integer.valueOf(i)});
            aux2[v] = card;

            TIntArrayList[] r = new TIntArrayList[card];

            for (int vv = 0; vv < card; vv++) {
                r[vv] = new TIntArrayList();
            }
            int[][] p_v = an.computeParentSetValues(this.bn.parents(v));

            for (int j = 0; j < p_v.length; j++) {
                double[] distr = DFg.generateUniformDistribution(card);

                for (int row : p_v[j]) {
                    r[DFg.sampleV(distr)].add(row);
                }
            }
            int[][] row = new int[card][];

            for (int vv = 0; vv < card; vv++) {
                row[vv] = r[vv].toArray();
                Arrays.sort(row[vv]);
            }
            aux3[v] = row;
        }
        dat.l_nm_var = aux;
        dat.l_n_arity = aux2;
        dat.row_values = aux3;

        dat.n_var += k;
    }

    protected void addHiddenVariables() {
        int r = 3;

        List<TIntArrayList> new_best = new ArrayList();

        for (int i = 0; i < this.dat.n_var; i++) {
            new_best.add(new TIntArrayList());
        }
        FastList<Integer> fs = new FastList(this.rand);

        for (int i = 0; i < this.dat.n_var; i++) {
            fs.add(Integer.valueOf(i));
        }
        addVars(r, new_best, fs);

        ParentSet[] str = new ParentSet[new_best.size()];

        for (int n = 0; n < new_best.size(); n++) {
            str[n] = new ParentSet(0.0D,
                    ((TIntArrayList) new_best.get(n)).toArray());
        }
        this.bn = new BayesianNetwork(str);
        this.bn.writeGraph(this.path + "/em0");
    }

    protected void addVars(int r, List<TIntArrayList> new_best, FastList<Integer> fs) {
        int cnt = 0;
        int k = 3;

        while ((cnt < this.n_hidden) && (fs.size() > 1)) {
            int[] f = new int[k];

            for (int i = 0; i < k; i++) {
                f[i] = ((Integer) fs.rand()).intValue();
                while (notOk(f, i)) {
                    f[i] = ((Integer) fs.rand()).intValue();
                }
            }
            TIntArrayList t = new TIntArrayList();

            for (int i = 0; i < k; i++) {
                t.add(f[i]);
            }
            new_best.add(t);
            cnt++;
        }
    }

    protected boolean notOk(int[] f, int i) {
        for (int j = 0; j < i; j++) {
            if (f[j] == f[i]) {
                return true;
            }
        }
        return false;
    }

    protected int getNextMissingRow() {
        int r;

        synchronized (this.lock) {
            if (this.todo >= this.samples.length) {
                return -1;
            }
            r = this.todo;
            this.todo += 1;
        }
        return r;
    }

    private class HEmSearcher
            implements Runnable {
        private HEmSearcher() {}

        public void run() {
            VariableElimination vEl = new VariableElimination(HiddenSEM.this.bn,
                    false);

            for (;;) {
                int r = HiddenSEM.this.getNextMissingRow();

                if (r == -1) {
                    return;
                }
                short[] sample = HiddenSEM.this.samples[r];

                TIntIntHashMap e = new TIntIntHashMap();

                for (int n = 0; n < HiddenSEM.this.n_var; n++) {
                    e.put(n, sample[n]);
                }
                for (int h = 0; h < HiddenSEM.this.n_hidden; h++) {
                    TIntArrayList q = new TIntArrayList();

                    q.add(HiddenSEM.this.n_var + h);
                    BayesianFactor res = vEl.query(q.toArray(), e);

                    HiddenSEM.this.addResult(r, h, (short) res.mostProbable());
                }
            }
        }
    }

    private void addResult(int r, int h, short i) {
        synchronized (this.lock) {
            this.completed[r][h] = i;
        }
    }
}
