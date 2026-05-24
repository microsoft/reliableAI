package ch.idsia.blip.core.learn.missing;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.analyze.Analyzer;
import ch.idsia.blip.core.io.bn.BnResReader;
import ch.idsia.blip.core.io.bn.BnUaiWriter;
import ch.idsia.blip.core.io.dat.DatFileLineReader;
import ch.idsia.blip.core.learn.param.ParLe;
import ch.idsia.blip.core.learn.solver.BaseSolver;
import ch.idsia.blip.core.learn.solver.WinAsobsSolver;
import ch.idsia.blip.core.learn.solver.brtl.BrutalMaxSolver;
import ch.idsia.blip.core.utils.data.FastList;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import ch.idsia.blip.core.utils.other.DFGenerator;
import ch.idsia.blip.core.utils.ParentSet;
import ch.idsia.blip.core.utils.RandomStuff;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;


public class HiddenSEM2 extends HardMissingSEM {
    protected int n_hidden;
    private short[][] dataLines;
    private TreeSet<BaseSolver.Solution> bests;
    private List<Map<int[], Double>> scores;
    private String jkl;
    private String base;
    private String nm;

    public HiddenSEM2(int verbose) {
        this.verbose = verbose;
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

    public void init(String p, int time, int w, int n_h) {
        super.init(p, time, w);

        this.bests = new TreeSet();

        this.n_hidden = n_h;
    }

    public void go(String dat_path, String nm)
        throws Exception {
        this.dat = RandomStuff.getDataSet(dat_path);
        getOrigData(dat_path);
        this.dat_path = dat_path;

        this.nm = nm;

        prepare();

        this.start = System.currentTimeMillis();

        logf("First learning (%.2f)\n", new Object[] { Double.valueOf(elaps())});

        addHiddenVariables();

        addHiddenValues(this.dat, this.n_hidden);

        performEM("em");

        this.bn = performSEM();

        performEM("ssem");

        this.bn.writeGraph(this.path + "-g");
        BnUaiWriter.ex(this.bn, this.path + ".uai");

        this.bn = winasobs();
        this.bn.writeGraph(this.path + "-w");
        BnUaiWriter.ex(this.bn, this.path + ".win.uai");
    }

    private BayesianNetwork winasobs()
        throws Exception {
        String res = this.base + ".win.res";
        WinAsobsSolver gs = new WinAsobsSolver();

        prp(gs, RandomStuff.getScoreReader(this.jkl, 0), this.time_for_variable,
                Integer.valueOf(this.thread_pool_size));
        gs.go(res);
        BayesianNetwork newBn = BnResReader.ex(res);

        return ParLe.ex(newBn, this.dat);
    }

    public void prepare() {
        super.prepare();

        this.n_var = this.dat.n_var;

        this.scores = new ArrayList();
        for (int i = 0; i < this.n_var; i++) {
            this.scores.add(new TreeMap());
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
    }

    private void getOrigData(String dat_path)
        throws IOException {
        DatFileLineReader dr = new DatFileLineReader(dat_path);

        dr.readMetaData();

        this.dataLines = new short[this.dat.n_datapoints][];
        int i = 0;

        while (!dr.concluded) {
            this.dataLines[(i++)] = dr.next();
        }
    }

    private BayesianNetwork performSEM()
        throws Exception {
        this.bn = maximizSEM(0);

        int i = 1;

        for (;;) {
            logf("SEM iteration %d, expectation (%.2f) \n",
                    new Object[] { Integer.valueOf(i), Double.valueOf(elaps())});

            expect(this.bn, this.dat, i, "sem");

            BayesianNetwork newBn = maximizSEM(i);

            if ((stationaryStr(this.bn, newBn)) || (i > 15)) {
                return newBn;
            }
            this.bn = newBn;
            i++;
        }
    }

    private BayesianNetwork maximizSEM(int i)
        throws Exception {
        BayesianNetwork newBn = learn("sem" + i);

        return ParLe.ex(newBn, this.dat);
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

            expect(this.bn, this.dat, i, s);

            BayesianNetwork newBn = ParLe.ex(this.bn, this.dat);

            if ((stationary(this.bn, newBn)) || (i > 15)) {
                break;
            }
            this.bn = newBn;
            i++;
        }
    }

    private void expect(BayesianNetwork bn, DataSet dat, int i, String s)
        throws IOException, InterruptedException {
        String bn_path = this.path + "/" + this.nm + "-" + s + i + ".uai";

        BnUaiWriter.ex(bn, bn_path);

        daooptOld(bn_path);
    }

    private void daoopt(String bn_path)
        throws IOException, InterruptedException {
        String r = RandomStuff.f("./daoopt-new -f %s -e %s",
                new Object[] { bn_path, this.dat_path});
        Process proc = Runtime.getRuntime().exec(r, new String[0],
                new File(System.getProperty("user.home") + "/Tools"));

        RandomStuff.exec(proc);

        proc.getInputStream().close();
        proc.getOutputStream().close();
        proc.getErrorStream().close();

        DataSet d = RandomStuff.getDataSet(bn_path + ".out");

        for (int j = 0; j < this.n_hidden; j++) {
            for (int v = 0; v < this.dat.l_n_arity[(this.n_var + j)]; v++) {
                if (d.row_values[j].length > v) {
                    this.dat.row_values[(this.n_var + j)][v] = d.row_values[j][v];
                } else {
                    this.dat.row_values[(this.n_var + j)][v] = new int[0];
                }
            }
        }
    }

    private void daooptOld(String bn_path)
        throws IOException, InterruptedException {
        TIntArrayList[][] newValues = new TIntArrayList[this.n_hidden][];

        for (int n = 0; n < this.n_hidden; n++) {
            int card = this.dat.l_n_arity[(this.n_var + n)];

            newValues[n] = new TIntArrayList[card];
            for (int v = 0; v < card; v++) {
                newValues[n][v] = new TIntArrayList();
            }
        }
        for (int l = 0; l < this.dat.n_datapoints; l++) {
            daooptExOld(bn_path, l, newValues);
        }
        for (int n = 0; n < this.n_hidden; n++) {
            int card = this.dat.l_n_arity[(this.n_var + n)];

            this.dat.row_values[(this.n_var + n)] = new int[card][];
            for (int v = 0; v < card; v++) {
                this.dat.row_values[(this.n_var + n)][v] = newValues[n][v].toArray();
            }
        }
    }

    private void daooptExOld(String bn_path, int l, TIntArrayList[][] newValues)
        throws IOException, InterruptedException {
        String evid_path = bn_path + ".evid";

        PrintWriter w = new PrintWriter(evid_path, "UTF-8");

        RandomStuff.wf(w, "%d ", new Object[] { Integer.valueOf(this.n_var)});
        for (int j = 0; j < this.n_var; j++) {
            RandomStuff.wf(w, "%d %d ",
                    new Object[] {
                Integer.valueOf(j),
                Short.valueOf(this.dataLines[l][j])});
        }
        w.close();

        String r = RandomStuff.f("./daoopt -f %s -e %s",
                new Object[] { bn_path, evid_path});
        Process proc = Runtime.getRuntime().exec(r, new String[0],
                new File(System.getProperty("user.home") + "/Tools"));
        ArrayList<String> out = RandomStuff.exec(proc);
        String res = (String) out.get(out.size() - 1);

        String[] aux = res.split(" ");

        for (int h = 0; h < this.n_hidden; h++) {
            int v = Integer.valueOf(aux[(this.n_var + h + 3)]).intValue();

            newValues[h][v].add(l);
        }
        proc.getInputStream().close();
        proc.getOutputStream().close();
        proc.getErrorStream().close();
    }

    protected void addHiddenVariables() {
        int r = 3;

        List<TIntArrayList> new_best = new ArrayList();

        for (int i = 0; i < this.n_var; i++) {
            new_best.add(new TIntArrayList());
        }
        FastList<Integer> fs = new FastList(this.rand);

        for (int i = 0; i < this.n_var; i++) {
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

    protected BayesianNetwork addHiddenVariables2(BayesianNetwork base) {
        List<TIntArrayList> new_best = new ArrayList();

        for (int i = 0; i < this.n_var; i++) {
            TIntArrayList t = new TIntArrayList();

            t.addAll(base.parents(i));
            new_best.add(t);
        }
        int nextHidden = this.n_var;

        for (int n = 0; n < this.n_var; n++) {
            for (int p : base.parents(n)) {
                ((TIntArrayList) new_best.get(n)).add(nextHidden);
                ((TIntArrayList) new_best.get(p)).add(nextHidden);

                new_best.add(new TIntArrayList());
                nextHidden++;
            }
        }
        ParentSet[] str = new ParentSet[new_best.size()];

        for (int n = 0; n < new_best.size(); n++) {
            str[n] = new ParentSet(0.0D,
                    ((TIntArrayList) new_best.get(n)).toArray());
        }
        BayesianNetwork bn = new BayesianNetwork(str);

        bn.writeGraph(this.path + "/em0");
        return bn;
    }

    private BayesianNetwork learn(String g)
        throws Exception {
        this.base = RandomStuff.f("%s/%s", new Object[] { this.path, g});

        this.jkl = (this.base + ".jkl");
        SemIndependenceScorer is = new SemIndependenceScorer();

        is.setScores(this.scores, this.n_var);
        HashMap<String, String> o = new HashMap();

        o.put("ph_scores", this.jkl);
        o.put("max_pset_size", String.valueOf(this.tw - 1));
        o.put("time_for_variable", String.valueOf(this.time_for_variable));
        o.put("scoreNm", "bdeu");
        o.put("alpha", "10.0");
        o.put("thread_pool_size", String.valueOf(this.thread_pool_size));
        is.init(o);
        is.verbose = 0;
        is.go(this.dat);

        String res = this.base + ".res";
        BrutalMaxSolver gs = new BrutalMaxSolver();

        o = new HashMap();
        o.put("time_for_variable", String.valueOf(this.time_for_variable));
        o.put("thread_pool_size", String.valueOf(this.thread_pool_size));
        o.put("out_solutions", "10");
        o.put("tw", String.valueOf(this.tw));
        o.put("res_path", res);
        gs.init(o);

        gs.init(RandomStuff.getScoreReader(this.jkl, 0));
        gs.verbose = 0;
        gs.out_solutions = 10;
        gs.tryFirst = this.bests;
        gs.go();

        this.bests.addAll(gs.open);

        BayesianNetwork newbn = BnResReader.ex(res);

        newbn.writeGraph(this.base);

        return newbn;
    }
}
