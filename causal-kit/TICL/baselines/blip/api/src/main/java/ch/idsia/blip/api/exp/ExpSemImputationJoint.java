package ch.idsia.blip.api.exp;


import ch.idsia.blip.api.learn.scorer.IndependenceScorerApi;
import ch.idsia.blip.api.learn.solver.win.WinAsobsSolverApi;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.common.SamGe;
import ch.idsia.blip.core.io.dat.BaseFileLineReader;
import ch.idsia.blip.core.io.dat.DatFileLineWriter;
import ch.idsia.blip.core.inference.ve.BayesianFactor;
import ch.idsia.blip.core.inference.ve.VariableElimination;
import ch.idsia.blip.core.learn.missing.HardJointMissingSEM;
import ch.idsia.blip.core.learn.missing.HardMissingSEM;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import ch.idsia.blip.core.utils.data.hash.TIntIntHashMap;
import ch.idsia.blip.core.utils.RandomStuff;

import java.io.File;
import java.io.IOException;

import static ch.idsia.blip.core.utils.RandomStuff.*;
import static ch.idsia.blip.core.utils.RandomStuff.logExp;


public class ExpSemImputationJoint extends ExpSemImputation {

    public static void main(String[] args)
            throws IOException {
        try {
            if (args.length > 1) {
                if (args[0].equals("prepare")) {
                    new ExpSemImputationJoint().prepareNow(args);
                } else if (args[0].equals("go")) {
                    new ExpSemImputationJoint().goNow(args);
                } else if (args[0].equals("measure")) {
                    new ExpSemImputationJoint().measureNow(args);
                }
            } else {
                new ExpSemImputationJoint().test2();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    protected  void test() throws Exception {
        thread = 0;


        for (String nm: new String[]{"cancer", "earthquake"}) {

            locPath = System.getProperty("user.home")
                    + "/Desktop/test/";

            BayesianNetwork orig = getBayesianNetwork(locPath + nm + ".net");

            String nmPath = locPath + "/" + nm + "/";

            File fi = new File(nmPath);
            if (fi.exists()) {
                RandomStuff.deleteFolder(fi);
            }

            fi.mkdirs();

            String complete = nmPath + "complete.dat";

            SamGe.ex(orig, complete, 5000);

            orig(nmPath, orig.l_nm_var);

            for (int per : new int[]{0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70,}) {

                String perPath = nmPath + "/" + per + "/";
                if (!new File(perPath).exists())
                    new File(perPath).mkdirs();

                String missing = perPath + "missing.dat";
                File fa = new File(missing);
                // if (!fi.exists()) {

                BaseFileLineReader dr = getDataSetReader(complete);
                DatFileLineWriter dflw = new DatFileLineWriter(orig,
                        RandomStuff.getWriter(missing));

                try {
                    dr.readMetaData();
                    dflw.writeMetaData();

                    short[] sample;

                    while (!dr.concluded) {
                        sample = dr.next();
                        dflw.next(makeMissing(sample, per));
                    }

                    dflw.close();

                } catch (IOException e) {
                    RandomStuff.logExp(e);
                }


                //   }

                // ### MARGINALS


                goSem(perPath, "sem-mar", missing, new HardMissingSEM());

                // ### JOINT

                goSem(perPath, "sem-map", missing, new HardJointMissingSEM());
            }
        }

    }

    private void orig(String locPath, String[] keys) {

        String train = locPath + "complete.dat";
        String jkl = locPath + "jkl";

        IndependenceScorerApi.main(
                new String[] { "", "-d", train, "-j", jkl, "-b", "0", "-t", "5", "-n", "5"});

        String res = locPath + "res";

        WinAsobsSolverApi.main(
                new String[] { "", "-j", jkl, "-r", res, "-b", "0", "-t", "5"});

        BayesianNetwork bn = getBayesianNetwork(res);
        bn.l_nm_var = keys;
        bn.writeGraph(locPath + "bn");
    }

    private void goSem(String locPath, String name, String missing, HardMissingSEM mSem) throws Exception {

        String compl = locPath + name + "-compl.dat";
        sem = locPath + name + "/";

        mSem.init(sem, this.max_time, this.tw);
        mSem.thread_pool_size = this.thread;
        BayesianNetwork bn = mSem.go(missing);

        RandomStuff.writeBayesianNetwork(bn, sem + "new.uai");
        mSem.writeCompletedDataSet(missing, bn, compl);
        bn.writeGraph(locPath + name + "-graph.l");
    }

    private short[] makeMissing(short[] s, int per) {


            for (int n = 0; n < 2; n++) {
                int ra = randInt(0, 100);

                if (ra < per) {
                    s[n] = -1;
                }
            }


        return s;
    }


    protected  void test2() throws Exception {
        thread =1;

        locPath = System.getProperty("user.home")
                + "/Desktop/SEM/imputation2/";

        goNow(new String[]{"", locPath, "accidents.test", "1","15"});

        prepareNow(new String[]{"", locPath, "accidents.test"});
        for (int f = 1; f <= this.max_fold; f++) {
            for (int p = 0; p < this.percs.length; p++) {
                int per = this.percs[p];
                goNow(new String[]{"", locPath, "accidents.test", String.valueOf(f), String.valueOf(per)});
            }
        }
    }

    protected void update(VariableElimination vEl, short[] sm, TIntArrayList qu, TIntIntHashMap e) {

        int nd = 0;

        while (nd < qu.size()) {

            int l = Math.min(15, qu.size() - nd);
            int[] q = new int[l];
            for (int j = 0; j < l; j++)
                q[j] = qu.get(nd + j);

            BayesianFactor f = vEl.query(q, e);
            int r = f.mostProbable();
            for (int i = f.dom.length - 1; i >= 0; i--) {
                int var = f.dom[i];
                int val = r / f.stride[i];
                r -= val * f.stride[i];
                sm[var] = (short) val;
            }

            nd += l;
        }
    }

    /*
    protected void update(VariableElimination vEl, short[] sm, TIntArrayList qu, TIntIntHashMap ev) {

        String cmd = f("./daoopt -f %s -e %s", bnPath, ev);

        Process proc = null;
        try {

            String e = sem + "ev";

            //  writeQuery(qu, q);
            writeEvidence(e, ev);

            proc = Runtime.getRuntime().exec(cmd, new String[0],
                    new File(System.getProperty("user.home") + "/Tools"));

            ArrayList<String> out = RandomStuff.exec(proc);

            proc.getInputStream().close();
            proc.getOutputStream().close();
            proc.getErrorStream().close();

            String output = out.get(out.size() - 1);
            String[] aux = output.split(" ");

            TIntIntHashMap res = new TIntIntHashMap();

            for (int i = 0; i < qu.size(); i++) {
                int n = qu.get(i);

                sm[n] = Short.valueOf(aux[(n + 3)]);
            }

        } catch (IOException exp) {
            logExp(exp);
        } catch (InterruptedException exp) {
            logExp(exp);
        }

    }*/


    protected HardMissingSEM getHardMissingSEM() {
        return new HardJointMissingSEM();
        // return new HardMPEMissingSEM();
    }
}

