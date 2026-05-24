package ch.idsia.blip.core.io;


import ch.idsia.TheTest;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.common.SamGe;
import ch.idsia.blip.core.io.bn.*;
import ch.idsia.blip.core.learn.param.ParLeBayes;
import ch.idsia.blip.core.learn.scorer.IndependenceScorer;
import ch.idsia.blip.core.learn.solver.AsobsSolver;
import ch.idsia.blip.core.learn.solver.brtl.BrutalSolver;
import ch.idsia.blip.core.learn.solver.ktree.S2PlusSolver;
import org.junit.Test;

import java.io.File;
import java.io.FileNotFoundException;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class BnErg extends TheTest {

    @Test
    public void testOpen() throws FileNotFoundException {

        String net = f("%s.net", basePath, "simple");
        String net2 = f("%s2.net", basePath, "simple");
        String uai = f("%s.uai", basePath, "simple");

        BayesianNetwork bn = BnNetReader.ex(net);

        BnErgWriter.ex(uai, bn);

        BayesianNetwork bn2 = BnErgReader.ex(uai);

        BnNetWriter.ex(bn2, net2);

    }

    @Test
    public void testRethink() throws Exception {

        basePath += "uai/";
        String r = "4";
        int n = 100000;

        String uai = f("%s.uai", r);

        BayesianNetwork b = BnErgReader.ex(uai);

        String g = f("%s.png", r);

        if (!(new File(g).exists())) {
            b.writeGraph(r);
        }

        String d = f("%s-%d.dat", r, n);

        if (!(new File(d).exists())) {
            SamGe.ex(b, r, n);
        }

        DataSet dat = getDataSet(d);

        String s = f("%s.jkl", r);

        if (!(new File(s).exists())) {
            IndependenceScorer is = new IndependenceScorer();

            is.ph_scores = s;
            is.max_exec_time = 5;
            is.scoreNm = "bdeu";
            is.alpha = 1;
            is.max_pset_size = 3;
            is.go(dat);
        }

        String n2 = f("%s.res", r);
        BrutalSolver sv = new BrutalSolver();

        sv.init(getScoreReader(s), 30, 5);
        sv.go(n2);

        sv.writeGraph(r + "-new");
        BayesianNetwork bn_new = new BayesianNetwork(sv.best_str);

        ParLeBayes p = new ParLeBayes(1);

        p.go(bn_new, dat);

        String uai2 = f("%s-2.uai", r);

        BnErgWriter.ex(uai2, p.bn);

    }

    @Test
    public void sfdfd() throws Exception {

        int n = 100000;
        int tw = 4;
        int learning = 600;

        String net = "100-BN_1";
        String path = "/home/loskana/Desktop/inference";
        String r = f("%s/%s/%s", path, net, net);

        String erg = f("%s.erg", r);
        BayesianNetwork b = BnErgReader.ex(erg);

        String g = f("%s.png", r);

        if (!(new File(g).exists())) {
            b.writeGraph(r);
        }

        String d = f("%s-%d.dat", r, n);

        if (!(new File(d).exists())) {
            SamGe.ex(b, r, n);
        }

        DataSet dat = getDataSet(d);

        String s = f("%s.jkl", r);

        if (!(new File(s).exists())) {
            IndependenceScorer is = new IndependenceScorer();

            is.ph_scores = s;
            is.max_exec_time = 5;
            is.scoreNm = "bdeu";
            is.alpha = 1;
            is.max_pset_size = 5;
            is.go(dat);
        }

        // S2+
        String s2 = f("%s.s2", r);

        if (!(new File(s2).exists())) {
            S2PlusSolver sv = new S2PlusSolver();

            sv.init(getScoreReader(s), learning);
            sv.tw = tw;
            sv.ph_dat = d;
            sv.ph_astar = path + "/astar/";
            sv.ph_work = f("%s/%s/gob", path, net);
            sv.ph_gobnilp = path + "/gobnilp";
            sv.go(s2);
            sv.writeGraph(r + "-s2");
        }

        // Brutal
        String bru = f("%s.brutal", r);

        if (!(new File(bru).exists())) {
            BrutalSolver sv = new BrutalSolver();

            sv.init(getScoreReader(s), learning);
            sv.tw = tw;
            sv.go(bru);
            sv.writeGraph(r + "-bru");
        }
        BayesianNetwork BnBru = BnResReader.ex(bru);

        learn(dat, bru, BnBru);

        String gob = f("%s.gob", r);

        if (!(new File(gob).exists())) {
            String cmd = f("%s/gobnilp %s.jkl > %s.gob 2>&1 ", path, r, r);

            p(cmd);
            Process p = Runtime.getRuntime().exec(
                    new String[] { "bash", "-c", cmd});
            int exitVal = waitForProc(p, 1000);
        }
        BayesianNetwork BnGob = GobnilpReader.ex(gob);

        learn(dat, gob, BnGob);

        String asobs = f("%s.asobs", r);

        if (!(new File(asobs).exists())) {
            AsobsSolver sv = new AsobsSolver();

            sv.init(getScoreReader(s), b.n_var * 5);
            sv.go(asobs);
            sv.writeGraph(r + "-aso");
        }
        BayesianNetwork BnAsobs = BnResReader.ex(asobs);

        learn(dat, asobs, BnAsobs);

        // True
        BnUaiWriter.ex(f("%s.uai", r), b);

        /*

         // True
         String real = f("%s.erg",  r);
         BayesianNetwork BnReal = BnErgReader.ex(real);
         learn(dat, f("%s.real",  r), BnReal);


         // Asobs
         String asobs = f("%s.asobs",  r);
         BayesianNetwork BnAsobs = BnResReader.ex(asobs);
         learn(dat, asobs, BnAsobs);


         String res = f("%s.asobs",  r);
         if (!(new File(res).exists())) {
         AsobsSolver sv = new AsobsSolver();
         sv.init(120, getScoreReader(s));
         sv.ex(res);
         sv.writeGraph( r + "-aso");
         }


         */






        /*
         String d = f("%s_10000.dat",  r);

         DataSet dat = getDataSet(d);

         String s = f("%s-3.jkl",  r);
         IndependenceScorer is = new IndependenceScorer();
         is.ph_scores = s;
         is.max_exec_time = 1;
         is.scoreNm = "bdeu";
         is.alpha = 1;
         is.max_pset_size = 3;
         is.ex(dat);

         */

    }

    @Test
    public void jahsdsdd() throws Exception {
        File folder = new File("/home/loskana/Desktop/inference/nets");

        for (File f : folder.listFiles()) {
            if (!f.isFile()) {
                continue;
            }

            if (!f.getName().endsWith(".erg")) {
                continue;
            }

            String n = f.getAbsolutePath();

            BayesianNetwork b = BnErgReader.ex(n);

            BnUaiWriter.ex(f("%s.uai", n.replace(".erg", "")), b);
        }
    }

    private void learn(DataSet dat, String s, BayesianNetwork bn) {
        ParLeBayes p = new ParLeBayes(1);

        bn = p.go(bn, dat);
        BnErgWriter.ex(s, bn);
        BnUaiWriter.ex(s, bn);
    }
}
