package ch.idsia.blip.api.exp;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.analyze.Entropy;
import ch.idsia.blip.core.utils.analyze.LogLikelihood;
import ch.idsia.blip.core.io.bn.BnResReader;
import ch.idsia.blip.core.io.ScoreReader;
import ch.idsia.blip.core.utils.ParentSet;

import java.io.IOException;
import java.io.Writer;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class ExpEntr {

    public static void main(String[] args) {
        try {
            new ExpEntr().see6(args[0], args[1], args[2]);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void see(String dat, String firstBn, String secondBn) throws IOException {

        DataSet d = getDataSet(dat);

        BnResReader red1 = new BnResReader();
        BayesianNetwork bn1 = red1.go(firstBn);

        BnResReader red2 = new BnResReader();
        BayesianNetwork bn2 = red2.go(secondBn);

        Entropy e = new Entropy(d);
        LogLikelihood l = new LogLikelihood(d);

        pf("var,entropy,log-likelihood,");
        pf("num parents 1, num children 1, bic 1,");
        pf("num parents 2, num children 2, bic 2\n");
        for (int i = 0; i < d.n_var; i++) {
            pf("%d,%.4f,%.2f,", i, e.computeH(i), l.computeLL(i));
            pf("%d, %d, %.2f,", bn1.parents(i).length, bn1.childrens(i).length,
                    red1.scores.get(i));
            pf("%d, %d, %.2f\n", bn2.parents(i).length, bn2.childrens(i).length,
                    red2.scores.get(i));
        }

        pf("\n");
        pf(",normal,ent\n");
        pf("arcs,%d,%d \n", bn1.numEdges(), bn2.numEdges());
        pf("moralized,%d,%d\n", bn1.moralize().numEdges(),
                bn2.moralize().numEdges());

    }

    private void see2(String dat, String firstBn, String secondBn) throws IOException {

        BnResReader red1 = new BnResReader();
        BayesianNetwork bn1 = red1.go(firstBn);

        BnResReader red2 = new BnResReader();
        BayesianNetwork bn2 = red2.go(secondBn);

        pf("%s,", dat);
        pf("arcs,%d,%d,", bn1.numEdges(), bn2.numEdges());
        pf("moralized,%d,%d\n", bn1.moralize().numEdges(),
                bn2.moralize().numEdges());
    }

    private void see3(String dat, String firstBn, String secondBn) throws IOException {
        DataSet d = getDataSet(dat);

        BnResReader red1 = new BnResReader();

        red1.go(firstBn);
        BnResReader red2 = new BnResReader();

        red2.go(secondBn);
        Entropy e = new Entropy(d);

        for (int i = 0; i < d.n_var; i++) {
            pf("%.4f %.4f\n", e.computeH(i),
                    red1.scores.get(i) - red2.scores.get(i));
        }

    }

    private void see4(String dat, String jkl, String out) throws IOException {
        DataSet d = getDataSet(dat);

        ScoreReader sc = new ScoreReader(jkl);

        sc.readScores();

        Entropy e = new Entropy(d);
        HashMap<Integer, Double> ents = new HashMap<Integer, Double>();

        for (int i = 0; i < d.n_var; i++) {
            ents.put(i, e.computeH(i));
        }

        Map<Integer, Double> ents_new = sortByValues(ents);
        Iterator<Integer> it = ents_new.keySet().iterator();
        int i = 0;
        String path = f("%s-low.dat", out);
        Writer w = getWriter(path);

        while (it.hasNext() && i < 20) {
            normPSets(w, sc, it.next());
            i++;
        }
        w.close();

        ents_new = sortInvByValues(ents);
        it = ents_new.keySet().iterator();
        i = 0;
        path = f("%s-high.dat", out);
        w = getWriter(path);
        while (it.hasNext() && i < 20) {
            normPSets(w, sc, it.next());
            i++;
        }
        w.close();
    }

    private void see6(String dat, String jkl, String out) throws IOException {
        DataSet d = getDataSet(dat);

        ScoreReader sc = new ScoreReader(jkl);

        sc.readScores();

        Entropy e = new Entropy(d);
        HashMap<Integer, Double> ents = new HashMap<Integer, Double>();

        for (int i = 0; i < d.n_var; i++) {
            ents.put(i, e.computeH(i));
        }

        Map<Integer, Double> ents_new = sortByValues(ents);
        Iterator<Integer> it = ents_new.keySet().iterator();
        int i = 0;

        while (it.hasNext() && i < 20) {
            String path = f("%s/low-%d.dat", out, i);
            Writer w = getWriter(path);

            diffPSets(w, sc, it.next());
            i++;
            w.close();
        }

        ents_new = sortInvByValues(ents);
        it = ents_new.keySet().iterator();
        i = 0;
        while (it.hasNext() && i < 20) {
            String path = f("%s/high-%d.dat", out, i);
            Writer w = getWriter(path);

            diffPSets(w, sc, it.next());
            i++;
            w.close();
        }

    }

    private void diffPSets(Writer w, ScoreReader sc, Integer var) throws IOException {

        ParentSet[] ps = sc.m_scores[var];

        double min = ps[0].sk;

        int cnt = 0;

        for (ParentSet p : ps) {

            double sk = p.sk;

            // sk = (sk - min) / (max - min);
            sk = sk - min;
            wf(w, "%d %.4f\n", cnt, Math.abs(sk));
            cnt++;
        }
    }

    private void normPSets(Writer w, ScoreReader sc, int var) throws IOException {

        ParentSet[] ps = sc.m_scores[var];

        double min = ps[0].sk;
        double max = ps[ps.length - 1].sk;

        double step = ps.length / 100.0;

        int cnt = 0;

        int wj = 0;

        for (ParentSet p : ps) {

            if (cnt >= wj * step) {
                double sk = p.sk;

                // sk = (sk - min) / (max - min);
                sk = sk - min;
                wf(w, "%d %.4f\n", wj, Math.abs(sk));
                wj++;
            }

            cnt++;

        }
    }

    public static void exec(String h) throws IOException {
        Process proc = Runtime.getRuntime().exec(h, new String[0]);
        int exitVal = waitForProc(proc, 100000);
    }

    private void see5(String dat, String firstBn, String secondBn) throws IOException {
        DataSet d = getDataSet(dat);

        BnResReader red1 = new BnResReader();

        red1.go(firstBn);
        BnResReader red2 = new BnResReader();

        red2.go(secondBn);

        Entropy e = new Entropy(d);
        HashMap<Integer, Double> ents = new HashMap<Integer, Double>();

        for (int i = 0; i < d.n_var; i++) {
            ents.put(i, e.computeH(i));
        }

        Map<Integer, Double> ents_new = sortByValues(ents);
        Iterator<Integer> it = ents_new.keySet().iterator();

        double tot = 0;

        while (it.hasNext()) {
            int i = it.next();

            tot += (red2.scores.get(i) - red1.scores.get(i));
            pf("%.4f %.4f\n", ents_new.get(i), tot);
        }

    }
}
