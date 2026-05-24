import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.common.SamGe;
import ch.idsia.blip.core.io.bn.BnNetReader;
import ch.idsia.blip.core.io.bn.BnNetWriter;
import ch.idsia.blip.core.io.bn.BnResReader;
import ch.idsia.blip.core.io.bn.BnUaiWriter;
import ch.idsia.blip.core.utils.score.BDeu;
import ch.idsia.blip.core.utils.score.BDeuWeight;
import ch.idsia.blip.core.utils.score.BIC;
import ch.idsia.blip.core.utils.score.Score;
import ch.idsia.blip.core.learn.param.ParLeBayes;
import ch.idsia.blip.core.learn.param.ParLeWeight;
import ch.idsia.blip.core.learn.scorer.IndependenceScorer;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class ExpUstat {

    String path = "/home/loskana/Documents/bazaar/2016/Ustat/U3/data_new/";

    @Test
    public void testScores() throws Exception {

        // score("apprendisti");
        // score("attivi_occupati");
        // score("disoccupati");
        // score("inattivi");
        score("totale");
    }

    @Test
    public void testParle() throws Exception {
        parle("totale");
    }

    private void parle(String s) throws IOException {
        double[] weight = getWeights(s);

        DataSet dat = getDatas(s);

        BayesianNetwork bn = BnResReader.ex(f(path + "../result/%s.res", s));
        ParLeWeight pl = new ParLeWeight(1, weight);
        BayesianNetwork newBn = pl.go(bn, dat);

        BnNetWriter.ex(newBn, f(path + "../result/%s.net", s));

        bn = BnResReader.ex(f(path + "../result/%s.res", s));
        ParLeBayes p = new ParLeBayes(1);

        newBn = p.go(bn, dat);
        BnNetWriter.ex(newBn, f(path + "../result/%s-def.net", s));
    }

    @Test
    public void debug() throws Exception {
        parledeb("totale", "finale-7");
        // parledeb("totale", "test-2");
        // parledeb("totale", "void");
        // parledeb("totale", "test-1");
    }

    private void parledeb(String s, String h) throws IOException {
        double[] weight = getWeights(s);

        DataSet dat = getDatas(s);

        BayesianNetwork bn = BnResReader.ex(f(path + "../final/%s.res", h));
        ParLeWeight pl = new ParLeWeight(1, weight);
        BayesianNetwork newBn = pl.go(bn, dat);

        BnNetWriter.ex(newBn, f(path + "../final/%s.net", h));
        BnUaiWriter.ex(newBn, f(path + "../final/%s.uai", h));

        bn = BnResReader.ex(f(path + "../final/%s.res", h));
        ParLeBayes p = new ParLeBayes(1);

        newBn = p.go(bn, dat);
        BnNetWriter.ex(newBn, f(path + "../final/%s-def.net", h));
        BnUaiWriter.ex(newBn, f(path + "../final/%s-def.uai", h));
    }

    private void score(String s) throws Exception {

        double[] weight = getWeights(s);

        DataSet dat = getDatas(s);

        BDeuWeight bW = new BDeuWeight(1, dat, weight);

        //
        // BDeu pb = new BDeu(1, dat);
        //
        // p(bW.computeScore(0));
        // p(pb.computeScore(0));
        //
        // p("");
        //
        // p(bW.computeScore(1));
        // p(pb.computeScore(1));

        scores(s, bW, dat, f("-1.jkl"));
        // scores(set, pb, dat, f("-%d-old.jkl", hgf));

    }

    private DataSet getDatas(String s) throws IOException {
        String d = path + s + ".dat";
        DataSet dat = getDataSet(d);

        return dat;
    }

    private double[] getWeights(String s) throws IOException {
        String we = path + s + "-weight";

        BufferedReader br = getReader(we);
        String l;
        ArrayList<Double> ls = new ArrayList<Double>();
        double sum = 0;

        while ((l = br.readLine()) != null) {
            double h = Double.valueOf(l);

            ls.add(h);
            sum += h;
        }

        double[] weight = new double[ls.size()];
        double n_sum = 0;

        for (int i = 0; i < ls.size(); i++) {
            // weight[i] = ls.get(i) / sum * ls.size();
            // n_sum += weight[i];
            weight[i] = ls.get(i) / 10;
        }
        return weight;
    }

    private void scores(String s, Score b, DataSet dat, String h) throws Exception {
        IndependenceScorer is = new IndependenceScorer();

        is.ph_scores = path + s + h;
        is.dat = dat;
        is.prepare();
        is.score = b;
        is.max_exec_time = 600;
        is.thread_pool_size = 6;
        is.n_var = b.dat.n_var;
        is.max_pset_size = 1;

        // is.go(dat, 0);
        is.searchAll();

    }

    @Test
    public void bdeucheck() throws Exception {
        String p2 = "../experiments/child";

        BayesianNetwork bn = BnNetReader.ex(p2 + ".net");

        for (int s : new int[] { 100, 500, 1000, 5000, 50000}) {
            String d = f("%s-%d.dat", p2, s);

            SamGe.ex(bn, d, s);

            for (String sc : new String[] { "bic", "bdeu"}) {
                String j = f("%s-%d-%s.jkl", p2, s, sc);
                IndependenceScorer is = new IndependenceScorer();

                is.max_pset_size = 3;
                is.ph_scores = j;
                is.scoreNm = sc;
                is.choice_variables = "0";
                is.go(getDataSet(d));

            }
        }

    }

    @Test
    public void testnew() throws Exception {

        BDeu b = new BDeu(1, getDatas("totale"));

        p(b.computeScore(2));
        p(b.computeScore(2, new int[] { 16}));

        BIC bi = new BIC(getDatas("totale"));

        p(bi.computeScore(2));
        p(bi.computeScore(2, new int[] { 16}));

        IndependenceScorer is = new IndependenceScorer();

        is.max_pset_size = 1;
        is.ph_scores = "heer";
        is.scoreNm = "bic";
        is.choice_variables = "2";
        is.go(getDatas("totale"));
    }
}
