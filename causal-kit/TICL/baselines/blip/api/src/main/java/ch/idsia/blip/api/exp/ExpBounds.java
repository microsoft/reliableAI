package ch.idsia.blip.api.exp;


import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.analyze.MutualInformation;
import ch.idsia.blip.core.learn.scorer.SeqNewScorer;
import ch.idsia.blip.core.learn.scorer.SeqUltScorer;

import java.io.IOException;
import java.io.Writer;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class ExpBounds {

    String path = "experiments/uci_data/out";

    public static void main(String[] args) {
        try {
            new ExpBounds().see2();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void see2() throws IOException {

        for (String net : new String[] {
            "car", "nursery", "breast", "adult",
            "zoo", "letter", "mushroom", "wdbc", "lung"}) {
            String s = f("%s/%s.dat", path, net);
            DataSet dat = getDataSet(s);
            MutualInformation mi = new MutualInformation(dat);

            double t = 0.0;

            for (int i = 0; i < dat.n_var; i++) {
                t += dat.l_n_arity[i];
            }

            double m = 0.0;
            int cnt = 0;

            for (int i1 = 0; i1 < dat.n_var; i1++) {
                for (int i2 = i1 + 1; i2 < dat.n_var; i2++) {
                    m += mi.computeMi(i1, i2);
                    cnt++;
                }
            }

            pf("%s %.2f  %.4f\n", net, t / dat.n_var, m / cnt);
        }
    }

    public void see() throws Exception {

        Writer wr = getWriter(f("%s/res2", path));

        wf(wr, "%10s \t %6s \t %6s \t %6s \t %8s \t %8s \t %8s \t %8s \n",
                "Name", "n_var", "n_dp", "k", "Def", "Old", "New", "Old+New");

        for (String net : new String[] {
            "car", "nursery", "breast", "adult",
            "zoo", "letter", "mushroom", "wdbc", "lung"}) {

            for (int k : new int[] { 3, 4, 5}) {

                String s = f("%s/%s.dat", path, net);
                DataSet dat = getDataSet(s);

                SeqNewScorer seq = new SeqNewScorer();

                wf(wr, "%10s \t %6d \t %6d \t %6d \t", net, dat.n_var,
                        dat.n_datapoints, k);

                g(s, k, seq, net, wr);

                // IndependenceScorer is = new IndependenceScorer();
                // g(set, is, "is", net);

                wf(wr, "\n");
                wr.flush();
            }
        }
    }

    private void g(String s, int k, SeqNewScorer seq, String net, Writer wr) throws Exception {
        String res = f("%s/res/%s-%d", path, net, k);

        // if (!new File(res).exists()) {
        seq.max_pset_size = k;
        seq.scoreNm = "bic";
        seq.max_exec_time = 0;
        seq.verbose = 1;
        seq.ph_scores = f("%s/jkl/%s-%d", path, net, k);
        seq.wr = getWriter(res);
        // seq.thread_pool_size = 1;
        seq.go(s);
        seq.wr.close();
        seq.thread_pool_size = 1;

        wf(wr, " %8d \t %8d \t %8d \t %8d \t", seq.def_cnt, seq.old_cnt,
                seq.new_cnt, seq.both_cnt);
        // }
    }

    public void test() {
        SeqUltScorer seq = new SeqUltScorer();

        seq.max_pset_size = 3;
        seq.scoreNm = "bic";
        seq.max_exec_time = 0;
        seq.verbose = 1;
        seq.thread_pool_size = 1;
        // seq.go(set);
    }
}
