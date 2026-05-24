package ch.idsia.blip.core.utils.other;


import ch.idsia.blip.core.App;
import ch.idsia.blip.core.io.ScoreReader;
import ch.idsia.blip.core.io.ScoreWriter;
import ch.idsia.blip.core.utils.ParentSet;
import ch.idsia.blip.core.utils.tw.KTree;

import java.io.IOException;
import java.util.HashMap;
import java.util.TreeSet;


public class KTreeScore extends App {

    private String ph_output;
    private String ph_scores;
    private int num_outputs;
    private int max_tw;

    public void go() throws IOException {

        prepare();

        ScoreReader sc = new ScoreReader(ph_scores.toString(), 0);

        sc.readScores();

        ParentSet[][] orig = sc.m_scores.clone();

        TreeSet<OpenScores> open = new TreeSet<OpenScores>();
        double worst_score = 0;
        int last_improvement = 0;

        while (last_improvement < 100) {

            last_improvement++;
            System.out.println(last_improvement);

            // sample a tree
            KTree k = KTree.sample(sc.n_var, max_tw, this);
            // get the parent sets
            ParentSet[][] sel = k.selectScores(orig);
            // get the number of selected scores
            double sk = k.score(sel);

            boolean toDropWorst = false;

            // If there is no space in the queue
            if (open.size() >= num_outputs) {
                // and it is terrible
                if (sk < worst_score) {
                    continue;
                }
                toDropWorst = true;
            }

            if (toDropWorst) {
                open.pollLast();
            }

            open.add(new OpenScores(sel, sk));
            last_improvement = 0;
            worst_score = open.last().sk;
        }

        for (int i = 0; i < num_outputs; i++) {
            ParentSet[][] sel = open.pollFirst().p;
            ScoreWriter sw = new ScoreWriter(ph_output + "-" + i + ".jkl");

            System.out.println(sw.path);

            sw.go(sel);
            sw.close();
        }
    }

    @Override
    public void init(HashMap<String, String> options) {
        super.init(options);
        this.max_tw = gInt("tw");
        this.num_outputs = gInt("num_outputs");
        this.ph_output = gStr("ph_output");
        this.ph_scores = gStr("ph_scores");
    }

    private class OpenScores implements Comparable<OpenScores> {
        private final ParentSet[][] p;
        private final double sk;
        public int n;

        public OpenScores(ParentSet[][] p, double sk) {
            this.p = p;
            this.sk = sk;

            n = 0;
            for (ParentSet[] p1 : p) {
                n += p1.length;
            }
        }

        public int compareTo(OpenScores other) {
            if (sk < other.sk) {
                return 1;
            }
            return -1;
        }
    }

}
