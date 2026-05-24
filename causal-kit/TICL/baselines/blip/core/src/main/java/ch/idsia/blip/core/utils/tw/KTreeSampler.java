package ch.idsia.blip.core.utils.tw;


import ch.idsia.blip.core.App;
import ch.idsia.blip.core.utils.analyze.MutualInformation;
import ch.idsia.blip.core.utils.ParentSet;

import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.logExp;


public class KTreeSampler {

    private static final Logger log = Logger.getLogger(
            KTreeSampler.class.getName());

    private final int n_var;
    private final int maxTreeWidth;
    private final MutualInformation mi;
    private final ParentSet[][] m_scores;

    public KTreeSampler(int n_var, int maxTreeWidth, MutualInformation mi, ParentSet[][] m_scores, App base) {
        this.n_var = n_var;
        this.maxTreeWidth = maxTreeWidth;
        this.mi = mi;
        this.m_scores = m_scores;

        this.base = base;
    }

    private double best_is = -Double.MAX_VALUE;

    private final App base;

    public KTree go() {

        double is;
        boolean go = false;

        KTree K = null;

        while (K == null) {
            try {
                KTree K1 = KTree.decode(
                        Dandelion.sample(n_var, maxTreeWidth, base));

                is = K1.informativeScore(mi, m_scores);

                // if the score is better than before then the tw-tree is ok
                if (is > best_is) {
                    best_is = is;
                    go = true;
                } else {

                    // test if the score is indeed better
                    // System.out.println("is: " + is + ", best: " + is + "ratio: " + is / best_is);
                    if (base.rand.nextDouble() < (is / best_is)) {
                        go = true;
                    }
                }

                if (go) {
                    K = K1;
                }

            } catch (Exception e) {
                logExp(log, e);
            }
        }

        return K;
    }
}
