package ch.idsia.blip.api.exp;


import ch.idsia.blip.api.learn.scorer.IndependenceScorerApi;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.score.BIC;
import ch.idsia.blip.core.utils.score.K2;
import ch.idsia.blip.core.utils.score.Score;
import ch.idsia.blip.core.utils.data.ArrayUtils;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class ExpK2 {

    String p = "/home/loskana/Desktop/winmine/";

    public static void main(String[] args) {
        try {
            new ExpK2().test2();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void test() throws Exception {
        String f = f("%s/data1/accidents.ts.dat", p);
        DataSet d = getDataSet(f);

        K2 score = new K2(d);

        int[] s = new int[] { 5, 95, 96, 49, 14, 50, 83, 100, 98, 107};

        p(score.computeScore(8, s));

        IndependenceScorerApi.main(
                new String[] {
            "", "-d", f, "-j", p + "out", "-b", "1", "-c",
            "k2", "-t", "60000", "-u", "1", "-n", "12"
        });
    }

    public void test2() throws Exception {
        String f = f("%s/data1/accidents.ts.dat", p);
        DataSet d = getDataSet(f);

        Score bic = new BIC(d);

        checkExpansion(bic, new int[] { 2, 8}, 5);

        checkExpansion(bic, new int[] { 2, 5, 8}, 3);

        checkExpansion(bic, new int[] { 1, 4, 8}, 0);
    }

    private void checkExpansion(Score bic, int[] old_pset, int new_p) throws Exception {
        int[] new_pset = ArrayUtils.expandArray(old_pset, new_p);

        // FIRST THING
        int[][] res1 = bic.computeParentSetValues(new_pset);

        // NEXT THING
        int[][] old = bic.computeParentSetValues(old_pset);
        int[][] res2 = bic.expandParentSetValues(new_pset, old, new_p);

        pf("%d - %d\n\n", res1.length, res2.length);

        for (int i = 0; i < res1.length; i++) {
            pf("\t %d - %d \n", res1[i].length, res2[i].length);
            if (res1[i].length != res2[i].length) {
                throw new Exception("ahhhahahahahahaha");
            }
        }
    }
}

