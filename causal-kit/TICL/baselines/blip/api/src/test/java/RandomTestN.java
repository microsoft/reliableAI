import ch.idsia.blip.api.common.SamGeApi;
import ch.idsia.blip.api.learn.scorer.SeqScorerApi;
import ch.idsia.blip.api.learn.solver.win.WinAsobsSolverApi;
import ch.idsia.blip.api.common.LLEvalApi;
import ch.idsia.blip.api.common.Evaluate;
import ch.idsia.blip.api.learn.param.ParLeApi;
import ch.idsia.blip.api.learn.param.ParLeSmoothApi;
import ch.idsia.blip.core.learn.param.ParLe;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.math.FastMath;
import org.junit.Test;

import static ch.idsia.blip.core.utils.RandomStuff.p;


public class RandomTestN {

    @Test
    public void testt2() throws Exception {
        int[] a = new int[] { 0, 3, 9};
        int[] b = new int[] { 1, 2};

        p(ArrayUtils.union(a, b));

        a = new int[] { 1, 2};
        b = new int[] { 0, 3, 9};
        p(ArrayUtils.union(a, b));

        a = new int[] {};
        b = new int[] { 0, 3, 7};
        p(ArrayUtils.union(a, b));

        a = new int[] { 0, 3, 7};
        b = new int[] {};
        p(ArrayUtils.union(a, b));
    }

    @Test
    public void testFail() throws Exception {

        String p = "/home/loskana/Desktop/";

        SeqScorerApi.main(
                new String[] {
            "", "-j", p + "here2", "-d", p + "here", "-t",
            "600", "-n", "6", "-pc", "bic", "-pa", "1.0"
        });
    }

    @Test
    public void testScores() throws Exception {

        String p = "/home/loskana/Desktop/chordalysis/comparison/networks/";

        SamGeApi.main(
                new String[] {
            "", "-n", p + "data/andes.net", "-d",
            p + "data/andes.dat", "-set", "5000"
        });

        ParLeApi.main(
                new String[] {
            "", "-d", p + "data/andes.dat", "-r",
            p + "nets/chord-05/andes", "-n", p + "nets/chord-05/andes.uai"
        });

        LLEvalApi.main(
                new String[] {
            "", "-d", p + "data/andes.dat", "-n",
            p + "nets/chord-05/andes.uai"
        });

    }

    @Test
    public void test() {
        String p = "/home/loskana/Desktop/SPN/kmax/";

        ParLeSmoothApi.main(
                new String[] {
            "", "-d",
            p + "../SLSPN_Release/data/nltcs.ts.data", "-n",
            p + "work/nltcs/res-1.uai", "-r", p + "work/nltcs/res-1", "-va",
            p + "../SLSPN_Release/data/nltcs.valid.data", "-pb", "7", "-v", "1"
        });

    }

    @Test
    public void test2() {
        String p = "/home/loskana/Desktop/SPN/kmax/";

        LLEvalApi.main(
                new String[] {
            "", "-d", p + "data/msnbc.ts.dat", "-n",
            p + "work/msnbc/res-1.uai"
        });
    }

    @Test
    public void entTest() {

        // double[] set = new double[] {0.3, 0.4, 0.2};
        double[] s = new double[] { 0.001, 0.999};

        for (int b = 2; b < 20; b++) {
            p(ent(s, b));
        }
    }

    private double ent(double[] s, int b) {
        double t = 0.0;

        for (double e : s) {
            t += e * FastMath.log(e) / FastMath.log(b);
        }
        return t;
    }

    @Test
    public void testWin() {
        String s = "/home/loskana/Desktop/";

        WinAsobsSolverApi.main(
                new String[] {
            "", "-j", s + "net30.d.jkl", "-r", s + "results",
            "-t", "999", "-win", "2", "-v", "2"});
    }

    @Test
    public void testest(){
        Evaluate.main(new String[] {
                "",
                "-n", "/home/loskana/Dropbox/blip/Rexample/test/res/water/hc.net",
                "-d", "/home/loskana/Dropbox/blip/Rexample/test/res/water/arff"
        });
    }

    @Test
    public void testest2(){
        ParLeApi.main(new String[] {
                "",
                "-r", "/home/loskana/Dropbox/blip/Rexample/test/res/insurance/hc.net",
                "-d", "/home/loskana/Dropbox/blip/Rexample/test/res/insurance/arff",
                "-n", "/home/loskana/Dropbox/blip/Rexample/test/res/insurance/hc2.net"
        });
    }

}
