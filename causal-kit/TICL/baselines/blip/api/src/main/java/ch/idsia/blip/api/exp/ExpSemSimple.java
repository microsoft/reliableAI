package ch.idsia.blip.api.exp;


import ch.idsia.blip.core.learn.missing.HardMissingSEM;
import ch.idsia.blip.core.learn.missing.HardSimpleMissingSEM;

import java.io.IOException;


public class ExpSemSimple extends ExpSemImputation {

    public ExpSemSimple() {
        this.suffix = "simple";
    }

    public static void main(String[] args)
            throws IOException {
        try {
            if (args.length > 1) {
                if (args[0].equals("prepare")) {
                    new ExpSemSimple().prepareNow(args);
                } else if (args[0].equals("go")) {
                    new ExpSemSimple().goNow(args);
                } else if (args[0].equals("measure")) {
                    new ExpSemSimple().measureNow(args);
                }
            } else {
                new ExpSemSimple().test2();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
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

    protected HardMissingSEM getHardMissingSEM() {
        return new HardSimpleMissingSEM();
        // return new HardMPEMissingSEM();
    }
}

