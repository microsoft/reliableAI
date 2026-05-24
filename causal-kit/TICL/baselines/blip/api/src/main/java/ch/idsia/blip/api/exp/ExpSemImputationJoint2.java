package ch.idsia.blip.api.exp;


import java.io.IOException;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class ExpSemImputationJoint2 extends ExpSemImputationJoint {

    public static void main(String[] args)
            throws IOException {
        try {
            if (args.length > 1) {
                if (args[0].equals("prepare")) {
                    new ExpSemImputationJoint2().prepareNow(args);
                } else if (args[0].equals("go")) {
                    new ExpSemImputationJoint2().goNow(args);
                } else if (args[0].equals("measure")) {
                    new ExpSemImputationJoint2().measureNow(args);
                }
            } else {
                new ExpSemImputationJoint2().test();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    protected void measureNow(String[] args) throws IOException {
        this.competit = "simple";
        p("ciao");
        super.measureNow(args);
    }
}

