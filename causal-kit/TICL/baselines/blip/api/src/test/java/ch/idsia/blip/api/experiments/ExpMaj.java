package ch.idsia.blip.api.experiments;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.api.learn.solver.AsobsAvgSolverApi;
import org.junit.Test;

import java.io.IOException;


public class ExpMaj extends TheTest {

    String path = basePath + "majRule/nets";

    @Test
    public void gen() throws IOException, InterruptedException {

        String base = "/home/loskana/Dropbox/genetics/G2/cross-validation/";

        String h = base + "jkl-1";
        String d = base + "data-1.dat";

        String h2 = base + "res-1.csv";

        AsobsAvgSolverApi.main(new String[] {
            "-set", h, "-r", h2, "-t", "600", "-d", d, "-pb", "7"
        });

    }

    @Test
    public void testing() {
        String base = "/home/loskana/Desktop/";
        String dat = base + "dati.dat";
        String jkl = base + "dati.jkl";
        String res = base + "dati.csv";

        AsobsAvgSolverApi.main(
                new String[] {
            "solver.asobs.avg", "-r", res, "-t", "600", "-d", dat, "-pb", "7",
            "-j", jkl
        });
    }

    /*
     @Test
     public void res() throws IOException, InterruptedException {

     String n = path + "child.net";

     AsobsAvgSolverApi.main(new String[] {
     "-set", h,
     "-r", h2,
     "-t", "10",
     "-pb", "1",
     "-d", d
     });

     }
     */
}
