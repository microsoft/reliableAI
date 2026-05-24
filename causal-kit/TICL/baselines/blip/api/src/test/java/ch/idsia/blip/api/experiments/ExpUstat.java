package ch.idsia.blip.api.experiments;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.api.learn.scorer.IndependenceScorerApi;
import ch.idsia.blip.api.learn.solver.AsobsAvgSolverApi;
import org.junit.Test;

import java.io.File;
import java.io.IOException;


public class ExpUstat extends TheTest {

    String path = basePath + "majRule/nets";

    @Test
    public void gen() throws IOException, InterruptedException {

        String p = "/home/loskana/Desktop/Ustat/SAKE2010-2015_Stephani/Total_SAKE2015/";

        String jkl = p + "jkl";
        String dat = p + "SAKE2015.CSV.dat";
        String res = p + "res.csv";

        if (!new File(jkl).exists()) {
            IndependenceScorerApi.main(
                    new String[] {
                "-d", dat, "-set", jkl, "-n", "4", "-t", "20", "-pc", "bdeu",
                "-pa", "1", "-v", "2"

            });
        }

        AsobsAvgSolverApi.main(new String[] {
            "-set", jkl, "-r", res, "-t", "60", "-d", dat,
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
