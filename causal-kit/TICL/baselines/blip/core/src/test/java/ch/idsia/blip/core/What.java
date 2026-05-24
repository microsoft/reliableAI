package ch.idsia.blip.core;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.io.GobnilpReader;
import org.junit.Test;

import java.io.File;
import java.io.IOException;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class What {

    @Test
    public void testWhut() throws IOException {
        String path = "/home/loskana/Desktop/tw/uci/";
        File folder = new File(path);

        for (File f : folder.listFiles()) {
            if (!f.isFile()) {
                continue;
            }

            if (!f.getName().endsWith(".dat")) {
                continue;
            }

            String n = f.getAbsolutePath();

            n = n.replace(".dat", "");

            String gob = f("%s/gob", n);

            if (!(new File(gob).exists())) {
                String cmd = f(
                        "/home/loskana/Desktop/inference/gobnilp %s/jkl > %s/gob 2>&1 ",
                        n, n);
                // p(cmd);
                Process p = Runtime.getRuntime().exec(
                        new String[] { "bash", "-c", cmd});
                int exitVal = waitForProc(p, 1000);
            }

            BayesianNetwork b = GobnilpReader.ex(gob);

            p(n);
            p(b.n_var);
            p(b.treeWidth());

        }
    }
}
