package ch.idsia.blip.api.experiments;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.io.bn.BnErgReader;
import ch.idsia.blip.core.io.bn.BnUaiWriter;
import org.junit.Test;

import java.io.File;
import java.io.IOException;

import static ch.idsia.blip.core.utils.RandomStuff.f;
import static ch.idsia.blip.core.utils.RandomStuff.p;


public class ExpUai06Final extends TheTest {

    String path = basePath + "uai06/final";

    @Test
    public void gen() throws IOException {

        File[] fList = new File(path).listFiles();

        for (File file : fList) {
            if (file.isDirectory()) {
                p(file.getName());
                String s = f("%s/%s/%s.erg", path, file.getName(),
                        file.getName());
                String s2 = f("%s/%s/%s.uai", path, file.getName(),
                        file.getName());
                BayesianNetwork bn = BnErgReader.ex(s);

                BnUaiWriter.ex(s2, bn);
            }
        }
    }

}
