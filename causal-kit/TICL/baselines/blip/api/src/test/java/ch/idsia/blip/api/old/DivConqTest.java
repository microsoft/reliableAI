package ch.idsia.blip.api.old;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.other.DivConq;
import ch.idsia.blip.core.utils.other.IncorrectCallException;
import ch.idsia.blip.core.utils.ParentSet;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.List;

import static ch.idsia.blip.core.utils.RandomStuff.getDataSet;
import static ch.idsia.blip.core.utils.RandomStuff.getScoreReader;


public class DivConqTest extends TheTest {

    @Test
    public void someTimes() throws Exception {

        // hereGoesNothing("alarm-10000.dat", 2);
        hereGoesNothing("child-5000");

        // hereGoesNothing("d-10000.dat", 10);
    }

    private void hereGoesNothing(String s) throws IOException, IncorrectCallException {

        s = basePath + "scorer/" + s;

        ParentSet[][] sc = getScoreReader(s + ".scores", 1);
        DataSet dat = getDataSet(s + ".dat");

        BufferedWriter writer = new BufferedWriter(
                new OutputStreamWriter(System.out));

        DivConq dv = new DivConq();

        dv.findKMedoids(dat);

    }

    @Test
    public void testMI() throws IOException, IncorrectCallException {

        String s = basePath + "scorer/child-5000";

        DataSet dat = getDataSet(s + ".dat");

        BufferedWriter writer = new BufferedWriter(
                new OutputStreamWriter(System.out));

        DivConq dv = new DivConq();

        dv.k = 5;
        dv.verbose = 2;
        List<TIntArrayList> clusters = dv.findKMedoids(dat);

        System.out.println("\n MIS: \n");
        for (int i = 0; i < dat.n_var; i++) {
            for (int j = i + 1; j < dat.n_var; j++) {
                System.out.printf("%d - %d - %.5f \n", i, j, dv.mi.getMI(i, j));
            }
        }

        for (TIntArrayList cl : clusters) {
            System.out.println(cl);
        }
    }

}
