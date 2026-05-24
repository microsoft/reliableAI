package ch.idsia.blip.core.io;


import ch.idsia.TheTest;
import ch.idsia.blip.core.utils.DataSet;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;

import static ch.idsia.blip.core.utils.RandomStuff.getDataSet;


public class DataFileReaderTest extends TheTest {

    @Test
    public void testSimple() throws IOException {

        DataSet dat_rd = getDataSet(basePath + "child-10000.dat");

        for (int n = 0; n < dat_rd.n_var; n++) {
            System.out.println("## Variable: " + n);
            for (int v = 0; v < dat_rd.l_n_arity[n]; v++) {
                System.out.println(
                        "value: " + v + " - "
                        + Arrays.toString(dat_rd.row_values[n][v]));
            }
        }
    }

    @Test
    public void testMissing() throws IOException {

        DataSet dat_rd = getDataSet(basePath + "child-missing-10000.dat");

        for (int n = 0; n < dat_rd.n_var; n++) {
            System.out.println("## Variable: " + n);
            for (int v = 0; v < dat_rd.l_n_arity[n]; v++) {
                System.out.println(
                        "value: " + v + " - "
                        + Arrays.toString(dat_rd.row_values[n][v]));
            }
            System.out.println(
                    "missing: "
                            + Arrays.toString(
                                    dat_rd.row_values[n][dat_rd.l_n_arity[n]]));

        }
    }
}
