package ch.idsia.blip.core.common;


import ch.idsia.blip.core.utils.arcs.Undirected;
import ch.idsia.blip.core.utils.graph.UndToGraph;
import ch.idsia.blip.core.io.dat.DatFileLineReader;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.logExp;
import static ch.idsia.blip.core.utils.RandomStuff.p;


public class MatrixArcVisualizer {

    private static String ph_mtx;

    private static String ph_dat;

    private static Double eps;

    private static String ph_out;

    private static final Logger log = Logger.getLogger(
            MatrixArcVisualizer.class.getName());

    private static DatFileLineReader dat;

    public static void ex(String ph_mtx, String ph_dat, Double eps, String ph_out) {
        try {
            new MatrixArcVisualizer().go(ph_mtx, ph_dat, eps, ph_out);
        } catch (IOException e) {
            logExp(log, e);
        }
    }

    public void go(String ph_mtx, String ph_dat, Double eps, String ph_out) throws IOException {
        // Read data file

        p(ph_dat);
        dat = new DatFileLineReader(ph_dat);
        dat.readMetaData();

        // Read arc matrix, init undirected
        BufferedReader rd = new BufferedReader(new FileReader(ph_mtx));
        String[] aux = rd.readLine().split(" ");
        int n_var = Integer.valueOf(aux[0]);
        int n_arcs = Integer.valueOf(aux[1]);
        int m = (int) (dat.n_var * eps);
        Undirected n = new Undirected(dat.n_var);

        for (int i = 0; i < m && i < n_arcs; i++) {
            aux = rd.readLine().split(",");
            n.mark(Integer.valueOf(aux[0].trim()),
                    Integer.valueOf(aux[2].trim()));
        }

        n.names = dat.l_s_names;

        UndToGraph utg = new UndToGraph();

        utg.go(n, ph_out);

    }
}
