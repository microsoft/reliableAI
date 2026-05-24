package ch.idsia.blip.core.io;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.arcs.NamedDirected;
import ch.idsia.blip.core.io.bn.BnResReader;
import ch.idsia.blip.core.io.dat.DatFileLineReader;
import ch.idsia.blip.core.utils.other.IncorrectCallException;

import java.io.IOException;


public class GraphWriter {

    public static void go(String path_res, String path_dat, String out) throws IOException, IncorrectCallException {
        go(BnResReader.ex(path_res), new DatFileLineReader(path_dat), out);
    }

    private static void go(BayesianNetwork f, DatFileLineReader dr, String out) throws IOException {
        dr.readMetaData();

        NamedDirected d = f.directed();

        d.names = dr.l_s_names;
        d.graph(out);
    }
}
