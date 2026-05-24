package ch.idsia.blip.api;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.arcs.Directed;
import ch.idsia.blip.core.utils.arcs.Undirected;
import ch.idsia.blip.core.io.bn.BnNetReader;
import ch.idsia.blip.core.io.bn.BnNetWriter;
import ch.idsia.blip.core.utils.tw.TreeWidth;
import ch.idsia.blip.core.utils.ParentSet;

import java.io.*;


public class TheTest {

    protected String basePath = "../experiments/";

    protected static void writeGraph(String s, Directed dir) {

        // System.out.println(set);

        dir.graph(s + "-dir");

        Undirected und = dir.moralize();

        und.graph(s + "-mor");

        TreeWidth t = new TreeWidth();
        int h = t.exec(und);

        // System.out.println("#### Treewidth: "  + h);
        t.ar.graph(s + "-cor");
    }

    protected BayesianNetwork getBnFromFile(String s) throws IOException {
        File f_bn_original = new File(basePath + s);
        BufferedReader rd_orig = new BufferedReader(
                new FileReader(f_bn_original));

        return BnNetReader.ex(rd_orig);
    }

    protected void printBnToFile(BayesianNetwork bn, String fileName) throws FileNotFoundException, UnsupportedEncodingException {
        OutputStreamWriter bn_stream = new OutputStreamWriter(
                new FileOutputStream(basePath + fileName), "utf-8");
        BufferedWriter bn_wr = new BufferedWriter(bn_stream);

        BnNetWriter.ex(bn, bn_wr);
    }

    public boolean notExceedingTw(BayesianNetwork bn, int maxTw) {

        // System.out.println(" -> " + tw.treewidth());

        return TreeWidth.go(bn) <= maxTw;
    }

    public boolean notExceedingTw(ParentSet[] best_str, int maxTw) {

        BayesianNetwork bn = new BayesianNetwork(best_str.length);

        for (int i = 0; i < best_str.length; i++) {
            // System.out.print(Arrays.toString(best_str[thread].parents));
            bn.setParents(i, best_str[i].parents);
        }

        return TreeWidth.go(bn) <= maxTw;

    }
}
