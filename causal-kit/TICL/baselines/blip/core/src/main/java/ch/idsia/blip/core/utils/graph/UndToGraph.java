package ch.idsia.blip.core.utils.graph;


import ch.idsia.blip.core.utils.arcs.Undirected;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class UndToGraph {

    TreeSet<String> highligth;

    public String path_highlith;
    private int max_time = 100;

    public void go(Undirected u, String s) throws IOException {
        File index = new File(s);

        if (!index.exists()) {
            index.mkdir();
        }

        List<Undirected> ls = UndirectedSeparator.go(u);

        PrintWriter w = new PrintWriter(s + "/excluded", "UTF-8");

        Map<Undirected, Integer> sized = new HashMap<Undirected, Integer>();

        for (Undirected n_u : ls) {
            sized.put(n_u, n_u.n);
        }
        sized = sortInvByValues(sized);
        // pngSingle(bn, s);

        int i = 0;

        // Print each bn separated
        for (Undirected n_u : sized.keySet()) {

            if (n_u.n <= 1) {
                wf(w, twoIsBetter(n_u));
                continue;
            }

            String s1 = f("%s/%d", s, i);

            pngSingle(n_u, s1);
            i++;
        }

        w.flush();
        w.close();
    }

    private String twoIsBetter(Undirected b) {

        if (b.n == 1) {
            return f("%s \n", b.name(0));
        } else {
            return null;
        }
    }

    private void pngSingle(Undirected und, String s) throws IOException {
        und.write(s + ".dot");
        String h = f("neato -Tpng %s.dot -o %s.png", s, s);

        exec(h);
    }

    private void exec(String h) throws IOException {
        Process proc = Runtime.getRuntime().exec(h, new String[0]);
        int exitVal = waitForProc(proc, max_time * 1000);
    }
}
