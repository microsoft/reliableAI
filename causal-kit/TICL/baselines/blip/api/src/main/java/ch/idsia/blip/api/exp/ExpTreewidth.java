package ch.idsia.blip.api.exp;

import ch.idsia.blip.core.utils.arcs.Undirected;
import ch.idsia.blip.core.utils.tw.TreeWidth;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;

import static ch.idsia.blip.core.utils.RandomStuff.getReader;

public class ExpTreewidth extends ExpUKang {

    String path = "/home/loskana/Documents/bazaar/2016/UKang";

    public static void main(String[] argv) throws Exception {

        new ExpTreewidth().go();
    }

    private void go() throws IOException {
        check();
    }

    private void check() throws IOException {
        // String s = path + "/BTS-1.0/out/subgraph.csv";
        // String d = ",";
        String s = path + "/PolBlogs/tw7/4316/0";
        String d = "\\s+";

        Undirected u = readUndirected(s, d);

        int v = TreeWidth.go(u);
        System.out.println(v);
    }

    static public Undirected readUndirected(String pol, String del) throws IOException {

        BufferedReader br = getReader(pol);
        String l;
        ArrayList<int[]> list = new ArrayList<int[]>();
        ArrayList<String> names = new ArrayList<String>();

        while ((l = br.readLine()) != null) {

            if ("".equals(l.trim())) {
                continue;
            }

            String[] aux =l.split(del);
            if (aux.length < 2)
                continue;

            int i = c(aux[0], names);
            int j = c(aux[1], names);

            list.add(new int[] { i, j});
        }

        Undirected u = new Undirected(names.size());

        for (int[] aux : list) {
            u.mark(aux[0], aux[1]);
        }

        u.names = new String[u.n];
        for (int i = 0; i < u.n; i++) {
            u.names[i] = names.get(i);
        }

        return u;
    }
}
