package ch.idsia.blip.api.exp;


import ch.idsia.blip.core.utils.arcs.Und;
import ch.idsia.blip.core.utils.graph.UndSeparator;
import ch.idsia.blip.core.learn.solver.brtl.BrutalUndirectedSolver;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;

import static ch.idsia.blip.core.utils.graph.UndSeparator.go;
import static ch.idsia.blip.core.utils.RandomStuff.*;


public class ExpUKang {

    String path = "/home/loskana/Documents/bazaar/2016/UKang";

    int tw = 2;

    String s;
    String s2;
    String out;

    public static void main(String[] argv) throws Exception {

        ExpUKang u = new ExpUKang();

        if (argv.length > 0) {
            u.tw = Integer.valueOf(argv[1]);
            u.path = argv[0];

            u.s2 = u.path + "edges.txt";
            u.out = f("%stw%d/", u.path, u.tw);
        } else {
            u.init();
        }
        u.work();
    }

    private void work() throws Exception {
        Und u = readUnd(s2);

        u.write(s + "orig");
        List<Und> lu = UndSeparator.go(u);

        p(lu.size());

        BrutalUndirectedSolver sv = new BrutalUndirectedSolver();
        int thread = 1;

        sv.max_exec_time = 24 * 3500;
        sv.setUnd(u);
        sv.thread_pool_size = thread;
        sv.tw = tw;

        sv.verbose = 2;

        sv.behaviour = 3; // bts

        // sv.behaviour = 2; // greedy
        // sv.behaviour = 1; // normal

        File f = new File(out);

        if (!f.exists()) {
            p(f.mkdirs());
        }

        sv.go(out);
    }

    private void init() throws Exception {
        pol();
        // med();
        // test();
        // camp();
    }

    private void camp() {
        tw = 5;

        s = f("%s/Campaigns/", path);
        s2 = s + "edges.txt";
        out = f("%stw%d/", s, tw);
    }

    private void pol() {
        tw = 8;

        s = f("%s/PolBlogs/", path);
        s2 = s + "polblogs-edges.txt";
        out = f("%stw%d/", s, tw);
    }

    private void med() {
        tw = 5;

        s = f("%s/PubMed/", path);
        s2 = s + "edges.txt";
        out = f("%stw%d/", s, tw);
    }

    public void test() throws Exception {
        tw = 2;

        s = f("%s/test/", path);
        s2 = s + "edges.txt";
        out = f("%stw%d/", s, tw);
    }

    public void write() throws Exception {
        init();

        BufferedReader r = getReader(s2);
        int n = Integer.valueOf(r.readLine());

        File out = new File(this.out);

        for (String d : out.list()) {
            p(d);

            File dir = new File(f("%s/%s", out.getAbsolutePath(), d));

            if (!dir.exists()) {
                continue;
            }

            for (String p : dir.list()) {
                if (!p.endsWith(".dot")) {
                    continue;
                }

                p = f("%s/%s", dir.getAbsolutePath(), p.replace(".dot", ""));

                Und u = readUnd(p);

                List<Und> l_u = go(u);

                pf("%s %d \n", p, l_u.size());

                /*
                 String h = String.format("dot -Tpng %s.dot -o %s.png", p, p);
                 cmdTimeout(h, false, false, 1000000);



                 p += "-out";
                 new File(p).mkdirs();

                 int i = 0;
                 for (Und n_u: l_u) {
                 n_u.write(f("%s/%d", p, i));
                 // String h = f("dot -Tpng %s.dot -o %s.png", p, p);
                 // exec(h);
                 i++;
                 }*/

            }
        }

    }

    public void polEasy() throws Exception {
        String s = f("%s/test/", path);
        String s2 = s + "test.txt";

        // rand(s2);

        // go(set, s2, 6);
    }

    private void rand(String s2) throws IOException {
        int n = 50;
        Writer w = getWriter(s2);

        wf(w, "%d\n", n);
        java.util.Random r = getRandom();

        for (int i = 0; i < 100; i++) {
            int v1 = r.nextInt(n);
            int v2 = v1;

            while (v2 == v1) {
                v2 = r.nextInt(n);
            }
            wf(w, "%d %d\n", v1, v2);
        }
        w.close();
    }

    static public Und readUnd(String pol) throws IOException {

        BufferedReader br = getReader(pol);
        String l;
        ArrayList<int[]> list = new ArrayList<int[]>();
        ArrayList<String> names = new ArrayList<String>();

        while ((l = br.readLine()) != null) {
            if ("".equals(l.trim())) {
                continue;
            }
            String[] aux = l.split("\\s+");
            int i = c(aux[0], names);
            int j = c(aux[1], names);

            list.add(new int[] { i, j});
        }

        Und u = new Und(names.size());

        for (int[] aux : list) {
            u.mark(aux[0], aux[1]);
        }

        u.names = new String[u.n];
        for (int i = 0; i < u.n; i++) {
            u.names[i] = names.get(i);
        }

        return u;
    }

    protected static int c(String s, ArrayList<String> names) {
        int h = names.indexOf(s);

        if (h != -1) {
            return h;
        }

        names.add(s);
        return names.size() - 1;
    }

    static public Integer i(String s) {
        return Integer.valueOf(s.trim());
    }
}
