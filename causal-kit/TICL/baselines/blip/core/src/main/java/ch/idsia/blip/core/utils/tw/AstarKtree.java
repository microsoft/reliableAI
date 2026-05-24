package ch.idsia.blip.core.utils.tw;


import ch.idsia.blip.core.utils.analyze.MutualInformation;
import ch.idsia.blip.core.utils.arcs.Undirected;
import ch.idsia.blip.core.learn.solver.ktree.S2PlusSolver;
import ch.idsia.blip.core.utils.RandomStuff;

import java.io.*;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class AstarKtree {

    private static final Logger log = Logger.getLogger(
            AstarKtree.class.getName());

    private final int n;
    private final int k;
    private MutualInformation mi;

    private int index;

    String run = "matlab -nodisplay -nodesktop -nojvm < %s > log 2>&1 & ";

    private String path;

    private long start;

    private int count;

    public AstarKtree(int n_var, int k, S2PlusSolver solv) {
        this.n = n_var;
        this.k = k;

        this.mi = solv.mi;

        new File(solv.ph_work).mkdir();
        copyDirectory(new File(solv.ph_astar), new File(solv.ph_work));
        path = solv.ph_work;
    }

    public KTree go(int[] R, double availableTime, int i, int index) {

        start = System.currentTimeMillis();

        try {

            // Write file command
            String cmd = f("%s/cmd%d.m", path, i);
            File f = new File(cmd);

            if (!f.exists()) {
                f.createNewFile();
                FileWriter fw = new FileWriter(f.getAbsoluteFile());
                BufferedWriter bw = new BufferedWriter(fw);

                writeCommands(bw, i);
                bw.close();
                update(f("file %s printed", cmd));
            }

            // Execute
            String run = "matlab -nodisplay -nodesktop -nojvm < cmd%d.m > log%d-%d 2>&1 ";
            String r = f(run, i, i, index);
            Process proc = Runtime.getRuntime().exec(r, new String[0],
                    new File(path));

            update(f("available time: %.2f", availableTime / 1000.0));
            int exitVal = waitForProc(proc, availableTime);

            update("executed");

            // Check if there is output
            File f1 = new File(f("%s/out%d", path, i));

            if (!f1.exists()) {
                log.severe("NO KTREE CREATED!");
                return null;
            }

            File f2 = new File(f("%s/out%d-%d", path, i, index));

            f1.renameTo(f2);
            KTree k = readKtree(f2);

            update("done");
            pf("%d\n", ++count);
            return k;

        } catch (IOException e) {
            RandomStuff.logExp(log, e);
        }

        return null;
    }

    private void update(String s) {
        pf("%s %.2f \n", s, (System.currentTimeMillis() - start) / 1000.0);
    }

    private KTree readKtree(File f) throws IOException {
        Undirected u = new Undirected(n);

        BufferedReader br = new BufferedReader(new FileReader(f));

        String s;
        int i1 = 0;

        while ((s = br.readLine()) != null) {
            String[] aux = s.split(",");

            for (int i2 = i1 + 1; i2 < n; i2++) {
                if (aux[i2].equals("1")) {
                    u.mark(i1, i2);
                }
            }
            i1++;
        }

        u.graph(path + "graph");

        KTree k = new KTree(u);

        return k;
    }

    private void writeCommands(BufferedWriter c, int i) throws IOException {

        c.write(f("n = %d; \n", n));
        c.write(f("tw = %d; \n", k));
        // A := matrix([[1, 5], [2, 3]])
        c.write("M = zeros(n); \n");
        for (int i1 = 0; i1 < n; i1++) {

            for (int i2 = i1 + 1; i2 < n; i2++) {
                c.write(
                        f("M(%d, %d) = %.8f; \n", i2 + 1, i1 + 1,
                        mi.getMI(i1, i2)));
            }
        }
        c.write("M=tril(M,-1)+tril(M)';\n");
        // c.graph("]\n");

        c.write("t = cputime; \n");
        c.write(f("i = %d; \n", i));
        // c.graph(f("while (cputime - t) < %d \n", solv.max_exec_time) );
        c.write("    rng shuffle; \n");
        c.write("    R = randsample(n,tw+1)'; \n");
        c.write("    disp(i); \n");
        c.write("    disp(R); \n");
        c.write("    T = AstarKtree(R, M); \n");
        c.write("    dlmwrite(strcat('out', int2str(i)),T);  \n");
        c.write("    disp((cputime - t)); \n");
        // c.graph("    thread = thread+1; \n");
        // c.graph("end \n");

        c.write("exit");
    }

    private String prune(String res) {
        String[] p = res.split("ans =");

        return p[1].replace(">>", "").trim();
    }
}
