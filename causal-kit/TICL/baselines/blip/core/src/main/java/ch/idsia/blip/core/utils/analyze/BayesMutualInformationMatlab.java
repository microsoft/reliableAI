package ch.idsia.blip.core.utils.analyze;


import ch.idsia.blip.core.utils.DataSet;

import java.io.*;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class BayesMutualInformationMatlab extends BayesMutualInformation {

    private static final Logger log = Logger.getLogger(
            BayesMutualInformationMatlab.class.getName());

    private final String path = "../utils/";

    private double[][] n_w;
    private double[] n_z;
    private int x_ar;
    private int y_ar;
    private int z_ar;

    public BayesMutualInformationMatlab(DataSet dat) {
        super(dat);
    }

    @Override
    public double computeCMI(int x, int y, int z) {
        return computeCMI(x, y, new int[] { z});
    }

    @Override

    /**
     * Bayesian PR of CMI: MI(X, Y | Z)
     */
    public double computeCMI(int x, int y, int[] z) {

        if (z.length == 0) {
            return computeMi(x, y);
        }

        int[][] x_r = dat.row_values[x];

        x_ar = dat.l_n_arity[x];

        int[][] y_r = dat.row_values[y];

        y_ar = dat.l_n_arity[y];

        int[][] z_r = getZRowsNoMissing(z);

        z_ar = z_r.length;

        double ess = (x_ar - 1) * (y_ar - 1) * z_ar;

        n_z = getCounts(z_r, z_ar, ess);

        int w_ar = x_ar * y_ar;

        n_w = getCondJointCounts(x_r, x_ar, y_r, y_ar, w_ar, z_r, z_ar, ess);

        double d = 0;

        try {
            d = execute();
        } catch (IOException e) {
            logExp(log, e);
        }

        return d;

    }

    private double execute() throws IOException {

        String cmd = path + "cmd.m";
        File f = new File(cmd);

        if (!f.exists()) {
            f.createNewFile();
        }

        new File(path + "out").delete();
        new File(path + "out.png").delete();

        FileWriter fw = new FileWriter(f.getAbsoluteFile());
        BufferedWriter bw = new BufferedWriter(fw);

        writeCommands(bw);
        bw.close();

        String run = "matlab -nodisplay -nodesktop < cmd.m > log 2>&1 ";
        Process proc = Runtime.getRuntime().exec(run, new String[0],
                new File(path));
        int exitVal = waitForProc(proc, 10000);

        double cmi = -1;

        if (new File(path + "out").exists()) {

            BufferedReader br = new BufferedReader(new FileReader(path + "out"));

            cmi = Double.valueOf(br.readLine().trim());
        }

        return cmi;
    }

    private void writeCommands(BufferedWriter w) throws IOException {
        wf(w, "N_z = %d; \n", z_ar);
        wf(w, "N_x = %d; \n", x_ar);
        wf(w, "N_y = %d; \n", y_ar);

        wf(w, "n_z = zeros(%d, 1); \n", n_z.length);
        for (int i = 0; i < n_z.length; i++) {
            wf(w, "n_z(%d) = %.8f; \n", i + 1, n_z[i]);
        }

        wf(w, "nz_w = zeros(%d, %d); \n", z_ar, n_w[0].length);
        for (int i = 0; i < z_ar; i++) {
            for (int j = 0; j < n_w[i].length; j++) {
                wf(w, "nz_w(%d, %d) = %.8f; \n", i + 1, j + 1, n_w[i][j]);
            }
        }

        wf(w, "[c, h] = cmi(N_z, n_z, N_x, N_y, nz_w);\n");
        wf(w, "dlmwrite('out',mean(h));\n");
        wf(w, "figure;\n");
        wf(w, "aa = hist(h,80);\n");
        wf(w, "hist(h,80);\n");
        wf(w, "m = max(aa);\n");
        wf(w, "line([c c], [0 m]);\n");
        wf(w, "saveas(gcf,'out.png')\n");
    }

    public void out(String s) {

        File f1 = new File(path + "out.png");

        if (f1.exists()) {
            try {
                copySomething(f1, new File(s));
            } catch (IOException e) {
                logExp(log, e);
            }

        }
    }
}
