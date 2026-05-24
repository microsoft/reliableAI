package ch.idsia.blip.core.utils.matrix;


import ch.idsia.blip.core.utils.RandomStuff;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;

import static ch.idsia.blip.core.utils.RandomStuff.getReader;
import static ch.idsia.blip.core.utils.RandomStuff.wf;


public class UpperMatrix {

    // size of matrix
    public final int size;

    // matrix
    public double[] vec;

    // number of variables
    public int n;

    public UpperMatrix(int n) {
        this.n = n;
        size = (n * (n - 1)) / 2;
        vec = new double[size];
    }

    public int index(int v1, int v2) {
        return RandomStuff.index(n, v1, v2);
    }

    public void inc(int v1, int v2) {
        if (v1 != v2) {
            vec[index(v1, v2)]++;
        }
    }

    public double value(int v1, int v2) {
        if (v1 != v2) {
            return vec[index(v1, v2)];
        } else {
            return 0;
        }
    }

    public void inc(int i, double v) {
        if (i >= 0 && i < size) {
            vec[i] += v;
        }
    }

    public double value(int i) {
        if (i >= 0 && i < size) {
            return vec[i];
        }
        return 0;
    }

    public void write(Writer w) throws IOException {
        wf(w, "%d \n", n);
        for (int v1 = 0; v1 < n; v1++) {
            for (int v2 = v1 + 1; v2 < n; v2++) {
                wf(w, "%d - %d - %.3f\n", v1, v2, Math.abs(value(v1, v2)));
            }
        }
        w.close();
    }

    public static UpperMatrix read(String s) throws IOException {
        BufferedReader br = new BufferedReader(getReader(s));
        int n = Integer.valueOf(br.readLine().trim());
        UpperMatrix u = new UpperMatrix(n);
        String l;

        while ((l = br.readLine()) != null) {
            String[] aux = l.split("-");

            u.inc(in(aux[0]), in(aux[1]), f(aux[2]));
        }
        return u;
    }

    private void inc(int v1, int v2, double f) {
        inc(index(v1, v2), f);
    }

    private static double f(String s) {
        return Float.valueOf(s.trim());
    }

    private static int in(String s) {
        return Integer.valueOf(s.trim());
    }
}
