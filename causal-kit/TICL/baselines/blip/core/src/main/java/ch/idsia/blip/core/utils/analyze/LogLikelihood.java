package ch.idsia.blip.core.utils.analyze;


import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.data.ArrayUtils;


public class LogLikelihood extends Analyzer {

    /**
     * Alpha for counts
     */
    double alpha = 1.0;

    /**
     * Num of variables
     */
    private int r;

    public LogLikelihood(DataSet dat) {
        super(dat);
    }

    public double computeLL(int x) {

        double ll = 0;

        double alpha = 1;
        double alpha_i = (alpha / dat.l_n_arity[x]);

        double p;
        int k;

        for (int v = 0; v < dat.l_n_arity[x]; v++) {
            k = dat.row_values[x][v].length;
            p = (k + alpha_i) / (dat.n_datapoints + alpha);

            ll += k * log(p);
        }

        return ll;
    }

    public double computeLL(int x, int y) {

        double mi = 0;

        int x_ar = dat.l_n_arity[x];
        int y_ar = dat.l_n_arity[y];

        for (int x_i = 0; x_i < x_ar; x_i++) {

            int[] r_x = dat.row_values[x][x_i];

            for (int y_i = 0; y_i < y_ar; y_i++) {

                int[] r_y = dat.row_values[y][y_i];
                double n_y = r_y.length; // + 1.0/y_ar);

                double n_xy = ArrayUtils.intersectN(r_x, r_y); // ;+ 1.0/(x_ar*y_ar);

                if (n_y == 0 || n_xy == 0) {
                    continue;
                }

                mi += n_xy * log(n_xy / n_y);

                // System.out.printf(" %.5f * log ( %.5f / %.5f) - %.5f * %.5f \n", p_xy, p_xy, p_x * p_y, p_xy, log(p_xy / (p_x * p_y)));
            }
        }

        // System.out.printf("mi: x %d, y %d -> %.5f\n", x, y, mi);

        return mi;
    }

}
