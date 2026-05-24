package ch.idsia.blip.core.utils.other;


import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.analyze.MutualInformation;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.getRandom;


public class DivConq {

    /**
     * Logger
     */
    private static final Logger log = Logger.getLogger(DivConq.class.getName());

    /**
     * Mutual Information computer
     */
    public MutualInformation mi;

    /**
     * Number of cluster
     */
    public int k = 3;

    /**
     * Verbosity flag
     */
    public int verbose;

    /**
     * Random generator
     */
    private Random rand;

    /**
     * Input datapoint file
     */
    private DataSet dat;

    private DivConq(int k) {
        this.k = k;
    }

    public DivConq() {}

    // Find the tw-medoids from the variables.
    public List<TIntArrayList> findKMedoids(DataSet dat) throws IOException {

        // Prepare structures
        mi = new MutualInformation(dat);
        mi.compute();

        rand = getRandom();

        List<TIntArrayList> best_clu = null;
        Double best_sk = 0.0;

        for (int i = 0; i < dat.n_var; i++) {
            Pair<List<TIntArrayList>, Double> res = tryOneRun();

            if (verbose > 1) {
                System.out.println(res.getSecond() + " &&& ");
                for (TIntArrayList cl : res.getFirst()) {
                    System.out.print(cl + " ");
                }
                System.out.println();
            }

            if (res.getSecond() < best_sk) {
                best_clu = res.getFirst();
                best_sk = res.getSecond();
            }
        }

        return best_clu;
    }

    private Pair<List<TIntArrayList>, Double> tryOneRun() {
        List<TIntArrayList> cluster = new ArrayList<TIntArrayList>();
        TIntArrayList medoids = new TIntArrayList();

        // Initialize
        while (medoids.size() != this.k) {
            int m = rand.nextInt(dat.n_var);

            if (!medoids.contains(m)) {
                medoids.add(m);
            }
        }

        if (verbose > 2) {
            System.out.println("Initial medoids " + medoids);
        }

        boolean change = true;
        int i = 0;

        while (change) {
            if (verbose > 2) {
                System.out.println("\n### Iter " + i++);
            }

            // Clean cluster
            cluster.clear();
            for (int j = 0; j < this.k; j++) {
                cluster.add(new TIntArrayList());
            }

            // Associate each data point to the closest medoid
            for (int v = 0; v < dat.n_var; v++) {
                int c = -1;
                double min_d = Double.MAX_VALUE;

                // For every medoids
                for (int j = 0; j < this.k; j++) {
                    double d = mi.distance(medoids.get(j), v);

                    if (d < min_d) {
                        min_d = d;
                        c = j;
                    }
                }

                cluster.get(c).add(v);
            }

            if (verbose > 2) {
                System.out.println("Clusters " + cluster);
            }

            change = false;
            // For every medoid
            for (int j = 0; j < this.k; j++) {
                int m = medoids.get(j);

                for (int v = 0; v < dat.n_var; v++) {
                    // Select only non-medoids
                    if (medoids.contains(v)) {
                        continue;
                    }

                    if (totalDistance(v) < totalDistance(m)) {
                        medoids.set(j, v);
                        change = true;
                    }
                }
            }

            if (verbose > 2) {
                System.out.println("Medoids " + medoids);
            }
        }

        double tot = 0;

        for (int m : medoids.toArray()) {
            tot += totalDistance(m);
        }

        return new Pair<List<TIntArrayList>, Double>(cluster, tot);
    }

    /**
     * Compute total distance from medoid
     *
     * @param j medoid variable index
     * @return total distance from all the dand nodes
     */
    private double totalDistance(int j) {
        double d = 0;

        for (int v = 0; v < dat.n_var; v++) {
            d += mi.distance(j, v);
        }
        return d;
    }

}
