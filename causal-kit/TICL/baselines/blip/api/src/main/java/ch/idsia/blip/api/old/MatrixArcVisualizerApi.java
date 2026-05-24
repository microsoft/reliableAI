package ch.idsia.blip.api.old;


import ch.idsia.blip.api.Api;
import ch.idsia.blip.core.common.MatrixArcVisualizer;
import org.kohsuke.args4j.Option;


public class MatrixArcVisualizerApi extends Api {

    @Option(name = "-m", required = true, usage = "Matrix arcs file path")
    private static String ph_mtx;

    @Option(name = "-d", required = true, usage = "Datafile path")
    private static String ph_dat;

    @Option(name = "-e", usage = "Epsilon value")
    private static Double eps = 1.0;

    @Option(name = "-o", required = true, usage = "Output path")
    private static String ph_out;

    public static void main(String[] args) {
        defaultMain(args, new MatrixArcVisualizerApi());
    }

    @Override
    public void exec() throws Exception {
        MatrixArcVisualizer.ex(ph_mtx, ph_dat, eps, ph_out);
    }

}
