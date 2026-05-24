package ch.idsia.blip.api.exp;

import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.io.bn.BnNetReader;
import ch.idsia.blip.core.io.bn.BnNetWriter;
import ch.idsia.blip.core.utils.other.BnGenerator;

public class Random {

    public static void main(String[] argv) throws Exception {
        new Random().test2();
    }

    public void test() {
        BnGenerator bngen = new BnGenerator(22);
        BayesianNetwork bn = bngen.create();

        bn.l_nm_var = new String[] {
                "GHTR",
                "FFT",
                "WEF",
                "EEG",
                "YUG",
                "F6H",
                "LBLevel",
                "QRLevel",
                "GRLevel",
                "MriScan",
                "KK9",
                "FSD",
                "Age",
                "DSWU",
                "P08",
                "D65T",
                "YUT",
                "LK7",
                "R43F",
                "EMG",
                "ECG",
                "Disease"
        };

        String h = "/home/loskana/Desktop/dm/ciao";
        bn.writeGraph(h);
        BnNetWriter.ex(bn, h+".net");

        bn = BnNetReader.ex(h+".net" );
    }

    public void test2() {
        String h = "/home/loskana/Desktop/";
        BayesianNetwork bn = BnNetReader.ex(h+"alarm.net");

        bn.l_nm_var = new String[] {
                "FFT",
                "WEF",

                "EEG",

                "GHTR",

                "LBLevel",

                "GRLevel",

                "LK7",

                "YUG",

                "F6H",

                "QRLevel",

                "ECG",

                "D65T",

                "DSWU",

                "KK9",

                "MriScan",

                "FSD",

                "R43F",

                "Age",

                "YUT",

                "P08",

                "EMG",

                "RG6",

                "Disease"
        };

        BnNetWriter.ex(bn, h+"alarm2.net");

    }
}
