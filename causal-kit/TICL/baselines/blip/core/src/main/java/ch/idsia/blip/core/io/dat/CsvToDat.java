package ch.idsia.blip.core.io.dat;


import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class CsvToDat {

    private static final Logger log = Logger.getLogger(CsvToDat.class.getName());

    private ArrayList<int[]> data;

    private List<List<String>> values;

    private int n_var;

    public static void go(String input, String output) {
        try {
            CsvToDat inst = new CsvToDat();

            inst.readAll(input);
            inst.write(output);
        } catch (Exception e) {
            logExp(log, e);
        }
    }

    private void write(String output) throws IOException {

        BufferedWriter wr = new BufferedWriter(
                new OutputStreamWriter(new FileOutputStream(output), "utf-8"));

        wf(wr, f("%d\n", n_var));

        for (int i = 0; i < n_var; i++) {
            if (i != 0) {
                wf(wr, " ");
            }
            wf(wr, "N" + i);
        }

        wf(wr, "\n");

        for (int i = 0; i < n_var; i++) {
            if (i != 0) {
                wf(wr, " ");
            }
            wf(wr, f("%d", values.get(i).size()));
        }

        wf(wr, f("\n%d\n", data.size()));

        for (int[] aData : data) {
            for (int i = 0; i < n_var; i++) {
                if (i != 0) {
                    wf(wr, " ");
                }
                wf(wr, f("%d", aData[i]));
            }
            wf(wr, "\n");
        }

        wr.close();
    }

    private void readAll(String input) throws Exception {

        data = new ArrayList<int[]>();

        BufferedReader br = new BufferedReader(new FileReader(input));
        String strLine = null;
        StringTokenizer st = null;
        int lineNumber = 0, tokenNumber = 0;

        String fileName;

        while ((fileName = br.readLine()) != null) {
            lineNumber++;
            String[] result = fileName.split(",");

            if (n_var == 0) {
                n_var = result.length;
                values = new ArrayList<List<String>>();
                for (int i = 0; i < n_var; i++) {
                    values.add(new ArrayList<String>());
                }
            }
            if (n_var != result.length) {
                throw new Exception(
                        f(
                                "Different number of values! Found %d at first line, %d at line %d",
                                n_var, result.length, lineNumber));
            }

            int[] row = new int[n_var];

            for (int i = 0; i < result.length; i++) {
                String v = result[i];
                List<String> l_v = values.get(i);

                if (!l_v.contains(v)) {
                    l_v.add(v);
                }
                row[i] = l_v.indexOf(v);
            }
            data.add(row);
        }
    }
}
