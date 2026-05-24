package ch.idsia.blip.core.io.dat;


import ch.idsia.blip.core.utils.RandomStuff;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.logging.Logger;


public class ArffFileLineReader extends BaseFileLineReader {

    private static final Logger log = Logger.getLogger(
            ArffFileLineReader.class.getName());

    private String relation;
    public ArrayList<HashMap<String, Integer>> values;
    private int[] l_n_arity;

    public String[][] l_v_names;

    public ArffFileLineReader(String s)
        throws FileNotFoundException {
        super(s);
    }

    protected String[] getSplit(String ln) {
        return ln.split(",");
    }

    public short[] next() {

        try {
            int i = 0;

            for (String aux : getSplit(this.nextLine)) {
                if ("?".equals(aux)) {
                    samp[i] = -1;
                } else {
                    int v = (Integer) ((HashMap) this.values.get(i)).get(aux);

                    samp[i] = ((short) v);
                }
                i++;
            }
            this.n_datapoints += 1;

            this.nextLine = this.rd_dat.readLine();
            if (this.nextLine == null) {
                this.concluded = true;
            } else {
                this.nextLine = this.nextLine.trim();
                if (this.nextLine.length() == 0) {
                    this.concluded = true;
                }
            }
        } catch (IOException e) {
            RandomStuff.logExp(log, e);
        }
        return samp;
    }

    public boolean readMetaData() {
        try {
            this.nextLine = readLn();
            if (!this.nextLine.startsWith("@relation")) {
                RandomStuff.p("ERROR! Arff file does not start with relation");
            }
            this.relation = splitSpace(this.nextLine)[1];
            this.nextLine = readLn();

            this.n_var = 0;
            ArrayList<String> names = new ArrayList<String>();
            ArrayList<String[]> v_aux = new ArrayList<String[]>();

            while (!this.nextLine.equals("@data")) {
                while (this.nextLine.equals("")) {
                    this.nextLine = readLn();
                }
                if (!this.nextLine.startsWith("@attribute ")) {
                    RandomStuff.p(
                            "ERROR! Arff file does not start with relation");
                }
                String ln = this.nextLine.replace("@attribute ", "");
                String value;
                String name;

                if (ln.startsWith("'")) {
                    ln = ln.substring(1);
                    int g = ln.indexOf("'");

                    name = ln.substring(0, g);
                    value = ln.substring(g + 1);
                } else {
                    int g = ln.indexOf(" ");

                    name = ln.substring(0, g);
                    value = ln.substring(g);
                }
                names.add(name);
                String[] v = getSplit(
                        value.replace("{", "").replace("}", "").trim());
                String[] va = new String[v.length];

                for (int j = 0; j < v.length; j++) {
                    va[j] = v[j].trim();
                }
                v_aux.add(va);

                this.n_var += 1;

                this.nextLine = readLn();
            }
            this.n_var = this.n_var;
            this.l_s_names = new String[this.n_var];
            this.l_n_arity = new int[this.n_var];
            this.l_v_names = new String[this.n_var][];

            this.values = new ArrayList(this.n_var);
            for (int i = 0; i < this.n_var; i++) {
                this.l_s_names[i] = ( names.get(i));
                this.l_n_arity[i] = (v_aux.get(i)).length;

                this.l_v_names[i] = ( v_aux.get(i));

                HashMap<String, Integer> h = new HashMap();

                for (int j = 0; j < this.l_n_arity[i]; j++) {
                    h.put(this.l_v_names[i][j], Integer.valueOf(j));
                }
                this.values.add(h);
            }
            while ((this.nextLine.equals("")) || (this.nextLine.equals("@data"))) {
                this.nextLine = readLn();
            }


            samp = new short[n_var];
        } catch (IOException e) {
            RandomStuff.logExp(log, e);
            return true;
        }
        return false;
    }
}
