package ch.idsia.blip.core.utils;


import ch.idsia.blip.core.io.ScoreReader;
import ch.idsia.blip.core.io.bn.*;
import ch.idsia.blip.core.io.dat.*;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import ch.idsia.blip.core.utils.data.hash.TIntIntHashMap;
import ch.idsia.blip.core.utils.other.IncorrectCallException;
import ch.idsia.blip.core.utils.other.Worker;

import java.io.*;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;


@SuppressWarnings("ALL")
public class RandomStuff {

    private static final Logger log = Logger.getLogger(
            RandomStuff.class.getName());

    public static String rmExt(String str) {
        if (str.contains(".")) {
            return str.substring(0, str.lastIndexOf('.'));
        } else {
            return str;
        }
    }

    public static Random getRandom() {
        return new Random(System.currentTimeMillis());
    }

    public static int index(int s, int v1, int v2) {
        if (v1 > v2) {
            int t = v1;

            v1 = v2;
            v2 = t;
        }

        int n = 0;

        for (int i = 0; i < v1; i++) {
            n += (s - i - 2);
        }
        n += (v2 - 1);
        return n;
    }

    public static int waitForProc(Process process, double timeout) {

        Worker worker = new Worker(process);

        worker.start();
        try {
            worker.join((long) timeout);
            if (worker.exit != null) {
                return worker.exit;
            }
        } catch (InterruptedException ex) {
            worker.interrupt();
            Thread.currentThread().interrupt();
            logExp(log, ex);
        } finally {
            process.destroy();
        }
        return -1;
    }

    public static <T extends Comparable<? super T>> List<T> asSortedList(Collection<T> c) {
        List<T> list = new ArrayList<T>(c);

        java.util.Collections.sort(list);
        return list;
    }

    public static <K, V extends Comparable> Map<K, V> sortByValues(Map<K, V> map) {
        List<Map.Entry<K, V>> entries = new LinkedList<Map.Entry<K, V>>(
                map.entrySet());

        Collections.sort(entries, new Comparator<Map.Entry<K, V>>() {

            public int compare(Map.Entry<K, V> o1, Map.Entry<K, V> o2) {
                return o1.getValue().compareTo(o2.getValue());
            }
        });

        // LinkedHashMap will keep the keys in the order they are inserted
        // which is currently sorted on natural ordering
        Map<K, V> sortedMap = new LinkedHashMap<K, V>();

        for (Map.Entry<K, V> entry : entries) {
            sortedMap.put(entry.getKey(), entry.getValue());
        }

        return sortedMap;
    }

    public static <K, V extends Comparable> List<K> sortByValuesList(Map<K, V> map) {
        List<Map.Entry<K, V>> entries = new LinkedList<Map.Entry<K, V>>(
                map.entrySet());

        Collections.sort(entries, new Comparator<Map.Entry<K, V>>() {

            public int compare(Map.Entry<K, V> o1, Map.Entry<K, V> o2) {
                return o1.getValue().compareTo(o2.getValue());
            }
        });

        // LinkedHashMap will keep the keys in the order they are inserted
        // which is currently sorted on natural ordering
        List<K> sortedList = new ArrayList<K>();

        for (Map.Entry<K, V> entry : entries) {
            sortedList.add(entry.getKey());
        }

        return sortedList;
    }

    public static boolean doubleEquals(double s, double v) {
        return doubleEquals(s, v, 5);
    }

    public static boolean doubleEquals(double s, double v, int p) {
        if (!(Math.abs(s - v) < Math.pow(2, -p))) {
            return false;
        } else {
            return true;
        }
    }

    public static <K, V extends Comparable> Map<K, V> sortInvByValues(Map<K, V> map) {

        List<Map.Entry<K, V>> entries = new LinkedList<Map.Entry<K, V>>(
                map.entrySet());

        Collections.sort(entries, new Comparator<Map.Entry<K, V>>() {

            public int compare(Map.Entry<K, V> o1, Map.Entry<K, V> o2) {
                return -o1.getValue().compareTo(o2.getValue());
            }
        });

        // LinkedHashMap will keep the keys in the order they are inserted
        // which is currently sorted on natural ordering
        Map<K, V> sortedMap = new LinkedHashMap<K, V>();

        for (Map.Entry<K, V> entry : entries) {
            sortedMap.put(entry.getKey(), entry.getValue());
        }

        return sortedMap;
    }

    public static <K, V extends Comparable> List<K> sortInvByValuesList(Map<K, V> map) {
        List<Map.Entry<K, V>> entries = new LinkedList<Map.Entry<K, V>>(
                map.entrySet());

        Collections.sort(entries, new Comparator<Map.Entry<K, V>>() {

            public int compare(Map.Entry<K, V> o1, Map.Entry<K, V> o2) {
                return -o1.getValue().compareTo(o2.getValue());
            }
        });

        // LinkedHashMap will keep the keys in the order they are inserted
        // which is currently sorted on natural ordering
        List<K> sortedList = new ArrayList<K>();

        for (Map.Entry<K, V> entry : entries) {
            sortedList.add(entry.getKey());
        }

        return sortedList;
    }

    public static float listMeanL(List<? extends Number> list) {
        long sum = 0;

        for (Number l : list) {
            sum += l.doubleValue();
        }
        return sum / list.size();
    }

    public static float listMeanF(List<Float> list) {
        float sum = 0;

        for (float l : list) {
            sum += l;
        }
        return sum / list.size();
    }

    public static float listMeanD(List<Double> list) {
        float sum = 0;

        for (double l : list) {
            sum += l;
        }
        return sum / list.size();
    }

    public static float listMeanInt(List<Integer> list) {
        Integer sum = 0;

        for (Integer l : list) {
            sum += l;
        }
        return sum / list.size();
    }

    /**
     * Re-print a message on the current screen line
     *
     * @param msg Useful message for the user
     */
    public static void printR(String msg) {
        System.out.print('\r' + msg);
    }

    public static long factorial(int n) {
        long fact = 1; // this  will be the result

        for (int i = 1; i <= n; i++) {
            fact *= i;
        }
        return fact;
    }

    public static String printOrdMap(Map<Integer, Integer> map) {
        String stri = "{ ";
        boolean first = true;
        List<Integer> keys = new ArrayList<Integer>(map.keySet());

        Collections.sort(keys);
        for (int key : keys) {
            if (first) {
                first = false;
            } else {
                stri += ", ";
            }

            stri += String.format("%d: %d", key, map.get(key));
        }
        stri += " }";
        return stri;
    }

    public static boolean different(TIntArrayList v1, TIntArrayList v2) {
        for (int e1 : v1.toArray()) {
            if (!v2.contains(e1)) {
                return true;
            }
        }
        for (int e2 : v2.toArray()) {
            if (!v1.contains(e2)) {
                return true;
            }
        }

        return false;
    }

    public static void logExp(Throwable trw) {
        log.log(Level.SEVERE, trw.toString(), trw);
    }

    public static void logExp(Logger log, Throwable trw) {
        log.log(Level.SEVERE, trw.toString(), trw);
    }

    public static String readFile(String s) throws IOException {

        File f = new File(s);

        StringBuilder fileContents = new StringBuilder((int) f.length());
        Scanner scanner = new Scanner(f);
        String lineSeparator = System.getProperty("line.separator");

        try {
            while (scanner.hasNextLine()) {
                fileContents.append(scanner.nextLine() + lineSeparator);
            }
            return fileContents.toString();
        } finally {
            scanner.close();
        }
    }

    public static DataSet getDataSet(String dat_path) {
        return getDataSet(dat_path, false);
    }

    /**
     * Get a new data file reader from the argument
     *
     * @param ph path to the file
     * @return data file reader
     * @throws IncorrectCallException        file path not valid
     * @throws java.io.FileNotFoundException data file not found
     */
    public static DataSet getDataSet(String ph, boolean readMissing) {

        try {
            BufferedReader reader;

            if (ph == null) {
                throw new IncorrectCallException(
                        "No data point path provided: " + ph);
            }
            File f_dat = new File(ph);

            if (!f_dat.isFile()) {
                throw new IncorrectCallException(
                        "No valid data point path provided: " + ph);
            }

            DatFileReader dr;

            if (ph.endsWith(".data")) {
                dr = new DataFileReader();
            } else if (ph.endsWith("dat")) {
                dr = new DatFileReader();
            } else if (ph.endsWith("arff")) {
                dr = new ArffFileReader();
            } else {
                dr = new AnyFileReader();
            }

            dr.init(ph, readMissing);
            return dr.read();
        } catch (Exception ex) {
            logExp(log, ex);
        }

        return null;
    }

    public static void writeDataSet(DataSet dat, String ph) {

        try {
            DatFileWriter dWr = null;

            if (ph.endsWith(".data")) {
                dWr = new DataFileWriter();
            } else if (ph.endsWith("dat")) {
                dWr = new DatFileWriter();
            } else if (ph.endsWith("arff")) {
                dWr = new ArffFileWriter();
            }

            dWr.go(dat, ph);
        } catch (Exception ex) {
            logExp(log, ex);
        }
    }

    public static BaseFileLineReader getDataSetReader(String ph) {

        try {
            BufferedReader reader;

            if (ph == null) {
                throw new IncorrectCallException(
                        "No data point path provided: " + ph);
            }
            File f_dat = new File(ph);

            if (!f_dat.isFile()) {
                throw new IncorrectCallException(
                        "No valid data point path provided: " + ph);
            }

            if (ph.endsWith(".data")) {
                return new DataFileLineReader(ph);
            } else if (ph.endsWith("dat")) {
                return new DatFileLineReader(ph);
            } else if (ph.endsWith("arff")) {
                return new ArffFileLineReader(ph);
            } else {
                return new AnyFileLineReader(ph);
            }
        } catch (Exception ex) {
            logExp(log, ex);
        }

        return null;
    }

    public static DataSet getDataFromArff(String ph) {

        try {
            BufferedReader reader;

            if (ph == null) {
                throw new IncorrectCallException(
                        "No data point path provided: " + ph);
            }
            File f_dat = new File(ph);

            if (!f_dat.isFile()) {
                throw new IncorrectCallException(
                        "No valid data point path provided: " + ph);
            }

            ArffFileReader rd = new ArffFileReader();

            rd.init(ph, true);

            return rd.read();
        } catch (Exception ex) {
            logExp(log, ex);
        }

        return null;
    }

    public static Writer getWriter(String ph) {
        return getWriter(ph, false);
    }
    /**
     * Get writer for the output scores
     *
     * @param ph (optional) file path
     * @return writer to use to output result
     * @throws java.io.UnsupportedEncodingException error in stream creation
     * @throws java.io.FileNotFoundException        file path not valid
     */
    public static Writer getWriter(String ph, boolean suppress) {
        Writer writer = null;

        try {

            if (ph != null) {
                writer = new BufferedWriter(
                        new OutputStreamWriter(new FileOutputStream(ph), "utf-8"));
            } else {
                writer = new BufferedWriter(new OutputStreamWriter(System.out));
            }
        } catch (Exception e) {
            if (!suppress)
            logExp(e);
        }

        return writer;
    }

    public static BufferedReader getReader(String format, Object... args) throws FileNotFoundException {
        return getReader(f(format, args));
    }

    public static BufferedReader getReader(String ph) throws FileNotFoundException {
        return new BufferedReader(new FileReader(ph));
    }

    /**
     * Get directory
     *
     * @param ph dir path
     * @return directory
     * @throws java.io.UnsupportedEncodingException error in stream creation
     * @throws java.io.FileNotFoundException        file path not valid
     */
    public static File getDirectory(String ph) throws IncorrectCallException {

        if (ph == null) {
            throw new IncorrectCallException("No path provided: " + ph);
        }
        File f = new File(ph);

        if (!(f.exists() && f.isDirectory())) {
            throw new IncorrectCallException("No valid path provided: " + ph);
        }
        return f;
    }

    public static ParentSet[][] getScoreReader(String ph) throws IncorrectCallException, IOException {
        return getScoreReader(ph, 0);
    }

    /**
     * Get a new score file reader from the argument
     *
     * @param ph path to the file
     * @return score file reader
     * @throws IncorrectCallException        file path not valid
     * @throws java.io.FileNotFoundException if file is not found
     */
    public static ParentSet[][] getScoreReader(String ph, int debug)
        throws IncorrectCallException, IOException {

        if (ph == null) {
            throw new IncorrectCallException("No score path provided: " + ph);
        }

        if (!new File(ph).exists()) {
            throw new IncorrectCallException("Score file not found: " + ph);
        }

        ScoreReader s = new ScoreReader(ph, 10);

        s.readScores();

        return s.m_scores;
    }

    public static void closeIt(Logger log, Closeable writer) {
        try {
            if (writer != null) {
                writer.close();
            }
        } catch (Exception exp) {
            logExp(log, exp);
        }
    }

    public static void checkPath(String ph) throws IncorrectCallException, IOException {
        if (ph == null) {
            throw new IncorrectCallException("No path provided: " + ph);
        }
        File f = new File(ph);

        if (!f.createNewFile()) {
            throw new IncorrectCallException(
                    "Impossible to create path provided: " + ph);
        }
        f.delete();
    }

    public static <K, V extends Comparable<? super V>>
            SortedSet<Map.Entry<K, V>> entriesSortedByValues(Map<K, V> map) {
        return entriesSortedByValues(map, false);
    }

    public static <K, V extends Comparable<? super V>>
            SortedSet<Map.Entry<K, V>> entriesSortedByValues(Map<K, V> map, final boolean invert) {
        SortedSet<Map.Entry<K, V>> sortedEntries = new TreeSet<Map.Entry<K, V>>(
                new Comparator<Map.Entry<K, V>>() {

            public int compare(Map.Entry<K, V> e1, Map.Entry<K, V> e2) {
                int res = e1.getValue().compareTo(e2.getValue());

                if (invert) {
                    res = -res;
                }
                return res != 0 ? res : 1;
            }
        });

        sortedEntries.addAll(map.entrySet());
        return sortedEntries;
    }

    public static <K, V> void p(SortedSet<Map.Entry<K, V>> entries) {
        for (Map.Entry<K, V> entry : entries) {
            pf("%s - %s \n", entry.getKey().toString(),
                    entry.getValue().toString());
        }
    }

    public static <K extends Comparable<? super K>, V>
            SortedSet<Map.Entry<K, V>> entriesSortedByKey(Map<K, V> map) {
        return entriesSortedByKey(map, false);
    }

    public static <K extends Comparable<? super K>, V>
            SortedSet<Map.Entry<K, V>> entriesSortedByKey(Map<K, V> map, final boolean b) {
        SortedSet<Map.Entry<K, V>> sortedEntries = new TreeSet<Map.Entry<K, V>>(
                new Comparator<Map.Entry<K, V>>() {

            public int compare(Map.Entry<K, V> e1, Map.Entry<K, V> e2) {
                int res = e1.getKey().compareTo(e2.getKey());

                if (b) {
                    res = -res;
                }
                return res != 0 ? res : 1;
            }
        });

        sortedEntries.addAll(map.entrySet());
        return sortedEntries;
    }

    static public String cmd(String cmd) throws IOException {

        // p(cmd);

        Process proc = Runtime.getRuntime().exec(cmd);

        BufferedReader stdInput = new BufferedReader(new
                InputStreamReader(proc.getInputStream()));

        BufferedReader stdError = new BufferedReader(new
                InputStreamReader(proc.getErrorStream()));

        String s = null;
        StringBuilder out = new StringBuilder();

        while ((s = stdInput.readLine()) != null) {
            out.append(s);
        }

        // read any errors from the attempted command
        StringBuilder err = new StringBuilder();
        boolean e = false;

        while ((s = stdError.readLine()) != null) {
            err.append(s);
            System.out.println(s);
            e = true;
        }

        if (e) {
            System.out.println("errors: " + err);
        }

        return out.toString();
    }

    static public void cmd(ProcessBuilder pb, String s) throws IOException {
        pb.directory(new File(s));

        // redirect stdout, stderr, etc
        procOutput(pb.start());
    }

    static public void cmdOutput(String cmd) throws IOException {

        // read the output from the command
        System.out.println("cmd: " + cmd + "\noutput: ");

        procOutput(Runtime.getRuntime().exec(cmd));
    }

    static public void procOutput(Process proc) throws IOException {
        BufferedReader stdInput = new BufferedReader(new
                InputStreamReader(proc.getInputStream()));

        BufferedReader stdError = new BufferedReader(new
                InputStreamReader(proc.getErrorStream()));

        String s = null;

        while ((s = stdInput.readLine()) != null) {
            System.out.println(s);
        }

        // read any errors from the attempted command
        // System.out.printf("errors: ");
        StringBuilder st = new StringBuilder();

        while ((s = stdError.readLine()) != null) {
            st.append(s);
        }

        if (st.length() > 0) {
            pf("errors: %s \n", st.toString());
        }
    }

    public static void pf(String format, Object... args) {
        System.out.printf(format, args);
    }

    public static String f(String format, Object... args) {
        return String.format(format, args);
    }

    public static void wf(Writer writer, String format, Object... args) throws IOException {
        writer.write(String.format(format, args));
    }

    public static void p(String s) {
        System.out.println(s);
    }

    public static void p(Object s) {
        System.out.println(String.valueOf(s));
    }

    public static void p(int[] s) {
        System.out.println(Arrays.toString(s));
    }

    public static boolean exists(String s) {
        File f = new File(s);

        return f.exists() && !f.isDirectory();
    }

    public static void deleteFolder(File folder) {
        File[] files = folder.listFiles();

        if (files != null) { // some JVMs return null for empty dirs
            for (File f : files) {
                if (f.isDirectory()) {
                    deleteFolder(f);
                } else {
                    f.delete();
                }
            }
        }
        folder.delete();
    }

    public static void copyDirectory(File src, File trg) {
        try {
            if (src.isDirectory()) {
                if (!trg.exists()) {
                    trg.mkdir();
                }

                for (File f : src.listFiles()) {
                    String s = f.getName();

                    if (f.isFile()) {
                        copySomething(new File(src, s), new File(trg, s));
                    } else {
                        copyDirectory(new File(src, s), new File(trg, s));
                    }

                }
                // String[] children = sourceLocation.listFiles();
                // for (int thread = 0; thread < children.length; thread++) {
                // copyDirectory(new File(sourceLocation, children[thread]),
                // new File(targetLocation, children[thread]));

            } else {

                copySomething(src, trg);
            }
        } catch (IOException ex) {
            logExp(log, ex);
        }
    }

    public static void copySomething(String f1, String f2) throws IOException {
        copySomething(new File(f1), new File(f2));
    }

    public static void copySomething(File f1, File f2) throws IOException {
        InputStream in = new FileInputStream(f1);
        OutputStream out = new FileOutputStream(f2);

        // Copy the bits from instream to outstream
        byte[] buf = new byte[1024];
        int len;

        while ((len = in.read(buf)) > 0) {
            out.write(buf, 0, len);
        }
        in.close();
        out.close();
    }

    public static BayesianNetwork getBayesianNetwork(String s_bn) {

        BufferedReader rd_bn = null;

        try {
            rd_bn = new BufferedReader(new FileReader(s_bn));
        } catch (FileNotFoundException e) {
            p("Warning! Bayesian network not found! " + s_bn);
            return null;
        }

        BayesianNetwork bn = null;

        if (s_bn.endsWith(".net")) {
            bn = BnNetReader.ex(rd_bn);
        } else if (s_bn.endsWith(".uai")) {
            bn = BnUaiReader.ex(rd_bn);
        } else if (s_bn.endsWith(".erg")) {
            bn = BnErgReader.ex(rd_bn);
        } else {
            bn = BnResReader.ex(rd_bn);
        }
        return bn;
    }

    public static void writeBayesianNetwork(BayesianNetwork bn, String path) {

        if (path.endsWith(".uai")) {
            BnUaiWriter.ex(path, bn);
        } else if (path.endsWith(".erg")) {
            BnErgWriter.ex(path, bn);
        } else {
            BnNetWriter.ex(bn, path);
        }

    }

    public static String clean(String s) {
        return s.trim().toLowerCase();
    }

    public static void deleteDir(File file) {

        File[] contents = file.listFiles();

        if (contents != null) {
            for (File f : contents) {
                deleteDir(f);
            }
        }
        file.delete();

    }

    public static ArrayList<String> exec(Process proc) throws IOException, InterruptedException {
        InputStream stdin = proc.getInputStream();
        InputStreamReader isr = new InputStreamReader(stdin);
        BufferedReader br = new BufferedReader(isr);

        String line = null;
        ArrayList<String> o = new ArrayList<String>();

        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.length() > 0) {
                o.add(line);
            }
        }

        int exitVal = proc.waitFor();

        // System.out.println("Process exitValue: " + exitVal);

        return o;
    }

    private static boolean incrementPset(int[] pset, int i, int n_var) {

        if (i < 0) {
            return false;
        }

        // Try to increment set at position thread
        pset[i]++;

        // Check if we have to backtrack
        if (pset[i] > (n_var - (pset.length - i))) {
            boolean cnt = incrementPset(pset, i - 1, n_var);

            if (cnt) {
                pset[i] = pset[i - 1] + 1;
            }
            return cnt;
        }

        return true;
    }

    public static Double ijgp(String bn_path, short[] sample) throws IOException, InterruptedException {

        String evid_path = bn_path + ".evid";

        // Write evidence
        PrintWriter w = new PrintWriter(evid_path, "UTF-8");

        // wf(w, "/* Evidence */ \n");
        wf(w, "%d ", sample.length);
        for (int j = 0; j < sample.length; j++) {
            wf(w, "%d %d ", j, sample[j]);
        }
        w.close();

        return cmpIjgp(bn_path, evid_path);
    }

    public static Double ijgp(String bn_path, TIntIntHashMap a) throws IOException, InterruptedException {

        String evid_path = bn_path + ".evid";

        // Write evidence
        PrintWriter w = new PrintWriter(evid_path, "UTF-8");

        // wf(w, "/* Evidence */ \n");
        wf(w, "%d ", a.size());
        int[] keys = a.keySet().toArray();

        Arrays.sort(keys);
        for (int k : keys) {
            wf(w, "%d %d ", k, a.get(k));
        }
        w.close();

        return cmpIjgp(bn_path, evid_path);
    }

    private static Double cmpIjgp(String bn_path, String evid_path) throws IOException, InterruptedException {
        String r = f("./ijgp  %s %s 10 PR", bn_path, evid_path);
        String p = System.getProperty("user.home") + "/Tools";
        Process proc = Runtime.getRuntime().exec(r, new String[0], new File(p));
        ArrayList<String> out = exec(proc);

        // To close them
        proc.getInputStream().close();
        proc.getOutputStream().close();
        proc.getErrorStream().close();

        String bn_name = new File(bn_path).getName();

        String res = f("%s/%s.PR", p, bn_name);
        BufferedReader rd = getReader(res);

        rd.readLine();
        bn_name = rd.readLine();
        rd.close();

        double v = 0;

        try {
            v = Double.valueOf(bn_name);
        } catch (Exception e) {
            logExp(log, e);
        }

        new File(res).delete();
        return v;
    }

    public static void cloneStr(ParentSet[] a, ParentSet[] b) {
        System.arraycopy(a, 0, b, 0, a.length);
    }

    public static String[] getContent(String s) throws IOException {
        BufferedReader r = getReader(s);
        String l;
        String cnt = "";
        while ((l = r.readLine()) != null) {
            cnt += l + " ";
        }
        return cnt.trim().split("\\s+");
    }


    public static TIntIntHashMap getOutput(ArrayList<String> out, TIntArrayList q) throws IOException {
        String output = out.get(out.size() - 1);
        String[] aux = output.split(" ");

        TIntIntHashMap res = new TIntIntHashMap();

        for (int i = 0; i < q.size(); i++) {
            int n = q.get(i);

            res.put(n, Short.valueOf(aux[(n + 3)]));
        }

        return  res;
    }


    public static void writeQuery(String qu, TIntArrayList q) throws IOException {
        Writer w = getWriter(qu);
        wf(w, "%d", q.size());
        for (int i = 0; i < q.size(); i++)
            wf(w, " %d", q.get(i));
        w.close();
    }

    public static void writeEvidence(String ev, TIntIntHashMap e) throws IOException {
        Writer w = getWriter(ev);
        wf(w, "%d", e.keys().length);
        for (int i = e.keys().length -1 ; i >= 0 ; i--) {
            int k = e.keys()[i];
            wf(w, " %d %d", k, e.get(k));
        }
        w.close();
    }
}
