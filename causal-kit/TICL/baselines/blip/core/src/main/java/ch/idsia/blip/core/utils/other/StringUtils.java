package ch.idsia.blip.core.utils.other;


import ch.idsia.blip.core.utils.data.array.TIntArrayList;

import java.util.List;

import static ch.idsia.blip.core.utils.RandomStuff.f;


public class StringUtils {

    private static final String EMPTY = "";

    public static String join(final double[] array) {
        if (array == null) {
            return null;
        }
        return join(array, " ", 0, array.length);
    }

    private static String join(double[] array, String s, int startIndex, int endIndex) {
        if (array == null) {
            return null;
        }
        final int noOfItems = endIndex - startIndex;

        if (noOfItems <= 0) {
            return EMPTY;
        }
        final StringBuilder buf = new StringBuilder(noOfItems * 16);

        for (int i = startIndex; i < endIndex; i++) {
            if (i > startIndex) {
                buf.append(s);
            }

            buf.append(f("%.4f", array[i]));

        }
        return buf.toString();
    }

    public static String join(final String[] array) {
        if (array == null) {
            return null;
        }
        return join(array, " ", 0, array.length);
    }

    public static String join(final String[] array, String separator) {
        if (array == null) {
            return null;
        }
        return join(array, separator, 0, array.length);
    }

    private static String join(final String[] array, final String separator, final int startIndex, final int endIndex) {
        if (array == null) {
            return null;
        }
        final int noOfItems = endIndex - startIndex;

        if (noOfItems <= 0) {
            return EMPTY;
        }
        final StringBuilder buf = new StringBuilder(noOfItems * 16);

        for (int i = startIndex; i < endIndex; i++) {
            if (i > startIndex) {
                buf.append(separator);
            }
            if (array[i] != null) {
                buf.append(array[i]);
            }
        }
        return buf.toString();
    }

    public static String join(final List<? extends Object> array) {
        if (array == null) {
            return null;
        }
        return join(array, " ", 0, array.size());
    }

    public static String join(final List<? extends Object> array, String separator) {
        if (array == null) {
            return null;
        }
        return join(array, separator, 0, array.size());
    }

    private static String join(final List<? extends Object> array, String separator, final int startIndex, final int endIndex) {
        if (array == null) {
            return null;
        }
        final int noOfItems = endIndex - startIndex;

        if (noOfItems <= 0) {
            return EMPTY;
        }
        final StringBuilder buf = new StringBuilder(noOfItems * 16);

        for (int i = startIndex; i < endIndex; i++) {
            if (i > startIndex) {
                buf.append(separator);
            }
            if (array.get(i) != null) {
                buf.append(String.valueOf(array.get(i)));
            }
        }
        return buf.toString();
    }

    public static String join(final TIntArrayList array) {
        if (array == null) {
            return null;
        }
        return join(array, " ", 0, array.size());
    }

    public static String join(TIntArrayList array, String separator) {
        if (array == null) {
            return null;
        }
        return join(array, separator, 0, array.size());
    }

    private static String join(final TIntArrayList array, String separator, final int startIndex, final int endIndex) {
        if (array == null) {
            return null;
        }
        final int noOfItems = endIndex - startIndex;

        if (noOfItems <= 0) {
            return EMPTY;
        }
        final StringBuilder buf = new StringBuilder(noOfItems * 16);

        for (int i = startIndex; i < endIndex; i++) {
            if (i > startIndex) {
                buf.append(separator);
            }

            buf.append(String.valueOf(array.get(i)));

        }
        return buf.toString();
    }

    public static String join(final int[] array) {
        if (array == null) {
            return null;
        }
        return join(array, " ", 0, array.length);
    }

    public static String join(int[] array, String separator) {
        if (array == null) {
            return null;
        }
        return join(array, separator, 0, array.length);
    }

    private static String join(final int[] array, String separator, final int startIndex, final int endIndex) {
        if (array == null) {
            return null;
        }
        final int noOfItems = endIndex - startIndex;

        if (noOfItems <= 0) {
            return EMPTY;
        }
        final StringBuilder buf = new StringBuilder(noOfItems * 16);

        for (int i = startIndex; i < endIndex; i++) {
            if (i > startIndex) {
                buf.append(separator);
            }

            buf.append(String.valueOf(array[i]));

        }
        return buf.toString();
    }
}
