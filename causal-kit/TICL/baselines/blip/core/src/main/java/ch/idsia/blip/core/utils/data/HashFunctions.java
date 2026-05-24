package ch.idsia.blip.core.utils.data;


public final class HashFunctions {

    /**
     * Returns a hashcode for the specified value.
     *
     * @return a hash code value for the specified value.
     */
    public static int hash(double value) {
        assert !Double.isNaN(value) : "Values of NaN are not supported.";

        long bits = Double.doubleToLongBits(value);

        return (int) (bits ^ (bits >>> 32));
        // return (int) Double.doubleToLongBits(value*663608941.737);
        // this avoids excessive hashCollisions in the case values are
        // of the form (1.0, 2.0, 3.0, ...)
    }

    /**
     * Returns a hashcode for the specified value.
     *
     * @return a hash code value for the specified value.
     */
    public static int hash(float value) {
        assert !Float.isNaN(value) : "Values of NaN are not supported.";

        return Float.floatToIntBits(value * 663608941.737f);
        // this avoids excessive hashCollisions in the case values are
        // of the form (1.0, 2.0, 3.0, ...)
    }

    /**
     * Returns a hashcode for the specified value.
     *
     * @return a hash code value for the specified value.
     */
    public static int hash(int value) {
        return value;
    }

    /**
     * Returns a hashcode for the specified value.
     *
     * @return a hash code value for the specified value.
     */
    public static int hash(long value) {
        return ((int) (value ^ (value >>> 32)));
    }

    /**
     * Returns a hashcode for the specified object.
     *
     * @return a hash code value for the specified object.
     */
    public static int hash(Object object) {
        return object == null ? 0 : object.hashCode();
    }

    /**
     * In profiling, it has been found to be faster to have our own local implementation
     * of "ceil" rather than to call to {@link Math#ceil(double)}.
     */
    public static int fastCeil(float v) {
        int possible_result = (int) v;

        if (v - possible_result > 0) {
            possible_result++;
        }
        return possible_result;
    }
}
