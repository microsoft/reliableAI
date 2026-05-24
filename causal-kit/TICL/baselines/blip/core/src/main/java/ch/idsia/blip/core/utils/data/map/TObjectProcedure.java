package ch.idsia.blip.core.utils.data.map;


public interface TObjectProcedure<T> {

    /**
     * Executes this procedure. A false return value indicates that
     * the application executing this procedure should not invoke this
     * procedure again.
     *
     * @param object an <code>Object</code> value
     * @return true if additional invocations of the procedure are
     * allowed.
     */
    public boolean execute(T object);
}
