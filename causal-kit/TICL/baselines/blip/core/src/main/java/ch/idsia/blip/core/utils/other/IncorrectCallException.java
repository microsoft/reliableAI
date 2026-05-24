package ch.idsia.blip.core.utils.other;


/**
 * Exception raised when a method of the ch.idsia.blip.api is invoked via command line with  incorrect parameters.
 */
public class IncorrectCallException extends Exception {

    /**
     * Default constructor
     *
     * @param s problem message.
     */
    public IncorrectCallException(String s) {
        super(s);
    }
}
