package cs224n.deep;

import java.lang.*;
import org.ejml.simple.SimpleMatrix;

/**
 * An objective function takes an input vector and gives an answer.
 * @author Chris Billovits
 */
public interface ObjectiveFunction {

    /**
     * Evaluates an objective function on top of a Matrix vector input.
     */   
    public double valueAt(SimpleMatrix input);

}
